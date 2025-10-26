#!/usr/bin/env python3
"""
Historical Data Collector

This script manages large-scale historical data collection with database persistence,
pause/resume functionality, and progress tracking. It uses the trading bot's 
historical data service for actual data collection.

Key Features:
- Database persistence for progress tracking
- Pause/resume functionality
- Batch processing with progress monitoring
- Error handling and retry logic
- Integration with historical data service

Usage:
    python historical_data_collector.py [COMMAND] [OPTIONS]

Commands:
    create-job     Create a new data collection job
    run-job        Run/resume a data collection job
    pause-job      Pause a running job
    status         Show job status
    list-jobs      List all jobs
"""

import asyncio
import aiohttp
import json
import time
import argparse
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, and_, or_

# Import our common modules
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from common.config import get_settings
from common.db import get_db_session, execute_with_retry, initialize_database
from common.models import DataCollectionJob, DataCollectionSymbol, DataCollectionRequest
from common.logging import configure_service_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('historical_data_collection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HistoricalDataCollector:
    """Manages historical data collection with database persistence."""
    
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.settings = get_settings()
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def load_symbols_from_file(self, symbols_file: str) -> List[str]:
        """Load symbols from file."""
        if not Path(symbols_file).exists():
            raise FileNotFoundError(f"Symbols file not found: {symbols_file}")
        
        with open(symbols_file, 'r') as f:
            symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
        
        logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")
        return symbols
    
    def calculate_time_chunks(self, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime]]:
        """Calculate time chunks for the specified date range."""
        chunks = []
        current_time = start_date
        
        # Create 1-hour chunks (max duration for 5-second bars)
        while current_time < end_date:
            chunk_end = min(current_time + timedelta(hours=1), end_date)
            chunks.append((current_time, chunk_end))
            current_time = chunk_end
        
        logger.info(f"Created {len(chunks)} time chunks from {start_date} to {end_date}")
        return chunks
    
    def create_job(self, name: str, symbols: List[str], start_date: datetime, 
                   end_date: datetime, bar_size: str = "5 secs", 
                   what_to_show: str = "TRADES", use_rth: bool = True,
                   description: str = None) -> int:
        """Create a new data collection job in the database."""
        
        # Calculate total requests needed
        time_chunks = self.calculate_time_chunks(start_date, end_date)
        total_requests = len(symbols) * len(time_chunks)
        
        with get_db_session() as db:
            try:
                logger.info(f"Creating job record for: {name}")
                job = DataCollectionJob(
                    name=name,
                    description=description,
                    start_date=start_date,
                    end_date=end_date,
                    bar_size=bar_size,
                    what_to_show=what_to_show,
                    use_rth=use_rth,
                    status='pending',
                    total_symbols=len(symbols),
                    total_requests=total_requests
                )
                db.add(job)
                db.commit()
                db.refresh(job)
                logger.info(f"Job record created with ID: {job.id}")
            except Exception as e:
                logger.error(f"Error creating job record: {e}")
                raise
            
            # Create symbol records
            symbol_records = []
            try:
                logger.info(f"Creating {len(symbols)} symbol records")
                for symbol in symbols:
                    symbol_record = DataCollectionSymbol(
                        job_id=job.id,
                        symbol=symbol,
                        total_requests=len(time_chunks)
                    )
                    db.add(symbol_record)
                    symbol_records.append(symbol_record)
                
                # Commit symbols first to get their IDs
                db.commit()
                # Refresh all symbol records to ensure they have proper IDs
                for symbol_record in symbol_records:
                    db.refresh(symbol_record)
                logger.info(f"Symbol records committed successfully")
            except Exception as e:
                logger.error(f"Error creating symbol records: {e}")
                raise
            
            # Create request records
            try:
                logger.info(f"Creating request records for {len(symbol_records)} symbols")
                # Query symbols back from database to ensure they have proper IDs
                db_symbols = db.query(DataCollectionSymbol).filter(DataCollectionSymbol.job_id == job.id).all()
                logger.info(f"Retrieved {len(db_symbols)} symbols from database")
                
                request_count = 0
                for db_symbol in db_symbols:
                    logger.info(f"Processing symbol {db_symbol.symbol} (ID: {db_symbol.id})")
                    for start_time, end_time in time_chunks:
                        logger.info(f"Creating request for {db_symbol.symbol} from {start_time} to {end_time}")
                        logger.info(f"Types: job.id={type(job.id)}, symbol_id={type(db_symbol.id)}, symbol={type(db_symbol.symbol)}, start_time={type(start_time)}, end_time={type(end_time)}")
                        logger.info(f"Values: job.id={job.id}, symbol_id={db_symbol.id}, symbol={db_symbol.symbol}")
                        logger.info(f"About to create DataCollectionRequest object")
                        try:
                            request_record = DataCollectionRequest()
                            logger.info(f"Empty object created")
                            request_record.job_id = job.id
                            logger.info(f"Set job_id")
                            request_record.symbol_id = int(db_symbol.id)
                            logger.info(f"Set symbol_id")
                            request_record.symbol = db_symbol.symbol
                            logger.info(f"Set symbol")
                            request_record.start_time = start_time
                            logger.info(f"Set start_time")
                            request_record.end_time = end_time
                            logger.info(f"Set end_time")
                        except Exception as e:
                            logger.error(f"Error creating request record step by step: {e}")
                            raise
                        logger.info(f"Request record created, adding to session")
                        db.add(request_record)
                        request_count += 1
                        logger.info(f"Request count: {request_count}")
                
                logger.info(f"About to commit {request_count} request records")
                db.commit()
                logger.info(f"Created {request_count} request records successfully")
            except Exception as e:
                logger.error(f"Error creating request records: {e}")
                raise
            
            logger.info(f"Created job {job.id}: {name}")
            logger.info(f"  Symbols: {len(symbols)}")
            logger.info(f"  Total requests: {total_requests}")
            logger.info(f"  Date range: {start_date} to {end_date}")
            
            return job.id
    
    def get_job_status(self, job_id: int) -> Dict:
        """Get detailed status of a job."""
        with get_db_session() as db:
            job = db.query(DataCollectionJob).filter(DataCollectionJob.id == job_id).first()
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            # Get symbol statuses
            symbols = db.query(DataCollectionSymbol).filter(DataCollectionSymbol.job_id == job_id).all()
            
            # Get request statuses
            requests = db.query(DataCollectionRequest).filter(DataCollectionRequest.job_id == job_id).all()
            
            return {
                "job_id": job.id,
                "name": job.name,
                "description": job.description,
                "status": job.status,
                "start_date": job.start_date.isoformat(),
                "end_date": job.end_date.isoformat(),
                "bar_size": job.bar_size,
                "total_symbols": job.total_symbols,
                "completed_symbols": job.completed_symbols,
                "failed_symbols": job.failed_symbols,
                "total_requests": job.total_requests,
                "completed_requests": job.completed_requests,
                "failed_requests": job.failed_requests,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "symbol_statuses": {
                    symbol.status: len([s for s in symbols if s.status == symbol.status])
                    for symbol in symbols
                },
                "request_statuses": {
                    request.status: len([r for r in requests if r.status == request.status])
                    for request in requests
                }
            }
    
    def list_jobs(self, status: str = None) -> List[Dict]:
        """List all jobs, optionally filtered by status."""
        with get_db_session() as db:
            query = db.query(DataCollectionJob)
            if status:
                query = query.filter(DataCollectionJob.status == status)
            
            jobs = query.order_by(DataCollectionJob.created_at.desc()).all()
            
            return [
                {
                    "job_id": job.id,
                    "name": job.name,
                    "status": job.status,
                    "total_symbols": job.total_symbols,
                    "completed_symbols": job.completed_symbols,
                    "total_requests": job.total_requests,
                    "completed_requests": job.completed_requests,
                    "created_at": job.created_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None
                }
                for job in jobs
            ]
    
    def pause_job(self, job_id: int) -> bool:
        """Pause a running job."""
        with get_db_session() as db:
            job = db.query(DataCollectionJob).filter(DataCollectionJob.id == job_id).first()
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            if job.status not in ['running', 'pending']:
                raise ValueError(f"Cannot pause job {job_id} with status {job.status}")
            
            job.status = 'paused'
            db.commit()
            
            logger.info(f"Paused job {job_id}: {job.name}")
            return True
    
    def resume_job(self, job_id: int) -> bool:
        """Resume a paused job."""
        with get_db_session() as db:
            job = db.query(DataCollectionJob).filter(DataCollectionJob.id == job_id).first()
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            if job.status != 'paused':
                raise ValueError(f"Cannot resume job {job_id} with status {job.status}")
            
            job.status = 'running'
            if not job.started_at:
                job.started_at = datetime.now(timezone.utc)
            db.commit()
            
            logger.info(f"Resumed job {job_id}: {job.name}")
            return True
    
    async def make_historical_request(self, symbol: str, start_time: datetime, 
                                    end_time: datetime, bar_size: str = "5 secs") -> Dict:
        """Make a single historical data request using the historical service API."""
        request_data = {
            "symbol": symbol,
            "bar_size": bar_size,
            "what_to_show": "TRADES",
            "duration": "1 H",  # 1 hour max for 5-second bars
            "end_datetime": end_time.isoformat(),
            "use_rth": True
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/historical/request",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    return {"error": f"HTTP {response.status}: {error_text}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def run_job(self, job_id: int, max_concurrent: int = 10) -> Dict:
        """Run or resume a data collection job."""
        
        with get_db_session() as db:
            job = db.query(DataCollectionJob).filter(DataCollectionJob.id == job_id).first()
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            if job.status not in ['pending', 'paused', 'running']:
                raise ValueError(f"Cannot run job {job_id} with status {job.status}")
            
            # Update job status
            job.status = 'running'
            if not job.started_at:
                job.started_at = datetime.now(timezone.utc)
            db.commit()
        
        logger.info(f"Starting job {job_id}: {job.name}")
        
        try:
            # Get pending symbols
            with get_db_session() as db:
                pending_symbols = db.query(DataCollectionSymbol).filter(
                    and_(
                        DataCollectionSymbol.job_id == job_id,
                        DataCollectionSymbol.status.in_(['pending', 'running'])
                    )
                ).all()
            
            logger.info(f"Found {len(pending_symbols)} symbols to process")
            
            # Process symbols
            for i, symbol_record in enumerate(pending_symbols):
                try:
                    # Check if job was paused during processing
                    with get_db_session() as db:
                        job_check = db.query(DataCollectionJob).filter(DataCollectionJob.id == job_id).first()
                        if job_check.status == 'paused':
                            logger.info(f"Job {job_id} was paused, stopping processing")
                            break
                    
                    logger.info(f"Processing symbol {i+1}/{len(pending_symbols)}: {symbol_record.symbol}")
                    
                    # Update symbol status
                    with get_db_session() as db:
                        symbol = db.query(DataCollectionSymbol).filter(DataCollectionSymbol.id == symbol_record.id).first()
                        symbol.status = 'running'
                        symbol.started_at = datetime.now(timezone.utc)
                        db.commit()
                    
                    # Get pending requests for this symbol
                    with get_db_session() as db:
                        pending_requests = db.query(DataCollectionRequest).filter(
                            and_(
                                DataCollectionRequest.symbol_id == symbol_record.id,
                                DataCollectionRequest.status == 'pending'
                            )
                        ).all()
                    
                    # Process requests for this symbol
                    for request_record in pending_requests:
                        try:
                            # Check if job was paused during request processing
                            with get_db_session() as db:
                                job_check = db.query(DataCollectionJob).filter(DataCollectionJob.id == job_id).first()
                                if job_check.status == 'paused':
                                    logger.info(f"Job {job_id} was paused during request processing")
                                    break
                            
                            # Make the request
                            result = await self.make_historical_request(
                                symbol_record.symbol,
                                request_record.start_time,
                                request_record.end_time,
                                job.bar_size
                            )
                            
                            # Update request status
                            with get_db_session() as db:
                                request = db.query(DataCollectionRequest).filter(
                                    DataCollectionRequest.id == request_record.id
                                ).first()
                                
                                if "error" in result:
                                    request.status = 'failed'
                                    request.error_message = result["error"]
                                    request.completed_at = datetime.now(timezone.utc)
                                else:
                                    request.status = 'queued'
                                    request.request_id = result.get("request_id")
                                    request.started_at = datetime.now(timezone.utc)
                                
                                db.commit()
                            
                            # Update symbol progress
                            with get_db_session() as db:
                                symbol = db.query(DataCollectionSymbol).filter(
                                    DataCollectionSymbol.id == symbol_record.id
                                ).first()
                                
                                if "error" in result:
                                    symbol.failed_requests += 1
                                else:
                                    symbol.completed_requests += 1
                                
                                db.commit()
                            
                            # Update job progress
                            with get_db_session() as db:
                                job = db.query(DataCollectionJob).filter(DataCollectionJob.id == job_id).first()
                                
                                if "error" in result:
                                    job.failed_requests += 1
                                else:
                                    job.completed_requests += 1
                                
                                db.commit()
                            
                            logger.info(f"  Request {request_record.id}: {'queued' if 'error' not in result else 'failed'}")
                            
                        except Exception as e:
                            logger.error(f"  Error processing request {request_record.id}: {e}")
                            
                            # Update request as failed
                            with get_db_session() as db:
                                request = db.query(DataCollectionRequest).filter(
                                    DataCollectionRequest.id == request_record.id
                                ).first()
                                request.status = 'failed'
                                request.error_message = str(e)
                                request.completed_at = datetime.now(timezone.utc)
                                db.commit()
                    
                    # Check if symbol is complete
                    with get_db_session() as db:
                        symbol = db.query(DataCollectionSymbol).filter(
                            DataCollectionSymbol.id == symbol_record.id
                        ).first()
                        
                        if symbol.completed_requests + symbol.failed_requests >= symbol.total_requests:
                            symbol.status = 'completed'
                            symbol.completed_at = datetime.now(timezone.utc)
                            
                            # Update job symbol counts
                            job = db.query(DataCollectionJob).filter(DataCollectionJob.id == job_id).first()
                            if symbol.failed_requests == 0:
                                job.completed_symbols += 1
                            else:
                                job.failed_symbols += 1
                            
                            db.commit()
                            
                            logger.info(f"✅ Completed symbol {symbol_record.symbol}")
                        else:
                            logger.info(f"⚠️ Partial completion for {symbol_record.symbol}")
                    
                    # Check if job is complete
                    with get_db_session() as db:
                        job = db.query(DataCollectionJob).filter(DataCollectionJob.id == job_id).first()
                        if job.completed_symbols + job.failed_symbols >= job.total_symbols:
                            job.status = 'completed'
                            job.completed_at = datetime.now(timezone.utc)
                            db.commit()
                            
                            logger.info(f"🎉 Job {job_id} completed!")
                            break
                
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol_record.symbol}: {e}")
                    
                    # Update symbol as failed
                    with get_db_session() as db:
                        symbol = db.query(DataCollectionSymbol).filter(
                            DataCollectionSymbol.id == symbol_record.id
                        ).first()
                        symbol.status = 'failed'
                        symbol.error_message = str(e)
                        symbol.completed_at = datetime.now(timezone.utc)
                        
                        # Update job counts
                        job = db.query(DataCollectionJob).filter(DataCollectionJob.id == job_id).first()
                        job.failed_symbols += 1
                        
                        db.commit()
        
        except KeyboardInterrupt:
            logger.info(f"Job {job_id} interrupted by user")
            # Job status remains 'running' so it can be resumed
            with get_db_session() as db:
                job = db.query(DataCollectionJob).filter(DataCollectionJob.id == job_id).first()
                logger.info(f"Job {job_id} can be resumed later")
        
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            # Mark job as failed
            with get_db_session() as db:
                job = db.query(DataCollectionJob).filter(DataCollectionJob.id == job_id).first()
                job.status = 'failed'
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
        
        # Return final status
        return self.get_job_status(job_id)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Historical Data Collector")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create job command
    create_parser = subparsers.add_parser('create-job', help='Create a new data collection job')
    create_parser.add_argument('--name', required=True, help='Job name')
    create_parser.add_argument('--symbols-file', required=True, help='File containing symbols (one per line)')
    create_parser.add_argument('--start-date', required=True, help='Start date in ISO format (e.g., 2024-01-01T00:00:00Z)')
    create_parser.add_argument('--end-date', required=True, help='End date in ISO format (e.g., 2024-01-11T00:00:00Z)')
    create_parser.add_argument('--bar-size', default='5 secs', help='Bar size (default: 5 secs)')
    create_parser.add_argument('--description', help='Job description')
    
    # Run job command
    run_parser = subparsers.add_parser('run-job', help='Run or resume a data collection job')
    run_parser.add_argument('--job-id', type=int, required=True, help='Job ID to run')
    run_parser.add_argument('--max-concurrent', type=int, default=10, help='Max concurrent requests')
    
    # Pause job command
    pause_parser = subparsers.add_parser('pause-job', help='Pause a running job')
    pause_parser.add_argument('--job-id', type=int, required=True, help='Job ID to pause')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show job status')
    status_parser.add_argument('--job-id', type=int, required=True, help='Job ID to check')
    
    # List jobs command
    list_parser = subparsers.add_parser('list-jobs', help='List all jobs')
    list_parser.add_argument('--status', help='Filter by status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize database
    initialize_database()
    
    async with HistoricalDataCollector() as collector:
        if args.command == 'create-job':
            # Parse dates
            start_date = datetime.fromisoformat(args.start_date.replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(args.end_date.replace('Z', '+00:00'))
            
            # Load symbols
            symbols = collector.load_symbols_from_file(args.symbols_file)
            
            # Create job
            job_id = collector.create_job(
                name=args.name,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                bar_size=args.bar_size,
                description=args.description
            )
            
            print(f"Created job {job_id}: {args.name}")
            
        elif args.command == 'run-job':
            result = await collector.run_job(args.job_id, args.max_concurrent)
            print(f"Job {args.job_id} completed:")
            print(json.dumps(result, indent=2))
            
        elif args.command == 'pause-job':
            collector.pause_job(args.job_id)
            print(f"Paused job {args.job_id}")
            
        elif args.command == 'status':
            result = collector.get_job_status(args.job_id)
            print(json.dumps(result, indent=2))
            
        elif args.command == 'list-jobs':
            jobs = collector.list_jobs(args.status)
            print(json.dumps(jobs, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        sys.exit(1)