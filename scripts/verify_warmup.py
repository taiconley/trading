import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add backend to path
if os.path.exists("/app/src"):
    sys.path.insert(0, "/app/src")
else:
    sys.path.insert(0, "/home/taiconley/Desktop/Projects/trading/backend/src")

from common.db import get_db_session
from common.models import Candle, Symbol, Strategy
from services.strategy.main import StrategyService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def verify_warmup():
    # 1. Setup Test Data
    symbol = "TEST_WARMUP"
    
    # Create DB session
    db_host = os.getenv("POSTGRES_HOST", "postgres") # Default to postgres service name
    engine = create_engine(f"postgresql://bot:botpw@{db_host}:5432/trading")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        # Ensure symbol exists
        db.execute(text(f"INSERT INTO symbols (symbol, currency, active, updated_at) VALUES ('{symbol}', 'USD', true, NOW()) ON CONFLICT (symbol) DO NOTHING"))
        
        # Clear existing candles for test symbol
        db.execute(text(f"DELETE FROM candles WHERE symbol = '{symbol}'"))
        
        # Insert "old" data (e.g., 1 hour ago, creating a gap)
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(hours=1)
        
        # Insert a few candles 1 hour ago
        for i in range(10):
            ts = old_time - timedelta(seconds=i*5)
            db.execute(text(
                f"INSERT INTO candles (symbol, tf, ts, open, high, low, close, volume) "
                f"VALUES ('{symbol}', '5 secs', '{ts}', 100, 101, 99, 100, 100)"
            ))
        
        # Create a dummy strategy config using this symbol
        strategy_id = "test_strategy"
        db.execute(text(f"DELETE FROM strategies WHERE strategy_id = '{strategy_id}'"))
        
        params = {
            "lookback_window": 240,
            "spread_history_bars": 1000,
            "stats_aggregation_seconds": 60,
            "min_hedge_lookback": 120,
            "symbols": [symbol]
        }
        import json
        params_json = json.dumps(params)
        
        db.execute(text(
            f"INSERT INTO strategies (strategy_id, name, enabled, params_json, created_at) "
            f"VALUES ('{strategy_id}', 'Test Strategy', true, '{params_json}', NOW())"
        ))
        
        db.commit()
        
        logger.info(f"Test data setup complete for {symbol}. Gap created ending at {old_time}")
        
        # 2. Initialize Strategy Service (Mocking dependencies where needed)
        service = StrategyService()
        
        # Mock the strategy runner loading to pick up our test strategy
        # In reality, we can just manually trigger _backfill_bar_cache after mocking strategy_runners
        
        # We need to mock the strategy object structure
        class MockConfig:
            def __init__(self, params):
                self.enabled = True
                self.symbols = params['symbols']
                self.lookback_window = params['lookback_window']
                self.spread_history_bars = params['spread_history_bars']
                self.stats_aggregation_seconds = params['stats_aggregation_seconds']
                self.min_hedge_lookback = params['min_hedge_lookback']

        class MockStrategy:
            def __init__(self, config):
                self.config = config

        class MockRunner:
            def __init__(self, strategy):
                self.strategy = strategy

        service.strategy_runners = {
            strategy_id: MockRunner(MockStrategy(MockConfig(params)))
        }
        
        # Mock _request_historical_data to avoid actual network calls but verify logic
        original_request = service._request_historical_data
        requested_gaps = []
        
        async def mock_request(symbol, bar_size, duration, end_datetime=None):
            logger.info(f"MOCK REQUEST: {symbol} {duration} ending {end_datetime}")
            requested_gaps.append({
                "symbol": symbol,
                "duration": duration,
                "end_datetime": end_datetime
            })
            return True
            
        service._request_historical_data = mock_request
        
        # 3. Run Backfill
        logger.info("Running _backfill_bar_cache...")
        await service._backfill_bar_cache()
        
        # 4. Verify Results
        if not requested_gaps:
            logger.error("FAILED: No historical data requests were made!")
            return
            
        logger.info(f"Captured {len(requested_gaps)} requests.")
        for req in requested_gaps:
            logger.info(f"Request: {req}")
            
        # Verify we requested a gap of approx 3600 seconds (1 hour)
        # The logic calculates required history: max(240, 1000, 120) * 60 = 60000 seconds (approx 16 hours)
        # So it should request a tail gap filling from 1 hour ago to now.
        
        found_tail_gap = False
        for req in requested_gaps:
            if "S" in req['duration']:
                seconds = int(req['duration'].replace(" S", ""))
                if seconds > 3000: # At least nearly an hour
                    found_tail_gap = True
                    logger.info("SUCCESS: Found request covering the tail gap.")
        
        if not found_tail_gap:
            logger.error("FAILED: Did not find a request covering the expected tail gap.")
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(verify_warmup())
