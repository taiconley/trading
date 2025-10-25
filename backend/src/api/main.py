"""
API Gateway for Trading Bot

Aggregates all microservices into a unified REST API for the frontend.
No authentication required (local system only).

Service Ports:
- Account: 8001
- Market Data: 8002
- Historical: 8003
- Trader: 8004
- Strategy: 8005
- Optimizer: 8006
- Backtester: 8007 (on-demand)
"""
import httpx
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import json
import logging
from sqlalchemy import desc, asc, text

# Import common modules at the top
from common.db import get_db_session
from common.models import (
    Tick, Candle, WatchlistEntry, Strategy, BacktestRun, BacktestTrade,
    OptimizationRun, OptimizationResult, ParameterSensitivity,
    Signal, Execution, Order
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service URLs (internal Docker network)
SERVICES = {
    "account": "http://backend-account:8001",
    "marketdata": "http://backend-marketdata:8002",
    "historical": "http://backend-historical:8003",
    "trader": "http://backend-trader:8004",
    "strategy": "http://backend-strategy:8005",
    # Note: optimizer and backtester are on-demand CLI tools, not always-running services
}

app = FastAPI(
    title="Trading Bot API Gateway",
    description="Unified API for all trading bot services",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend (localhost access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],  # Vite dev server, production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP client with timeout
http_client = httpx.AsyncClient(timeout=30.0)


# ============================================================================
# Helper Functions
# ============================================================================

async def proxy_get(service: str, path: str, params: Optional[Dict] = None):
    """Proxy GET request to a service."""
    try:
        url = f"{SERVICES[service]}{path}"
        response = await http_client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from {service}: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        logger.error(f"Request error to {service}: {e}")
        raise HTTPException(status_code=503, detail=f"Service {service} unavailable")
    except Exception as e:
        logger.error(f"Unexpected error proxying to {service}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def proxy_post(service: str, path: str, data: Optional[Dict] = None, json_data: Optional[Dict] = None):
    """Proxy POST request to a service."""
    try:
        url = f"{SERVICES[service]}{path}"
        if json_data:
            response = await http_client.post(url, json=json_data)
        else:
            response = await http_client.post(url, data=data)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from {service}: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        logger.error(f"Request error to {service}: {e}")
        raise HTTPException(status_code=503, detail=f"Service {service} unavailable")
    except Exception as e:
        logger.error(f"Unexpected error proxying to {service}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def proxy_put(service: str, path: str, json_data: Dict):
    """Proxy PUT request to a service."""
    try:
        url = f"{SERVICES[service]}{path}"
        response = await http_client.put(url, json=json_data)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from {service}: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        logger.error(f"Request error to {service}: {e}")
        raise HTTPException(status_code=503, detail=f"Service {service} unavailable")
    except Exception as e:
        logger.error(f"Unexpected error proxying to {service}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def proxy_delete(service: str, path: str):
    """Proxy DELETE request to a service."""
    try:
        url = f"{SERVICES[service]}{path}"
        response = await http_client.delete(url)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from {service}: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        logger.error(f"Request error to {service}: {e}")
        raise HTTPException(status_code=503, detail=f"Service {service} unavailable")
    except Exception as e:
        logger.error(f"Unexpected error proxying to {service}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Root & Health Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Trading Bot API Gateway",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/healthz")
async def health_check():
    """Simple health check for gateway itself."""
    return {"status": "healthy", "service": "api-gateway"}


@app.get("/api/health")
async def aggregate_health():
    """
    Aggregate health status from all services using HealthMonitor.
    
    Returns comprehensive system health including:
    - Overall system status (healthy, degraded, unhealthy)
    - Individual service health with timestamps
    - Stale service detection
    - Critical service status
    """
    from common.health_monitor import get_health_monitor
    
    try:
        monitor = get_health_monitor()
        system_health = await monitor.get_system_health()
        return system_health
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        # Fallback to basic health check
        return {
            "status": "unknown",
            "message": f"Health monitoring error: {str(e)}",
            "gateway": "healthy",
            "timestamp": None,
        }


@app.get("/api/health/detailed")
async def detailed_health():
    """
    Detailed health check that includes live service pings.
    
    This endpoint actively pings each service's health endpoint
    in addition to database-backed health monitoring.
    """
    from common.health_monitor import get_health_monitor
    
    # Get database health
    try:
        monitor = get_health_monitor()
        db_health = await monitor.get_system_health()
    except Exception as e:
        logger.error(f"Error getting DB health: {e}")
        db_health = {"error": str(e)}
    
    # Ping live services
    live_services = {}
    for service_name, service_url in SERVICES.items():
        try:
            response = await http_client.get(f"{service_url}/healthz", timeout=2.0)
            if response.status_code == 200:
                live_services[service_name] = {
                    "status": "healthy",
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "details": response.json()
                }
            else:
                live_services[service_name] = {
                    "status": "unhealthy",
                    "code": response.status_code
                }
        except Exception as e:
            live_services[service_name] = {
                "status": "unavailable",
                "error": str(e)
            }
    
    return {
        "database_health": db_health,
        "live_services": live_services,
        "gateway": "healthy"
    }


@app.get("/api/health/circuit-breakers")
async def circuit_breaker_status():
    """Get status of all circuit breakers."""
    from common.db import get_circuit_breaker_status
    from common.circuit_breaker import get_tws_circuit_breaker
    
    try:
        db_cb = get_circuit_breaker_status()
        tws_cb = get_tws_circuit_breaker().get_state()
        
        return {
            "circuit_breakers": {
                "database": db_cb,
                "tws": tws_cb
            }
        }
    except Exception as e:
        logger.error(f"Error getting circuit breaker status: {e}")
        return {"error": str(e)}


# ============================================================================
# Account Service Endpoints
# ============================================================================

@app.get("/api/account")
async def get_account_stats():
    """Get account summary statistics."""
    return await proxy_get("account", "/account/stats")


@app.get("/api/positions")
async def get_positions():
    """Get current positions."""
    return await proxy_get("account", "/account/stats")  # Account stats includes positions


# ============================================================================
# Trader Service Endpoints
# ============================================================================

@app.get("/api/orders")
async def get_orders(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 100
):
    """Get order history with optional filters."""
    params = {"limit": limit}
    if status:
        params["status"] = status
    if symbol:
        params["symbol"] = symbol
    return await proxy_get("trader", "/orders", params=params)


@app.get("/api/orders/{order_id}")
async def get_order(order_id: int):
    """Get specific order details."""
    return await proxy_get("trader", f"/orders/{order_id}")


@app.post("/api/orders")
async def place_order(order: Dict = Body(...)):
    """Place a new order."""
    return await proxy_post("trader", "/orders", json_data=order)


@app.post("/api/orders/{order_id}/cancel")
async def cancel_order(order_id: int):
    """Cancel an order."""
    return await proxy_post("trader", f"/cancel/{order_id}")


# ============================================================================
# Market Data Endpoints
# ============================================================================

@app.get("/api/ticks")
async def get_recent_ticks(symbol: str, limit: int = 100):
    """Get recent tick data for a symbol."""
    with get_db_session() as db:
        ticks = db.query(Tick).filter(
            Tick.symbol == symbol
        ).order_by(desc(Tick.ts)).limit(limit).all()
        
        return {
            "symbol": symbol,
            "count": len(ticks),
            "ticks": [
                {
                    "timestamp": tick.ts.isoformat(),
                    "bid": float(tick.bid) if tick.bid else None,
                    "ask": float(tick.ask) if tick.ask else None,
                    "last": float(tick.last) if tick.last else None,
                    "bid_size": tick.bid_size,
                    "ask_size": tick.ask_size,
                    "last_size": tick.last_size
                }
                for tick in ticks
            ]
        }


@app.get("/api/subscriptions")
async def get_subscriptions():
    """Get current market data subscriptions."""
    return await proxy_get("marketdata", "/subscriptions")


@app.post("/api/watchlist")
async def update_watchlist(data: Dict = Body(...)):
    """Add or remove symbols from watchlist."""
    # Direct database update
    from datetime import datetime, timezone
    
    action = data.get("action")  # "add" or "remove"
    symbol = data.get("symbol")
    
    if not symbol or action not in ["add", "remove"]:
        raise HTTPException(status_code=400, detail="Invalid request")
    
    with get_db_session() as db:
        if action == "add":
            # Check if already exists
            existing = db.query(WatchlistEntry).filter(WatchlistEntry.symbol == symbol).first()
            if not existing:
                watchlist_entry = WatchlistEntry(symbol=symbol, added_at=datetime.now(timezone.utc))
                db.add(watchlist_entry)
                db.commit()
                return {"message": f"Added {symbol} to watchlist"}
            else:
                return {"message": f"{symbol} already in watchlist"}
        else:  # remove
            deleted = db.query(WatchlistEntry).filter(WatchlistEntry.symbol == symbol).delete()
            db.commit()
            if deleted:
                return {"message": f"Removed {symbol} from watchlist"}
            else:
                return {"message": f"{symbol} not in watchlist"}


@app.get("/api/watchlist")
async def get_watchlist():
    """Get all symbols in watchlist."""
    
    with get_db_session() as db:
        watchlist = db.query(WatchlistEntry).all()
        return {
            "symbols": [w.symbol for w in watchlist],
            "count": len(watchlist)
        }


# ============================================================================
# Historical Data Endpoints
# ============================================================================

@app.post("/api/historical/request")
async def request_historical_data(request: Dict = Body(...)):
    """Request historical data for a symbol."""
    return await proxy_post("historical", "/historical/request", json_data=request)


@app.post("/api/historical/bulk")
async def bulk_historical_request(body: Optional[Dict] = None):
    """Request historical data for all watchlist symbols."""
    return await proxy_post("historical", "/historical/bulk", json_data=body if body else {})


@app.get("/api/historical/queue")
async def get_historical_queue():
    """Get historical data request queue status."""
    return await proxy_get("historical", "/queue/status")


@app.get("/api/historical/datasets")
async def get_available_datasets():
    """Get list of available historical datasets (symbol + timeframe combinations)."""
    
    with get_db_session() as db:
        # Query distinct symbol, timeframe combinations with count and date range
        query = """
        SELECT 
            symbol,
            tf as timeframe,
            COUNT(*) as bar_count,
            MIN(ts) as start_date,
            MAX(ts) as end_date
        FROM candles
        GROUP BY symbol, tf
        ORDER BY symbol, tf
        """
        
        result = db.execute(text(query))
        datasets = []
        
        for row in result:
            datasets.append({
                "symbol": row[0],
                "timeframe": row[1],
                "bar_count": row[2],
                "start_date": row[3].isoformat() if row[3] else None,
                "end_date": row[4].isoformat() if row[4] else None
            })
        
        return {"datasets": datasets, "count": len(datasets)}


@app.get("/api/historical/candles")
async def get_candles(
    symbol: str,
    timeframe: str,
    limit: int = 1000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get candle data for a specific symbol and timeframe."""
    
    with get_db_session() as db:
        query = db.query(Candle).filter(
            Candle.symbol == symbol.upper(),
            Candle.tf == timeframe
        )
        
        # Apply date filters if provided
        if start_date:
            query = query.filter(Candle.ts >= start_date)
        if end_date:
            query = query.filter(Candle.ts <= end_date)
        
        # Order by timestamp and limit
        candles = query.order_by(Candle.ts.asc()).limit(limit).all()
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": len(candles),
            "candles": [
                {
                    "timestamp": c.ts.isoformat(),
                    "open": float(c.open),
                    "high": float(c.high),
                    "low": float(c.low),
                    "close": float(c.close),
                    "volume": int(c.volume) if c.volume else 0
                }
                for c in candles
            ]
        }


@app.delete("/api/historical/dataset")
async def delete_dataset(symbol: str, timeframe: str):
    """Delete all candle data for a specific symbol and timeframe."""
    
    with get_db_session() as db:
        deleted_count = db.query(Candle).filter(
            Candle.symbol == symbol.upper(),
            Candle.tf == timeframe
        ).delete()
        
        db.commit()
        
        return {
            "message": f"Deleted {deleted_count} candles for {symbol} ({timeframe})",
            "symbol": symbol,
            "timeframe": timeframe,
            "deleted_count": deleted_count
        }


# ============================================================================
# Strategy Management Endpoints
# ============================================================================

@app.get("/api/strategies")
async def get_strategies():
    """Get all strategies from database."""
    
    with get_db_session() as db:
        strategies = db.query(Strategy).all()
        return {
            "strategies": [
                {
                    "id": s.strategy_id,
                    "name": s.name,
                    "enabled": s.enabled,
                    "params": s.params_json,
                    "created_at": s.created_at.isoformat()
                }
                for s in strategies
            ]
        }


@app.post("/api/strategies/{strategy_id}/enable")
async def enable_strategy(strategy_id: str, data: Dict = Body(...)):
    """Enable or disable a strategy."""
    
    enabled = data.get("enabled", True)
    
    with get_db_session() as db:
        strategy = db.query(Strategy).filter(Strategy.strategy_id == strategy_id).first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy.enabled = enabled
        db.commit()
        
        return {
            "message": f"Strategy {'enabled' if enabled else 'disabled'}",
            "strategy_id": strategy_id,
            "enabled": enabled
        }


@app.put("/api/strategies/{strategy_id}/params")
async def update_strategy_params(strategy_id: str, params: Dict = Body(...)):
    """Update strategy parameters."""
    
    with get_db_session() as db:
        strategy = db.query(Strategy).filter(Strategy.strategy_id == strategy_id).first()
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy.params_json = params
        db.commit()
        
        return {
            "message": "Strategy parameters updated",
            "strategy_id": strategy_id,
            "params": params
        }


# ============================================================================
# Backtest Endpoints
# ============================================================================

@app.post("/api/backtests")
async def run_backtest(config: Dict = Body(...)):
    """
    Trigger a new backtest.
    Note: Backtester runs on-demand via CLI. This endpoint queues the request.
    """
    raise HTTPException(
        status_code=501,
        detail="Backtests must be run via CLI. Use: docker compose exec backend-historical python /app/src/services/backtester/main.py cli ..."
    )


@app.get("/api/backtests")
async def list_backtests(limit: int = 50):
    """List recent backtests from database."""
    
    with get_db_session() as db:
        runs = db.query(BacktestRun).order_by(desc(BacktestRun.created_at)).limit(limit).all()
        
        return {
            "backtests": [
                {
                    "id": r.id,
                    "strategy_name": r.strategy_name,
                    "params": r.params_json,
                    "start_ts": r.start_ts.isoformat() if r.start_ts else None,
                    "end_ts": r.end_ts.isoformat() if r.end_ts else None,
                    "pnl": float(r.pnl) if r.pnl else None,
                    "sharpe": float(r.sharpe) if r.sharpe else None,
                    "maxdd": float(r.maxdd) if r.maxdd else None,
                    "trades": r.trades,
                    "created_at": r.created_at.isoformat()
                }
                for r in runs
            ]
        }


@app.get("/api/backtests/{run_id}")
async def get_backtest(run_id: int):
    """Get backtest results from database."""
    
    with get_db_session() as db:
        run = db.query(BacktestRun).filter(BacktestRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Backtest run not found")
        
        return {
            "id": run.id,
            "strategy_name": run.strategy_name,
            "params": run.params_json,
            "start_ts": run.start_ts.isoformat() if run.start_ts else None,
            "end_ts": run.end_ts.isoformat() if run.end_ts else None,
            "pnl": float(run.pnl) if run.pnl else None,
            "sharpe": float(run.sharpe) if run.sharpe else None,
            "maxdd": float(run.maxdd) if run.maxdd else None,
            "trades": run.trades,
            "created_at": run.created_at.isoformat()
        }


@app.get("/api/backtests/{run_id}/trades")
async def get_backtest_trades(run_id: int):
    """Get trades from a backtest."""
    
    with get_db_session() as db:
        trades = db.query(BacktestTrade).filter(BacktestTrade.run_id == run_id).all()
        
        return {
            "trades": [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": float(t.qty),
                    "entry_ts": t.entry_ts.isoformat() if t.entry_ts else None,
                    "entry_px": float(t.entry_px) if t.entry_px else None,
                    "exit_ts": t.exit_ts.isoformat() if t.exit_ts else None,
                    "exit_px": float(t.exit_px) if t.exit_px else None,
                    "pnl": float(t.pnl) if t.pnl else None
                }
                for t in trades
            ]
        }


# ============================================================================
# Optimizer Endpoints
# ============================================================================

@app.post("/api/optimizations")
async def run_optimization(config: Dict = Body(...)):
    """
    Start a new parameter optimization.
    Note: Optimizer runs on-demand via CLI. This endpoint queues the request.
    """
    raise HTTPException(
        status_code=501,
        detail="Optimizations must be run via CLI. Use: docker compose exec backend-historical python /app/src/services/optimizer/main.py optimize ..."
    )


@app.get("/api/optimizations")
async def list_optimizations(limit: int = 50):
    """List recent optimizations from database."""
    
    with get_db_session() as db:
        runs = db.query(OptimizationRun).order_by(desc(OptimizationRun.created_at)).limit(limit).all()
        
        return {
            "optimizations": [
                {
                    "id": r.id,
                    "strategy_name": r.strategy_name,
                    "algorithm": r.algorithm,
                    "symbols": r.symbols,
                    "status": r.status,
                    "total_combinations": r.total_combinations,
                    "completed_combinations": r.completed_combinations,
                    "best_params": r.best_params,
                    "best_score": float(r.best_score) if r.best_score else None,
                    "created_at": r.created_at.isoformat()
                }
                for r in runs
            ]
        }


@app.get("/api/optimizations/{run_id}")
async def get_optimization(run_id: int):
    """Get optimization status and results from database."""
    
    with get_db_session() as db:
        run = db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Optimization run not found")
        
        duration = None
        if run.start_time and run.end_time:
            duration = (run.end_time - run.start_time).total_seconds()
        
        return {
            "id": run.id,
            "strategy_name": run.strategy_name,
            "algorithm": run.algorithm,
            "symbols": run.symbols,
            "timeframe": run.timeframe,
            "param_ranges": run.param_ranges,
            "objective": run.objective,
            "status": run.status,
            "total_combinations": run.total_combinations,
            "completed_combinations": run.completed_combinations,
            "best_params": run.best_params,
            "best_score": float(run.best_score) if run.best_score else None,
            "duration_seconds": duration,
            "created_at": run.created_at.isoformat()
        }


@app.get("/api/optimizations/{run_id}/results")
async def get_optimization_results(run_id: int, top_n: int = 20):
    """Get top parameter combinations from optimization."""
    
    with get_db_session() as db:
        results = db.query(OptimizationResult).filter(
            OptimizationResult.run_id == run_id
        ).order_by(desc(OptimizationResult.score)).limit(top_n).all()
        
        return {
            "results": [
                {
                    "params": r.params_json,
                    "score": float(r.score) if r.score else None,
                    "sharpe_ratio": float(r.sharpe_ratio) if r.sharpe_ratio else None,
                    "total_return": float(r.total_return) if r.total_return else None,
                    "max_drawdown": float(r.max_drawdown) if r.max_drawdown else None,
                    "win_rate": float(r.win_rate) if r.win_rate else None,
                    "profit_factor": float(r.profit_factor) if r.profit_factor else None,
                    "total_trades": r.total_trades
                }
                for r in results
            ]
        }


@app.get("/api/optimizations/{run_id}/analysis")
async def get_sensitivity_analysis(run_id: int):
    """Get parameter sensitivity analysis from database."""
    
    with get_db_session() as db:
        sensitivity = db.query(ParameterSensitivity).filter(
            ParameterSensitivity.run_id == run_id
        ).order_by(asc(ParameterSensitivity.importance_rank)).all()
        
        if not sensitivity:
            raise HTTPException(
                status_code=404,
                detail="No sensitivity analysis found. Run sensitivity analysis first."
            )
        
        return {
            "sensitivity": [
                {
                    "parameter": s.parameter_name,
                    "sensitivity_score": float(s.sensitivity_score) if s.sensitivity_score else None,
                    "correlation": float(s.correlation_with_objective) if s.correlation_with_objective else None,
                    "importance_rank": s.importance_rank,
                    "mean_score": float(s.mean_score) if s.mean_score else None,
                    "std_score": float(s.std_score) if s.std_score else None,
                    "min_score": float(s.min_score) if s.min_score else None,
                    "max_score": float(s.max_score) if s.max_score else None,
                    "interactions": s.interactions
                }
                for s in sensitivity
            ]
        }


@app.get("/api/optimizations/{run_id}/pareto")
async def get_pareto_frontier(run_id: int, objectives: str = "sharpe_ratio,max_drawdown"):
    """
    Get Pareto frontier for multi-objective optimization.
    Note: This requires running Pareto analysis via CLI first.
    """
    raise HTTPException(
        status_code=501,
        detail="Pareto frontier analysis must be run via optimizer CLI"
    )


@app.post("/api/optimizations/{run_id}/stop")
async def stop_optimization(run_id: int):
    """Stop a running optimization (not implemented for CLI-based optimizations)."""
    raise HTTPException(
        status_code=501,
        detail="Cannot stop CLI-based optimizations remotely"
    )


# ============================================================================
# Signals & Analytics Endpoints
# ============================================================================

@app.get("/api/signals")
async def get_signals(
    strategy_id: Optional[int] = None,
    symbol: Optional[str] = None,
    limit: int = 100
):
    """Get recent strategy signals."""
    
    with get_db_session() as db:
        query = db.query(Signal)
        
        if strategy_id:
            query = query.filter(Signal.strategy_id == strategy_id)
        if symbol:
            query = query.filter(Signal.symbol == symbol)
        
        signals = query.order_by(desc(Signal.ts)).limit(limit).all()
        
        return {
            "count": len(signals),
            "signals": [
                {
                    "id": s.id,
                    "strategy_id": s.strategy_id,
                    "symbol": s.symbol,
                    "signal_type": s.signal_type,
                    "strength": float(s.strength) if s.strength else None,
                    "timestamp": s.ts.isoformat(),
                    "meta": s.meta_json
                }
                for s in signals
            ]
        }


@app.get("/api/executions")
async def get_executions(limit: int = 100):
    """Get recent trade executions."""
    
    with get_db_session() as db:
        executions = db.query(Execution).order_by(desc(Execution.ts)).limit(limit).all()
        
        return {
            "count": len(executions),
            "executions": [
                {
                    "id": e.id,
                    "order_id": e.order_id,
                    "trade_id": e.trade_id,
                    "symbol": e.symbol,
                    "qty": float(e.qty),
                    "price": float(e.price),
                    "timestamp": e.ts.isoformat()
                }
                for e in executions
            ]
        }


# ============================================================================
# WebSocket Proxies
# ============================================================================

# Store active WebSocket connections
active_websockets: Dict[str, List[WebSocket]] = {
    "account": [],
    "market": [],
    "orders": []
}


@app.websocket("/ws/account")
async def websocket_account(websocket: WebSocket):
    """WebSocket proxy for account updates."""
    await websocket.accept()
    active_websockets["account"].append(websocket)
    
    try:
        # Connect to account service WebSocket
        async with httpx.AsyncClient() as client:
            # For now, send periodic updates (polling)
            while True:
                try:
                    # Get account stats
                    stats = await proxy_get("account", "/account/stats")
                    await websocket.send_json(stats)
                except Exception as e:
                    logger.error(f"Error in account WebSocket: {e}")
                
                await asyncio.sleep(2)  # Update every 2 seconds
                
    except WebSocketDisconnect:
        active_websockets["account"].remove(websocket)


@app.websocket("/ws/market")
async def websocket_market(websocket: WebSocket):
    """WebSocket proxy for market data updates."""
    await websocket.accept()
    active_websockets["market"].append(websocket)
    
    try:
        while True:
            # Poll recent ticks from database
            from sqlalchemy import desc
            
            with get_db_session() as db:
                watchlist = db.query(WatchlistEntry).all()
                market_data = {}
                
                for watch in watchlist[:10]:  # Limit to 10 symbols
                    latest = db.query(Tick).filter(
                        Tick.symbol == watch.symbol
                    ).order_by(desc(Tick.ts)).first()
                    
                    if latest:
                        market_data[watch.symbol] = {
                            "bid": float(latest.bid) if latest.bid else None,
                            "ask": float(latest.ask) if latest.ask else None,
                            "last": float(latest.last) if latest.last else None,
                            "timestamp": latest.ts.isoformat()
                        }
                
                if market_data:
                    await websocket.send_json(market_data)
            
            await asyncio.sleep(1)  # Update every second
            
    except WebSocketDisconnect:
        active_websockets["market"].remove(websocket)


@app.websocket("/ws/orders")
async def websocket_orders(websocket: WebSocket):
    """WebSocket proxy for order status updates."""
    await websocket.accept()
    active_websockets["orders"].append(websocket)
    
    try:
        while True:
            # Poll recent orders
            from sqlalchemy import desc
            
            with get_db_session() as db:
                recent_orders = db.query(Order).order_by(desc(Order.updated_at)).limit(20).all()
                
                orders_data = {
                    "orders": [
                        {
                            "id": o.id,
                            "symbol": o.symbol,
                            "side": o.side,
                            "qty": float(o.qty),
                            "order_type": o.order_type,
                            "status": o.status,
                            "updated_at": o.updated_at.isoformat()
                        }
                        for o in recent_orders
                    ]
                }
                
                await websocket.send_json(orders_data)
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        active_websockets["orders"].remove(websocket)


# ============================================================================
# Startup & Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("API Gateway starting up...")
    logger.info(f"Configured services: {list(SERVICES.keys())}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("API Gateway shutting down...")
    await http_client.aclose()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
