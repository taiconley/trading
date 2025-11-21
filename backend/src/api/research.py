from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
import logging
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sqlalchemy import func

from common.models import Candle
from common.db import get_db_session
from research.potential_pairs_analyzer import (
    load_symbol_data,
    analyze_pair,
    SymbolProfile,
    AnalysisParams,
    timeframe_to_seconds,
    SymbolData
)

router = APIRouter(prefix="/api/research", tags=["research"])
logger = logging.getLogger(__name__)

class PairAnalysisRequest(BaseModel):
    symbol_a: str
    symbol_b: str
    timeframe: str = "5 secs"
    lookback_days: int = 5
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@router.post("/analyze-pair")
async def analyze_pair_endpoint(request: PairAnalysisRequest):
    """
    Analyze a specific pair of symbols on demand.
    """
    try:
        # Determine time window
        now = datetime.now(timezone.utc)
        if request.end_date:
            end_dt = datetime.fromisoformat(request.end_date.replace("Z", "+00:00"))
        else:
            end_dt = now
            
        # Setup params
        bar_seconds = timeframe_to_seconds(request.timeframe)
        
        # Smart lookback calculation
        if request.start_date:
            start_dt = datetime.fromisoformat(request.start_date.replace("Z", "+00:00"))
        else:
            # If lookback_days is default (5) but timeframe is large, adjust it
            # We need at least min_bars (100) + some buffer
            min_required_seconds = 150 * bar_seconds
            min_required_days = min_required_seconds / 86400
            
            # Use the larger of user request or minimum required
            effective_lookback = max(request.lookback_days, int(min_required_days) + 1)
            
            start_dt = end_dt - timedelta(days=effective_lookback)

        params = AnalysisParams(
            timeframe=request.timeframe,
            bar_seconds=bar_seconds,
            entry_z=2.0,
            exit_z=0.5,
            min_bars=100,  # Lower threshold for on-demand analysis
            min_spread_std=1e-6,
            min_trades=1   # Lower threshold
        )

        # Load data
        with get_db_session() as session:
            data_a = load_symbol_data(session, request.symbol_a, request.timeframe, start_dt, end_dt)
            data_b = load_symbol_data(session, request.symbol_b, request.timeframe, start_dt, end_dt)

            if not data_a or not data_b:
                raise HTTPException(status_code=404, detail=f"Data not found for one or both symbols in timeframe {request.timeframe}")

            # Create dummy profiles as they are only used for volume stats in the result
            # and we can calculate what we need or pass placeholders
            
            # Calculate actual volume for the result
            vol_a = float(np.mean(data_a.close * data_a.volume)) if len(data_a.close) > 0 else 0.0
            vol_b = float(np.mean(data_b.close * data_b.volume)) if len(data_b.close) > 0 else 0.0
            
            profiles = {
                request.symbol_a: SymbolProfile(request.symbol_a, len(data_a.close), vol_a, start_dt, end_dt),
                request.symbol_b: SymbolProfile(request.symbol_b, len(data_b.close), vol_b, start_dt, end_dt)
            }

            result = analyze_pair(
                request.symbol_a,
                request.symbol_b,
                data_a,
                data_b,
                profiles,
                params
            )

            if not result:
                # If analysis returns None (e.g. cointegration failed or not enough overlap), 
                # we still want to return the raw data so the user can see it.
                # We'll construct a partial result.
                return {
                    "status": "insufficient_data_or_no_match",
                    "symbol_a": request.symbol_a,
                    "symbol_b": request.symbol_b,
                    "timeframe": request.timeframe,
                    "data_a": {
                        "timestamps": [pd.Timestamp(ts).isoformat() for ts in data_a.timestamps],
                        "close": data_a.close.tolist()
                    },
                    "data_b": {
                        "timestamps": [pd.Timestamp(ts).isoformat() for ts in data_b.timestamps],
                        "close": data_b.close.tolist()
                    }
                }

            # Serialize numpy arrays in result
            # The 'analyze_pair' function returns a dict with some numpy types and a 'equity_curve' list
            # We need to make sure everything is JSON serializable
            
            # Helper to convert numpy types
            def serialize(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            clean_result = {k: serialize(v) for k, v in result.items()}
            
            # Add raw price data for plotting
            clean_result["price_data"] = {
                 "timestamps": [pd.Timestamp(ts).isoformat() for ts in data_a.timestamps], 
                 "symbol_a": {
                     "timestamps": [pd.Timestamp(ts).isoformat() for ts in data_a.timestamps],
                     "close": data_a.close.tolist()
                 },
                 "symbol_b": {
                     "timestamps": [pd.Timestamp(ts).isoformat() for ts in data_b.timestamps],
                     "close": data_b.close.tolist()
                 }
            }

            return clean_result

    except Exception as e:
        logger.error(f"Error analyzing pair: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/availability")
async def get_data_availability(symbol: str, timeframe: str):
    """
    Get the available date range for a specific symbol and timeframe.
    """
    with get_db_session() as db:
        result = db.query(
            func.min(Candle.ts),
            func.max(Candle.ts),
            func.count(Candle.ts)
        ).filter(
            Candle.symbol == symbol.upper(),
            Candle.tf == timeframe
        ).first()
        
        if not result or not result[0]:
             return {
                "symbol": symbol,
                "timeframe": timeframe,
                "available": False,
                "start_date": None,
                "end_date": None,
                "count": 0
            }
            
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "available": True,
            "start_date": result[0].isoformat(),
            "end_date": result[1].isoformat(),
            "count": result[2]
        }
