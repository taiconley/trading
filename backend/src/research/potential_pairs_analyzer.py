"""
Offline analysis tool to score potential trading pairs from historical candles.

Run inside the backend container, for example:

    docker compose exec backend-api python -m src.research.potential_pairs_analyzer \
        --timeframe "5 secs" \
        --lookback-days 5 \
        --min-bars 3000 \
        --min-dollar-volume 200000 \
        --max-symbols 60 \
        --parallelism 8 \
        --replace-existing

The script loads symbol data from the `candles` table, evaluates each pair, and
stores the results in the `potential_pairs` table for further review.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, delete, func, select
from sqlalchemy.orm import Session

from common.config import get_settings
from common.db import execute_with_retry, get_engine
from common.models import Candle, Symbol

try:
    from statsmodels.tsa.stattools import adfuller, coint
except ImportError as e:  # pragma: no cover - optional dependency
    LOGGER.warning(f"statsmodels not available: {e}. Cointegration and ADF tests will be skipped.")
    LOGGER.warning("Install statsmodels with: pip install statsmodels")
    adfuller = None
    coint = None
except Exception as e:  # pragma: no cover - defensive
    LOGGER.warning(f"Unexpected error importing statsmodels: {e}")
    adfuller = None
    coint = None


LOGGER = logging.getLogger("pairs.research")


@dataclass
class SymbolProfile:
    """Liquidity and coverage information for a symbol."""

    symbol: str
    bars: int
    avg_dollar_volume: float
    first_ts: datetime
    last_ts: datetime


@dataclass
class SymbolData:
    """Aligned time series for a symbol."""

    timestamps: np.ndarray
    close: np.ndarray
    volume: np.ndarray


@dataclass
class AnalysisParams:
    timeframe: str
    bar_seconds: int
    entry_z: float
    exit_z: float
    min_bars: int
    min_spread_std: float
    min_trades: int


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify candidate pairs for the pairs trading strategy.")
    parser.add_argument("--timeframe", default="5 secs", help="Timeframe to evaluate (must match candles.tf).")
    parser.add_argument("--lookback-days", type=int, default=5, help="Trailing calendar days to analyze when --start-date is omitted.")
    parser.add_argument("--start-date", type=str, default=None, help="Inclusive UTC start timestamp (e.g. 2024-01-01T13:30:00Z).")
    parser.add_argument("--end-date", type=str, default=None, help="Inclusive UTC end timestamp.")
    parser.add_argument("--min-bars", type=int, default=2500, help="Minimum overlapping bars required for a pair.")
    parser.add_argument("--min-dollar-volume", type=float, default=200_000.0, help="Minimum average dollar volume per bar for symbols.")
    parser.add_argument("--max-symbols", type=int, default=80, help="Maximum number of symbols to include after filtering by liquidity.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional hard cap on the number of pairs to analyze.")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Z-score entry threshold used for simulation.")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Z-score exit threshold used for simulation.")
    parser.add_argument("--min-trades", type=int, default=5, help="Minimum completed trades required to keep a pair.")
    parser.add_argument("--parallelism", type=int, default=max(1, min(8, (get_settings().development.debug and 2) or (os_cpu_count() - 6))), help="Worker threads for pair evaluation.")
    parser.add_argument("--replace-existing", action="store_true", help="Delete existing potential_pairs rows overlapping the window/timeframe.")
    parser.add_argument("--status", default="candidate", choices=["candidate", "validated", "rejected"], help="Status flag to assign to new rows.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(args=args)


def os_cpu_count() -> int:
    count = 1
    try:
        import os

        cpu_count = os.cpu_count()
        if cpu_count:
            count = cpu_count
    except Exception:  # pragma: no cover - defensive
        pass
    return count


def parse_timestamp(value: Optional[str], fallback: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return fallback
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def ensure_utc_datetime(value) -> datetime:
    ts = pd.Timestamp(value).to_pydatetime()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def timeframe_to_seconds(timeframe: str) -> int:
    tf = timeframe.strip().lower()
    if tf.endswith("secs"):
        return int(tf.split()[0])
    if tf.endswith("sec"):
        return int(tf.split()[0])
    if tf.endswith("mins"):
        return int(tf.split()[0]) * 60
    if tf.endswith("min"):
        return int(tf.split()[0]) * 60
    if tf.endswith("hours") or tf.endswith("hour"):
        return int(tf.split()[0]) * 3600
    if tf.endswith("day") or tf.endswith("days"):
        return int(tf.split()[0]) * 86400
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def load_symbol_profiles(
    session: Session,
    timeframe: str,
    start: Optional[datetime],
    end: Optional[datetime],
    min_bars: int,
    min_avg_dollar_volume: float,
    max_symbols: int,
) -> List[SymbolProfile]:
    query = (
        select(
            Candle.symbol,
            func.count().label("bars"),
            func.avg(Candle.close * Candle.volume).label("avg_dollar_volume"),
            func.min(Candle.ts).label("first_ts"),
            func.max(Candle.ts).label("last_ts"),
        )
        .join(Symbol, Candle.symbol == Symbol.symbol)
        .where(Candle.tf == timeframe, Symbol.active.is_(True))
    )

    if start is not None:
        query = query.where(Candle.ts >= start)
    if end is not None:
        query = query.where(Candle.ts <= end)

    query = query.group_by(Candle.symbol)

    rows = session.execute(query).all()
    profiles: List[SymbolProfile] = []

    for row in rows:
        avg_dollar_volume = float(row.avg_dollar_volume or 0.0)
        if row.bars < min_bars:
            continue
        if avg_dollar_volume < min_avg_dollar_volume:
            continue
        profiles.append(
            SymbolProfile(
                symbol=row.symbol,
                bars=row.bars,
                avg_dollar_volume=avg_dollar_volume,
                first_ts=row.first_ts,
                last_ts=row.last_ts,
            )
        )

    profiles.sort(key=lambda p: p.avg_dollar_volume, reverse=True)
    if max_symbols is not None:
        profiles = profiles[: max_symbols]
    return profiles


def load_symbol_data(
    session: Session,
    symbol: str,
    timeframe: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Optional[SymbolData]:
    query = (
        select(Candle.ts, Candle.close, Candle.volume)
        .where(Candle.symbol == symbol, Candle.tf == timeframe)
        .order_by(Candle.ts)
    )

    if start is not None:
        query = query.where(Candle.ts >= start)
    if end is not None:
        query = query.where(Candle.ts <= end)

    rows = session.execute(query).all()
    if not rows:
        return None

    timestamps = np.array([pd.Timestamp(r.ts).to_datetime64() for r in rows], dtype="datetime64[ns]")
    close = np.array([float(r.close) for r in rows], dtype=np.float64)
    volume = np.array([float(r.volume) for r in rows], dtype=np.float64)

    return SymbolData(timestamps=timestamps, close=close, volume=volume)


def symbol_data_to_frame(data: SymbolData) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "timestamp": data.timestamps,
            "close": data.close,
            "volume": data.volume,
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


def compute_adf(series: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute Augmented Dickey-Fuller test for stationarity.
    
    Returns:
        Tuple of (test_statistic, pvalue) or (None, None) if test fails.
        Test statistic: more negative = more stationary (typically < -3.0 for strong stationarity)
        P-value: lower = more significant (typically < 0.05 for stationarity)
    """
    if adfuller is None:
        return None, None
    if len(series) < 20:
        return None, None
    try:
        result = adfuller(series, maxlag=1, regression="c")
        # result[0] = test statistic, result[1] = p-value
        return float(result[0]), float(result[1])
    except Exception as e:
        LOGGER.debug(f"ADF test failed for series of length {len(series)}: {e}", exc_info=True)
        return None, None


def compute_cointegration(series_a: np.ndarray, series_b: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute cointegration test between two series.
    
    Returns:
        Tuple of (test_statistic, pvalue) or (None, None) if test fails.
        Test statistic: more negative = stronger cointegration
        P-value: lower = more significant (typically < 0.05 for cointegration)
    """
    if coint is None:
        return None, None
    if len(series_a) < 20 or len(series_b) < 20:
        return None, None
    if len(series_a) != len(series_b):
        LOGGER.debug(f"Series length mismatch: {len(series_a)} vs {len(series_b)}")
        return None, None
    try:
        result = coint(series_a, series_b)
        # result[0] = test statistic, result[1] = p-value
        return float(result[0]), float(result[1])
    except Exception as e:
        LOGGER.debug(f"Cointegration test failed for series of length {len(series_a)}: {e}", exc_info=True)
        return None, None


def estimate_half_life_minutes(spread: np.ndarray, bar_seconds: int) -> Optional[float]:
    if len(spread) < 3:
        return None

    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)

    if np.all(np.isclose(spread_diff, 0.0)):
        return None

    try:
        beta = np.linalg.lstsq(spread_lag[:, np.newaxis], spread_diff, rcond=None)[0][0]
    except np.linalg.LinAlgError:
        return None

    if beta >= 0:
        return None

    half_life_bars = -np.log(2) / beta
    if not np.isfinite(half_life_bars) or half_life_bars <= 0:
        return None

    return float(half_life_bars * (bar_seconds / 60.0))


def simple_spread_backtest(
    zscore: np.ndarray,
    spread: np.ndarray,
    params: AnalysisParams,
) -> Tuple[int, int, float, float, float, float, List[float], List[int], List[float]]:
    position = 0  # 1 = long spread, -1 = short spread
    entry_spread = 0.0
    entry_index = 0
    closed_pnls: List[float] = []
    holding_bars: List[int] = []
    equity_curve: List[float] = []
    equity = 0.0

    for idx in range(len(zscore)):
        z = zscore[idx]
        s = spread[idx]

        if position == 0:
            if z >= params.entry_z:
                position = -1
                entry_spread = s
                entry_index = idx
            elif z <= -params.entry_z:
                position = 1
                entry_spread = s
                entry_index = idx
        else:
            current_pnl = (s - entry_spread) * position
            exit_condition = False

            if position == 1 and z >= -params.exit_z:
                exit_condition = True
            elif position == -1 and z <= params.exit_z:
                exit_condition = True

            if exit_condition:
                pnl = current_pnl
                closed_pnls.append(pnl)
                holding_bars.append(idx - entry_index)
                equity += pnl
                position = 0
                entry_spread = 0.0
            else:
                equity_curve.append(equity + current_pnl)
                continue

        equity_curve.append(equity)

    total_trades = len(closed_pnls)
    wins = sum(1 for pnl in closed_pnls if pnl > 0)
    total_pnl = float(sum(closed_pnls))
    max_drawdown = compute_max_drawdown(np.array(equity_curve)) if equity_curve else 0.0
    sharpe = compute_sharpe(np.array(equity_curve))
    profit_factor = compute_profit_factor(closed_pnls)

    return total_trades, wins, total_pnl, max_drawdown, sharpe, profit_factor, closed_pnls, holding_bars, equity_curve


def compute_max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdowns = peak - equity
    return float(drawdowns.max()) if drawdowns.size else 0.0


def compute_sharpe(equity: np.ndarray) -> float:
    if equity.size < 2:
        return 0.0
    returns = np.diff(equity)
    if np.allclose(returns, 0.0):
        return 0.0
    mean = returns.mean()
    std = returns.std()
    if std == 0:
        return 0.0
    return float((mean / std) * math.sqrt(len(returns)))


def compute_profit_factor(pnls: Iterable[float]) -> float:
    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p < 0]
    total_losses = sum(losses)
    if not wins or total_losses == 0:
        return 0.0
    return float(sum(wins) / total_losses)


def to_decimal(value: Optional[float]) -> Optional[Decimal]:
    if value is None or math.isnan(value):
        return None
    return Decimal(str(value))


def analyze_pair(
    symbol_a: str,
    symbol_b: str,
    data_a: SymbolData,
    data_b: SymbolData,
    profiles: Dict[str, SymbolProfile],
    params: AnalysisParams,
) -> Optional[Dict[str, object]]:
    df_a = symbol_data_to_frame(data_a)
    df_b = symbol_data_to_frame(data_b)
    merged = df_a.join(df_b, how="inner", lsuffix="_a", rsuffix="_b")

    if merged.empty or len(merged) < params.min_bars:
        return None

    price_a = merged["close_a"].to_numpy(dtype=np.float64)
    price_b = merged["close_b"].to_numpy(dtype=np.float64)

    if np.any(price_a <= 0) or np.any(price_b <= 0):
        return None

    log_a = np.log(price_a)
    log_b = np.log(price_b)
    X = np.vstack([log_b, np.ones(len(log_b))]).T
    beta, alpha = np.linalg.lstsq(X, log_a, rcond=None)[0]

    spread = log_a - (alpha + beta * log_b)
    spread_mean = float(spread.mean())
    spread_std = float(spread.std())

    if spread_std < params.min_spread_std:
        return None

    zscore = (spread - spread_mean) / spread_std

    adf_statistic, adf_pvalue = compute_adf(spread)
    coint_statistic, coint_pvalue = compute_cointegration(log_a, log_b)
    half_life_minutes = estimate_half_life_minutes(spread, params.bar_seconds)

    (
        total_trades,
        wins,
        total_pnl,
        max_drawdown,
        sharpe,
        profit_factor,
        pnls,
        holding_bars,
        equity_curve,
    ) = simple_spread_backtest(zscore, spread, params)

    if total_trades < params.min_trades:
        return None

    win_rate = wins / total_trades if total_trades else 0.0
    avg_holding_minutes = (
        (np.mean(holding_bars) * params.bar_seconds / 60.0) if holding_bars else 0.0
    )

    profile_a = profiles[symbol_a]
    profile_b = profiles[symbol_b]

    return {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "timeframe": params.timeframe,
        "window_start": ensure_utc_datetime(merged.index[0]),
        "window_end": ensure_utc_datetime(merged.index[-1]),
        "sample_bars": len(merged),
        "avg_dollar_volume_a": profile_a.avg_dollar_volume,
        "avg_dollar_volume_b": profile_b.avg_dollar_volume,
        "hedge_ratio": float(beta),
        "hedge_intercept": float(alpha),
        "adf_statistic": adf_statistic,  # Test statistic (more negative = more stationary)
        "adf_pvalue": adf_pvalue,  # P-value (lower = more significant)
        "coint_statistic": coint_statistic,  # Test statistic (more negative = stronger)
        "coint_pvalue": coint_pvalue,  # P-value (lower = more significant)
        "half_life_minutes": half_life_minutes,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
        "simulated_entry_z": params.entry_z,
        "simulated_exit_z": params.exit_z,
        "pair_sharpe": sharpe,
        "pair_profit_factor": profit_factor,
        "pair_max_drawdown": max_drawdown,
        "pair_avg_holding_minutes": avg_holding_minutes,
        "pair_total_trades": total_trades,
        "pair_win_rate": win_rate,
        "total_pnl": total_pnl,
        "pnls": pnls,
        "equity_curve": equity_curve,
        "spread_series": spread,
        "zscore_series": zscore,
    }


def insert_results(results: List[Dict[str, object]], status: str, replace_existing: bool, timeframe: str, start: Optional[datetime], end: Optional[datetime]) -> None:
    if not results:
        LOGGER.warning("No candidate pairs to insert.")
        return

    engine = get_engine()
    metadata = MetaData()
    potential_pairs = Table("potential_pairs", metadata, autoload_with=engine)

    rows = []
    for result in results:
        rows.append(
            {
                "symbol_a": result["symbol_a"],
                "symbol_b": result["symbol_b"],
                "timeframe": result["timeframe"],
                "window_start": result["window_start"],
                "window_end": result["window_end"],
                "sample_bars": result["sample_bars"],
                "avg_dollar_volume_a": to_decimal(result["avg_dollar_volume_a"]),
                "avg_dollar_volume_b": to_decimal(result["avg_dollar_volume_b"]),
                "hedge_ratio": to_decimal(result["hedge_ratio"]),
                "hedge_intercept": to_decimal(result["hedge_intercept"]),
                "adf_statistic": to_decimal(result["adf_statistic"]) if result["adf_statistic"] is not None else None,
                "adf_pvalue": to_decimal(result["adf_pvalue"]) if result["adf_pvalue"] is not None else None,
                "coint_statistic": to_decimal(result["coint_statistic"]) if result["coint_statistic"] is not None else None,
                "coint_pvalue": to_decimal(result["coint_pvalue"]) if result["coint_pvalue"] is not None else None,
                "half_life_minutes": to_decimal(result["half_life_minutes"]) if result["half_life_minutes"] is not None else None,
                "spread_mean": to_decimal(result["spread_mean"]),
                "spread_std": to_decimal(result["spread_std"]),
                "simulated_entry_z": to_decimal(result["simulated_entry_z"]),
                "simulated_exit_z": to_decimal(result["simulated_exit_z"]),
                "pair_sharpe": to_decimal(result["pair_sharpe"]),
                "pair_profit_factor": to_decimal(result["pair_profit_factor"]),
                "pair_max_drawdown": to_decimal(result["pair_max_drawdown"]),
                "pair_avg_holding_minutes": to_decimal(result["pair_avg_holding_minutes"]),
                "pair_total_trades": result["pair_total_trades"],
                "pair_win_rate": to_decimal(result["pair_win_rate"]),
                "status": status,
                "meta": {
                    "total_pnl": result["total_pnl"],
                    "trade_pnls": result["pnls"],
                    "equity_curve": result["equity_curve"],
                    "adf_statistic": result.get("adf_statistic"),  # Store ADF test statistic in meta
                    "coint_statistic": result.get("coint_statistic"),  # Store cointegration test statistic in meta
                },
            }
        )

    def _write(session: Session):
        if replace_existing:
            stmt = delete(potential_pairs).where(potential_pairs.c.timeframe == timeframe)
            if start is not None:
                stmt = stmt.where(potential_pairs.c.window_end >= start)
            if end is not None:
                stmt = stmt.where(potential_pairs.c.window_start <= end)
            session.execute(stmt)
        session.execute(potential_pairs.insert(), rows)

    execute_with_retry(_write)
    LOGGER.info("Inserted %d rows into potential_pairs", len(rows))



def run_analysis(args: Optional[Sequence[str]] = None) -> None:
    namespace = parse_args(args)
    namespace.parallelism = max(1, min(namespace.parallelism, 26))
    logging.basicConfig(level=logging.DEBUG if namespace.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    now = datetime.now(timezone.utc)
    default_end = parse_timestamp(namespace.end_date, now)
    default_start = parse_timestamp(
        namespace.start_date,
        (default_end - timedelta(days=namespace.lookback_days)) if default_end else now - timedelta(days=namespace.lookback_days),
    )

    LOGGER.info(
        "Analyzing pairs for timeframe=%s, window=[%s, %s]",
        namespace.timeframe,
        default_start.isoformat() if default_start else "earliest",
        default_end.isoformat() if default_end else "latest",
    )

    bar_seconds = timeframe_to_seconds(namespace.timeframe)
    params = AnalysisParams(
        timeframe=namespace.timeframe,
        bar_seconds=bar_seconds,
        entry_z=namespace.entry_z,
        exit_z=namespace.exit_z,
        min_bars=namespace.min_bars,
        min_spread_std=1e-6,
        min_trades=namespace.min_trades,
    )

    def _load_profiles(session: Session) -> List[SymbolProfile]:
        return load_symbol_profiles(
            session=session,
            timeframe=namespace.timeframe,
            start=default_start,
            end=default_end,
            min_bars=namespace.min_bars,
            min_avg_dollar_volume=namespace.min_dollar_volume,
            max_symbols=namespace.max_symbols,
        )

    profiles = execute_with_retry(_load_profiles)

    if not profiles:
        LOGGER.warning("No symbols met the liquidity/coverage requirements.")
        return

    LOGGER.info("Selected %d symbols for pairing.", len(profiles))

    profile_map = {profile.symbol: profile for profile in profiles}

    def _load_all_data(session: Session) -> Dict[str, SymbolData]:
        data_map: Dict[str, SymbolData] = {}
        for profile in profiles:
            data = load_symbol_data(
                session=session,
                symbol=profile.symbol,
                timeframe=namespace.timeframe,
                start=default_start,
                end=default_end,
            )
            if data is None:
                LOGGER.warning("No data loaded for %s; skipping.", profile.symbol)
                continue
            data_map[profile.symbol] = data
        return data_map

    symbol_data_map = execute_with_retry(_load_all_data)

    missing_symbols = set(profile_map) - set(symbol_data_map)
    if missing_symbols:
        LOGGER.warning("Skipping %d symbols with missing data: %s", len(missing_symbols), ", ".join(sorted(missing_symbols)))
        for symbol in missing_symbols:
            profile_map.pop(symbol, None)

    # Use symbols in original order (from data map) rather than alphabetical
    # This avoids bias where pairs with alphabetically first symbols appear more frequently
    symbols = list(symbol_data_map.keys())
    combinations = list(itertools.combinations(symbols, 2))

    if namespace.max_pairs is not None:
        combinations = combinations[: namespace.max_pairs]

    LOGGER.info("Evaluating %d symbol pairs.", len(combinations))

    results: List[Dict[str, object]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=namespace.parallelism) as executor:
        future_to_pair = {
            executor.submit(
                analyze_pair,
                sym_a,
                sym_b,
                symbol_data_map[sym_a],
                symbol_data_map[sym_b],
                profile_map,
                params,
            ): (sym_a, sym_b)
            for sym_a, sym_b in combinations
        }

        for future in concurrent.futures.as_completed(future_to_pair):
            pair = future_to_pair[future]
            try:
                result = future.result()
            except Exception as exc:
                LOGGER.exception("Pair analysis failed for %s/%s: %s", pair[0], pair[1], exc)
                continue
            if result:
                results.append(result)

    LOGGER.info("Completed analysis. %d pairs passed filters.", len(results))
    insert_results(results, status=namespace.status, replace_existing=namespace.replace_existing, timeframe=namespace.timeframe, start=default_start, end=default_end)


if __name__ == "__main__":
    run_analysis()

