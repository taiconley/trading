"""
Pairs Trading Strategy

A statistical arbitrage strategy that trades two correlated stocks by going
long one and short the other when their price ratio deviates from the mean.

Classic pairs trading strategy:
1. Calculate the price ratio between two stocks
2. Calculate the z-score of the ratio
3. When z-score is high: short expensive stock, long cheap stock
4. When z-score is low: long expensive stock, short cheap stock
5. Exit when ratio returns to mean (z-score near 0)
"""

from collections import deque
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .base import BaseStrategy, StrategyConfig, StrategySignal, SignalType, StrategyState
from .registry import strategy
from .pairs_trading_config import PAIRS_TRADING_CONFIG

try:  # Optional dependency for stationarity tests
    from statsmodels.tsa.stattools import adfuller, coint
except Exception:  # pragma: no cover - best effort import
    adfuller = None
    coint = None


class PairsTradingConfig(StrategyConfig):
    """Configuration for Pairs Trading Strategy."""
    
    # Pairs definition - list of [stock_a, stock_b] pairs
    # Example: pairs = [["AAPL", "MSFT"], ["JPM", "BAC"], ["XOM", "CVX"]]
    pairs: List[List[str]] = []
    
    # Pairs parameters
    lookback_window: int = 240  # Window for calculating mean and std (240 bars = 20 minutes at 5-sec bars)
    entry_threshold: float = 2.0  # Z-score threshold to enter trade
    exit_threshold: float = 0.5   # Z-score threshold to exit trade
    
    # Position sizing
    position_size: int = 100  # Shares per leg
    
    # Order execution type
    execution_type: str = "ADAPTIVE"  # Default: Limit order. Options: "MKT", "LMT", "ADAPTIVE", "PEG BEST", "PEG MID"
    use_adaptive: bool = False  # If True, use Adaptive order (overrides execution_type)
    use_pegged: bool = False  # If True, use pegged order (overrides execution_type)
    pegged_type: Optional[str] = None  # "BEST" or "MID" for pegged orders
    adaptive_priority: str = "Normal"  # "Patient", "Normal", "Urgent" for Adaptive orders
    pegged_offset: float = 0.01  # Offset from midpoint/best for pegged orders (in dollars)
    
    # Risk management (for 5-second bars: 720 bars = 1 hour)
    max_hold_bars: int = 720  # Maximum bars to hold a position
    stop_loss_zscore: float = 3.0  # Exit if z-score gets worse
    cooldown_bars: int = 60  # Cooling-off period after stop loss exits
    
    # Intraday-only trading
    market_close_hour: int = 16  # US market close hour (4 PM Eastern)
    market_close_minute: int = 0  # Market close minute
    close_before_eod_minutes: int = 5  # Close all positions N minutes before market close
    timezone: str = "US/Eastern"  # Trading session timezone

    # Statistical modelling
    spread_history_bars: int = 1000  # History length for spread calculations
    hedge_refresh_bars: int = 30  # How often to refresh hedge ratio (in bars)
    min_hedge_lookback: int = 120  # Minimum bars required before hedge regression
    stationarity_checks_enabled: bool = True
    adf_pvalue_threshold: float = 0.05  # Maximum acceptable p-value for ADF test
    cointegration_pvalue_threshold: float = 0.05  # Maximum acceptable p-value for cointegration test
    stationarity_check_interval: int = 60  # Bars between stationarity checks
    volatility_adaptation_enabled: bool = True
    volatility_window: int = 240  # Bars used for volatility adaptation
    volatility_ema_alpha: float = 0.2  # EMA smoothing for baseline volatility
    min_volatility_ratio: float = 0.75  # Lower clamp for volatility ratio
    max_volatility_ratio: float = 1.5   # Upper clamp for volatility ratio
    min_exit_volatility_ratio: float = 0.8  # Lower clamp for exit adjustment
    max_exit_volatility_ratio: float = 1.3  # Upper clamp for exit adjustment
    
    class Config:
        extra = "allow"


@strategy(
    name="Pairs_Trading",
    description="Intraday statistical arbitrage strategy for 5-second bars, trading price ratio deviations between multiple pairs",
    default_config=PAIRS_TRADING_CONFIG
)
class PairsTradingStrategy(BaseStrategy):
    """
    Pairs Trading Strategy.
    
    Monitors the price ratio between two stocks and trades when the ratio
    deviates significantly from its historical mean.
    """
    
    def __init__(self, config: StrategyConfig):
        # Convert base config to Pairs-specific config
        if not isinstance(config, PairsTradingConfig):
            pairs_config_data = config.dict() if hasattr(config, 'dict') else config.__dict__
            # Merge parameters into the main config
            if 'parameters' in pairs_config_data:
                pairs_config_data.update(pairs_config_data['parameters'])
            config = PairsTradingConfig(**pairs_config_data)
        
        # Parse pairs from config
        if hasattr(config, 'pairs') and config.pairs is not None:
            # pairs is explicitly set (could be [] for no pairs, which is valid)
            self.pairs = config.pairs
        elif hasattr(config, 'symbols') and len(config.symbols) >= 2:
            # Fallback: treat consecutive symbols as pairs
            # e.g., ["AAPL", "MSFT", "JPM", "BAC"] -> [["AAPL", "MSFT"], ["JPM", "BAC"]]
            if len(config.symbols) % 2 != 0:
                raise ValueError(f"Pairs trading requires even number of symbols, got {len(config.symbols)}")
            self.pairs = [[config.symbols[i], config.symbols[i+1]] for i in range(0, len(config.symbols), 2)]
        else:
            raise ValueError("Pairs trading requires 'pairs' parameter or even number of 'symbols'")
        
        # Extract all unique symbols for the config
        all_symbols = list(set([sym for pair in self.pairs for sym in pair]))
        config.symbols = all_symbols
        
        # Ensure spread history length covers lookback
        if config.spread_history_bars < config.lookback_window:
            config.spread_history_bars = max(config.lookback_window, config.spread_history_bars)
        
        super().__init__(config)

        # Trading session timezone
        try:
            self.market_timezone = ZoneInfo(self.config.timezone)
        except Exception:
            fallback_tz = "US/Eastern"
            self.market_timezone = ZoneInfo(fallback_tz)
            self.log_warning(
                f"Invalid timezone '{self.config.timezone}' provided. Falling back to {fallback_tz}."
            )
            self.config.timezone = fallback_tz
        
        # Pre-compute clamps
        self._vol_ratio_bounds = (
            min(self.config.min_volatility_ratio, self.config.max_volatility_ratio),
            max(self.config.min_volatility_ratio, self.config.max_volatility_ratio)
        )
        self._exit_vol_ratio_bounds = (
            min(self.config.min_exit_volatility_ratio, self.config.max_exit_volatility_ratio),
            max(self.config.min_exit_volatility_ratio, self.config.max_exit_volatility_ratio)
        )
        
        # State tracking for each pair
        # Key is "SYMBOL_A/SYMBOL_B"
        self._pair_states: Dict[str, Dict[str, Any]] = {}
        
        for pair in self.pairs:
            pair_key = f"{pair[0]}/{pair[1]}"
            # Ensure price_history is long enough for all use cases:
            # - spread_history_bars: for spread calculations
            # - lookback_window: for z-score lookback
            # - min_hedge_lookback: for hedge ratio calculation
            maxlen = max(
                self.config.spread_history_bars,
                self.config.lookback_window,
                self.config.min_hedge_lookback
            )
            self._pair_states[pair_key] = {
                'stock_a': pair[0],
                'stock_b': pair[1],
                'price_history_a': deque(maxlen=maxlen),
                'price_history_b': deque(maxlen=maxlen),
                'spread_history': deque(maxlen=self.config.spread_history_bars),
                'hedge_ratio': 1.0,
                'hedge_intercept': 0.0,
                'bars_since_hedge': 0,
                'bars_since_stationarity': 0,
                'bars_since_entry': 0,
                'adf_pvalue': None,
                'cointegration_pvalue': None,
                'baseline_spread_std': None,
                'current_position': 'flat',  # "flat", "long_a_short_b", "short_a_long_b"
                'entry_zscore': 0.0,
                'entry_timestamp': None,
                'entry_prices': {},
                'entry_spread': None,
                'bars_in_trade': 0,
                'cooldown_remaining': 0,
            }
        
        self.log_info(f"Initialized pairs trading with {len(self.pairs)} pairs: {self.pairs}")
    
    @property
    def supports_multi_symbol(self) -> bool:
        """This strategy needs to see both symbols together."""
        return True
    
    @classmethod
    def get_parameters_schema(cls) -> Dict[str, Any]:
        """Return parameter schema for validation."""
        return {
            "lookback_window": {
                "type": int,
                "required": False,
                "min": 5,
                "max": 2000,
                "description": "Bars used for spread/z-score calculation"
            },
            "entry_threshold": {
                "type": float,
                "required": False,
                "min": 0.5,
                "max": 5.0,
                "description": "Z-score threshold to enter trade"
            },
            "exit_threshold": {
                "type": float,
                "required": False,
                "min": 0.0,
                "max": 2.0,
                "description": "Z-score threshold to exit trade"
            },
            "position_size": {
                "type": int,
                "required": False,
                "min": 1,
                "description": "Position size (shares per leg)"
            },
            "cooldown_bars": {
                "type": int,
                "required": False,
                "min": 0,
                "description": "Bars to wait after a stop-loss exit before re-entering"
            },
            "hedge_refresh_bars": {
                "type": int,
                "required": False,
                "min": 1,
                "description": "Frequency (in bars) to refresh hedge ratio"
            },
            "stationarity_check_interval": {
                "type": int,
                "required": False,
                "min": 10,
                "description": "Bars between ADF/cointegration checks"
            },
            "adf_pvalue_threshold": {
                "type": float,
                "required": False,
                "min": 0.0,
                "max": 0.25,
                "description": "Maximum ADF p-value to allow trading"
            },
            "cointegration_pvalue_threshold": {
                "type": float,
                "required": False,
                "min": 0.0,
                "max": 0.25,
                "description": "Maximum cointegration p-value to allow trading"
            },
            "volatility_window": {
                "type": int,
                "required": False,
                "min": 20,
                "description": "Bars used to adapt entry/exit thresholds"
            }
        }
    
    async def on_start(self, instruments: Dict[str, Any]) -> None:
        """Initialize strategy state."""
        self.set_state(StrategyState.RUNNING)
        
        # Reset all pair states
        for pair_key in self._pair_states:
            state = self._pair_states[pair_key]
            state['price_history_a'].clear()
            state['price_history_b'].clear()
            state['spread_history'].clear()
            state['hedge_ratio'] = 1.0
            state['hedge_intercept'] = 0.0
            state['bars_since_hedge'] = 0
            state['bars_since_stationarity'] = 0
            state['bars_since_entry'] = 0
            state['adf_pvalue'] = None
            state['cointegration_pvalue'] = None
            state['baseline_spread_std'] = None
            self._pair_states[pair_key]['current_position'] = 'flat'
            self._pair_states[pair_key]['entry_zscore'] = 0.0
            self._pair_states[pair_key]['entry_timestamp'] = None
            self._pair_states[pair_key]['entry_prices'] = {}
            self._pair_states[pair_key]['entry_spread'] = None
            self._pair_states[pair_key]['bars_in_trade'] = 0
            self._pair_states[pair_key]['cooldown_remaining'] = 0
        
        self.log_info(
            f"Started intraday pairs trading strategy with {len(self.pairs)} pairs, "
            f"entry_threshold={self.config.entry_threshold}, "
            f"exit_threshold={self.config.exit_threshold}, "
            f"max_hold_bars={self.config.max_hold_bars}"
        )
    
    async def on_bar_multi(self, symbols: List[str], timeframe: str, 
                          bars_data: Dict[str, pd.DataFrame]) -> List[StrategySignal]:
        """Process bar data for all pairs simultaneously."""
        
        # Suppress verbose logs during optimization (strategy_id starts with 'opt_')
        is_optimization = self.config.strategy_id.startswith('opt_')
        
        # DEBUG: Log data availability (only occasionally to avoid spam)
        if not is_optimization:
            import random
            if random.random() < 0.05:  # 5% of the time
                bar_counts = {sym: len(df) for sym, df in bars_data.items()}
                self.log_info(f"[DATA] Processing {len(bars_data)} symbols, bar counts: {bar_counts}")
        
        all_signals = []
        # Process each pair independently
        for pair in self.pairs:
            stock_a, stock_b = pair[0], pair[1]
            pair_key = f"{stock_a}/{stock_b}"
            
            # Ensure we have data for both stocks in this pair
            if stock_a not in bars_data or stock_b not in bars_data:
                continue
            
            bars_a = bars_data[stock_a]
            bars_b = bars_data[stock_b]
            
            # Ensure both DataFrames have the same length and are aligned
            if len(bars_a) != len(bars_b):
                # Align by timestamp if possible
                common_timestamps = bars_a['timestamp'].isin(bars_b['timestamp'])
                if common_timestamps.sum() == 0:
                    continue
                bars_a = bars_a[common_timestamps]
                bars_b = bars_b[bars_b['timestamp'].isin(bars_a['timestamp'])]
            
            if len(bars_a) == 0 or len(bars_b) == 0:
                continue
            
            # Get pair state
            pair_state = self._pair_states[pair_key]
            
            # Process all bars in the DataFrame to build up history quickly
            # This is important on startup when we have historical data
            initial_spread_len = len(pair_state['spread_history'])
            
            for idx in range(len(bars_a)):
                if idx >= len(bars_b):
                    break
                
                bar_a = bars_a.iloc[idx]
                bar_b = bars_b.iloc[idx]
                
                # Extract timestamp from bar
                if 'timestamp' not in bar_a.index:
                    continue
                timestamp = pd.to_datetime(bar_a['timestamp'])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.to_pydatetime().replace(tzinfo=self.market_timezone)
                else:
                    timestamp = timestamp.to_pydatetime().astimezone(self.market_timezone)
                
                # Get prices
                price_a = float(bar_a['close'])
                price_b = float(bar_b['close'])
                
                if price_b == 0 or price_a <= 0 or price_b <= 0:
                    continue
                
                # Update counters
                pair_state['bars_since_hedge'] += 1
                pair_state['bars_since_stationarity'] += 1
                pair_state['bars_since_entry'] += 1
                
                # Reduce cooldown when flat
                if pair_state['current_position'] == "flat" and pair_state['cooldown_remaining'] > 0:
                    pair_state['cooldown_remaining'] = max(0, pair_state['cooldown_remaining'] - 1)
                
                # Append price history
                pair_state['price_history_a'].append(price_a)
                pair_state['price_history_b'].append(price_b)
                
                # Need enough data to build hedge ratio before computing spreads
                if len(pair_state['price_history_a']) < self.config.min_hedge_lookback:
                    continue  # Skip spread calculation until we have enough price history
                
                # Refresh hedge ratio periodically
                if pair_state['bars_since_hedge'] >= self.config.hedge_refresh_bars or pair_state['hedge_ratio'] is None:
                    self._refresh_hedge_ratio(pair_key, pair_state)
                
                # Compute current spread using hedge ratio
                spread = self._compute_spread(pair_state, price_a, price_b)
                if spread is None:
                    continue
                pair_state['spread_history'].append(spread)
            
            # After processing all bars, check if we have enough data for trading decisions
            # Only process trading logic on the LAST bar to avoid duplicate signals
            if len(bars_a) == 0:
                continue
            
            last_timestamp = self._extract_timestamp(bars_a)
            if last_timestamp is None:
                continue
            
            close_dt = datetime.combine(
                last_timestamp.date(),
                time(self.config.market_close_hour, self.config.market_close_minute),
                tzinfo=self.market_timezone
            ) - timedelta(minutes=self.config.close_before_eod_minutes)
            is_near_close = last_timestamp >= close_dt
            
            # Get current prices (from last bar)
            price_a = float(bars_a['close'].iloc[-1])
            price_b = float(bars_b['close'].iloc[-1])
            
            if pair_state['current_position'] != "flat":
                pair_state['bars_in_trade'] += 1
            
            # Update adaptive thresholds and volatility ratio
            vol_ratio, adjusted_entry, adjusted_exit = self._get_adaptive_thresholds(pair_state)
            
            # Update stationarity metrics periodically
            self._update_stationarity_metrics(pair_key, pair_state)
            
            # Require sufficient spread history for statistics
            spread_hist_len = len(pair_state['spread_history'])
            if spread_hist_len < self.config.lookback_window:
                # DEBUG: Log first few times to show progress (suppress during optimization)
                if not is_optimization and (spread_hist_len % 20 == 0 or spread_hist_len < 10):
                    self.log_info(
                        f"[WARMING UP] {stock_a}/{stock_b}: {spread_hist_len}/{self.config.lookback_window} bars collected"
                    )
                continue
            
            # Get current spread for z-score calculation
            current_spread = self._compute_spread(pair_state, price_a, price_b)
            if current_spread is None:
                continue
            
            recent_spreads = list(pair_state['spread_history'])[-self.config.lookback_window:]
            mean_spread = float(np.mean(recent_spreads))
            std_spread = float(np.std(recent_spreads))
            if std_spread == 0:
                continue
            
            zscore = (current_spread - mean_spread) / std_spread
            pair_state['last_zscore'] = zscore
            
            # DEBUG: Log z-score calculations every ~30 bars (suppress during optimization)
            if not is_optimization and pair_state['bars_since_entry'] % 30 == 0:
                self.log_info(
                    f"[Z-SCORE] {stock_a}/{stock_b}: z={zscore:.3f}, spread={current_spread:.4f}, "
                    f"mean={mean_spread:.4f}, std={std_spread:.4f}, pos={pair_state['current_position']}"
                )
            
            # Stationarity gating
            if self.config.stationarity_checks_enabled:
                adf_pval = pair_state.get('adf_pvalue')
                coint_pval = pair_state.get('cointegration_pvalue')
                
                if adfuller is not None and adf_pval is not None:
                    if adf_pval > self.config.adf_pvalue_threshold:
                        if not is_optimization:
                            self.log_info(
                                f"[STATIONARITY BLOCK] {stock_a}/{stock_b}: "
                                f"ADF p-value {adf_pval:.4f} > threshold {self.config.adf_pvalue_threshold}, "
                                f"z={zscore:.3f}, skipping entry"
                            )
                        continue
                
                if coint is not None and coint_pval is not None:
                    if coint_pval > self.config.cointegration_pvalue_threshold:
                        if not is_optimization:
                            self.log_info(
                                f"[STATIONARITY BLOCK] {stock_a}/{stock_b}: "
                                f"Cointegration p-value {coint_pval:.4f} > threshold {self.config.cointegration_pvalue_threshold}, "
                                f"z={zscore:.3f}, skipping entry"
                            )
                        continue
                
                # Log when stationarity checks pass but we're near entry
                if not is_optimization and abs(zscore) > adjusted_entry * 0.8:
                    adf_str = f"{adf_pval:.4f}" if adf_pval is not None else "N/A"
                    coint_str = f"{coint_pval:.4f}" if coint_pval is not None else "N/A"
                    self.log_info(
                        f"[STATIONARITY OK] {stock_a}/{stock_b}: "
                        f"ADF={adf_str}, Coint={coint_str}, "
                        f"z={zscore:.3f}, entry={adjusted_entry:.3f}"
                    )
            
            # Force exit near the close
            if is_near_close and pair_state['current_position'] != "flat":
                signals = self._generate_exit_signals(
                    pair_key=pair_key,
                    exit_reason="end_of_day",
                    zscore=zscore,
                    spread=current_spread,
                    mean_spread=mean_spread,
                    std_spread=std_spread,
                    price_a=price_a,
                    price_b=price_b,
                    stock_a=stock_a,
                    stock_b=stock_b,
                    timestamp=last_timestamp,
                    volatility_ratio=vol_ratio
                )
                all_signals.extend(signals)
                continue
            
            signals = []
            if pair_state['current_position'] == "flat":
                cooldown = pair_state.get('cooldown_remaining', 0)
                if cooldown > 0:
                    if not is_optimization:
                        self.log_info(
                            f"[SKIP ENTRY] {stock_a}/{stock_b}: z={zscore:.3f}, "
                            f"entry_threshold={adjusted_entry:.3f}, cooldown={cooldown}"
                        )
                    continue
                
                # Look for entry signals
                if zscore > adjusted_entry:
                    # Ratio is too high: short A, long B
                    if not is_optimization:
                        self.log_info(
                            f"[ENTRY SIGNAL] {stock_a}/{stock_b}: z={zscore:.3f} > "
                            f"entry={adjusted_entry:.3f}, generating SHORT A / LONG B signals"
                        )
                    signals = self._generate_entry_signals(
                        pair_key=pair_key,
                        position_type="short_a_long_b",
                        zscore=zscore,
                        spread=current_spread,
                        mean_spread=mean_spread,
                        std_spread=std_spread,
                        price_a=price_a,
                        price_b=price_b,
                        stock_a=stock_a,
                        stock_b=stock_b,
                        timestamp=last_timestamp,
                        entry_threshold=adjusted_entry,
                        volatility_ratio=vol_ratio
                    )
                    
                elif zscore < -adjusted_entry:
                    # Ratio is too low: long A, short B
                    if not is_optimization:
                        self.log_info(
                            f"[ENTRY SIGNAL] {stock_a}/{stock_b}: z={zscore:.3f} < "
                            f"-entry={-adjusted_entry:.3f}, generating LONG A / SHORT B signals"
                        )
                    signals = self._generate_entry_signals(
                        pair_key=pair_key,
                        position_type="long_a_short_b",
                        zscore=zscore,
                        spread=current_spread,
                        mean_spread=mean_spread,
                        std_spread=std_spread,
                        price_a=price_a,
                        price_b=price_b,
                        stock_a=stock_a,
                        stock_b=stock_b,
                        timestamp=last_timestamp,
                        entry_threshold=adjusted_entry,
                        volatility_ratio=vol_ratio
                    )
                else:
                    # Not at entry threshold yet
                    if not is_optimization and abs(zscore) > adjusted_entry * 0.8:
                        # Log when close to entry (80% of threshold)
                        self.log_info(
                            f"[NEAR ENTRY] {stock_a}/{stock_b}: z={zscore:.3f}, "
                            f"entry={adjusted_entry:.3f}, proximity={abs(zscore)/adjusted_entry:.1%}"
                        )
            
            else:
                # Already in a position - check for exit signals
                
                # Exit condition 1: Z-score returns to normal
                should_exit_mean_reversion = abs(zscore) < adjusted_exit
                
                # Exit condition 2: Hit stop loss (z-score got worse)
                should_exit_stop_loss = False
                if pair_state['current_position'] == "short_a_long_b" and zscore > self.config.stop_loss_zscore:
                    should_exit_stop_loss = True
                elif pair_state['current_position'] == "long_a_short_b" and zscore < -self.config.stop_loss_zscore:
                    should_exit_stop_loss = True
                
                # Exit condition 3: Held too long
                should_exit_time = pair_state['bars_in_trade'] >= self.config.max_hold_bars
                
                if should_exit_mean_reversion or should_exit_stop_loss or should_exit_time:
                    exit_reason = (
                        "mean_reversion" if should_exit_mean_reversion else
                        "stop_loss" if should_exit_stop_loss else
                        "max_hold_time"
                    )
                    
                    signals = self._generate_exit_signals(
                        pair_key=pair_key,
                        exit_reason=exit_reason,
                        zscore=zscore,
                        spread=spread,
                        mean_spread=mean_spread,
                        std_spread=std_spread,
                        price_a=price_a,
                        price_b=price_b,
                        stock_a=stock_a,
                        stock_b=stock_b,
                        timestamp=last_timestamp,
                        volatility_ratio=vol_ratio
                    )
            
            # Add signals from this pair to the overall list
            all_signals.extend(signals)
        
        return all_signals
    
    def _extract_timestamp(self, bars: pd.DataFrame) -> Optional[datetime]:
        """Extract the latest timestamp from a bars DataFrame in strategy timezone."""
        if bars.empty or 'timestamp' not in bars.columns:
            return None
        ts = pd.to_datetime(bars['timestamp'].iloc[-1])
        if ts.tzinfo is None:
            return ts.to_pydatetime().replace(tzinfo=self.market_timezone)
        else:
            if hasattr(ts, 'tz_convert'):
                ts = ts.tz_convert(self.market_timezone)
                return ts.to_pydatetime()
            return ts.to_pydatetime().astimezone(self.market_timezone)

    def _refresh_hedge_ratio(self, pair_key: str, pair_state: Dict[str, Any]) -> None:
        """Recalculate hedge ratio using OLS on log prices."""
        try:
            prices_a = np.array(pair_state['price_history_a'], dtype=float)
            prices_b = np.array(pair_state['price_history_b'], dtype=float)
            if len(prices_a) < 2 or np.any(prices_a <= 0) or np.any(prices_b <= 0):
                return
            log_a = np.log(prices_a)
            log_b = np.log(prices_b)
            X = np.vstack([log_b, np.ones(len(log_b))]).T
            beta, alpha = np.linalg.lstsq(X, log_a, rcond=None)[0]
            pair_state['hedge_ratio'] = float(beta)
            pair_state['hedge_intercept'] = float(alpha)
            pair_state['bars_since_hedge'] = 0
        except Exception as exc:  # pragma: no cover - defensive logging
            self.log_warning(f"Failed to refresh hedge ratio for {pair_key}: {exc}")

    def _compute_spread(self, pair_state: Dict[str, Any], price_a: float, price_b: float) -> Optional[float]:
        """Compute log-price spread using current hedge parameters."""
        if price_a <= 0 or price_b <= 0:
            return None
        log_a = np.log(price_a)
        log_b = np.log(price_b)
        return log_a - (pair_state['hedge_intercept'] + pair_state['hedge_ratio'] * log_b)

    def _get_adaptive_thresholds(self, pair_state: Dict[str, Any]) -> tuple[float, float, float]:
        """Return volatility ratio and adapted entry/exit thresholds."""
        base_entry = self.config.entry_threshold
        base_exit = self.config.exit_threshold
        if not self.config.volatility_adaptation_enabled:
            return 1.0, base_entry, base_exit
        if len(pair_state['spread_history']) < max(self.config.volatility_window, 10):
            return 1.0, base_entry, base_exit
        window = list(pair_state['spread_history'])[-self.config.volatility_window:]
        window_std = float(np.std(window))
        if window_std <= 0:
            return 1.0, base_entry, base_exit
        if pair_state['baseline_spread_std'] is None:
            pair_state['baseline_spread_std'] = window_std
        else:
            alpha = self.config.volatility_ema_alpha
            pair_state['baseline_spread_std'] = (
                alpha * window_std + (1 - alpha) * pair_state['baseline_spread_std']
            )
        baseline = pair_state['baseline_spread_std'] or window_std
        vol_ratio = float(window_std / baseline) if baseline > 0 else 1.0
        vol_ratio = float(np.clip(vol_ratio, *self._vol_ratio_bounds))
        exit_ratio = float(np.clip(vol_ratio, *self._exit_vol_ratio_bounds))
        entry_threshold = base_entry * vol_ratio
        exit_threshold = base_exit * exit_ratio
        return vol_ratio, entry_threshold, exit_threshold

    def _update_stationarity_metrics(self, pair_key: str, pair_state: Dict[str, Any]) -> None:
        """Periodically refresh ADF and cointegration statistics."""
        if not self.config.stationarity_checks_enabled:
            return
        if pair_state['bars_since_stationarity'] < self.config.stationarity_check_interval:
            return
        pair_state['bars_since_stationarity'] = 0
        # ADF on spread
        if adfuller is not None and len(pair_state['spread_history']) >= self.config.lookback_window:
            try:
                series = list(pair_state['spread_history'])[-self.config.lookback_window:]
                pair_state['adf_pvalue'] = float(adfuller(series)[1])
            except Exception as exc:  # pragma: no cover - defensive logging
                self.log_warning(f"ADF test failed for {pair_key}: {exc}")
        # Cointegration between price legs
        if coint is not None and len(pair_state['price_history_a']) >= self.config.min_hedge_lookback:
            prices_a = np.array(pair_state['price_history_a'], dtype=float)
            prices_b = np.array(pair_state['price_history_b'], dtype=float)
            if np.any(prices_a <= 0) or np.any(prices_b <= 0):
                return
            try:
                result = coint(np.log(prices_a), np.log(prices_b))
                pair_state['cointegration_pvalue'] = float(result[1])
            except Exception as exc:  # pragma: no cover - defensive logging
                self.log_warning(f"Cointegration test failed for {pair_key}: {exc}")
    
    def _get_execution_params(self) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """
        Determine execution type and algorithm parameters based on config.
        
        Returns:
            Tuple of (execution_type, algo_strategy, algo_params)
        """
        if self.config.use_adaptive:
            return (
                "ADAPTIVE",
                "Adaptive",
                {"adaptivePriority": self.config.adaptive_priority}
            )
        elif self.config.use_pegged:
            if self.config.pegged_type == "BEST":
                return (
                    "PEG BEST",
                    None,
                    {"offset": self.config.pegged_offset} if self.config.pegged_offset != 0.01 else None
                )
            elif self.config.pegged_type == "MID":
                return (
                    "PEG MID",
                    None,
                    {"offset": self.config.pegged_offset} if self.config.pegged_offset != 0.01 else None
                )
            else:
                # Default to MID if pegged_type not specified
                return (
                    "PEG MID",
                    None,
                    {"offset": self.config.pegged_offset} if self.config.pegged_offset != 0.01 else None
                )
        else:
            # Use configured execution_type (default: "LMT")
            # Special handling for ADAPTIVE orders to ensure algo params are set
            if self.config.execution_type == "ADAPTIVE":
                return (
                    "ADAPTIVE",
                    "Adaptive",
                    {"adaptivePriority": self.config.adaptive_priority}
                )
            return (self.config.execution_type, None, None)
    
    def _generate_entry_signals(
        self,
        pair_key: str,
        position_type: str,
        zscore: float,
        spread: float,
        mean_spread: float,
        std_spread: float,
        price_a: float,
        price_b: float,
        stock_a: str,
        stock_b: str,
        timestamp: datetime,
        entry_threshold: float,
        volatility_ratio: float
    ) -> List[StrategySignal]:
        """Generate entry signals for the pair."""
        signals = []
        pair_state = self._pair_states[pair_key]
        metadata_common = {
            'pair': pair_key,
            'position_type': position_type,
            'zscore': zscore,
            'spread': spread,
            'mean_spread': mean_spread,
            'std_spread': std_spread,
            'hedge_ratio': pair_state['hedge_ratio'],
            'hedge_intercept': pair_state['hedge_intercept'],
            'volatility_ratio': volatility_ratio,
            'adaptive_entry_threshold': entry_threshold,
            'adf_pvalue': pair_state.get('adf_pvalue'),
            'cointegration_pvalue': pair_state.get('cointegration_pvalue'),
            'timestamp': timestamp.isoformat()
        }
        signal_strength = Decimal(str(min(abs(zscore) / max(entry_threshold, 1e-6), 1.0)))
        
        # Get execution parameters
        exec_type, algo_strategy, algo_params = self._get_execution_params()
        
        if position_type == "short_a_long_b":
            # Short A, Long B
            signals.append(self.create_signal(
                symbol=stock_a,
                signal_type=SignalType.SELL,
                strength=signal_strength,
                price=Decimal(str(price_a)),
                quantity=self.config.position_size,
                execution_type=exec_type,
                algo_strategy=algo_strategy,
                algo_params=algo_params,
                **metadata_common
            ))
            
            signals.append(self.create_signal(
                symbol=stock_b,
                signal_type=SignalType.BUY,
                strength=signal_strength,
                price=Decimal(str(price_b)),
                quantity=self.config.position_size,
                execution_type=exec_type,
                algo_strategy=algo_strategy,
                algo_params=algo_params,
                **metadata_common
            ))
            
            self.log_info(
                f"ENTRY [{pair_key}]: Short {stock_a} @ ${price_a:.2f}, "
                f"Long {stock_b} @ ${price_b:.2f}, "
                f"z-score={zscore:.2f}, spread={spread:.4f}, hedge_ratio={pair_state['hedge_ratio']:.4f}"
            )
        
        elif position_type == "long_a_short_b":
            # Long A, Short B
            signals.append(self.create_signal(
                symbol=stock_a,
                signal_type=SignalType.BUY,
                strength=signal_strength,
                price=Decimal(str(price_a)),
                quantity=self.config.position_size,
                execution_type=exec_type,
                algo_strategy=algo_strategy,
                algo_params=algo_params,
                **metadata_common
            ))
            
            signals.append(self.create_signal(
                symbol=stock_b,
                signal_type=SignalType.SELL,
                strength=signal_strength,
                price=Decimal(str(price_b)),
                quantity=self.config.position_size,
                execution_type=exec_type,
                algo_strategy=algo_strategy,
                algo_params=algo_params,
                **metadata_common
            ))
            
            self.log_info(
                f"ENTRY [{pair_key}]: Long {stock_a} @ ${price_a:.2f}, "
                f"Short {stock_b} @ ${price_b:.2f}, "
                f"z-score={zscore:.2f}, spread={spread:.4f}, hedge_ratio={pair_state['hedge_ratio']:.4f}"
            )
        
        # Update pair state
        pair_state['current_position'] = position_type
        pair_state['entry_zscore'] = zscore
        pair_state['bars_in_trade'] = 0
        pair_state['entry_timestamp'] = timestamp
        pair_state['entry_prices'] = {
            pair_state['stock_a']: price_a,
            pair_state['stock_b']: price_b
        }
        pair_state['entry_spread'] = spread
        
        return signals

    def _generate_exit_signals(
        self,
        pair_key: str,
        exit_reason: str,
        zscore: float,
        spread: float,
        mean_spread: float,
        std_spread: float,
        price_a: float,
        price_b: float,
        stock_a: str,
        stock_b: str,
        timestamp: datetime,
        volatility_ratio: float
    ) -> List[StrategySignal]:
        """Generate exit signals for the pair."""
        signals = []
        pair_state = self._pair_states[pair_key]
        metadata_common = {
            'pair': pair_key,
            'exit_reason': exit_reason,
            'zscore': zscore,
            'spread': spread,
            'mean_spread': mean_spread,
            'std_spread': std_spread,
            'hedge_ratio': pair_state['hedge_ratio'],
            'hedge_intercept': pair_state['hedge_intercept'],
            'volatility_ratio': volatility_ratio,
            'entry_zscore': pair_state['entry_zscore'],
            'bars_held': pair_state['bars_in_trade'],
            'entry_timestamp': pair_state.get('entry_timestamp').isoformat() if pair_state.get('entry_timestamp') else None,
            'entry_spread': pair_state.get('entry_spread'),
            'adf_pvalue': pair_state.get('adf_pvalue'),
            'cointegration_pvalue': pair_state.get('cointegration_pvalue'),
            'timestamp': timestamp.isoformat()
        }
        
        # Get execution parameters (same as entry)
        exec_type, algo_strategy, algo_params = self._get_execution_params()
        
        # Close both legs of the position
        if pair_state['current_position'] == "short_a_long_b":
            # Cover short A, Sell long B
            signals.append(self.create_signal(
                symbol=stock_a,
                signal_type=SignalType.BUY,  # Cover short
                strength=Decimal('1.0'),
                price=Decimal(str(price_a)),
                quantity=self.config.position_size,
                execution_type=exec_type,
                algo_strategy=algo_strategy,
                algo_params=algo_params,
                **metadata_common
            ))
            
            signals.append(self.create_signal(
                symbol=stock_b,
                signal_type=SignalType.SELL,  # Sell long
                strength=Decimal('1.0'),
                price=Decimal(str(price_b)),
                quantity=self.config.position_size,
                execution_type=exec_type,
                algo_strategy=algo_strategy,
                algo_params=algo_params,
                **metadata_common
            ))
        
        elif pair_state['current_position'] == "long_a_short_b":
            # Sell long A, Cover short B
            signals.append(self.create_signal(
                symbol=stock_a,
                signal_type=SignalType.SELL,  # Sell long
                strength=Decimal('1.0'),
                price=Decimal(str(price_a)),
                quantity=self.config.position_size,
                execution_type=exec_type,
                algo_strategy=algo_strategy,
                algo_params=algo_params,
                **metadata_common
            ))
            
            signals.append(self.create_signal(
                symbol=stock_b,
                signal_type=SignalType.BUY,  # Cover short
                strength=Decimal('1.0'),
                price=Decimal(str(price_b)),
                quantity=self.config.position_size,
                execution_type=exec_type,
                algo_strategy=algo_strategy,
                algo_params=algo_params,
                **metadata_common
            ))
        
        self.log_info(
            f"EXIT [{pair_key}] ({exit_reason}): Close {pair_state['current_position']}, "
            f"z-score={zscore:.2f} (entry={pair_state['entry_zscore']:.2f}), "
            f"bars_held={pair_state['bars_in_trade']}"
        )
        
        # Reset pair state
        pair_state['current_position'] = "flat"
        pair_state['entry_zscore'] = 0.0
        pair_state['bars_in_trade'] = 0
        pair_state['entry_timestamp'] = None
        pair_state['entry_prices'] = {}
        pair_state['entry_spread'] = None
        if exit_reason == "stop_loss":
            pair_state['cooldown_remaining'] = max(
                pair_state['cooldown_remaining'],
                self.config.cooldown_bars
            )
        
        return signals
    
    async def on_bar(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> List[StrategySignal]:
        """
        Single-symbol callback (not used for pairs trading).
        
        This strategy uses on_bar_multi() instead, but we need to implement
        this abstract method.
        """
        return []
    
    async def on_stop(self) -> None:
        """Cleanup when stopping."""
        self.set_state(StrategyState.STOPPING)
        
        # Check for open positions in any pair
        open_positions = [
            pair_key for pair_key, state in self._pair_states.items()
            if state['current_position'] != "flat"
        ]
        
        if open_positions:
            self.log_warning(f"Stopping with {len(open_positions)} open positions: {open_positions}")
        
        self.set_state(StrategyState.STOPPED)
        self.log_info("Pairs trading strategy stopped")
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for monitoring."""
        
        # Collect state for each pair
        pairs_state = {}
        for pair_key, state in self._pair_states.items():
            current_spread = state['spread_history'][-1] if state['spread_history'] else None
            last_zscore = state.get('last_zscore')
            
            # Calculate proximity to entry/exit thresholds
            entry_proximity = None
            exit_proximity = None
            if last_zscore is not None:
                # How close are we to entry threshold? (0 = at threshold, 1 = far away)
                if state['current_position'] == 'flat':
                    entry_proximity = abs(last_zscore) / self.config.entry_threshold if self.config.entry_threshold > 0 else None
                # How close are we to exit threshold? (0 = at threshold, 1 = far away)
                if state['current_position'] != 'flat':
                    exit_proximity = abs(last_zscore) / self.config.exit_threshold if self.config.exit_threshold > 0 else None
            
            # Calculate data readiness
            spread_history_len = len(state['spread_history'])
            price_history_len = len(state.get('price_history_a', []))
            has_sufficient_data = spread_history_len >= self.config.lookback_window
            data_readiness_pct = min(100, (spread_history_len / self.config.lookback_window * 100)) if self.config.lookback_window > 0 else 0
            
            pairs_state[pair_key] = {
                "position": state['current_position'],
                "current_spread": current_spread,
                "current_zscore": last_zscore,
                "spread_history_length": spread_history_len,
                "price_history_length": price_history_len,
                "bars_in_trade": state['bars_in_trade'],
                "entry_zscore": state['entry_zscore'],
                "hedge_ratio": state['hedge_ratio'],
                "adf_pvalue": state.get('adf_pvalue'),
                "cointegration_pvalue": state.get('cointegration_pvalue'),
                "cooldown_remaining": state.get('cooldown_remaining'),
                "entry_proximity": entry_proximity,  # 0-1, how close to entry threshold
                "exit_proximity": exit_proximity,  # 0-1, how close to exit threshold
                "has_sufficient_data": has_sufficient_data,
                "data_readiness_pct": data_readiness_pct,
                "lookback_window": self.config.lookback_window
            }
        
        return {
            "config": {
                "pairs": self.pairs,
                "lookback_window": self.config.lookback_window,
                "entry_threshold": self.config.entry_threshold,
                "exit_threshold": self.config.exit_threshold,
                "position_size": self.config.position_size,
                "max_hold_bars": self.config.max_hold_bars
            },
            "num_pairs": len(self.pairs),
            "pairs_state": pairs_state,
            "metrics": self.get_metrics().dict()
        }
