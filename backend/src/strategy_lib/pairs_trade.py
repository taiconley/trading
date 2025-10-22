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

from decimal import Decimal
from typing import Any, Dict, List
from datetime import datetime, time
import pandas as pd
import numpy as np

from .base import BaseStrategy, StrategyConfig, StrategySignal, SignalType, StrategyState
from .registry import strategy


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
    
    # Risk management (for 5-second bars: 720 bars = 1 hour)
    max_hold_bars: int = 720  # Maximum bars to hold a position
    stop_loss_zscore: float = 3.0  # Exit if z-score gets worse
    
    # Intraday-only trading
    market_close_hour: int = 16  # US market close hour (4 PM Eastern)
    market_close_minute: int = 0  # Market close minute
    close_before_eod_minutes: int = 5  # Close all positions N minutes before market close
    
    class Config:
        extra = "allow"


@strategy(
    name="Pairs_Trading",
    description="Intraday statistical arbitrage strategy for 5-second bars, trading price ratio deviations between multiple pairs",
    default_config={
        "pairs": [
            ["AAPL", "MSFT"],   # Tech Large Cap
            ["JPM", "BAC"],     # Banks
            ["GS", "MS"],       # Investment Banks
            ["XOM", "CVX"],     # Energy
            ["V", "MA"],        # Payments
            ["KO", "PEP"],      # Beverages
            ["WMT", "TGT"],     # Retail
            ["PFE", "MRK"],     # Pharma
            ["DIS", "NFLX"]     # Media
        ],
        "lookback_window": 240,  # 20 minutes at 5-sec bars
        "entry_threshold": 2.0,
        "exit_threshold": 0.5,
        "position_size": 100,
        "max_hold_bars": 720,  # 1 hour at 5-sec bars
        "stop_loss_zscore": 3.0,
        "market_close_hour": 16,
        "market_close_minute": 0,
        "close_before_eod_minutes": 5
    }
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
        if hasattr(config, 'pairs') and config.pairs:
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
        
        super().__init__(config)
        
        # State tracking for each pair
        # Key is "SYMBOL_A/SYMBOL_B"
        self._pair_states: Dict[str, Dict[str, Any]] = {}
        
        for pair in self.pairs:
            pair_key = f"{pair[0]}/{pair[1]}"
            self._pair_states[pair_key] = {
                'stock_a': pair[0],
                'stock_b': pair[1],
                'ratio_history': [],
                'current_position': 'flat',  # "flat", "long_a_short_b", "short_a_long_b"
                'entry_zscore': 0.0,
                'bars_in_trade': 0,
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
                "max": 100,
                "description": "Lookback window for mean/std calculation"
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
            }
        }
    
    async def on_start(self, instruments: Dict[str, Any]) -> None:
        """Initialize strategy state."""
        self.set_state(StrategyState.RUNNING)
        
        # Reset all pair states
        for pair_key in self._pair_states:
            self._pair_states[pair_key]['ratio_history'] = []
            self._pair_states[pair_key]['current_position'] = 'flat'
            self._pair_states[pair_key]['entry_zscore'] = 0.0
            self._pair_states[pair_key]['bars_in_trade'] = 0
        
        self.log_info(
            f"Started intraday pairs trading strategy with {len(self.pairs)} pairs, "
            f"entry_threshold={self.config.entry_threshold}, "
            f"exit_threshold={self.config.exit_threshold}, "
            f"max_hold_bars={self.config.max_hold_bars}"
        )
    
    async def on_bar_multi(self, symbols: List[str], timeframe: str, 
                          bars_data: Dict[str, pd.DataFrame]) -> List[StrategySignal]:
        """Process bar data for all pairs simultaneously."""
        
        all_signals = []
        
        # Check if we're near market close (force close all positions)
        now = datetime.now()
        
        # Calculate when to close positions (N minutes before market close)
        close_hour = self.config.market_close_hour
        close_minute = self.config.market_close_minute - self.config.close_before_eod_minutes
        if close_minute < 0:
            close_hour -= 1
            close_minute += 60
        close_positions_time = time(close_hour, close_minute)
        
        is_near_close = now.time() >= close_positions_time
        
        # Process each pair independently
        for pair in self.pairs:
            stock_a, stock_b = pair[0], pair[1]
            pair_key = f"{stock_a}/{stock_b}"
            
            # Ensure we have data for both stocks in this pair
            if stock_a not in bars_data or stock_b not in bars_data:
                continue
            
            bars_a = bars_data[stock_a]
            bars_b = bars_data[stock_b]
            
            # Get current prices
            price_a = float(bars_a['close'].iloc[-1])
            price_b = float(bars_b['close'].iloc[-1])
            
            if price_b == 0:  # Avoid division by zero
                continue
            
            # Get pair state
            pair_state = self._pair_states[pair_key]
            
            # Calculate price ratio
            ratio = price_a / price_b
            pair_state['ratio_history'].append(ratio)
            
            # Keep only recent history
            if len(pair_state['ratio_history']) > 200:
                pair_state['ratio_history'] = pair_state['ratio_history'][-200:]
            
            # Need enough history to calculate statistics
            if len(pair_state['ratio_history']) < self.config.lookback_window:
                continue
            
            # Calculate z-score
            recent_ratios = pair_state['ratio_history'][-self.config.lookback_window:]
            mean_ratio = np.mean(recent_ratios)
            std_ratio = np.std(recent_ratios)
            
            if std_ratio == 0:  # Avoid division by zero
                continue
            
            zscore = (ratio - mean_ratio) / std_ratio
            
            # Track bars in trade
            if pair_state['current_position'] != "flat":
                pair_state['bars_in_trade'] += 1
            
            # Generate signals based on z-score and market conditions
            signals = []
            
            # Force close all positions near market close
            if is_near_close and pair_state['current_position'] != "flat":
                signals = self._generate_exit_signals(
                    pair_key=pair_key,
                    exit_reason="end_of_day",
                    zscore=zscore,
                    price_a=price_a,
                    price_b=price_b,
                    stock_a=stock_a,
                    stock_b=stock_b
                )
                all_signals.extend(signals)
                continue  # Skip other checks for this pair
            
            if pair_state['current_position'] == "flat":
                # Look for entry signals
                if zscore > self.config.entry_threshold:
                    # Ratio is too high: short A, long B
                    signals = self._generate_entry_signals(
                        pair_key=pair_key,
                        position_type="short_a_long_b",
                        zscore=zscore,
                        price_a=price_a,
                        price_b=price_b,
                        stock_a=stock_a,
                        stock_b=stock_b
                    )
                    
                elif zscore < -self.config.entry_threshold:
                    # Ratio is too low: long A, short B
                    signals = self._generate_entry_signals(
                        pair_key=pair_key,
                        position_type="long_a_short_b",
                        zscore=zscore,
                        price_a=price_a,
                        price_b=price_b,
                        stock_a=stock_a,
                        stock_b=stock_b
                    )
            
            else:
                # Already in a position - check for exit signals
                
                # Exit condition 1: Z-score returns to normal
                should_exit_mean_reversion = abs(zscore) < self.config.exit_threshold
                
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
                        price_a=price_a,
                        price_b=price_b,
                        stock_a=stock_a,
                        stock_b=stock_b
                    )
            
            # Add signals from this pair to the overall list
            all_signals.extend(signals)
        
        return all_signals
    
    def _generate_entry_signals(self, pair_key: str, position_type: str, zscore: float,
                                price_a: float, price_b: float, stock_a: str, stock_b: str) -> List[StrategySignal]:
        """Generate entry signals for the pair."""
        signals = []
        pair_state = self._pair_states[pair_key]
        
        if position_type == "short_a_long_b":
            # Short A, Long B
            signals.append(self.create_signal(
                symbol=stock_a,
                signal_type=SignalType.SELL,
                strength=Decimal(str(min(abs(zscore) / 5.0, 1.0))),
                price=Decimal(str(price_a)),
                quantity=self.config.position_size,
                zscore=zscore,
                ratio=price_a / price_b,
                position_type=position_type,
                pair=pair_key
            ))
            
            signals.append(self.create_signal(
                symbol=stock_b,
                signal_type=SignalType.BUY,
                strength=Decimal(str(min(abs(zscore) / 5.0, 1.0))),
                price=Decimal(str(price_b)),
                quantity=self.config.position_size,
                zscore=zscore,
                ratio=price_a / price_b,
                position_type=position_type,
                pair=pair_key
            ))
            
            self.log_info(
                f"ENTRY [{pair_key}]: Short {stock_a} @ ${price_a:.2f}, "
                f"Long {stock_b} @ ${price_b:.2f}, "
                f"z-score={zscore:.2f}"
            )
        
        elif position_type == "long_a_short_b":
            # Long A, Short B
            signals.append(self.create_signal(
                symbol=stock_a,
                signal_type=SignalType.BUY,
                strength=Decimal(str(min(abs(zscore) / 5.0, 1.0))),
                price=Decimal(str(price_a)),
                quantity=self.config.position_size,
                zscore=zscore,
                ratio=price_a / price_b,
                position_type=position_type,
                pair=pair_key
            ))
            
            signals.append(self.create_signal(
                symbol=stock_b,
                signal_type=SignalType.SELL,
                strength=Decimal(str(min(abs(zscore) / 5.0, 1.0))),
                price=Decimal(str(price_b)),
                quantity=self.config.position_size,
                zscore=zscore,
                ratio=price_a / price_b,
                position_type=position_type,
                pair=pair_key
            ))
            
            self.log_info(
                f"ENTRY [{pair_key}]: Long {stock_a} @ ${price_a:.2f}, "
                f"Short {stock_b} @ ${price_b:.2f}, "
                f"z-score={zscore:.2f}"
            )
        
        # Update pair state
        pair_state['current_position'] = position_type
        pair_state['entry_zscore'] = zscore
        pair_state['bars_in_trade'] = 0
        
        return signals
    
    def _generate_exit_signals(self, pair_key: str, exit_reason: str, zscore: float,
                               price_a: float, price_b: float, stock_a: str, stock_b: str) -> List[StrategySignal]:
        """Generate exit signals for the pair."""
        signals = []
        pair_state = self._pair_states[pair_key]
        
        # Close both legs of the position
        if pair_state['current_position'] == "short_a_long_b":
            # Cover short A, Sell long B
            signals.append(self.create_signal(
                symbol=stock_a,
                signal_type=SignalType.BUY,  # Cover short
                strength=Decimal('1.0'),
                price=Decimal(str(price_a)),
                quantity=self.config.position_size,
                exit_reason=exit_reason,
                zscore=zscore,
                entry_zscore=pair_state['entry_zscore'],
                bars_held=pair_state['bars_in_trade'],
                pair=pair_key
            ))
            
            signals.append(self.create_signal(
                symbol=stock_b,
                signal_type=SignalType.SELL,  # Sell long
                strength=Decimal('1.0'),
                price=Decimal(str(price_b)),
                quantity=self.config.position_size,
                exit_reason=exit_reason,
                zscore=zscore,
                entry_zscore=pair_state['entry_zscore'],
                bars_held=pair_state['bars_in_trade'],
                pair=pair_key
            ))
        
        elif pair_state['current_position'] == "long_a_short_b":
            # Sell long A, Cover short B
            signals.append(self.create_signal(
                symbol=stock_a,
                signal_type=SignalType.SELL,  # Sell long
                strength=Decimal('1.0'),
                price=Decimal(str(price_a)),
                quantity=self.config.position_size,
                exit_reason=exit_reason,
                zscore=zscore,
                entry_zscore=pair_state['entry_zscore'],
                bars_held=pair_state['bars_in_trade'],
                pair=pair_key
            ))
            
            signals.append(self.create_signal(
                symbol=stock_b,
                signal_type=SignalType.BUY,  # Cover short
                strength=Decimal('1.0'),
                price=Decimal(str(price_b)),
                quantity=self.config.position_size,
                exit_reason=exit_reason,
                zscore=zscore,
                entry_zscore=pair_state['entry_zscore'],
                bars_held=pair_state['bars_in_trade'],
                pair=pair_key
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
            current_ratio = state['ratio_history'][-1] if state['ratio_history'] else None
            pairs_state[pair_key] = {
                "position": state['current_position'],
                "current_ratio": current_ratio,
                "ratio_history_length": len(state['ratio_history']),
                "bars_in_trade": state['bars_in_trade'],
                "entry_zscore": state['entry_zscore'],
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


