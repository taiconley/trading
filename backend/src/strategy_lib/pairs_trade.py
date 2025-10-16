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
import pandas as pd
import numpy as np

from .base import BaseStrategy, StrategyConfig, StrategySignal, SignalType, StrategyState
from .registry import strategy


class PairsTradingConfig(StrategyConfig):
    """Configuration for Pairs Trading Strategy."""
    
    # Must specify exactly 2 symbols for pairs trading
    # symbols[0] will be "Stock A", symbols[1] will be "Stock B"
    
    # Pairs parameters
    lookback_window: int = 20  # Window for calculating mean and std
    entry_threshold: float = 2.0  # Z-score threshold to enter trade
    exit_threshold: float = 0.5   # Z-score threshold to exit trade
    
    # Position sizing
    position_size: int = 100  # Shares per leg
    
    # Risk management
    max_hold_days: int = 20  # Maximum days to hold a position
    stop_loss_zscore: float = 3.0  # Exit if z-score gets worse
    
    class Config:
        extra = "allow"


@strategy(
    name="Pairs_Trading",
    description="Statistical arbitrage strategy trading price ratio deviations between two correlated stocks",
    default_config={
        "lookback_window": 20,
        "entry_threshold": 2.0,
        "exit_threshold": 0.5,
        "position_size": 100,
        "max_hold_days": 20,
        "stop_loss_zscore": 3.0
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
        
        super().__init__(config)
        
        # Validate that we have exactly 2 symbols
        if len(self.config.symbols) != 2:
            raise ValueError(f"Pairs trading requires exactly 2 symbols, got {len(self.config.symbols)}")
        
        # State tracking
        self._ratio_history: List[float] = []
        self._current_position: str = "flat"  # "flat", "long_a_short_b", "short_a_long_b"
        self._entry_zscore: float = 0.0
        self._entry_date = None
        self._days_in_trade = 0
        
        self.stock_a = self.config.symbols[0]
        self.stock_b = self.config.symbols[1]
        
        self.log_info(f"Initialized pairs trading: {self.stock_a} vs {self.stock_b}")
    
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
        self._ratio_history = []
        self._current_position = "flat"
        
        self.log_info(
            f"Started pairs trading strategy: {self.stock_a}/{self.stock_b}, "
            f"entry_threshold={self.config.entry_threshold}, "
            f"exit_threshold={self.config.exit_threshold}"
        )
    
    async def on_bar_multi(self, symbols: List[str], timeframe: str, 
                          bars_data: Dict[str, pd.DataFrame]) -> List[StrategySignal]:
        """Process bar data for both stocks simultaneously."""
        
        # Ensure we have data for both stocks
        if self.stock_a not in bars_data or self.stock_b not in bars_data:
            return []
        
        bars_a = bars_data[self.stock_a]
        bars_b = bars_data[self.stock_b]
        
        # Get current prices
        price_a = float(bars_a['close'].iloc[-1])
        price_b = float(bars_b['close'].iloc[-1])
        
        if price_b == 0:  # Avoid division by zero
            return []
        
        # Calculate price ratio
        ratio = price_a / price_b
        self._ratio_history.append(ratio)
        
        # Keep only recent history
        if len(self._ratio_history) > 200:
            self._ratio_history = self._ratio_history[-200:]
        
        # Need enough history to calculate statistics
        if len(self._ratio_history) < self.config.lookback_window:
            return []
        
        # Calculate z-score
        recent_ratios = self._ratio_history[-self.config.lookback_window:]
        mean_ratio = np.mean(recent_ratios)
        std_ratio = np.std(recent_ratios)
        
        if std_ratio == 0:  # Avoid division by zero
            return []
        
        zscore = (ratio - mean_ratio) / std_ratio
        
        # Track days in trade
        if self._current_position != "flat":
            self._days_in_trade += 1
        
        # Generate signals based on z-score
        signals = []
        
        if self._current_position == "flat":
            # Look for entry signals
            if zscore > self.config.entry_threshold:
                # Ratio is too high: short A, long B
                signals = self._generate_entry_signals(
                    position_type="short_a_long_b",
                    zscore=zscore,
                    price_a=price_a,
                    price_b=price_b
                )
                
            elif zscore < -self.config.entry_threshold:
                # Ratio is too low: long A, short B
                signals = self._generate_entry_signals(
                    position_type="long_a_short_b",
                    zscore=zscore,
                    price_a=price_a,
                    price_b=price_b
                )
        
        else:
            # Already in a position - check for exit signals
            
            # Exit condition 1: Z-score returns to normal
            should_exit_mean_reversion = abs(zscore) < self.config.exit_threshold
            
            # Exit condition 2: Hit stop loss (z-score got worse)
            should_exit_stop_loss = False
            if self._current_position == "short_a_long_b" and zscore > self.config.stop_loss_zscore:
                should_exit_stop_loss = True
            elif self._current_position == "long_a_short_b" and zscore < -self.config.stop_loss_zscore:
                should_exit_stop_loss = True
            
            # Exit condition 3: Held too long
            should_exit_time = self._days_in_trade >= self.config.max_hold_days
            
            if should_exit_mean_reversion or should_exit_stop_loss or should_exit_time:
                exit_reason = (
                    "mean_reversion" if should_exit_mean_reversion else
                    "stop_loss" if should_exit_stop_loss else
                    "max_hold_time"
                )
                
                signals = self._generate_exit_signals(
                    exit_reason=exit_reason,
                    zscore=zscore,
                    price_a=price_a,
                    price_b=price_b
                )
        
        return signals
    
    def _generate_entry_signals(self, position_type: str, zscore: float,
                                price_a: float, price_b: float) -> List[StrategySignal]:
        """Generate entry signals for the pair."""
        signals = []
        
        if position_type == "short_a_long_b":
            # Short A, Long B
            signals.append(self.create_signal(
                symbol=self.stock_a,
                signal_type=SignalType.SELL,
                strength=Decimal(str(min(abs(zscore) / 5.0, 1.0))),
                price=Decimal(str(price_a)),
                quantity=self.config.position_size,
                zscore=zscore,
                ratio=price_a / price_b,
                position_type=position_type
            ))
            
            signals.append(self.create_signal(
                symbol=self.stock_b,
                signal_type=SignalType.BUY,
                strength=Decimal(str(min(abs(zscore) / 5.0, 1.0))),
                price=Decimal(str(price_b)),
                quantity=self.config.position_size,
                zscore=zscore,
                ratio=price_a / price_b,
                position_type=position_type
            ))
            
            self.log_info(
                f"ENTRY: Short {self.stock_a} @ ${price_a:.2f}, "
                f"Long {self.stock_b} @ ${price_b:.2f}, "
                f"z-score={zscore:.2f}"
            )
        
        elif position_type == "long_a_short_b":
            # Long A, Short B
            signals.append(self.create_signal(
                symbol=self.stock_a,
                signal_type=SignalType.BUY,
                strength=Decimal(str(min(abs(zscore) / 5.0, 1.0))),
                price=Decimal(str(price_a)),
                quantity=self.config.position_size,
                zscore=zscore,
                ratio=price_a / price_b,
                position_type=position_type
            ))
            
            signals.append(self.create_signal(
                symbol=self.stock_b,
                signal_type=SignalType.SELL,
                strength=Decimal(str(min(abs(zscore) / 5.0, 1.0))),
                price=Decimal(str(price_b)),
                quantity=self.config.position_size,
                zscore=zscore,
                ratio=price_a / price_b,
                position_type=position_type
            ))
            
            self.log_info(
                f"ENTRY: Long {self.stock_a} @ ${price_a:.2f}, "
                f"Short {self.stock_b} @ ${price_b:.2f}, "
                f"z-score={zscore:.2f}"
            )
        
        # Update state
        self._current_position = position_type
        self._entry_zscore = zscore
        self._days_in_trade = 0
        
        return signals
    
    def _generate_exit_signals(self, exit_reason: str, zscore: float,
                               price_a: float, price_b: float) -> List[StrategySignal]:
        """Generate exit signals for the pair."""
        signals = []
        
        # Close both legs of the position
        if self._current_position == "short_a_long_b":
            # Cover short A, Sell long B
            signals.append(self.create_signal(
                symbol=self.stock_a,
                signal_type=SignalType.BUY,  # Cover short
                strength=Decimal('1.0'),
                price=Decimal(str(price_a)),
                quantity=self.config.position_size,
                exit_reason=exit_reason,
                zscore=zscore,
                entry_zscore=self._entry_zscore,
                days_held=self._days_in_trade
            ))
            
            signals.append(self.create_signal(
                symbol=self.stock_b,
                signal_type=SignalType.SELL,  # Sell long
                strength=Decimal('1.0'),
                price=Decimal(str(price_b)),
                quantity=self.config.position_size,
                exit_reason=exit_reason,
                zscore=zscore,
                entry_zscore=self._entry_zscore,
                days_held=self._days_in_trade
            ))
        
        elif self._current_position == "long_a_short_b":
            # Sell long A, Cover short B
            signals.append(self.create_signal(
                symbol=self.stock_a,
                signal_type=SignalType.SELL,  # Sell long
                strength=Decimal('1.0'),
                price=Decimal(str(price_a)),
                quantity=self.config.position_size,
                exit_reason=exit_reason,
                zscore=zscore,
                entry_zscore=self._entry_zscore,
                days_held=self._days_in_trade
            ))
            
            signals.append(self.create_signal(
                symbol=self.stock_b,
                signal_type=SignalType.BUY,  # Cover short
                strength=Decimal('1.0'),
                price=Decimal(str(price_b)),
                quantity=self.config.position_size,
                exit_reason=exit_reason,
                zscore=zscore,
                entry_zscore=self._entry_zscore,
                days_held=self._days_in_trade
            ))
        
        self.log_info(
            f"EXIT ({exit_reason}): Close {self._current_position}, "
            f"z-score={zscore:.2f} (entry={self._entry_zscore:.2f}), "
            f"days_held={self._days_in_trade}"
        )
        
        # Reset state
        self._current_position = "flat"
        self._entry_zscore = 0.0
        self._days_in_trade = 0
        
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
        
        # If we have an open position, log a warning
        if self._current_position != "flat":
            self.log_warning(f"Stopping with open position: {self._current_position}")
        
        self.set_state(StrategyState.STOPPED)
        self.log_info("Pairs trading strategy stopped")
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for monitoring."""
        current_ratio = self._ratio_history[-1] if self._ratio_history else None
        
        return {
            "config": {
                "stock_a": self.stock_a,
                "stock_b": self.stock_b,
                "lookback_window": self.config.lookback_window,
                "entry_threshold": self.config.entry_threshold,
                "exit_threshold": self.config.exit_threshold,
                "position_size": self.config.position_size
            },
            "position": self._current_position,
            "current_ratio": current_ratio,
            "ratio_history_length": len(self._ratio_history),
            "days_in_trade": self._days_in_trade,
            "entry_zscore": self._entry_zscore,
            "metrics": self.get_metrics().dict()
        }


