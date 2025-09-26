"""
Simple Moving Average Crossover Strategy

This strategy generates buy signals when a short-term SMA crosses above a long-term SMA,
and sell signals when the short-term SMA crosses below the long-term SMA.

This is a classic trend-following strategy that works well in trending markets.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from .base import BaseStrategy, StrategyConfig, StrategySignal, SignalType, StrategyState
from .registry import strategy


class SMAStrategyConfig(StrategyConfig):
    """Configuration for SMA Crossover Strategy."""
    
    # SMA parameters
    short_period: int = 20
    long_period: int = 50
    
    # Signal filtering
    min_cross_strength: float = 0.01  # Minimum percentage difference for valid cross
    volume_filter: bool = True  # Filter signals based on volume
    min_volume_ratio: float = 1.2  # Minimum volume vs average volume
    
    # Position sizing
    position_size_method: str = "fixed"  # "fixed", "percentage", "volatility"
    fixed_position_size: int = 100
    position_percentage: float = 0.1  # 10% of portfolio
    
    class Config:
        extra = "allow"


@strategy(
    name="SMA_Crossover",
    description="Simple Moving Average crossover strategy with configurable periods",
    default_config={
        "short_period": 20,
        "long_period": 50,
        "min_cross_strength": 0.01,
        "position_size_method": "fixed",
        "fixed_position_size": 100
    }
)
class SMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Generates signals based on crossovers between short-term and long-term SMAs.
    Includes volume filtering and configurable position sizing.
    """
    
    def __init__(self, config: StrategyConfig):
        # Convert base config to SMA-specific config
        if not isinstance(config, SMAStrategyConfig):
            sma_config_data = config.dict() if hasattr(config, 'dict') else config.__dict__
            # Merge parameters into the main config
            if 'parameters' in sma_config_data:
                sma_config_data.update(sma_config_data['parameters'])
            config = SMAStrategyConfig(**sma_config_data)
        
        super().__init__(config)
        
        # Strategy state
        self._last_signals: Dict[str, SignalType] = {}
        self._sma_data: Dict[str, Dict[str, List[float]]] = {}
        self._volume_averages: Dict[str, float] = {}
        
        # Validate periods
        if config.short_period >= config.long_period:
            raise ValueError("Short period must be less than long period")
        
        if config.long_period > config.lookback_periods:
            raise ValueError("Long period cannot exceed lookback periods")
    
    @classmethod
    def get_parameters_schema(cls) -> Dict[str, Any]:
        """Return parameter schema for validation."""
        return {
            "short_period": {
                "type": int,
                "required": True,
                "min": 1,
                "max": 200,
                "description": "Short-term SMA period"
            },
            "long_period": {
                "type": int,
                "required": True,
                "min": 2,
                "max": 500,
                "description": "Long-term SMA period"
            },
            "min_cross_strength": {
                "type": float,
                "required": False,
                "min": 0.0,
                "max": 0.1,
                "description": "Minimum cross strength (percentage)"
            },
            "volume_filter": {
                "type": bool,
                "required": False,
                "description": "Enable volume filtering"
            },
            "position_size_method": {
                "type": str,
                "required": False,
                "choices": ["fixed", "percentage", "volatility"],
                "description": "Position sizing method"
            },
            "fixed_position_size": {
                "type": int,
                "required": False,
                "min": 1,
                "description": "Fixed position size (shares)"
            }
        }
    
    async def on_start(self, instruments: Dict[str, Any]) -> None:
        """Initialize strategy state."""
        self.set_state(StrategyState.RUNNING)
        
        # Initialize data structures for each symbol
        for symbol in self.config.symbols:
            self._last_signals[symbol] = SignalType.HOLD
            self._sma_data[symbol] = {"short": [], "long": []}
            self._volume_averages[symbol] = 0.0
        
        self.log_info(f"Started SMA strategy with {self.config.short_period}/{self.config.long_period} periods")
        self.log_info(f"Monitoring symbols: {', '.join(self.config.symbols)}")
    
    async def on_bar(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> List[StrategySignal]:
        """Process new bar data and generate signals."""
        if symbol not in self.config.symbols:
            return []
        
        if bars.empty or len(bars) < self.config.long_period:
            self.log_warning(f"Insufficient data for {symbol}: {len(bars)} bars")
            return []
        
        try:
            # Calculate SMAs
            short_sma = self._calculate_sma(bars['close'], self.config.short_period)
            long_sma = self._calculate_sma(bars['close'], self.config.long_period)
            
            if pd.isna(short_sma) or pd.isna(long_sma):
                return []
            
            # Update internal state
            self._sma_data[symbol]["short"].append(float(short_sma))
            self._sma_data[symbol]["long"].append(float(long_sma))
            
            # Keep only recent values (memory management)
            max_history = 100
            for key in self._sma_data[symbol]:
                if len(self._sma_data[symbol][key]) > max_history:
                    self._sma_data[symbol][key] = self._sma_data[symbol][key][-max_history:]
            
            # Update volume average
            if self.config.volume_filter:
                volume_window = min(20, len(bars))
                self._volume_averages[symbol] = float(bars['volume'].tail(volume_window).mean())
            
            # Generate signals
            signals = self._generate_signals(symbol, bars, short_sma, long_sma)
            
            return signals
            
        except Exception as e:
            self.log_error(f"Error processing bars for {symbol}: {e}")
            return []
    
    async def on_stop(self) -> None:
        """Cleanup when stopping."""
        self.set_state(StrategyState.STOPPING)
        
        # Generate exit signals for all open positions
        exit_signals = []
        for symbol in self.config.symbols:
            if not self.is_flat(symbol):
                signal_type = SignalType.SELL if self.is_long(symbol) else SignalType.BUY
                exit_signal = self.create_signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=Decimal('1.0'),
                    quantity=abs(self.get_position(symbol)),
                    reason="strategy_stop",
                    exit_signal=True
                )
                exit_signals.append(exit_signal)
        
        self.set_state(StrategyState.STOPPED)
        self.log_info("SMA strategy stopped")
        return exit_signals
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return np.nan
        return float(prices.tail(period).mean())
    
    def _generate_signals(self, symbol: str, bars: pd.DataFrame, 
                         short_sma: float, long_sma: float) -> List[StrategySignal]:
        """Generate trading signals based on SMA crossover."""
        signals = []
        
        # Need at least 2 data points to detect crossover
        if len(self._sma_data[symbol]["short"]) < 2:
            return signals
        
        current_short = short_sma
        current_long = long_sma
        prev_short = self._sma_data[symbol]["short"][-2]
        prev_long = self._sma_data[symbol]["long"][-2]
        
        # Calculate cross strength (percentage difference)
        cross_strength = abs(current_short - current_long) / current_long
        
        # Check for crossover
        signal_type = None
        signal_strength = 0.0
        
        # Bullish crossover: short SMA crosses above long SMA
        if prev_short <= prev_long and current_short > current_long:
            if cross_strength >= self.config.min_cross_strength:
                signal_type = SignalType.BUY
                signal_strength = min(cross_strength * 10, 1.0)  # Scale strength
        
        # Bearish crossover: short SMA crosses below long SMA
        elif prev_short >= prev_long and current_short < current_long:
            if cross_strength >= self.config.min_cross_strength:
                signal_type = SignalType.SELL
                signal_strength = min(cross_strength * 10, 1.0)  # Scale strength
        
        # Generate signal if conditions are met
        if signal_type and signal_type != self._last_signals.get(symbol):
            # Apply volume filter if enabled
            if self._should_filter_by_volume(symbol, bars):
                self.log_info(f"Signal filtered by volume for {symbol}")
                return signals
            
            # Calculate position size
            position_size = self._calculate_position_size(symbol, signal_strength, bars)
            
            # Create signal
            signal = self.create_signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=Decimal(str(signal_strength)),
                price=Decimal(str(bars['close'].iloc[-1])),
                quantity=position_size,
                short_sma=current_short,
                long_sma=current_long,
                cross_strength=cross_strength,
                volume_ratio=self._get_volume_ratio(symbol, bars) if self.config.volume_filter else None
            )
            
            signals.append(signal)
            self._last_signals[symbol] = signal_type
            
            self.log_info(
                f"Generated {signal_type.value} signal for {symbol}: "
                f"strength={signal_strength:.3f}, size={position_size}, "
                f"SMA({self.config.short_period})={current_short:.2f}, "
                f"SMA({self.config.long_period})={current_long:.2f}"
            )
        
        return signals
    
    def _should_filter_by_volume(self, symbol: str, bars: pd.DataFrame) -> bool:
        """Check if signal should be filtered out based on volume."""
        if not self.config.volume_filter:
            return False
        
        current_volume = float(bars['volume'].iloc[-1])
        avg_volume = self._volume_averages.get(symbol, 0)
        
        if avg_volume == 0:
            return False
        
        volume_ratio = current_volume / avg_volume
        return volume_ratio < self.config.min_volume_ratio
    
    def _get_volume_ratio(self, symbol: str, bars: pd.DataFrame) -> float:
        """Get current volume ratio vs average."""
        if not self.config.volume_filter:
            return 1.0
        
        current_volume = float(bars['volume'].iloc[-1])
        avg_volume = self._volume_averages.get(symbol, current_volume)
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _calculate_position_size(self, symbol: str, signal_strength: float, 
                               bars: pd.DataFrame) -> int:
        """Calculate position size based on configuration."""
        if self.config.position_size_method == "fixed":
            return self.config.fixed_position_size
        
        elif self.config.position_size_method == "percentage":
            # This would require portfolio value - simplified for now
            return int(self.config.fixed_position_size * self.config.position_percentage)
        
        elif self.config.position_size_method == "volatility":
            # Calculate volatility-based position size
            if len(bars) < 20:
                return self.config.fixed_position_size
            
            returns = bars['close'].pct_change().dropna()
            volatility = float(returns.tail(20).std())
            
            # Inverse relationship with volatility
            vol_multiplier = max(0.5, min(2.0, 1.0 / (volatility * 100)))
            size = int(self.config.fixed_position_size * vol_multiplier * signal_strength)
            
            return max(1, size)
        
        return self.config.fixed_position_size
    
    async def on_parameter_update(self, new_params: Dict[str, Any]) -> None:
        """Handle parameter updates."""
        await super().on_parameter_update(new_params)
        
        # Validate SMA periods if they were updated
        short_period = new_params.get('short_period', self.config.short_period)
        long_period = new_params.get('long_period', self.config.long_period)
        
        if short_period >= long_period:
            self.log_error("Invalid parameters: short_period must be less than long_period")
            return
        
        # Clear SMA data if periods changed
        if ('short_period' in new_params or 'long_period' in new_params):
            self._sma_data.clear()
            for symbol in self.config.symbols:
                self._sma_data[symbol] = {"short": [], "long": []}
            self.log_info("Cleared SMA data due to period change")
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for monitoring."""
        return {
            "config": {
                "short_period": self.config.short_period,
                "long_period": self.config.long_period,
                "min_cross_strength": self.config.min_cross_strength,
                "volume_filter": self.config.volume_filter
            },
            "last_signals": self._last_signals,
            "sma_values": {
                symbol: {
                    "short": data["short"][-1] if data["short"] else None,
                    "long": data["long"][-1] if data["long"] else None
                }
                for symbol, data in self._sma_data.items()
            },
            "volume_averages": self._volume_averages,
            "metrics": self.get_metrics().dict()
        }
