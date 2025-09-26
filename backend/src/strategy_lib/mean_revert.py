"""
Mean Reversion Strategy

This strategy identifies when prices deviate significantly from their mean
and generates signals expecting prices to revert back to the mean.

Uses Bollinger Bands and RSI to identify overbought/oversold conditions.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from .base import BaseStrategy, StrategyConfig, StrategySignal, SignalType, StrategyState
from .registry import strategy


class MeanReversionConfig(StrategyConfig):
    """Configuration for Mean Reversion Strategy."""
    
    # Bollinger Bands parameters
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # RSI parameters
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # Mean reversion parameters
    price_deviation_threshold: float = 0.02  # 2% deviation from mean
    min_reversion_strength: float = 0.5  # Minimum signal strength
    
    # Exit parameters
    profit_target_pct: float = 0.03  # 3% profit target
    stop_loss_pct: float = 0.02  # 2% stop loss
    max_hold_bars: int = 50  # Maximum bars to hold position
    
    # Position sizing
    base_position_size: int = 100
    risk_per_trade_pct: float = 0.01  # 1% risk per trade
    
    class Config:
        extra = "allow"


@strategy(
    name="Mean_Reversion",
    description="Mean reversion strategy using Bollinger Bands and RSI",
    default_config={
        "bb_period": 20,
        "bb_std_dev": 2.0,
        "rsi_period": 14,
        "rsi_overbought": 70.0,
        "rsi_oversold": 30.0,
        "price_deviation_threshold": 0.02,
        "base_position_size": 100
    }
)
class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy using Bollinger Bands and RSI.
    
    Generates buy signals when:
    - Price is below lower Bollinger Band
    - RSI is oversold
    - Price deviation exceeds threshold
    
    Generates sell signals when:
    - Price is above upper Bollinger Band
    - RSI is overbought
    - Price deviation exceeds threshold
    """
    
    def __init__(self, config: StrategyConfig):
        # Convert base config to mean reversion specific config
        if not isinstance(config, MeanReversionConfig):
            mr_config_data = config.dict() if hasattr(config, 'dict') else config.__dict__
            # Merge parameters into the main config
            if 'parameters' in mr_config_data:
                mr_config_data.update(mr_config_data['parameters'])
            config = MeanReversionConfig(**mr_config_data)
        
        super().__init__(config)
        
        # Strategy state
        self._indicators: Dict[str, Dict[str, List[float]]] = {}
        self._entry_prices: Dict[str, Decimal] = {}
        self._entry_bars: Dict[str, int] = {}
        self._bars_since_entry: Dict[str, int] = {}
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate strategy-specific configuration."""
        if self.config.bb_period <= 0:
            raise ValueError("Bollinger Band period must be positive")
        
        if self.config.rsi_period <= 0:
            raise ValueError("RSI period must be positive")
        
        if not (0 < self.config.rsi_overbought <= 100):
            raise ValueError("RSI overbought level must be between 0 and 100")
        
        if not (0 <= self.config.rsi_oversold < 100):
            raise ValueError("RSI oversold level must be between 0 and 100")
        
        if self.config.rsi_oversold >= self.config.rsi_overbought:
            raise ValueError("RSI oversold level must be less than overbought level")
    
    @classmethod
    def get_parameters_schema(cls) -> Dict[str, Any]:
        """Return parameter schema for validation."""
        return {
            "bb_period": {
                "type": int,
                "required": True,
                "min": 5,
                "max": 100,
                "description": "Bollinger Bands period"
            },
            "bb_std_dev": {
                "type": float,
                "required": True,
                "min": 0.5,
                "max": 4.0,
                "description": "Bollinger Bands standard deviations"
            },
            "rsi_period": {
                "type": int,
                "required": True,
                "min": 2,
                "max": 50,
                "description": "RSI calculation period"
            },
            "rsi_overbought": {
                "type": float,
                "required": True,
                "min": 50,
                "max": 95,
                "description": "RSI overbought threshold"
            },
            "rsi_oversold": {
                "type": float,
                "required": True,
                "min": 5,
                "max": 50,
                "description": "RSI oversold threshold"
            },
            "price_deviation_threshold": {
                "type": float,
                "required": False,
                "min": 0.005,
                "max": 0.1,
                "description": "Minimum price deviation from mean"
            },
            "base_position_size": {
                "type": int,
                "required": False,
                "min": 1,
                "description": "Base position size (shares)"
            }
        }
    
    async def on_start(self, instruments: Dict[str, Any]) -> None:
        """Initialize strategy state."""
        self.set_state(StrategyState.RUNNING)
        
        # Initialize data structures for each symbol
        for symbol in self.config.symbols:
            self._indicators[symbol] = {
                "bb_upper": [],
                "bb_middle": [],
                "bb_lower": [],
                "rsi": [],
                "price_mean": []
            }
            self._entry_prices[symbol] = Decimal('0')
            self._entry_bars[symbol] = 0
            self._bars_since_entry[symbol] = 0
        
        self.log_info(f"Started Mean Reversion strategy")
        self.log_info(f"BB Period: {self.config.bb_period}, RSI Period: {self.config.rsi_period}")
        self.log_info(f"RSI Levels: {self.config.rsi_oversold}/{self.config.rsi_overbought}")
        self.log_info(f"Monitoring symbols: {', '.join(self.config.symbols)}")
    
    async def on_bar(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> List[StrategySignal]:
        """Process new bar data and generate signals."""
        if symbol not in self.config.symbols:
            return []
        
        required_bars = max(self.config.bb_period, self.config.rsi_period) + 10
        if bars.empty or len(bars) < required_bars:
            self.log_warning(f"Insufficient data for {symbol}: {len(bars)} bars (need {required_bars})")
            return []
        
        try:
            # Calculate indicators
            indicators = self._calculate_indicators(bars)
            
            if not indicators:
                return []
            
            # Update internal state
            self._update_indicator_history(symbol, indicators)
            
            # Update position tracking
            self._update_position_tracking(symbol)
            
            # Generate signals
            signals = self._generate_signals(symbol, bars, indicators)
            
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
        self.log_info("Mean Reversion strategy stopped")
        return exit_signals
    
    def _calculate_indicators(self, bars: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calculate technical indicators."""
        prices = bars['close']
        
        # Bollinger Bands
        bb_middle = prices.rolling(window=self.config.bb_period).mean()
        bb_std = prices.rolling(window=self.config.bb_period).std()
        bb_upper = bb_middle + (bb_std * self.config.bb_std_dev)
        bb_lower = bb_middle - (bb_std * self.config.bb_std_dev)
        
        # RSI
        rsi = self._calculate_rsi(prices, self.config.rsi_period)
        
        # Get current values
        current_bb_upper = bb_upper.iloc[-1]
        current_bb_middle = bb_middle.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Check for valid values
        if pd.isna(current_bb_upper) or pd.isna(current_rsi):
            return None
        
        return {
            "bb_upper": float(current_bb_upper),
            "bb_middle": float(current_bb_middle),
            "bb_lower": float(current_bb_lower),
            "rsi": float(current_rsi),
            "price_mean": float(current_bb_middle)
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _update_indicator_history(self, symbol: str, indicators: Dict[str, float]):
        """Update indicator history for the symbol."""
        for key, value in indicators.items():
            self._indicators[symbol][key].append(value)
        
        # Keep only recent history (memory management)
        max_history = 100
        for key in self._indicators[symbol]:
            if len(self._indicators[symbol][key]) > max_history:
                self._indicators[symbol][key] = self._indicators[symbol][key][-max_history:]
    
    def _update_position_tracking(self, symbol: str):
        """Update position tracking for exit conditions."""
        if not self.is_flat(symbol):
            self._bars_since_entry[symbol] += 1
        else:
            self._bars_since_entry[symbol] = 0
            self._entry_prices[symbol] = Decimal('0')
    
    def _generate_signals(self, symbol: str, bars: pd.DataFrame, 
                         indicators: Dict[str, float]) -> List[StrategySignal]:
        """Generate trading signals based on mean reversion conditions."""
        signals = []
        current_price = Decimal(str(bars['close'].iloc[-1]))
        
        # Check for exit conditions first
        exit_signal = self._check_exit_conditions(symbol, current_price, indicators)
        if exit_signal:
            signals.append(exit_signal)
            return signals
        
        # Don't generate new entry signals if already in position
        if not self.is_flat(symbol):
            return signals
        
        # Check for entry conditions
        entry_signal = self._check_entry_conditions(symbol, current_price, indicators)
        if entry_signal:
            # Record entry information
            self._entry_prices[symbol] = current_price
            self._entry_bars[symbol] = 0
            self._bars_since_entry[symbol] = 0
            
            signals.append(entry_signal)
        
        return signals
    
    def _check_entry_conditions(self, symbol: str, current_price: Decimal, 
                               indicators: Dict[str, float]) -> Optional[StrategySignal]:
        """Check for entry signal conditions."""
        bb_upper = indicators["bb_upper"]
        bb_lower = indicators["bb_lower"]
        bb_middle = indicators["bb_middle"]
        rsi = indicators["rsi"]
        
        current_price_float = float(current_price)
        
        # Calculate price deviation from mean
        price_deviation = abs(current_price_float - bb_middle) / bb_middle
        
        if price_deviation < self.config.price_deviation_threshold:
            return None
        
        signal_type = None
        signal_strength = 0.0
        
        # Oversold condition (buy signal)
        if (current_price_float < bb_lower and 
            rsi < self.config.rsi_oversold):
            
            signal_type = SignalType.BUY
            
            # Calculate signal strength based on multiple factors
            rsi_strength = (self.config.rsi_oversold - rsi) / self.config.rsi_oversold
            bb_strength = (bb_lower - current_price_float) / bb_lower
            deviation_strength = price_deviation / self.config.price_deviation_threshold
            
            signal_strength = min(1.0, (rsi_strength + bb_strength + deviation_strength) / 3)
        
        # Overbought condition (sell signal)
        elif (current_price_float > bb_upper and 
              rsi > self.config.rsi_overbought):
            
            signal_type = SignalType.SELL
            
            # Calculate signal strength
            rsi_strength = (rsi - self.config.rsi_overbought) / (100 - self.config.rsi_overbought)
            bb_strength = (current_price_float - bb_upper) / bb_upper
            deviation_strength = price_deviation / self.config.price_deviation_threshold
            
            signal_strength = min(1.0, (rsi_strength + bb_strength + deviation_strength) / 3)
        
        # Generate signal if conditions met and strength is sufficient
        if signal_type and signal_strength >= self.config.min_reversion_strength:
            position_size = self._calculate_position_size(symbol, signal_strength)
            
            signal = self.create_signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=Decimal(str(signal_strength)),
                price=current_price,
                quantity=position_size,
                bb_upper=bb_upper,
                bb_middle=bb_middle,
                bb_lower=bb_lower,
                rsi=rsi,
                price_deviation=price_deviation,
                entry_reason="mean_reversion"
            )
            
            self.log_info(
                f"Generated {signal_type.value} signal for {symbol}: "
                f"price=${current_price:.2f}, RSI={rsi:.1f}, "
                f"BB=[{bb_lower:.2f}, {bb_middle:.2f}, {bb_upper:.2f}], "
                f"strength={signal_strength:.3f}"
            )
            
            return signal
        
        return None
    
    def _check_exit_conditions(self, symbol: str, current_price: Decimal, 
                              indicators: Dict[str, float]) -> Optional[StrategySignal]:
        """Check for exit signal conditions."""
        if self.is_flat(symbol):
            return None
        
        entry_price = self._entry_prices.get(symbol, Decimal('0'))
        if entry_price == 0:
            return None
        
        position = self.get_position(symbol)
        is_long = position > 0
        
        # Calculate P&L
        if is_long:
            pnl_pct = float((current_price - entry_price) / entry_price)
        else:
            pnl_pct = float((entry_price - current_price) / entry_price)
        
        exit_reason = None
        
        # Profit target
        if pnl_pct >= self.config.profit_target_pct:
            exit_reason = "profit_target"
        
        # Stop loss
        elif pnl_pct <= -self.config.stop_loss_pct:
            exit_reason = "stop_loss"
        
        # Maximum hold period
        elif self._bars_since_entry[symbol] >= self.config.max_hold_bars:
            exit_reason = "max_hold_period"
        
        # Mean reversion completion (price returns to middle band)
        else:
            bb_middle = indicators["bb_middle"]
            if is_long and float(current_price) >= bb_middle * 0.99:
                exit_reason = "mean_reversion_complete"
            elif not is_long and float(current_price) <= bb_middle * 1.01:
                exit_reason = "mean_reversion_complete"
        
        if exit_reason:
            signal_type = SignalType.SELL if is_long else SignalType.BUY
            
            exit_signal = self.create_signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=Decimal('1.0'),
                price=current_price,
                quantity=abs(position),
                entry_price=float(entry_price),
                pnl_pct=pnl_pct,
                bars_held=self._bars_since_entry[symbol],
                exit_reason=exit_reason
            )
            
            self.log_info(
                f"Generated exit signal for {symbol}: {exit_reason}, "
                f"P&L={pnl_pct:.2%}, held={self._bars_since_entry[symbol]} bars"
            )
            
            return exit_signal
        
        return None
    
    def _calculate_position_size(self, symbol: str, signal_strength: float) -> int:
        """Calculate position size based on risk and signal strength."""
        # Base size adjusted by signal strength
        base_size = int(self.config.base_position_size * signal_strength)
        
        # Risk-based sizing (simplified - would need portfolio value in practice)
        risk_adjusted_size = max(1, int(base_size * self.config.risk_per_trade_pct * 100))
        
        return min(base_size, risk_adjusted_size)
    
    async def on_parameter_update(self, new_params: Dict[str, Any]) -> None:
        """Handle parameter updates."""
        await super().on_parameter_update(new_params)
        
        # Clear indicator data if calculation periods changed
        if any(param in new_params for param in ['bb_period', 'rsi_period']):
            for symbol in self.config.symbols:
                self._indicators[symbol] = {
                    "bb_upper": [],
                    "bb_middle": [],
                    "bb_lower": [],
                    "rsi": [],
                    "price_mean": []
                }
            self.log_info("Cleared indicator data due to period change")
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get current strategy state for monitoring."""
        return {
            "config": {
                "bb_period": self.config.bb_period,
                "bb_std_dev": self.config.bb_std_dev,
                "rsi_period": self.config.rsi_period,
                "rsi_overbought": self.config.rsi_overbought,
                "rsi_oversold": self.config.rsi_oversold
            },
            "current_indicators": {
                symbol: {
                    "bb_upper": data["bb_upper"][-1] if data["bb_upper"] else None,
                    "bb_middle": data["bb_middle"][-1] if data["bb_middle"] else None,
                    "bb_lower": data["bb_lower"][-1] if data["bb_lower"] else None,
                    "rsi": data["rsi"][-1] if data["rsi"] else None
                }
                for symbol, data in self._indicators.items()
            },
            "positions": {
                symbol: {
                    "size": self.get_position(symbol),
                    "entry_price": float(self._entry_prices.get(symbol, 0)),
                    "bars_held": self._bars_since_entry.get(symbol, 0)
                }
                for symbol in self.config.symbols
            },
            "metrics": self.get_metrics().dict()
        }
