"""
Backtesting Engine

This module implements the core backtesting engine that simulates strategy execution
on historical data with realistic order fills, commission, and slippage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from common.schemas import SignalType, OrderSide
from strategy_lib import BaseStrategy, StrategySignal


class OrderStatus(str, Enum):
    """Backtest order status."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class BacktestOrder:
    """Simulated order in backtest."""
    id: int
    symbol: str
    side: OrderSide
    quantity: int
    order_type: str  # MKT, LMT, STP
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: Optional[datetime] = None
    filled_price: Optional[Decimal] = None
    filled_quantity: int = 0
    commission: Decimal = Decimal('0.0')
    slippage: Decimal = Decimal('0.0')


@dataclass
class BacktestPosition:
    """Position tracking in backtest."""
    symbol: str
    quantity: int = 0
    avg_entry_price: Decimal = Decimal('0.0')
    realized_pnl: Decimal = Decimal('0.0')
    unrealized_pnl: Decimal = Decimal('0.0')
    entry_timestamp: Optional[datetime] = None
    
    def update_entry(self, price: Decimal, quantity: int, timestamp: datetime):
        """Update position on entry."""
        if self.quantity == 0:
            self.avg_entry_price = price
            self.quantity = quantity
            self.entry_timestamp = timestamp
        else:
            # Average in
            total_cost = self.avg_entry_price * Decimal(abs(self.quantity))
            new_cost = price * Decimal(abs(quantity))
            self.quantity += quantity
            if self.quantity != 0:
                self.avg_entry_price = (total_cost + new_cost) / Decimal(abs(self.quantity))
    
    def close_position(self, exit_price: Decimal, quantity: int) -> Decimal:
        """Close all or part of position and calculate realized P&L."""
        if self.quantity == 0:
            return Decimal('0.0')
        
        # Calculate P&L for the closed portion
        pnl = Decimal('0.0')
        if self.quantity > 0:  # Long position
            pnl = (exit_price - self.avg_entry_price) * Decimal(quantity)
        else:  # Short position
            pnl = (self.avg_entry_price - exit_price) * Decimal(abs(quantity))
        
        self.quantity -= quantity
        self.realized_pnl += pnl
        
        if self.quantity == 0:
            self.avg_entry_price = Decimal('0.0')
            self.entry_timestamp = None
        
        return pnl
    
    def update_unrealized(self, current_price: Decimal):
        """Update unrealized P&L."""
        if self.quantity == 0:
            self.unrealized_pnl = Decimal('0.0')
        elif self.quantity > 0:  # Long
            self.unrealized_pnl = (current_price - self.avg_entry_price) * Decimal(self.quantity)
        else:  # Short
            self.unrealized_pnl = (self.avg_entry_price - current_price) * Decimal(abs(self.quantity))


@dataclass
class BacktestTrade:
    """Completed trade in backtest."""
    symbol: str
    side: OrderSide
    entry_time: datetime
    entry_price: Decimal
    exit_time: datetime
    exit_price: Decimal
    quantity: int
    pnl: Decimal
    commission: Decimal
    slippage: Decimal
    net_pnl: Decimal


@dataclass
class BacktestMetrics:
    """Performance metrics from backtest."""
    total_pnl: Decimal = Decimal('0.0')
    total_return_pct: Decimal = Decimal('0.0')
    sharpe_ratio: Optional[Decimal] = None
    sortino_ratio: Optional[Decimal] = None
    annualized_volatility_pct: Optional[Decimal] = None
    value_at_risk_pct: Optional[Decimal] = None
    max_drawdown_pct: Decimal = Decimal('0.0')
    max_drawdown_duration_days: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal('0.0')
    avg_win: Decimal = Decimal('0.0')
    avg_loss: Decimal = Decimal('0.0')
    profit_factor: Optional[Decimal] = None
    largest_win: Decimal = Decimal('0.0')
    largest_loss: Decimal = Decimal('0.0')
    avg_trade_duration_days: Decimal = Decimal('0.0')
    avg_holding_period_hours: Decimal = Decimal('0.0')
    total_commission: Decimal = Decimal('0.0')
    total_slippage: Decimal = Decimal('0.0')
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    total_days: int = 0


class BacktestEngine:
    """
    Backtesting engine that simulates strategy execution on historical data.
    
    Features:
    - Realistic order fills with market/limit/stop orders
    - Commission and slippage modeling
    - Position tracking and P&L calculation
    - Performance metrics (Sharpe, drawdown, win rate, etc.)
    """
    
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: Decimal = Decimal('100000.0'),
        commission_per_share: Decimal = Decimal('0.005'),
        min_commission: Decimal = Decimal('1.0'),
        slippage_ticks: int = 1,
        tick_size: Decimal = Decimal('0.01')
    ):
        """
        Initialize backtest engine.
        
        Args:
            strategy: Strategy instance to backtest
            initial_capital: Starting capital
            commission_per_share: Commission per share
            min_commission: Minimum commission per order
            slippage_ticks: Slippage in ticks
            tick_size: Size of one tick
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        
        # State
        self.cash = initial_capital
        self.equity_curve: List[Tuple[datetime, Decimal]] = []
        self.positions: Dict[str, BacktestPosition] = {}
        self.orders: List[BacktestOrder] = []
        self.trades: List[BacktestTrade] = []
        self.order_id_counter = 0
        
        # Tracking
        self.daily_returns: List[Decimal] = []
        self.peak_equity = initial_capital
        self.max_drawdown = Decimal('0.0')
        
        self.logger = logging.getLogger("backtester.engine")
    
    async def run(
        self,
        bars_data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestMetrics:
        """
        Run backtest on historical data.
        
        Args:
            bars_data: Dictionary of symbol -> DataFrame with OHLCV data
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            BacktestMetrics with performance results
        """
        self.logger.info("Starting backtest")
        self.logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"Strategy: {self.strategy.config.name}")
        self.logger.info(f"Symbols: {', '.join(bars_data.keys())}")
        
        # Reset state
        self._reset_state()
        
        # Filter data by date range
        filtered_data = self._filter_by_date_range(bars_data, start_date, end_date)
        
        if not filtered_data:
            raise ValueError("No data available for backtest")
        
        # Get the union of all timestamps across symbols
        all_timestamps = self._get_all_timestamps(filtered_data)
        
        if len(all_timestamps) == 0:
            raise ValueError("No timestamps found in data")
        
        self.logger.info(f"Backtesting {len(all_timestamps)} bars from {all_timestamps[0]} to {all_timestamps[-1]}")
        
        # Initialize strategy
        instruments = {symbol: {"symbol": symbol} for symbol in filtered_data.keys()}
        await self.strategy.on_start(instruments)
        
        # Main backtest loop - iterate through each timestamp
        for i, timestamp in enumerate(all_timestamps):
            # Get current bar data for all symbols at this timestamp
            current_bars = {}
            for symbol, df in filtered_data.items():
                # Get bars up to and including current timestamp
                bars_until_now = df[df['timestamp'] <= timestamp]
                if not bars_until_now.empty:
                    current_bars[symbol] = bars_until_now
            
            # Process fills for pending orders
            self._process_pending_orders(current_bars, timestamp)
            
            # Generate signals - use multi-symbol mode if strategy supports it
            if self.strategy.supports_multi_symbol:
                # Multi-symbol mode: pass all symbols' data together
                symbols_ready = [
                    symbol for symbol, bars_df in current_bars.items()
                    if len(bars_df) >= self.strategy.config.lookback_periods
                ]
                
                if symbols_ready:
                    # Prepare bars data with only lookback period
                    bars_data_limited = {
                        symbol: current_bars[symbol].tail(self.strategy.config.lookback_periods)
                        for symbol in symbols_ready
                    }
                    
                    # Call multi-symbol strategy method
                    signals = await self.strategy.on_bar_multi(
                        symbols=symbols_ready,
                        timeframe=self.strategy.config.bar_timeframe,
                        bars_data=bars_data_limited
                    )
                    
                    # Process all signals
                    for signal in signals:
                        # Get bars for the signal's symbol
                        signal_bars = current_bars.get(signal.symbol)
                        if signal_bars is not None:
                            await self._process_signal(signal, signal_bars, timestamp)
            else:
                # Single-symbol mode: call strategy for each symbol individually
                for symbol, bars_df in current_bars.items():
                    if len(bars_df) >= self.strategy.config.lookback_periods:
                        # Call strategy with historical bars
                        signals = await self.strategy.on_bar(
                            symbol=symbol,
                            timeframe=self.strategy.config.bar_timeframe,
                            bars=bars_df.tail(self.strategy.config.lookback_periods)
                        )
                        
                        # Process signals
                        for signal in signals:
                            await self._process_signal(signal, bars_df, timestamp)
            
            # Update positions and equity
            self._update_positions(current_bars, timestamp)
            
            # Record progress (suppress during optimization to reduce log noise)
            is_optimization = self.strategy.config.strategy_id.startswith('opt_')
            if not is_optimization and i % 50 == 0:
                self.logger.info(f"Progress: {i}/{len(all_timestamps)} bars, Equity: ${self._get_total_equity(current_bars):,.2f}")
        
        # Stop strategy
        await self.strategy.on_stop()
        
        # Process any remaining pending orders before final close
        final_bars = {symbol: df for symbol, df in filtered_data.items()}
        self._process_pending_orders(final_bars, all_timestamps[-1])
        
        # Close any remaining positions at final prices
        self._close_all_positions(filtered_data, all_timestamps[-1])
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_timestamps[0], all_timestamps[-1])
        
        # Suppress detailed completion logs during optimization
        is_optimization = self.strategy.config.strategy_id.startswith('opt_')
        if not is_optimization:
            self.logger.info("Backtest completed")
            self.logger.info(f"Final equity: ${self._get_total_equity(filtered_data):,.2f}")
            self.logger.info(f"Total P&L: ${metrics.total_pnl:,.2f}")
            self.logger.info(f"Total trades: {metrics.total_trades}")
        else:
            # Minimal log for optimization
            self.logger.debug(f"Backtest completed: {metrics.total_trades} trades, P&L: ${metrics.total_pnl:,.2f}")
        
        return metrics
    
    def _reset_state(self):
        """Reset engine state for new backtest."""
        self.cash = self.initial_capital
        self.equity_curve = [(datetime.now(timezone.utc), self.initial_capital)]
        self.positions = {}
        self.orders = []
        self.trades = []
        self.order_id_counter = 0
        self.daily_returns = []
        self.peak_equity = self.initial_capital
        self.max_drawdown = Decimal('0.0')
    
    def _filter_by_date_range(
        self,
        bars_data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Dict[str, pd.DataFrame]:
        """Filter bars data by date range."""
        filtered = {}
        for symbol, df in bars_data.items():
            filtered_df = df.copy()
            if start_date:
                filtered_df = filtered_df[filtered_df['timestamp'] >= start_date]
            if end_date:
                filtered_df = filtered_df[filtered_df['timestamp'] <= end_date]
            if not filtered_df.empty:
                filtered[symbol] = filtered_df
        return filtered
    
    def _get_all_timestamps(self, bars_data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Get sorted list of all unique timestamps across all symbols."""
        all_timestamps = set()
        for df in bars_data.values():
            all_timestamps.update(df['timestamp'].tolist())
        return sorted(list(all_timestamps))
    
    async def _process_signal(self, signal: StrategySignal, bars_df: pd.DataFrame, timestamp: datetime):
        """Process a strategy signal and create orders."""
        if signal.signal_type == SignalType.HOLD:
            return
        
        # Determine order side
        if signal.signal_type == SignalType.BUY:
            side = OrderSide.BUY
        elif signal.signal_type == SignalType.SELL:
            side = OrderSide.SELL
        elif signal.signal_type == SignalType.EXIT:
            # Exit current position
            position = self.positions.get(signal.symbol)
            if not position or position.quantity == 0:
                return
            side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
            signal.quantity = abs(position.quantity)
        else:
            return
        
        # Create market order (simplified - always use market orders)
        order = BacktestOrder(
            id=self._get_next_order_id(),
            symbol=signal.symbol,
            side=side,
            quantity=signal.quantity or 100,
            order_type="MKT",
            created_at=timestamp
        )
        
        self.orders.append(order)
        self.logger.debug(f"Created order {order.id}: {order.side} {order.quantity} {order.symbol} @ {timestamp}")
    
    def _process_pending_orders(self, current_bars: Dict[str, pd.DataFrame], timestamp: datetime):
        """Process pending orders and fill them if conditions are met."""
        for order in self.orders:
            if order.status != OrderStatus.PENDING:
                continue
            
            if order.symbol not in current_bars:
                continue
            
            bars_df = current_bars[order.symbol]
            current_bar = bars_df[bars_df['timestamp'] == timestamp]
            
            if current_bar.empty:
                continue
            
            current_bar = current_bar.iloc[-1]
            
            # Simulate fill (simplified - fill at close with slippage)
            fill_price = self._calculate_fill_price(order, current_bar)
            
            if fill_price is None:
                continue
            
            # Calculate commission
            commission = max(
                self.commission_per_share * Decimal(order.quantity),
                self.min_commission
            )
            
            # Check if we have enough cash for buys
            if order.side == OrderSide.BUY:
                total_cost = fill_price * Decimal(order.quantity) + commission
                if total_cost > self.cash:
                    order.status = OrderStatus.REJECTED
                    self.logger.warning(f"Order {order.id} rejected: insufficient cash")
                    continue
                self.cash -= total_cost
            else:  # SELL
                self.cash += fill_price * Decimal(order.quantity) - commission
            
            # Fill order
            order.status = OrderStatus.FILLED
            order.filled_at = timestamp
            order.filled_price = fill_price
            order.filled_quantity = order.quantity
            order.commission = commission
            
            # Update position
            self._update_position_from_fill(order, timestamp)
            
            self.logger.debug(
                f"Filled order {order.id}: {order.side} {order.quantity} {order.symbol} "
                f"@ ${fill_price:.2f}, commission=${commission:.2f}"
            )
    
    def _calculate_fill_price(self, order: BacktestOrder, bar: pd.Series) -> Optional[Decimal]:
        """Calculate fill price with slippage."""
        if order.order_type == "MKT":
            # Market order fills at close price with slippage
            base_price = Decimal(str(bar['close']))
            slippage = self.tick_size * Decimal(self.slippage_ticks)
            
            if order.side == OrderSide.BUY:
                # Pay slippage on buys
                fill_price = base_price + slippage
            else:
                # Lose slippage on sells
                fill_price = base_price - slippage
            
            order.slippage = slippage * Decimal(order.quantity)
            return fill_price
        
        # TODO: Add limit and stop order logic
        return None
    
    def _update_position_from_fill(self, order: BacktestOrder, timestamp: datetime):
        """Update position tracking from filled order."""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = BacktestPosition(symbol=symbol)
        
        position = self.positions[symbol]
        
        # Determine quantity change (positive for longs, negative for shorts)
        qty_change = order.quantity if order.side == OrderSide.BUY else -order.quantity
        
        # Check if this closes a position (opposite direction)
        if position.quantity != 0 and np.sign(qty_change) != np.sign(position.quantity):
            # Save entry details BEFORE closing (close_position clears them)
            entry_ts = position.entry_timestamp
            entry_px = position.avg_entry_price
            position_side = OrderSide.BUY if position.quantity > 0 else OrderSide.SELL
            
            # Closing trade
            close_qty = min(abs(qty_change), abs(position.quantity))
            pnl = position.close_position(order.filled_price, close_qty if position.quantity > 0 else -close_qty)
            
            # Record trade using saved entry details
            if entry_ts:
                trade = BacktestTrade(
                    symbol=symbol,
                    side=position_side,
                    entry_time=entry_ts,
                    entry_price=entry_px,
                    exit_time=timestamp,
                    exit_price=order.filled_price,
                    quantity=close_qty,
                    pnl=pnl,
                    commission=order.commission,
                    slippage=order.slippage,
                    net_pnl=pnl - order.commission - order.slippage
                )
                self.trades.append(trade)
                self.logger.info(
                    f"Closed trade: {trade.side.value} {trade.quantity} {trade.symbol}, "
                    f"P&L: ${trade.net_pnl:,.2f}"
                )
            
            # If qty_change exceeds position size, open new position in opposite direction
            remaining_qty = abs(qty_change) - close_qty
            if remaining_qty > 0:
                qty_change = remaining_qty if order.side == OrderSide.BUY else -remaining_qty
                position.update_entry(order.filled_price, qty_change, timestamp)
        else:
            # Opening or adding to position
            position.update_entry(order.filled_price, qty_change, timestamp)
    
    def _update_positions(self, current_bars: Dict[str, pd.DataFrame], timestamp: datetime):
        """Update unrealized P&L for all positions."""
        total_equity = self.cash
        
        for symbol, position in self.positions.items():
            if position.quantity == 0:
                continue
            
            if symbol in current_bars:
                df = current_bars[symbol]
                current_bar = df[df['timestamp'] == timestamp]
                if not current_bar.empty:
                    current_price = Decimal(str(current_bar.iloc[-1]['close']))
                    position.update_unrealized(current_price)
                    total_equity += position.unrealized_pnl + (current_price * Decimal(abs(position.quantity)))
        
        # Record equity
        self.equity_curve.append((timestamp, total_equity))
        
        # Update drawdown
        if total_equity > self.peak_equity:
            self.peak_equity = total_equity
        else:
            drawdown = (self.peak_equity - total_equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
    
    def _close_all_positions(self, bars_data: Dict[str, pd.DataFrame], timestamp: datetime):
        """Close all remaining positions at final prices."""
        for symbol, position in self.positions.items():
            if position.quantity == 0:
                continue
            
            if symbol in bars_data:
                final_price = Decimal(str(bars_data[symbol].iloc[-1]['close']))
                
                # Save position details before closing
                entry_ts = position.entry_timestamp
                entry_px = position.avg_entry_price
                qty = abs(position.quantity)
                side = OrderSide.BUY if position.quantity > 0 else OrderSide.SELL
                
                # Close position and get P&L
                pnl = position.close_position(final_price, position.quantity)
                
                self.logger.info(f"Closing {symbol}: side={side.value}, qty={qty}, pnl=${pnl:.2f}, entry_ts={entry_ts}")
                
                if entry_ts:
                    trade = BacktestTrade(
                        symbol=symbol,
                        side=side,
                        entry_time=entry_ts,
                        entry_price=entry_px,
                        exit_time=timestamp,
                        exit_price=final_price,
                        quantity=qty,
                        pnl=pnl,
                        commission=Decimal('0.0'),
                        slippage=Decimal('0.0'),
                        net_pnl=pnl
                    )
                    self.trades.append(trade)
                    self.logger.info(f"Closed final position: {trade.symbol}, P&L: ${trade.net_pnl:,.2f}")
                else:
                    self.logger.warning(f"Position {symbol} has no entry timestamp!")
    
    def _get_total_equity(self, bars_data: Dict[str, pd.DataFrame]) -> Decimal:
        """Calculate total equity (cash + positions)."""
        total = self.cash
        
        for symbol, position in self.positions.items():
            if position.quantity == 0:
                continue
            
            if symbol in bars_data:
                last_price = Decimal(str(bars_data[symbol].iloc[-1]['close']))
                total += last_price * Decimal(abs(position.quantity))
        
        return total
    
    def _calculate_metrics(self, start_date: datetime, end_date: datetime) -> BacktestMetrics:
        """Calculate performance metrics."""
        metrics = BacktestMetrics()
        
        # Basic metrics
        final_equity = self.equity_curve[-1][1] if self.equity_curve else self.initial_capital
        metrics.total_pnl = final_equity - self.initial_capital
        metrics.total_return_pct = (metrics.total_pnl / self.initial_capital) * Decimal('100.0')
        metrics.max_drawdown_pct = self.max_drawdown * Decimal('100.0')
        metrics.start_date = start_date
        metrics.end_date = end_date
        metrics.total_days = (end_date - start_date).days
        
        # Trade metrics
        metrics.total_trades = len(self.trades)
        if metrics.total_trades > 0:
            wins = [t for t in self.trades if t.net_pnl > 0]
            losses = [t for t in self.trades if t.net_pnl < 0]
            
            metrics.winning_trades = len(wins)
            metrics.losing_trades = len(losses)
            metrics.win_rate = Decimal(metrics.winning_trades) / Decimal(metrics.total_trades) * Decimal('100.0')
            
            if wins:
                metrics.avg_win = sum(t.net_pnl for t in wins) / Decimal(len(wins))
                metrics.largest_win = max(t.net_pnl for t in wins)
            
            if losses:
                metrics.avg_loss = sum(t.net_pnl for t in losses) / Decimal(len(losses))
                metrics.largest_loss = min(t.net_pnl for t in losses)
            
            # Profit factor
            total_wins = sum(t.net_pnl for t in wins)
            total_losses = abs(sum(t.net_pnl for t in losses))
            if total_losses > 0:
                metrics.profit_factor = total_wins / total_losses
            
            # Average trade duration
            durations_seconds = [
                Decimal(str((t.exit_time - t.entry_time).total_seconds()))
                for t in self.trades
                if t.exit_time and t.entry_time
            ]
            if durations_seconds:
                avg_duration_seconds = sum(durations_seconds, Decimal('0')) / Decimal(len(durations_seconds))
                metrics.avg_trade_duration_days = avg_duration_seconds / Decimal('86400')
                metrics.avg_holding_period_hours = avg_duration_seconds / Decimal('3600')
            
            # Commission and slippage
            metrics.total_commission = sum(t.commission for t in self.trades)
            metrics.total_slippage = sum(t.slippage for t in self.trades)
        
        # Sharpe ratio (simplified - using daily returns)
        if len(self.equity_curve) > 1:
            returns: List[float] = []
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i - 1][1]
                curr_equity = self.equity_curve[i][1]
                if prev_equity > 0:
                    daily_return = (curr_equity - prev_equity) / prev_equity
                    returns.append(float(daily_return))
            
            if returns:
                returns_array = np.array(returns, dtype=float)
                mean_return = float(np.mean(returns_array))
                std_return = float(np.std(returns_array))
                
                if std_return > 0:
                    metrics.sharpe_ratio = Decimal(str((mean_return / std_return) * np.sqrt(252)))
                
                annualized_vol = std_return * np.sqrt(252)
                metrics.annualized_volatility_pct = Decimal(str(annualized_vol * 100))
                
                downside_returns = returns_array[returns_array < 0]
                if downside_returns.size > 0:
                    downside_std = float(np.std(downside_returns))
                    if downside_std > 0:
                        metrics.sortino_ratio = Decimal(str((mean_return / downside_std) * np.sqrt(252)))
                
                var_95 = float(np.percentile(returns_array, 5))
                metrics.value_at_risk_pct = Decimal(str(max(0.0, -var_95) * 100))
        
        return metrics
    
    def _get_next_order_id(self) -> int:
        """Get next order ID."""
        self.order_id_counter += 1
        return self.order_id_counter
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'symbol': t.symbol,
                'side': t.side.value,
                'entry_time': t.entry_time,
                'entry_price': float(t.entry_price),
                'exit_time': t.exit_time,
                'exit_price': float(t.exit_price),
                'quantity': t.quantity,
                'pnl': float(t.pnl),
                'commission': float(t.commission),
                'slippage': float(t.slippage),
                'net_pnl': float(t.net_pnl),
                'duration_days': (t.exit_time - t.entry_time).days
            }
            for t in self.trades
        ])
    
    def get_equity_curve_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame for plotting."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {'timestamp': ts, 'equity': float(eq)}
            for ts, eq in self.equity_curve
        ])
