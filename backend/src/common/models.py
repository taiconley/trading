"""
SQLAlchemy models for the trading bot database.

This module defines all database tables and relationships for the trading system.
Models are organized by functional area: reference/control, account/portfolio,
market data, trading, and signals/research.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
import sqlalchemy as sa
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Numeric, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


# =============================================================================
# Reference & Control Tables
# =============================================================================

class Symbol(Base):
    """Symbol reference data from Interactive Brokers."""
    __tablename__ = 'symbols'
    
    symbol = Column(String(20), primary_key=True, nullable=False)
    conid = Column(Integer, nullable=True, unique=True)  # IB contract ID
    primary_exchange = Column(String(20), nullable=True)
    currency = Column(String(3), nullable=False, default='USD')
    active = Column(Boolean, nullable=False, default=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    watchlist_entries = relationship("WatchlistEntry", back_populates="symbol_ref")
    ticks = relationship("Tick", back_populates="symbol_ref")
    candles = relationship("Candle", back_populates="symbol_ref")
    orders = relationship("Order", back_populates="symbol_ref")
    positions = relationship("Position", back_populates="symbol_ref")
    executions = relationship("Execution", back_populates="symbol_ref")
    signals = relationship("Signal", back_populates="symbol_ref")
    backtest_trades = relationship("BacktestTrade", back_populates="symbol_ref")


class WatchlistEntry(Base):
    """Symbols in the live market data watchlist."""
    __tablename__ = 'watchlist'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), ForeignKey('symbols.symbol'), nullable=False)
    added_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    symbol_ref = relationship("Symbol", back_populates="watchlist_entries")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('symbol', name='uq_watchlist_symbol'),
        Index('ix_watchlist_symbol', 'symbol'),
    )


class Strategy(Base):
    """Strategy configurations and parameters."""
    __tablename__ = 'strategies'
    
    strategy_id = Column(String(50), primary_key=True, nullable=False)
    name = Column(String(100), nullable=False)
    enabled = Column(Boolean, nullable=False, default=False)
    params_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    orders = relationship("Order", back_populates="strategy_ref")
    signals = relationship("Signal", back_populates="strategy_ref")


class RiskLimit(Base):
    """Risk management limits and controls."""
    __tablename__ = 'risk_limits'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(50), nullable=False, unique=True)
    value_json = Column(JSON, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Constraints
    __table_args__ = (
        Index('ix_risk_limits_key', 'key'),
    )


class RiskViolation(Base):
    """Risk limit violations and audit trail."""
    __tablename__ = 'risk_violations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    violation_type = Column(String(50), nullable=False)  # order_size, position_limit, daily_loss, etc
    severity = Column(String(20), nullable=False, default='warning')  # info, warning, critical
    account_id = Column(String(50), nullable=True)
    symbol = Column(String(20), nullable=True)
    strategy_id = Column(Integer, nullable=True)
    order_id = Column(Integer, nullable=True)
    limit_key = Column(String(50), nullable=False)
    limit_value = Column(Numeric(20, 4), nullable=True)
    actual_value = Column(Numeric(20, 4), nullable=True)
    message = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)  # Additional context
    action_taken = Column(String(50), nullable=False, default='rejected')  # rejected, warned, allowed
    resolved = Column(Boolean, nullable=False, default=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Foreign keys
    strategy_fk = relationship("Strategy", foreign_keys=[strategy_id], 
                              primaryjoin="RiskViolation.strategy_id == Strategy.strategy_id",
                              back_populates=None)
    
    # Constraints
    __table_args__ = (
        Index('ix_risk_violations_created_at', 'created_at'),
        Index('ix_risk_violations_type_severity', 'violation_type', 'severity'),
        Index('ix_risk_violations_account', 'account_id'),
        Index('ix_risk_violations_symbol', 'symbol'),
        Index('ix_risk_violations_resolved', 'resolved'),
        CheckConstraint("severity IN ('info', 'warning', 'critical')", 
                       name='ck_risk_violations_severity'),
        CheckConstraint("action_taken IN ('rejected', 'warned', 'allowed', 'emergency_stop')", 
                       name='ck_risk_violations_action'),
    )


class HealthStatus(Base):
    """Service health monitoring."""
    __tablename__ = 'health'
    
    service = Column(String(50), primary_key=True, nullable=False)
    status = Column(String(20), nullable=False)  # healthy, unhealthy, starting, stopping
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('healthy', 'unhealthy', 'starting', 'stopping')", 
                       name='ck_health_status'),
    )


class LogEntry(Base):
    """Application logs (optional database storage)."""
    __tablename__ = 'logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    service = Column(String(50), nullable=False)
    level = Column(String(10), nullable=False)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    msg = Column(Text, nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False, default=func.now())
    meta_json = Column(JSON, nullable=True)
    
    # Constraints
    __table_args__ = (
        Index('ix_logs_service_ts', 'service', 'ts'),
        Index('ix_logs_level_ts', 'level', 'ts'),
        CheckConstraint("level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')", 
                       name='ck_logs_level'),
    )


# =============================================================================
# Account & Portfolio Tables
# =============================================================================

class Account(Base):
    """IB account information."""
    __tablename__ = 'accounts'
    
    account_id = Column(String(20), primary_key=True, nullable=False)
    currency = Column(String(3), nullable=False, default='USD')
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    account_summaries = relationship("AccountSummary", back_populates="account_ref")
    positions = relationship("Position", back_populates="account_ref")
    orders = relationship("Order", back_populates="account_ref")


class AccountSummary(Base):
    """Account summary data from IB."""
    __tablename__ = 'account_summary'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String(20), ForeignKey('accounts.account_id'), nullable=False)
    tag = Column(String(50), nullable=False)  # NetLiquidation, TotalCashValue, etc.
    value = Column(String(50), nullable=False)  # Store as string to preserve precision
    currency = Column(String(3), nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    account_ref = relationship("Account", back_populates="account_summaries")
    
    # Constraints
    __table_args__ = (
        Index('ix_account_summary_account_tag_ts', 'account_id', 'tag', 'ts'),
    )


class Position(Base):
    """Current positions from IB."""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String(20), ForeignKey('accounts.account_id'), nullable=False)
    symbol = Column(String(20), ForeignKey('symbols.symbol'), nullable=False)
    conid = Column(Integer, nullable=True)
    qty = Column(Numeric(15, 2), nullable=False)
    avg_price = Column(Numeric(10, 4), nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    account_ref = relationship("Account", back_populates="positions")
    symbol_ref = relationship("Symbol", back_populates="positions")
    
    # Constraints
    __table_args__ = (
        Index('ix_positions_account_symbol', 'account_id', 'symbol'),
        Index('ix_positions_symbol', 'symbol'),
    )


# =============================================================================
# Market Data Tables
# =============================================================================

class Tick(Base):
    """Live market data ticks."""
    __tablename__ = 'ticks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), ForeignKey('symbols.symbol'), nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False, default=func.now())
    bid = Column(Numeric(10, 4), nullable=True)
    ask = Column(Numeric(10, 4), nullable=True)
    last = Column(Numeric(10, 4), nullable=True)
    bid_size = Column(Integer, nullable=True)
    ask_size = Column(Integer, nullable=True)
    last_size = Column(Integer, nullable=True)
    
    # Relationships
    symbol_ref = relationship("Symbol", back_populates="ticks")
    
    # Constraints
    __table_args__ = (
        Index('ix_ticks_symbol_ts', 'symbol', 'ts'),
    )


class Candle(Base):
    """Historical and live bar data."""
    __tablename__ = 'candles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), ForeignKey('symbols.symbol'), nullable=False)
    tf = Column(String(20), nullable=False)  # timeframe: "1 min", "5 mins", "1 day"
    ts = Column(DateTime(timezone=True), nullable=False)  # bar timestamp
    open = Column(Numeric(10, 4), nullable=False)
    high = Column(Numeric(10, 4), nullable=False)
    low = Column(Numeric(10, 4), nullable=False)
    close = Column(Numeric(10, 4), nullable=False)
    volume = Column(Integer, nullable=False, default=0)
    
    # Relationships
    symbol_ref = relationship("Symbol", back_populates="candles")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('symbol', 'tf', 'ts', name='uq_candles_symbol_tf_ts'),
        Index('ix_candles_symbol_tf_ts', 'symbol', 'tf', 'ts'),
    )


class HistoricalJob(Base):
    """Persisted historical data collection job."""
    __tablename__ = 'historical_jobs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_key = Column(String(128), nullable=False, unique=True)
    symbol = Column(String(20), ForeignKey('symbols.symbol'), nullable=False)
    bar_size = Column(String(20), nullable=False)
    what_to_show = Column(String(50), nullable=False, default='TRADES')
    use_rth = Column(Boolean, nullable=False, default=True)
    duration = Column(String(20), nullable=False)
    end_datetime = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default='pending')
    total_chunks = Column(Integer, nullable=False, default=0)
    completed_chunks = Column(Integer, nullable=False, default=0)
    failed_chunks = Column(Integer, nullable=False, default=0)
    priority = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)
    
    symbol_ref = relationship("Symbol")
    chunks = relationship("HistoricalJobChunk", back_populates="job", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_historical_jobs_status', 'status'),
        Index('ix_historical_jobs_symbol_tf', 'symbol', 'bar_size'),
    )


class HistoricalJobChunk(Base):
    """Individual request chunk for a historical job."""
    __tablename__ = 'historical_job_chunks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey('historical_jobs.id', ondelete='CASCADE'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    request_id = Column(String(128), nullable=False, unique=True)
    status = Column(String(20), nullable=False, default='pending')
    duration = Column(String(20), nullable=False)
    start_datetime = Column(DateTime(timezone=True), nullable=True)
    end_datetime = Column(DateTime(timezone=True), nullable=True)
    scheduled_for = Column(DateTime(timezone=True), nullable=False, default=func.now())
    priority = Column(Integer, nullable=False, default=0)
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=5)
    bars_expected = Column(Integer, nullable=True)
    bars_received = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    job = relationship("HistoricalJob", back_populates="chunks")
    
    __table_args__ = (
        UniqueConstraint('job_id', 'chunk_index', name='uq_historical_job_chunks_index'),
        Index('ix_historical_job_chunks_status', 'status'),
        Index('ix_historical_job_chunks_scheduled', 'scheduled_for'),
    )


class HistoricalCoverage(Base):
    """Coverage metadata for historical datasets."""
    __tablename__ = 'historical_coverage'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), ForeignKey('symbols.symbol'), nullable=False)
    timeframe = Column(String(20), nullable=False)
    min_ts = Column(DateTime(timezone=True), nullable=True)
    max_ts = Column(DateTime(timezone=True), nullable=True)
    total_bars = Column(Integer, nullable=False, default=0)
    last_updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    last_verified_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)
    
    symbol_ref = relationship("Symbol")
    
    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', name='uq_historical_coverage_symbol_tf'),
        Index('ix_historical_coverage_symbol_tf', 'symbol', 'timeframe'),
    )


# =============================================================================
# Trading Tables
# =============================================================================

class Order(Base):
    """Order records and lifecycle."""
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String(20), ForeignKey('accounts.account_id'), nullable=False)
    strategy_id = Column(String(50), ForeignKey('strategies.strategy_id'), nullable=True)
    symbol = Column(String(20), ForeignKey('symbols.symbol'), nullable=False)
    side = Column(String(4), nullable=False)  # BUY, SELL
    qty = Column(Numeric(15, 2), nullable=False)
    order_type = Column(String(15), nullable=False)  # MKT, LMT, STP, STP-LMT, ADAPTIVE, PEG BEST, PEG MID
    limit_price = Column(Numeric(10, 4), nullable=True)
    stop_price = Column(Numeric(10, 4), nullable=True)
    tif = Column(String(10), nullable=False, default='DAY')  # DAY, GTC, IOC, FOK
    status = Column(String(20), nullable=False, default='PendingSubmit')
    external_order_id = Column(String(50), nullable=True)  # IB order ID
    algo_strategy = Column(String(50), nullable=True)  # Algorithm strategy name (e.g., 'Adaptive')
    algo_params = Column(JSON, nullable=True)  # Algorithm-specific parameters as JSON
    placed_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    account_ref = relationship("Account", back_populates="orders")
    strategy_ref = relationship("Strategy", back_populates="orders")
    symbol_ref = relationship("Symbol", back_populates="orders")
    executions = relationship("Execution", back_populates="order_ref")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("side IN ('BUY', 'SELL')", name='ck_orders_side'),
        CheckConstraint("order_type IN ('MKT', 'LMT', 'STP', 'STP-LMT', 'ADAPTIVE', 'PEG BEST', 'PEG MID')", name='ck_orders_type'),
        CheckConstraint("tif IN ('DAY', 'GTC', 'IOC', 'FOK')", name='ck_orders_tif'),
        CheckConstraint("qty > 0", name='ck_orders_qty_positive'),
        Index('ix_orders_updated_at', 'updated_at'),
        Index('ix_orders_strategy_symbol', 'strategy_id', 'symbol'),
        Index('ix_orders_external_id', 'external_order_id'),
    )


class Execution(Base):
    """Trade executions."""
    __tablename__ = 'executions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=False)
    trade_id = Column(String(50), nullable=True)  # IB execution ID
    symbol = Column(String(20), ForeignKey('symbols.symbol'), nullable=False)
    qty = Column(Numeric(15, 2), nullable=False)
    price = Column(Numeric(10, 4), nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    order_ref = relationship("Order", back_populates="executions")
    symbol_ref = relationship("Symbol", back_populates="executions")
    
    # Constraints
    __table_args__ = (
        Index('ix_executions_order_id_ts', 'order_id', 'ts'),
        Index('ix_executions_symbol_ts', 'symbol', 'ts'),
        CheckConstraint("qty > 0", name='ck_executions_qty_positive'),
        CheckConstraint("price > 0", name='ck_executions_price_positive'),
    )


# =============================================================================
# Signals & Research Tables  
# =============================================================================

class Signal(Base):
    """Strategy signals for analysis."""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(50), ForeignKey('strategies.strategy_id'), nullable=False)
    symbol = Column(String(20), ForeignKey('symbols.symbol'), nullable=False)
    signal_type = Column(String(20), nullable=False)  # BUY, SELL, HOLD, etc.
    strength = Column(Numeric(5, 4), nullable=True)  # Signal strength 0.0-1.0
    ts = Column(DateTime(timezone=True), nullable=False, default=func.now())
    meta_json = Column(JSON, nullable=True)  # Additional signal metadata
    
    # Relationships
    strategy_ref = relationship("Strategy", back_populates="signals")
    symbol_ref = relationship("Symbol", back_populates="signals")
    
    # Constraints
    __table_args__ = (
        Index('ix_signals_strategy_ts', 'strategy_id', 'ts'),
        Index('ix_signals_symbol_ts', 'symbol', 'ts'),
        CheckConstraint("strength IS NULL OR (strength >= 0.0 AND strength <= 1.0)", 
                       name='ck_signals_strength_range'),
    )


class BacktestRun(Base):
    """Backtest execution results."""
    __tablename__ = 'backtest_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False)
    params_json = Column(JSON, nullable=True)
    start_ts = Column(DateTime(timezone=True), nullable=False)
    end_ts = Column(DateTime(timezone=True), nullable=False)
    
    # Core performance metrics
    pnl = Column(Numeric(15, 2), nullable=False)
    total_return_pct = Column(Numeric(8, 4), nullable=True)
    sharpe = Column(Numeric(8, 4), nullable=True)
    sortino_ratio = Column(Numeric(8, 4), nullable=True)
    annualized_volatility_pct = Column(Numeric(8, 4), nullable=True)
    value_at_risk_pct = Column(Numeric(8, 4), nullable=True)
    maxdd = Column(Numeric(8, 4), nullable=True)  # Max drawdown as percentage
    max_drawdown_duration_days = Column(Integer, nullable=True)
    
    # Trade statistics
    trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=True)
    losing_trades = Column(Integer, nullable=True)
    win_rate = Column(Numeric(8, 4), nullable=True)
    profit_factor = Column(Numeric(8, 4), nullable=True)
    
    # Trade performance
    avg_win = Column(Numeric(15, 2), nullable=True)
    avg_loss = Column(Numeric(15, 2), nullable=True)
    largest_win = Column(Numeric(15, 2), nullable=True)
    largest_loss = Column(Numeric(15, 2), nullable=True)
    
    # Trade timing
    avg_trade_duration_days = Column(Numeric(8, 4), nullable=True)
    avg_holding_period_hours = Column(Numeric(8, 4), nullable=True)
    
    # Costs
    total_commission = Column(Numeric(15, 2), nullable=True)
    total_slippage = Column(Numeric(15, 2), nullable=True)
    
    # Additional metadata
    total_days = Column(Integer, nullable=True)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    backtest_trades = relationship("BacktestTrade", back_populates="run_ref")
    
    # Constraints
    __table_args__ = (
        Index('ix_backtest_runs_strategy_created', 'strategy_name', 'created_at'),
        CheckConstraint("start_ts < end_ts", name='ck_backtest_runs_date_order'),
        CheckConstraint("trades >= 0", name='ck_backtest_runs_trades_positive'),
    )


class BacktestTrade(Base):
    """Individual trades from backtest runs."""
    __tablename__ = 'backtest_trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('backtest_runs.id'), nullable=False)
    symbol = Column(String(20), ForeignKey('symbols.symbol'), nullable=False)
    side = Column(String(4), nullable=False)  # BUY, SELL
    qty = Column(Numeric(15, 2), nullable=False)
    entry_ts = Column(DateTime(timezone=True), nullable=False)
    entry_px = Column(Numeric(10, 4), nullable=False)
    exit_ts = Column(DateTime(timezone=True), nullable=True)
    exit_px = Column(Numeric(10, 4), nullable=True)
    pnl = Column(Numeric(15, 2), nullable=True)
    
    # Relationships
    run_ref = relationship("BacktestRun", back_populates="backtest_trades")
    symbol_ref = relationship("Symbol", back_populates="backtest_trades")
    
    # Constraints
    __table_args__ = (
        Index('ix_backtest_trades_run_symbol', 'run_id', 'symbol'),
        CheckConstraint("side IN ('BUY', 'SELL')", name='ck_backtest_trades_side'),
        CheckConstraint("qty > 0", name='ck_backtest_trades_qty_positive'),
        CheckConstraint("entry_px > 0", name='ck_backtest_trades_entry_px_positive'),
        CheckConstraint("exit_px IS NULL OR exit_px > 0", name='ck_backtest_trades_exit_px_positive'),
        CheckConstraint("exit_ts IS NULL OR entry_ts <= exit_ts", name='ck_backtest_trades_date_order'),
    )


class PairsAnalysis(Base):
    """Pairs analysis results for statistical arbitrage strategies."""
    __tablename__ = 'potential_pairs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_a = Column(String(20), ForeignKey('symbols.symbol'), nullable=False)
    symbol_b = Column(String(20), ForeignKey('symbols.symbol'), nullable=False)
    timeframe = Column(String(10), nullable=False)  # e.g. '5s'
    window_start = Column(DateTime(timezone=True), nullable=False)
    window_end = Column(DateTime(timezone=True), nullable=False)
    sample_bars = Column(Integer, nullable=False)
    
    # Volume metrics
    avg_dollar_volume_a = Column(Numeric(18, 2), nullable=False)
    avg_dollar_volume_b = Column(Numeric(18, 2), nullable=False)
    
    # Hedge regression results
    hedge_ratio = Column(Numeric(18, 8), nullable=False)
    hedge_intercept = Column(Numeric(18, 8), nullable=False)
    
    # Statistical tests
    adf_statistic = Column(Numeric(18, 8), nullable=True)  # Test statistic (more negative = more stationary)
    adf_pvalue = Column(Numeric(10, 6), nullable=True)  # P-value (lower = more significant)
    coint_statistic = Column(Numeric(18, 8), nullable=True)  # Test statistic (more negative = stronger)
    coint_pvalue = Column(Numeric(10, 6), nullable=True)  # P-value (lower = more significant)
    half_life_minutes = Column(Numeric(18, 6), nullable=True)
    
    # Spread characteristics
    spread_mean = Column(Numeric(18, 8), nullable=True)
    spread_std = Column(Numeric(18, 8), nullable=True)
    
    # Simulated trading results
    simulated_entry_z = Column(Numeric(18, 8), nullable=True)
    simulated_exit_z = Column(Numeric(18, 8), nullable=True)
    pair_sharpe = Column(Numeric(18, 8), nullable=True)
    pair_profit_factor = Column(Numeric(18, 8), nullable=True)
    pair_max_drawdown = Column(Numeric(18, 8), nullable=True)
    pair_avg_holding_minutes = Column(Numeric(18, 6), nullable=True)
    pair_total_trades = Column(Integer, nullable=False, default=0)
    pair_win_rate = Column(Numeric(6, 3), nullable=True)
    
    # Status and metadata
    status = Column(String(20), nullable=False, default='candidate')
    meta = Column(JSON, nullable=True, default={})
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Constraints
    __table_args__ = (
        CheckConstraint("status IN ('candidate', 'validated', 'rejected')", name='ck_potential_pairs_status'),
        Index('idx_potential_pairs_symbols_timeframe', 'symbol_a', 'symbol_b', 'timeframe'),
        Index('idx_potential_pairs_window', 'window_start', 'window_end'),
        Index('idx_potential_pairs_status', 'status'),
    )
    
    # Relationships
    symbol_a_ref = relationship("Symbol", foreign_keys=[symbol_a])
    symbol_b_ref = relationship("Symbol", foreign_keys=[symbol_b])


class OptimizationRun(Base):
    """Parameter optimization run tracking."""
    __tablename__ = 'optimization_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False)
    algorithm = Column(String(50), nullable=False)  # 'grid_search', 'random_search', 'bayesian', etc.
    symbols = Column(JSON, nullable=False)  # List of symbols
    timeframe = Column(String(20), nullable=False)  # Bar timeframe
    param_ranges = Column(JSON, nullable=False)  # Parameter space definition
    objective = Column(String(50), nullable=False)  # 'sharpe_ratio', 'total_return', 'profit_factor', etc.
    status = Column(String(20), nullable=False, default='pending')  # 'pending', 'running', 'completed', 'failed', 'stopped'
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    total_combinations = Column(Integer, nullable=True)
    completed_combinations = Column(Integer, nullable=False, default=0)
    best_params = Column(JSON, nullable=True)  # Best parameter combination found
    best_score = Column(Numeric(15, 6), nullable=True)  # Best objective score
    error_message = Column(Text, nullable=True)  # Error details if failed
    config = Column(JSON, nullable=True)  # Additional configuration (cores, constraints, etc.)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    results = relationship("OptimizationResult", back_populates="run", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        Index('ix_optimization_runs_status_created', 'status', 'created_at'),
        Index('ix_optimization_runs_strategy_created', 'strategy_name', 'created_at'),
        CheckConstraint("status IN ('pending', 'running', 'completed', 'failed', 'stopped')", 
                       name='ck_optimization_runs_status'),
        CheckConstraint("completed_combinations >= 0", 
                       name='ck_optimization_runs_completed_positive'),
        CheckConstraint("total_combinations IS NULL OR total_combinations > 0", 
                       name='ck_optimization_runs_total_positive'),
    )


class OptimizationResult(Base):
    """Individual parameter combination result from optimization."""
    __tablename__ = 'optimization_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('optimization_runs.id'), nullable=False)
    params_json = Column(JSON, nullable=False)  # The specific parameters tested
    backtest_run_id = Column(Integer, ForeignKey('backtest_runs.id'), nullable=True)  # Link to backtest
    
    # Performance metrics (duplicated from backtest for quick access)
    score = Column(Numeric(15, 6), nullable=False)  # Objective function value
    
    # Core performance metrics
    sharpe_ratio = Column(Numeric(8, 4), nullable=True)
    sortino_ratio = Column(Numeric(8, 4), nullable=True)
    total_return_pct = Column(Numeric(8, 4), nullable=True)
    annualized_volatility_pct = Column(Numeric(8, 4), nullable=True)
    value_at_risk_pct = Column(Numeric(8, 4), nullable=True)
    max_drawdown_pct = Column(Numeric(8, 4), nullable=True)
    max_drawdown_duration_days = Column(Integer, nullable=True)
    
    # Trade statistics
    total_trades = Column(Integer, nullable=True)
    winning_trades = Column(Integer, nullable=True)
    losing_trades = Column(Integer, nullable=True)
    win_rate = Column(Numeric(8, 4), nullable=True)
    profit_factor = Column(Numeric(8, 4), nullable=True)
    
    # Trade performance
    avg_win = Column(Numeric(15, 2), nullable=True)
    avg_loss = Column(Numeric(15, 2), nullable=True)
    largest_win = Column(Numeric(15, 2), nullable=True)
    largest_loss = Column(Numeric(15, 2), nullable=True)
    
    # Trade timing
    avg_trade_duration_days = Column(Numeric(8, 4), nullable=True)
    avg_holding_period_hours = Column(Numeric(8, 4), nullable=True)
    
    # Costs
    total_commission = Column(Numeric(15, 2), nullable=True)
    total_slippage = Column(Numeric(15, 2), nullable=True)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    run = relationship("OptimizationRun", back_populates="results")
    backtest_run_ref = relationship("BacktestRun")
    
    # Constraints
    __table_args__ = (
        Index('ix_optimization_results_run_score', 'run_id', 'score'),
        Index('ix_optimization_results_run_created', 'run_id', 'created_at'),
    )


class ParameterSensitivity(Base):
    """Parameter sensitivity analysis results."""
    __tablename__ = 'parameter_sensitivity'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('optimization_runs.id'), nullable=False)
    parameter_name = Column(String(100), nullable=False)
    
    # Sensitivity metrics
    sensitivity_score = Column(Numeric(15, 6), nullable=False)  # Overall sensitivity measure
    correlation_with_objective = Column(Numeric(8, 6), nullable=True)  # Correlation coefficient
    importance_rank = Column(Integer, nullable=True)  # Ranking by importance (1=most important)
    
    # Statistical measures
    mean_score = Column(Numeric(15, 6), nullable=True)  # Mean objective value for this parameter
    std_score = Column(Numeric(15, 6), nullable=True)  # Standard deviation
    min_score = Column(Numeric(15, 6), nullable=True)  # Minimum objective value
    max_score = Column(Numeric(15, 6), nullable=True)  # Maximum objective value
    
    # Interaction effects (stored as JSON)
    # Format: {"param1": correlation, "param2": correlation, ...}
    interactions = Column(JSON, nullable=True)
    
    # Raw analysis data (stored as JSON for detailed drill-down)
    # Format: {"values": [...], "scores": [...], "statistics": {...}}
    analysis_data = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Relationships
    run = relationship("OptimizationRun")
    
    # Constraints
    __table_args__ = (
        Index('ix_parameter_sensitivity_run_param', 'run_id', 'parameter_name'),
        Index('ix_parameter_sensitivity_run_importance', 'run_id', 'importance_rank'),
        UniqueConstraint('run_id', 'parameter_name', name='uq_parameter_sensitivity_run_param'),
    )


# =============================================================================
# Additional Indexes for Performance
# =============================================================================

# These indexes are created as part of the table definitions above, but we can add
# additional composite indexes here if needed for specific query patterns

# Example of adding additional indexes after table creation:
# Index('ix_ticks_symbol_ts_desc', Tick.symbol, Tick.ts.desc())
# Index('ix_candles_symbol_tf_ts_desc', Candle.symbol, Candle.tf, Candle.ts.desc())
# Index('ix_orders_account_status_updated', Order.account_id, Order.status, Order.updated_at)

# =============================================================================
# Utility Functions
# =============================================================================

def get_utc_now():
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


class DataCollectionJob(Base):
    """Data collection job tracking."""
    __tablename__ = 'data_collection_jobs'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    bar_size = Column(String(50), nullable=False)
    what_to_show = Column(String(50), nullable=False, default='TRADES')
    use_rth = Column(Boolean, nullable=False, default=True)
    status = Column(String(50), nullable=False, default='pending')  # pending, running, paused, completed, failed
    total_symbols = Column(Integer, nullable=False)
    completed_symbols = Column(Integer, nullable=False, default=0)
    failed_symbols = Column(Integer, nullable=False, default=0)
    total_requests = Column(Integer, nullable=False)
    completed_requests = Column(Integer, nullable=False, default=0)
    failed_requests = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    symbols = relationship("DataCollectionSymbol", back_populates="job", cascade="all, delete-orphan")


class DataCollectionSymbol(Base):
    """Individual symbol tracking within a collection job."""
    __tablename__ = 'data_collection_symbols'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey('data_collection_jobs.id', ondelete='CASCADE'), nullable=False)
    symbol = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, default='pending')  # pending, running, completed, failed
    total_requests = Column(Integer, nullable=False)
    completed_requests = Column(Integer, nullable=False, default=0)
    failed_requests = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    job = relationship("DataCollectionJob", back_populates="symbols")
    requests = relationship("DataCollectionRequest", back_populates="symbol_record", cascade="all, delete-orphan")


class DataCollectionRequest(Base):
    """Individual request tracking within a symbol."""
    __tablename__ = 'data_collection_requests'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey('data_collection_jobs.id', ondelete='CASCADE'), nullable=False)
    symbol_id = Column(Integer, ForeignKey('data_collection_symbols.id', ondelete='CASCADE'), nullable=False)
    symbol = Column(String(50), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)
    request_id = Column(String(255), nullable=True)  # Historical service request ID
    status = Column(String(50), nullable=False, default='pending')  # pending, queued, completed, failed
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    job = relationship("DataCollectionJob")
    symbol_record = relationship("DataCollectionSymbol", back_populates="requests")


def create_all_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def drop_all_tables(engine):
    """Drop all tables from the database (use with caution)."""
    Base.metadata.drop_all(bind=engine)
