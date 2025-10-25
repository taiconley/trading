/**
 * API Client for Trading Bot Backend
 * 
 * Provides typed methods for all backend API endpoints
 */

import axios, { AxiosInstance } from 'axios';

// Use relative URLs - nginx will proxy /api/ to backend-api:8000
const API_BASE_URL = '';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // ============================================================================
  // Health & Status
  // ============================================================================

  async healthCheck() {
    const response = await this.client.get('/healthz');
    return response.data;
  }

  async getAggregateHealth() {
    const response = await this.client.get('/api/health');
    return response.data;
  }

  // ============================================================================
  // Account & Positions
  // ============================================================================

  async getAccountStats() {
    const response = await this.client.get('/api/account');
    return response.data;
  }

  async getPositions() {
    const response = await this.client.get('/api/positions');
    return response.data;
  }

  // ============================================================================
  // Orders
  // ============================================================================

  async getOrders(params?: { status?: string; symbol?: string; limit?: number }) {
    const response = await this.client.get('/api/orders', { params });
    return response.data;
  }

  async getOrder(orderId: number) {
    const response = await this.client.get(`/api/orders/${orderId}`);
    return response.data;
  }

  async placeOrder(order: {
    symbol: string;
    side: 'BUY' | 'SELL';
    qty: number;
    order_type: 'MKT' | 'LMT' | 'STP' | 'STP-LMT';
    limit_price?: number;
    stop_price?: number;
    tif?: string;
  }) {
    const response = await this.client.post('/api/orders', order);
    return response.data;
  }

  async cancelOrder(orderId: number) {
    const response = await this.client.post(`/api/orders/${orderId}/cancel`);
    return response.data;
  }

  // ============================================================================
  // Market Data
  // ============================================================================

  async getTicks(symbol: string, limit: number = 100) {
    const response = await this.client.get('/api/ticks', {
      params: { symbol, limit },
    });
    return response.data;
  }

  async getSubscriptions() {
    const response = await this.client.get('/api/subscriptions');
    return response.data;
  }

  async getWatchlist() {
    const response = await this.client.get('/api/watchlist');
    return response.data;
  }

  async updateWatchlist(action: 'add' | 'remove', symbol: string) {
    const response = await this.client.post('/api/watchlist', {
      action,
      symbol,
    });
    return response.data;
  }

  // ============================================================================
  // Historical Data
  // ============================================================================

  async requestHistoricalData(request: {
    symbol: string;
    bar_size: string;
    duration: string;
    end_datetime?: string;
  }) {
    const response = await this.client.post('/api/historical/request', request);
    return response.data;
  }

  async bulkHistoricalRequest(request?: {
    bar_size?: string;
    duration?: string;
    end_datetime?: string;
  }) {
    const response = await this.client.post('/api/historical/bulk', request || {});
    return response.data;
  }

  async getHistoricalQueue() {
    const response = await this.client.get('/api/historical/queue');
    return response.data;
  }

  async getHistoricalDatasets() {
    const response = await this.client.get('/api/historical/datasets');
    return response.data;
  }

  async getCandles(params: {
    symbol: string;
    timeframe: string;
    limit?: number;
    start_date?: string;
    end_date?: string;
  }) {
    const response = await this.client.get('/api/historical/candles', { params });
    return response.data;
  }

  async deleteDataset(symbol: string, timeframe: string) {
    const response = await this.client.delete('/api/historical/dataset', {
      params: { symbol, timeframe }
    });
    return response.data;
  }

  // ============================================================================
  // Strategies
  // ============================================================================

  async getStrategies() {
    const response = await this.client.get('/api/strategies');
    return response.data;
  }

  async enableStrategy(strategyId: string, enabled: boolean) {
    const response = await this.client.post(
      `/api/strategies/${strategyId}/enable`,
      { enabled }
    );
    return response.data;
  }

  async updateStrategyParams(strategyId: string, params: Record<string, any>) {
    const response = await this.client.put(
      `/api/strategies/${strategyId}/params`,
      params
    );
    return response.data;
  }

  // ============================================================================
  // Backtests
  // ============================================================================

  async getBacktests(limit: number = 50) {
    const response = await this.client.get('/api/backtests', {
      params: { limit },
    });
    return response.data;
  }

  async getBacktest(runId: number) {
    const response = await this.client.get(`/api/backtests/${runId}`);
    return response.data;
  }

  async getBacktestTrades(runId: number) {
    const response = await this.client.get(`/api/backtests/${runId}/trades`);
    return response.data;
  }

  // ============================================================================
  // Optimizations
  // ============================================================================

  async getOptimizations(limit: number = 50) {
    const response = await this.client.get('/api/optimizations', {
      params: { limit },
    });
    return response.data;
  }

  async getOptimization(runId: number) {
    const response = await this.client.get(`/api/optimizations/${runId}`);
    return response.data;
  }

  async getOptimizationResults(runId: number, topN: number = 20) {
    const response = await this.client.get(`/api/optimizations/${runId}/results`, {
      params: { top_n: topN },
    });
    return response.data;
  }

  async getOptimizationAnalysis(runId: number) {
    const response = await this.client.get(`/api/optimizations/${runId}/analysis`);
    return response.data;
  }

  // ============================================================================
  // Signals & Executions
  // ============================================================================

  async getSignals(params?: {
    strategy_id?: number;
    symbol?: string;
    limit?: number;
  }) {
    const response = await this.client.get('/api/signals', { params });
    return response.data;
  }

  async getExecutions(limit: number = 100) {
    const response = await this.client.get('/api/executions', {
      params: { limit },
    });
    return response.data;
  }
}

// Export singleton instance
export const api = new ApiClient();

// Export types
export interface Order {
  id: number;
  symbol: string;
  side: 'BUY' | 'SELL';
  qty: number;
  order_type: string;
  limit_price?: number;
  stop_price?: number;
  status: string;
  placed_at: string;
  updated_at: string;
}

export interface Position {
  symbol: string;
  qty: number;
  avg_price: number;
  market_value?: number;
  unrealized_pnl?: number;
}

export interface Backtest {
  id: number;
  strategy_name: string;
  params: Record<string, any>;
  start_ts?: string;
  end_ts?: string;
  
  // Core performance metrics
  pnl?: number;
  total_return_pct?: number;
  sharpe?: number;
  sortino_ratio?: number;
  annualized_volatility_pct?: number;
  value_at_risk_pct?: number;
  maxdd?: number;
  max_drawdown_duration_days?: number;
  
  // Trade statistics
  trades: number;
  winning_trades?: number;
  losing_trades?: number;
  win_rate?: number;
  profit_factor?: number;
  
  // Trade performance
  avg_win?: number;
  avg_loss?: number;
  largest_win?: number;
  largest_loss?: number;
  
  // Trade timing
  avg_trade_duration_days?: number;
  avg_holding_period_hours?: number;
  
  // Costs
  total_commission?: number;
  total_slippage?: number;
  
  // Additional metadata
  total_days?: number;
  created_at: string;
}

export interface Optimization {
  id: number;
  strategy_name: string;
  algorithm: string;
  symbols: string[];
  status: string;
  total_combinations: number;
  completed_combinations: number;
  best_params?: Record<string, any>;
  best_score?: number;
  created_at: string;
  
  // Best result metrics
  best_total_return_pct?: number;
  best_sharpe_ratio?: number;
  best_win_rate?: number;
  best_max_drawdown_pct?: number;
  best_sortino_ratio?: number;
  best_profit_factor?: number;
  best_total_trades?: number;
}

export interface Strategy {
  id: string;
  name: string;
  enabled: boolean;
  params: Record<string, any>;
  created_at: string;
}

