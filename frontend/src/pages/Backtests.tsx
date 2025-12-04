import React, { useEffect, useState } from 'react';
import { Card } from '../components/Card';
import { api, Backtest } from '../services/api';
import { RefreshCw, Eye, TrendingUp, TrendingDown, ArrowUpCircle, ArrowDownCircle } from 'lucide-react';
import { BacktestVisualization } from '../components/BacktestVisualization';

export function Backtests() {
  const [backtests, setBacktests] = useState<Backtest[]>([]);
  const [selectedBacktest, setSelectedBacktest] = useState<Backtest | null>(null);
  const [trades, setTrades] = useState<any[]>([]);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchBacktests = async () => {
    try {
      setError(null);
      const data = await api.getBacktests(50);
      setBacktests(data.backtests || []);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBacktests();
  }, []);

  const handleViewDetails = async (backtest: Backtest) => {
    try {
      setSelectedBacktest(backtest);
      setAnalysisData(null); // Reset analysis data

      // Fetch trades
      const data = await api.getBacktestTrades(backtest.id);

      // Map API field names to frontend-friendly names
      const mappedTrades = (data.trades || []).map((t: any) => ({
        id: t.id,
        symbol: t.symbol,
        side: t.side,
        quantity: t.qty ?? t.quantity,
        entry_time: t.entry_ts ?? t.entry_time,
        entry_price: t.entry_px ?? t.entry_price,
        exit_time: t.exit_ts ?? t.exit_time,
        exit_price: t.exit_px ?? t.exit_price,
        pnl: t.pnl
      }));

      setTrades(mappedTrades);

      // Fetch analysis data for visualization
      // Extract params from backtest object
      const params = backtest.params || {};
      let symbols = params.symbols || [];

      // Fallback: extract symbols from trades if not in params
      if ((!symbols || symbols.length === 0) && data.trades && data.trades.length > 0) {
        const uniqueSymbols = Array.from(new Set(data.trades.map((t: any) => t.symbol)));
        if (uniqueSymbols.length >= 2) {
          symbols = uniqueSymbols;
        }
      }

      if (symbols.length >= 2) {
        try {
          const analysis = await api.analyzePair({
            symbol_a: symbols[0],
            symbol_b: symbols[1],
            timeframe: params.timeframe || '5 mins',
            start_date: backtest.start_ts, // Use actual run times
            end_date: backtest.end_ts,
            strategy_params: params.strategy_params
          });
          setAnalysisData(analysis);
        } catch (e) {
          console.error('Failed to fetch analysis data for visualization', e);
        }
      }
    } catch (err: any) {
      setError(err.message);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (selectedBacktest) {
    return (
      <div className="space-y-6">
        <button
          onClick={() => {
            setSelectedBacktest(null);
            setTrades([]);
            setAnalysisData(null);
          }}
          className="text-blue-600 hover:text-blue-700 font-medium"
        >
          ← Back to list
        </button>

        <Card title={`Backtest #${selectedBacktest.id} - ${selectedBacktest.strategy_name}`}>
          {/* Core Performance Metrics */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-sm text-gray-500">Total P&L</p>
                <p
                  className={`text-2xl font-bold ${(selectedBacktest.pnl || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}
                >
                  ${selectedBacktest.pnl?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Total Return</p>
                <p
                  className={`text-2xl font-bold ${(selectedBacktest.total_return_pct || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}
                >
                  {selectedBacktest.total_return_pct?.toFixed(2) || 'N/A'}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Sharpe Ratio</p>
                <p className="text-2xl font-bold text-gray-900">
                  {selectedBacktest.sharpe?.toFixed(4) || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Sortino Ratio</p>
                <p className="text-2xl font-bold text-gray-900">
                  {selectedBacktest.sortino_ratio?.toFixed(4) || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Max Drawdown</p>
                <p className="text-2xl font-bold text-red-600">
                  {selectedBacktest.maxdd?.toFixed(2) || 'N/A'}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Volatility (Ann.)</p>
                <p className="text-2xl font-bold text-gray-900">
                  {selectedBacktest.annualized_volatility_pct?.toFixed(2) || 'N/A'}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Value at Risk</p>
                <p className="text-2xl font-bold text-orange-600">
                  {selectedBacktest.value_at_risk_pct?.toFixed(2) || 'N/A'}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Max DD Duration</p>
                <p className="text-2xl font-bold text-gray-900">
                  {selectedBacktest.max_drawdown_duration_days || 'N/A'} days
                </p>
              </div>
            </div>
          </div>

          {/* Trade Statistics */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Trade Statistics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-sm text-gray-500">Total Trades</p>
                <p className="text-2xl font-bold text-gray-900">
                  {selectedBacktest.trades || 0}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Winning Trades</p>
                <p className="text-2xl font-bold text-green-600">
                  {selectedBacktest.winning_trades || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Losing Trades</p>
                <p className="text-2xl font-bold text-red-600">
                  {selectedBacktest.losing_trades || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Win Rate</p>
                <p className="text-2xl font-bold text-gray-900">
                  {selectedBacktest.win_rate?.toFixed(2) || 'N/A'}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Profit Factor</p>
                <p className="text-2xl font-bold text-gray-900">
                  {selectedBacktest.profit_factor?.toFixed(2) || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Avg Win</p>
                <p className="text-2xl font-bold text-green-600">
                  ${selectedBacktest.avg_win?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Avg Loss</p>
                <p className="text-2xl font-bold text-red-600">
                  ${selectedBacktest.avg_loss?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Largest Win</p>
                <p className="text-2xl font-bold text-green-600">
                  ${selectedBacktest.largest_win?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || 'N/A'}
                </p>
              </div>
            </div>
          </div>

          {/* Trade Timing & Costs */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Trade Timing & Costs</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-sm text-gray-500">Avg Trade Duration</p>
                <p className="text-2xl font-bold text-gray-900">
                  {selectedBacktest.avg_trade_duration_days?.toFixed(2) || 'N/A'} days
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Avg Holding Period</p>
                <p className="text-2xl font-bold text-gray-900">
                  {selectedBacktest.avg_holding_period_hours?.toFixed(2) || 'N/A'} hours
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Total Commission</p>
                <p className="text-2xl font-bold text-orange-600">
                  ${selectedBacktest.total_commission?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || 'N/A'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Total Slippage</p>
                <p className="text-2xl font-bold text-orange-600">
                  ${selectedBacktest.total_slippage?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || 'N/A'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-md">
            <h4 className="font-medium text-gray-900 mb-2">Parameters</h4>
            <pre className="text-sm text-gray-700 overflow-x-auto">
              {JSON.stringify(selectedBacktest.params, null, 2)}
            </pre>
          </div>
        </Card>

        <Card title="Trades">
          {trades.length > 0 ? (
            <div className="overflow-x-auto max-h-96 overflow-y-auto border border-gray-200 rounded-lg">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50 sticky top-0 z-10">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Symbol
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Side
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Qty
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Entry Price
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Exit Price
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      P&L
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {trades.map((trade: any, idx: number) => (
                    <tr key={idx}>
                      <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                        {trade.symbol}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`px-2 py-1 text-xs font-medium rounded ${trade.side === 'BUY'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                            }`}
                        >
                          {trade.side}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                        {trade.qty}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                        ${trade.entry_px?.toFixed(2)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                        ${trade.exit_px?.toFixed(2)}
                      </td>
                      <td
                        className={`px-6 py-4 whitespace-nowrap font-medium ${trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                          }`}
                      >
                        ${trade.pnl?.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">No trades found</p>
          )}
        </Card>

        {/* Visualization Component */}
        {selectedBacktest && analysisData && (
          <BacktestVisualization
            trades={trades}
            analysisData={analysisData}
          />
        )}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      <Card
        title="Backtest Results"
        action={
          <button
            onClick={fetchBacktests}
            className="text-blue-600 hover:text-blue-700 text-sm font-medium flex items-center"
          >
            <RefreshCw className="w-4 h-4 mr-1" />
            Refresh
          </button>
        }
      >
        {backtests.length > 0 ? (
          <div className="overflow-x-auto max-h-96 overflow-y-auto border border-gray-200 rounded-lg">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50 sticky top-0 z-10">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Strategy
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    P&L
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Return %
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Sharpe
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Sortino
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Max DD %
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Win Rate %
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Trades
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {backtests.map((backtest) => (
                  <tr key={backtest.id}>
                    <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                      #{backtest.id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-900">
                      {backtest.strategy_name}
                    </td>
                    <td
                      className={`px-6 py-4 whitespace-nowrap font-medium ${(backtest.pnl || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}
                    >
                      {backtest.pnl ? (
                        <div className="flex items-center">
                          {backtest.pnl >= 0 ? (
                            <TrendingUp className="w-4 h-4 mr-1" />
                          ) : (
                            <TrendingDown className="w-4 h-4 mr-1" />
                          )}
                          ${backtest.pnl.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </div>
                      ) : (
                        'N/A'
                      )}
                    </td>
                    <td
                      className={`px-6 py-4 whitespace-nowrap font-medium ${(backtest.total_return_pct || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}
                    >
                      {backtest.total_return_pct?.toFixed(2) || 'N/A'}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                      {backtest.sharpe?.toFixed(3) || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                      {backtest.sortino_ratio?.toFixed(3) || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-red-600">
                      {backtest.maxdd?.toFixed(2) || 'N/A'}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                      {backtest.win_rate?.toFixed(1) || 'N/A'}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                      {backtest.trades}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(backtest.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <button
                        onClick={() => handleViewDetails(backtest)}
                        className="text-blue-600 hover:text-blue-700 flex items-center text-sm font-medium"
                      >
                        <Eye className="w-4 h-4 mr-1" />
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">
            No backtests found. Run backtests via CLI to see results here.
          </p>
        )}
      </Card>
    </div>
  );
}

