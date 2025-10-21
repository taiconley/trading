import React, { useEffect, useState } from 'react';
import { LineChart, Line, Scatter, ScatterChart, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, ReferenceDot } from 'recharts';
import { Card } from '../components/Card';
import { api, Backtest } from '../services/api';
import { RefreshCw, Eye, TrendingUp, TrendingDown, ArrowUpCircle, ArrowDownCircle } from 'lucide-react';

export function Backtests() {
  const [backtests, setBacktests] = useState<Backtest[]>([]);
  const [selectedBacktest, setSelectedBacktest] = useState<Backtest | null>(null);
  const [trades, setTrades] = useState<any[]>([]);
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
      const data = await api.getBacktestTrades(backtest.id);
      setTrades(data.trades || []);
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
          }}
          className="text-blue-600 hover:text-blue-700 font-medium"
        >
          ← Back to list
        </button>

        <Card title={`Backtest #${selectedBacktest.id} - ${selectedBacktest.strategy_name}`}>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-6">
            <div>
              <p className="text-sm text-gray-500">Total P&L</p>
              <p
                className={`text-2xl font-bold ${
                  (selectedBacktest.pnl || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                }`}
              >
                ${selectedBacktest.pnl?.toFixed(2) || '0.00'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Sharpe Ratio</p>
              <p className="text-2xl font-bold text-gray-900">
                {selectedBacktest.sharpe?.toFixed(4) || 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Max Drawdown</p>
              <p className="text-2xl font-bold text-red-600">
                {selectedBacktest.maxdd?.toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Trades</p>
              <p className="text-2xl font-bold text-gray-900">
                {selectedBacktest.trades || 0}
              </p>
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
                          className={`px-2 py-1 text-xs font-medium rounded ${
                            trade.side === 'BUY'
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
                        className={`px-6 py-4 whitespace-nowrap font-medium ${
                          trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
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
                    Sharpe
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Max DD
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
                      className={`px-6 py-4 whitespace-nowrap font-medium ${
                        (backtest.pnl || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {backtest.pnl ? (
                        <div className="flex items-center">
                          {backtest.pnl >= 0 ? (
                            <TrendingUp className="w-4 h-4 mr-1" />
                          ) : (
                            <TrendingDown className="w-4 h-4 mr-1" />
                          )}
                          ${backtest.pnl.toFixed(2)}
                        </div>
                      ) : (
                        'N/A'
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                      {backtest.sharpe?.toFixed(4) || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-red-600">
                      {backtest.maxdd?.toFixed(2)}%
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

