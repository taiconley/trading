import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart } from 'recharts';
import { Download, BarChart3, Clock, Search } from 'lucide-react';
import { Card } from '../components/Card';
import { api } from '../services/api';

interface Dataset {
  symbol: string;
  timeframe: string;
  bar_count: number;
  start_date: string;
  end_date: string;
}

interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export default function HistoricalData() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [loadingChart, setLoadingChart] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  
  // Request form state
  const [symbol, setSymbol] = useState('');
  const [barSize, setBarSize] = useState('1 min');
  const [duration, setDuration] = useState('1 D');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    loadDatasets();
    const interval = setInterval(loadDatasets, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  const loadDatasets = async () => {
    try {
      const data = await api.getHistoricalDatasets();
      setDatasets(data.datasets || []);
    } catch (error) {
      console.error('Failed to load datasets:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadCandles = async (dataset: Dataset) => {
    setLoadingChart(true);
    setSelectedDataset(dataset);
    try {
      console.log('Loading candles for:', dataset);
      const data = await api.getCandles({
        symbol: dataset.symbol,
        timeframe: dataset.timeframe,
        limit: 500
      });
      console.log('Received candles:', data);
      setCandles(data.candles || []);
    } catch (error: any) {
      console.error('Failed to load candles:', error);
      alert(`Failed to load chart: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoadingChart(false);
    }
  };

  const submitRequest = async () => {
    if (!symbol) {
      alert('Please enter a symbol');
      return;
    }
    
    setSubmitting(true);
    try {
      const response = await api.requestHistoricalData({
        symbol: symbol.toUpperCase(),
        bar_size: barSize,
        lookback: duration
      });
      alert(`Request queued: ${response.message || 'Success'}`);
      setSymbol('');
      setTimeout(loadDatasets, 2000); // Refresh after 2s
    } catch (error: any) {
      alert(`Failed to submit request: ${error.response?.data?.detail || error.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const bulkRequest = async () => {
    setSubmitting(true);
    try {
      const response = await api.bulkHistoricalRequest();
      alert(`Queued ${response.requests?.length || 0} requests for all watchlist symbols`);
      setTimeout(loadDatasets, 2000);
    } catch (error: any) {
      alert(`Failed to submit bulk request: ${error.response?.data?.detail || error.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const filteredDatasets = datasets.filter(d =>
    d.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    d.timeframe.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Format candle data for chart
  const chartData = candles.map(c => ({
    time: new Date(c.timestamp).toLocaleString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      hour: '2-digit', 
      minute: '2-digit' 
    }),
    open: c.open,
    high: c.high,
    low: c.low,
    close: c.close,
    volume: c.volume,
    // For line chart
    price: c.close
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Historical Data</h1>
        <p className="text-sm text-slate-600 mt-1">Request and visualize historical market data</p>
      </div>

      {/* Request Form */}
      <Card title="Request Historical Data">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Individual Request */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-slate-700">Individual Symbol</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Symbol</label>
                <input
                  type="text"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  placeholder="AAPL"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Bar Size</label>
                  <select
                    value={barSize}
                    onChange={(e) => setBarSize(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="5 secs">5 secs</option>
                    <option value="1 min">1 min</option>
                    <option value="5 mins">5 mins</option>
                    <option value="15 mins">15 mins</option>
                    <option value="1 hour">1 hour</option>
                    <option value="1 day">1 day</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Duration</label>
                  <select
                    value={duration}
                    onChange={(e) => setDuration(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="1 D">1 Day</option>
                    <option value="5 D">5 Days</option>
                    <option value="1 W">1 Week</option>
                    <option value="1 M">1 Month</option>
                    <option value="3 M">3 Months</option>
                    <option value="1 Y">1 Year</option>
                  </select>
                </div>
              </div>
              <button
                onClick={submitRequest}
                disabled={submitting}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Download className="w-5 h-5" />
                {submitting ? 'Submitting...' : 'Request Data'}
              </button>
            </div>
          </div>

          {/* Bulk Request */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-slate-700">Bulk Request</h3>
            <div className="bg-slate-50 rounded-lg p-4 space-y-3">
              <p className="text-sm text-slate-600">
                Request historical data for all symbols in your watchlist with the default timeframes (1 min, 5 mins, 1 day).
              </p>
              <button
                onClick={bulkRequest}
                disabled={submitting}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Clock className="w-5 h-5" />
                {submitting ? 'Submitting...' : 'Bulk Request All Symbols'}
              </button>
            </div>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <p className="text-xs text-blue-800">
                <strong>Note:</strong> Requests are rate-limited to 6 per minute to comply with TWS pacing rules.
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* Available Datasets */}
      <Card title={`Available Datasets (${filteredDatasets.length})`}>
        <div className="mb-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search symbols or timeframes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        </div>

        {loading ? (
          <div className="text-center py-8 text-slate-600">Loading datasets...</div>
        ) : filteredDatasets.length === 0 ? (
          <div className="text-center py-8 text-slate-600">No datasets found. Request some historical data above!</div>
        ) : (
          <div className="overflow-x-auto max-h-96 overflow-y-auto border border-gray-200 rounded-lg">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50 sticky top-0 z-10">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Symbol</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Timeframe</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Bars</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Date Range</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-700 uppercase tracking-wider">Action</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {filteredDatasets.map((dataset, idx) => (
                  <tr key={idx} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm font-semibold text-slate-900">{dataset.symbol}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-slate-600">{dataset.timeframe}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-slate-600">{dataset.bar_count.toLocaleString()}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">
                      {new Date(dataset.start_date).toLocaleDateString()} - {new Date(dataset.end_date).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <button
                        onClick={() => loadCandles(dataset)}
                        className="flex items-center gap-1 px-3 py-1 text-sm bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition-colors"
                      >
                        <BarChart3 className="w-4 h-4" />
                        View Chart
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* Chart Viewer */}
      {selectedDataset && (
        <Card title={`${selectedDataset.symbol} - ${selectedDataset.timeframe}`}>
          {loadingChart ? (
            <div className="text-center py-12 text-slate-600">Loading chart data...</div>
          ) : candles.length === 0 ? (
            <div className="text-center py-12 text-slate-600">No data available</div>
          ) : (
            <div className="space-y-6">
              {/* Price Chart */}
              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-3">Price Action</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis 
                      dataKey="time" 
                      tick={{ fontSize: 12, fill: '#64748b' }}
                      tickLine={{ stroke: '#cbd5e1' }}
                    />
                    <YAxis 
                      domain={['auto', 'auto']}
                      tick={{ fontSize: 12, fill: '#64748b' }}
                      tickLine={{ stroke: '#cbd5e1' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#fff', 
                        border: '1px solid #e2e8f0',
                        borderRadius: '8px',
                        padding: '8px'
                      }}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="price" 
                      name="Close Price"
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* OHLC Details */}
              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-3">OHLC Chart</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis 
                      dataKey="time" 
                      tick={{ fontSize: 12, fill: '#64748b' }}
                      tickLine={{ stroke: '#cbd5e1' }}
                    />
                    <YAxis 
                      yAxisId="price"
                      domain={['auto', 'auto']}
                      tick={{ fontSize: 12, fill: '#64748b' }}
                      tickLine={{ stroke: '#cbd5e1' }}
                    />
                    <YAxis 
                      yAxisId="volume"
                      orientation="right"
                      tick={{ fontSize: 12, fill: '#64748b' }}
                      tickLine={{ stroke: '#cbd5e1' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#fff', 
                        border: '1px solid #e2e8f0',
                        borderRadius: '8px',
                        padding: '8px'
                      }}
                    />
                    <Legend />
                    <Bar yAxisId="volume" dataKey="volume" name="Volume" fill="#cbd5e1" opacity={0.3} />
                    <Line yAxisId="price" type="monotone" dataKey="high" name="High" stroke="#10b981" strokeWidth={1} dot={false} />
                    <Line yAxisId="price" type="monotone" dataKey="low" name="Low" stroke="#ef4444" strokeWidth={1} dot={false} />
                    <Line yAxisId="price" type="monotone" dataKey="close" name="Close" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              {/* Data Stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-gray-200">
                <div>
                  <div className="text-xs text-slate-600">Bars Loaded</div>
                  <div className="text-lg font-bold text-slate-900">{candles.length.toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-xs text-slate-600">Highest Price</div>
                  <div className="text-lg font-bold text-green-600">
                    ${Math.max(...candles.map(c => c.high)).toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-600">Lowest Price</div>
                  <div className="text-lg font-bold text-red-600">
                    ${Math.min(...candles.map(c => c.low)).toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-slate-600">Avg Volume</div>
                  <div className="text-lg font-bold text-slate-900">
                    {(candles.reduce((sum, c) => sum + c.volume, 0) / candles.length).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </div>
                </div>
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

