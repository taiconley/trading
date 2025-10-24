import { useState, useEffect } from 'react';
import { LineChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart } from 'recharts';
import { Download, BarChart3, Search, Trash2, Calendar, Copy } from 'lucide-react';
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
  const [bulkMode, setBulkMode] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  
  // Date range mode
  const [dateRangeMode, setDateRangeMode] = useState<'relative' | 'absolute'>('relative');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

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

  const deleteDataset = async (dataset: Dataset) => {
    if (!confirm(`Are you sure you want to delete all ${dataset.bar_count.toLocaleString()} candles for ${dataset.symbol} (${dataset.timeframe})?`)) {
      return;
    }

    try {
      const response = await api.deleteDataset(dataset.symbol, dataset.timeframe);
      alert(`Successfully deleted ${response.deleted_count} candles`);
      
      // Clear selected dataset if it was the one deleted
      if (selectedDataset?.symbol === dataset.symbol && selectedDataset?.timeframe === dataset.timeframe) {
        setSelectedDataset(null);
        setCandles([]);
      }
      
      // Refresh datasets
      await loadDatasets();
    } catch (error: any) {
      console.error('Failed to delete dataset:', error);
      alert(`Failed to delete dataset: ${error.response?.data?.detail || error.message}`);
    }
  };

  const matchDateRange = (dataset: Dataset) => {
    // Convert dataset dates to date input format (YYYY-MM-DD)
    const start = new Date(dataset.start_date);
    const end = new Date(dataset.end_date);
    
    setStartDate(start.toISOString().split('T')[0]);
    setEndDate(end.toISOString().split('T')[0]);
    setDateRangeMode('absolute');
    
    // Scroll to the request form
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const submitRequest = async () => {
    // Validate input for individual mode
    if (!bulkMode && !symbol) {
      alert('Please enter a symbol');
      return;
    }
    
    // Validate date range in absolute mode
    if (dateRangeMode === 'absolute') {
      if (!endDate) {
        alert('Please enter an end date');
        return;
      }
      if (!startDate) {
        alert('Please enter a start date');
        return;
      }
      if (new Date(startDate) > new Date(endDate)) {
        alert('Start date must be before end date');
        return;
      }
    }
    
    setSubmitting(true);
    try {
      // Prepare request parameters
      let requestParams: any = {
        bar_size: barSize,
      };

      if (dateRangeMode === 'absolute') {
        // Convert end date to TWS format: "YYYYMMDD HH:MM:SS"
        const endDateTime = new Date(endDate);
        endDateTime.setHours(23, 59, 59); // Set to end of day
        const tws_end_datetime = endDateTime.toISOString()
          .replace('T', ' ')
          .split('.')[0]
          .replace(/-/g, '');
        
        requestParams.end_datetime = tws_end_datetime;
        
        // Calculate duration from start to end date
        const start = new Date(startDate);
        const end = new Date(endDate);
        const diffTime = Math.abs(end.getTime() - start.getTime());
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        // Convert to appropriate TWS duration string
        if (diffDays <= 1) {
          requestParams.duration = '1 D';
        } else if (diffDays <= 7) {
          requestParams.duration = `${diffDays} D`;
        } else if (diffDays <= 30) {
          const weeks = Math.ceil(diffDays / 7);
          requestParams.duration = `${weeks} W`;
        } else if (diffDays <= 365) {
          const months = Math.ceil(diffDays / 30);
          requestParams.duration = `${months} M`;
        } else {
          const years = Math.ceil(diffDays / 365);
          requestParams.duration = `${years} Y`;
        }
      } else {
        // Relative mode - use duration
        requestParams.duration = duration;
      }

      if (bulkMode) {
        // Bulk request for all watchlist symbols
        const response = await api.bulkHistoricalRequest(requestParams);
        const modeStr = dateRangeMode === 'absolute' 
          ? `from ${startDate} to ${endDate}` 
          : `${duration}`;
        alert(`Queued ${response.requests?.length || 0} requests for all watchlist symbols (${barSize}, ${modeStr})`);
      } else {
        // Individual symbol request
        const response = await api.requestHistoricalData({
          symbol: symbol.toUpperCase(),
          bar_size: barSize,
          duration: requestParams.duration,
          end_datetime: requestParams.end_datetime
        });
        alert(`Request queued: ${response.message || 'Success'}`);
        setSymbol('');
      }
      setTimeout(loadDatasets, 2000); // Refresh after 2s
    } catch (error: any) {
      alert(`Failed to submit request: ${error.response?.data?.detail || error.message}`);
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
        <div className="max-w-2xl mx-auto space-y-4">
          {/* Bulk Mode Checkbox */}
          <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg border border-slate-200">
            <input
              type="checkbox"
              id="bulkMode"
              checked={bulkMode}
              onChange={(e) => setBulkMode(e.target.checked)}
              className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
            />
            <label htmlFor="bulkMode" className="text-sm font-medium text-slate-700 cursor-pointer">
              Request data for all watchlist symbols
            </label>
          </div>

          {/* Date Range Mode Toggle */}
          <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <Calendar className="w-5 h-5 text-blue-600" />
            <div className="flex gap-2">
              <button
                onClick={() => setDateRangeMode('relative')}
                className={`px-3 py-1 text-sm font-medium rounded transition-colors ${
                  dateRangeMode === 'relative'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-slate-700 hover:bg-slate-100'
                }`}
              >
                Relative (Duration)
              </button>
              <button
                onClick={() => setDateRangeMode('absolute')}
                className={`px-3 py-1 text-sm font-medium rounded transition-colors ${
                  dateRangeMode === 'absolute'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-slate-700 hover:bg-slate-100'
                }`}
              >
                Absolute (Date Range)
              </button>
            </div>
          </div>

          {/* Request Form */}
          <div className="space-y-4">
            {/* Symbol Input - disabled when bulk mode is on */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Symbol {bulkMode && <span className="text-slate-400">(all watchlist symbols)</span>}
              </label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="AAPL"
                disabled={bulkMode}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:text-gray-500 disabled:cursor-not-allowed"
              />
            </div>

            {/* Bar Size */}
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

            {/* Duration (Relative Mode) or Date Range (Absolute Mode) */}
            {dateRangeMode === 'relative' ? (
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Duration (lookback)</label>
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
            ) : (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Start Date</label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">End Date</label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>
            )}

            {/* Submit Button */}
            <button
              onClick={submitRequest}
              disabled={submitting}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Download className="w-5 h-5" />
              {submitting ? 'Submitting...' : (bulkMode ? 'Request All Watchlist Symbols' : 'Request Data')}
            </button>

            {/* Info Notes */}
            <div className="space-y-2">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <p className="text-xs text-blue-800">
                  <strong>Note:</strong> Requests are rate-limited to 6 per minute to comply with TWS pacing rules.
                </p>
              </div>
              {dateRangeMode === 'absolute' && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                  <p className="text-xs text-amber-800">
                    <strong>Tip:</strong> Use the "Match Range" button in the datasets table below to copy date ranges from existing data.
                  </p>
                </div>
              )}
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
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => loadCandles(dataset)}
                          className="flex items-center gap-1 px-3 py-1 text-sm bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition-colors"
                          title="View chart"
                        >
                          <BarChart3 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => matchDateRange(dataset)}
                          className="flex items-center gap-1 px-3 py-1 text-sm bg-green-50 text-green-700 rounded-lg hover:bg-green-100 transition-colors"
                          title="Match date range for new request"
                        >
                          <Copy className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => deleteDataset(dataset)}
                          className="flex items-center gap-1 px-3 py-1 text-sm bg-red-50 text-red-700 rounded-lg hover:bg-red-100 transition-colors"
                          title="Delete dataset"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
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

