import { useState, useEffect, ChangeEvent } from 'react';
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

interface HistoricalJobSummary {
  id: number;
  job_key: string;
  symbol: string;
  bar_size: string;
  status: string;
  total_chunks: number;
  completed_chunks: number;
  failed_chunks: number;
  created_at?: string;
  updated_at?: string;
  started_at?: string;
  completed_at?: string;
}

interface QueueActiveRequest {
  id: string;
  symbol: string;
  bar_size: string;
  status: string;
  started_at?: string | null;
}

interface QueueCompletion {
  id: string;
  symbol: string;
  bar_size: string;
  status: string;
  bars_count: number;
  completed_at?: string | null;
  error?: string | null;
}

interface QueueSummary {
  queue_size: number;
  active_requests: QueueActiveRequest[];
  recent_completions: QueueCompletion[];
  db_summary?: {
    pending_chunks: number;
    in_progress_chunks: number;
    failed_chunks: number;
    skipped_chunks: number;
    total_jobs: number;
    completed_jobs: number;
    failed_jobs: number;
  };
}

export default function HistoricalData() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [loadingChart, setLoadingChart] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [queueStatus, setQueueStatus] = useState<QueueSummary | null>(null);
  const [jobs, setJobs] = useState<HistoricalJobSummary[]>([]);
  const [jobsLoading, setJobsLoading] = useState(true);
  const [uploadedSymbols, setUploadedSymbols] = useState<string[]>([]);
  const [uploadFileName, setUploadFileName] = useState('');
  const [uploadError, setUploadError] = useState<string | null>(null);
  
  // Request form state
  const [symbol, setSymbol] = useState('');
  const [barSize, setBarSize] = useState('1 min');
  const [duration, setDuration] = useState('1 D');
  const [requestMode, setRequestMode] = useState<'single' | 'watchlist' | 'uploaded'>('single');
  const [submitting, setSubmitting] = useState(false);
  
  // Date range mode
  const [dateRangeMode, setDateRangeMode] = useState<'relative' | 'absolute'>('relative');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

useEffect(() => {
  const refresh = () => {
    loadQueueStatus();
    loadJobs();
  };

  loadDatasets();
  refresh();

  const intervalId = setInterval(refresh, 10000); // Refresh tables every 10s
  return () => clearInterval(intervalId);
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
  
  const loadQueueStatus = async () => {
    try {
      const data = await api.getHistoricalQueue();
      const normalized: QueueSummary = {
        queue_size: data.queue_size ?? 0,
        active_requests: data.active_requests ?? [],
        recent_completions: data.recent_completions ?? [],
        db_summary: data.db_summary,
      };
      setQueueStatus(normalized);
    } catch (error) {
      console.error('Failed to load queue status:', error);
    }
  };

  const loadJobs = async () => {
    try {
      if (jobs.length === 0) {
        setJobsLoading(true);
      }
      const data = await api.getHistoricalJobs();
      setJobs(data.jobs || []);
    } catch (error) {
      console.error('Failed to load historical jobs:', error);
    } finally {
      setJobsLoading(false);
    }
  };

  const prepareRequestParams = () => {
    const requestParams: Record<string, any> = {
      bar_size: barSize
    };

    if (dateRangeMode === 'absolute') {
      if (!endDate) {
        throw new Error('Please enter an end date');
      }
      if (!startDate) {
        throw new Error('Please enter a start date');
      }
      if (new Date(startDate) > new Date(endDate)) {
        throw new Error('Start date must be before end date');
      }

      const endDateTime = new Date(endDate);
      endDateTime.setHours(23, 59, 59);
      const tws_end_datetime = endDateTime.toISOString()
        .replace('T', '-')
        .split('.')[0];
      requestParams.end_datetime = tws_end_datetime;

      const start = new Date(startDate);
      const end = new Date(endDate);
      const diffTime = Math.abs(end.getTime() - start.getTime());
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

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

      return {
        requestParams,
        modeDescription: `from ${startDate} to ${endDate}`
      };
    }

    requestParams.duration = duration;
    return {
      requestParams,
      modeDescription: duration
    };
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

  const handleSymbolFileUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    try {
      const file = event.target.files?.[0];
      if (!file) {
        return;
      }

      const text = await file.text();
      const symbols = text
        .split(/\r?\n/)
        .map((line) => line.trim().toUpperCase())
        .filter((line) => line.length > 0 && !line.startsWith('#'));

      if (symbols.length === 0) {
        setUploadError('No tickers found in the uploaded file.');
        setUploadedSymbols([]);
        setUploadFileName('');
      } else {
        setUploadedSymbols(symbols);
        setUploadFileName(file.name);
        setUploadError(null);
        setRequestMode('uploaded');
      }
    } catch (error: any) {
      console.error('Failed to read uploaded file:', error);
      setUploadError(error.message || 'Failed to read uploaded file.');
      setUploadedSymbols([]);
      setUploadFileName('');
    } finally {
      if (event.target) {
        event.target.value = '';
      }
    }
  };

  const clearUploadedSymbols = () => {
    setUploadedSymbols([]);
    setUploadFileName('');
    setUploadError(null);
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
    if (requestMode === 'single' && !symbol) {
      alert('Please enter a symbol');
      return;
    }

    if (requestMode === 'uploaded' && uploadedSymbols.length === 0) {
      alert('Please upload a .txt file containing ticker symbols before submitting.');
      return;
    }

    let requestConfig;
    try {
      requestConfig = prepareRequestParams();
    } catch (error: any) {
      alert(error.message || 'Invalid date range');
      return;
    }

    setSubmitting(true);
    try {
      const { requestParams, modeDescription } = requestConfig;

      if (requestMode === 'watchlist') {
        const response = await api.bulkHistoricalRequest(requestParams);
        const jobCount = response.jobs?.length || 0;
        alert(`Queued ${response.total_chunks || 0} chunks across ${jobCount} job(s) for all watchlist symbols (${barSize}, ${modeDescription})`);
      } else if (requestMode === 'uploaded') {
        const response = await api.bulkHistoricalUpload({
          symbols: uploadedSymbols,
          ...requestParams
        });
        const totalSymbols = response.symbols?.length || uploadedSymbols.length;
        alert(`Queued ${response.total_chunks || 0} chunks for ${totalSymbols} uploaded symbols (${barSize}, ${modeDescription})`);
        clearUploadedSymbols();
      } else {
        const response = await api.requestHistoricalData({
          symbol: symbol.toUpperCase(),
          ...requestParams
        });
        if (response.job_id) {
          alert(`Job #${response.job_id} queued (${response.chunks || 1} chunk${response.chunks === 1 ? '' : 's'})`);
        } else {
          alert(`Request queued: ${response.message || 'Success'}`);
        }
        setSymbol('');
      }

      setTimeout(() => {
        loadDatasets();
        loadQueueStatus();
        loadJobs();
      }, 2000);
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
          {/* Request Scope Selection */}
          <div className="space-y-2 p-3 bg-slate-50 rounded-lg border border-slate-200">
            <div className="text-sm font-semibold text-slate-700">Request Scope</div>
            <div className="flex flex-wrap gap-2">
              {( [
                { mode: 'single', label: 'Single Symbol' },
                { mode: 'watchlist', label: 'Watchlist Symbols' },
                { mode: 'uploaded', label: 'Uploaded List' },
              ] as const).map(({ mode, label }) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setRequestMode(mode)}
                  className={`px-3 py-1 text-sm font-medium rounded transition-colors border ${
                    requestMode === mode
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-100'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-500">
              {requestMode === 'single' && 'Fetch data for a specific symbol using the controls below.'}
              {requestMode === 'watchlist' && 'Run the request for every symbol currently on your watchlist.'}
              {requestMode === 'uploaded' && 'Use a .txt file with one ticker per line to drive this bulk request.'}
            </p>
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
            {/* Symbol Input - enabled for single mode */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Symbol {requestMode !== 'single' && <span className="text-slate-400">(not required for this mode)</span>}
              </label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="AAPL"
                disabled={requestMode !== 'single'}
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
              disabled={
                submitting ||
                (requestMode === 'single' && symbol.length === 0) ||
                (requestMode === 'uploaded' && uploadedSymbols.length === 0)
              }
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Download className="w-5 h-5" />
              {submitting ? 'Submitting...' : 'Request Data'}
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
              {requestMode === 'uploaded' && (
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-3">
                  <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2">
                      Upload Symbol List (.txt)
                    </label>
                    <input
                      type="file"
                      accept=".txt"
                      onChange={handleSymbolFileUpload}
                      className="w-full text-sm text-slate-600"
                    />
                  </div>
                  {uploadError && (
                    <div className="text-xs text-rose-600 bg-rose-50 border border-rose-200 rounded-lg p-2">
                      {uploadError}
                    </div>
                  )}
                  {uploadedSymbols.length > 0 && (
                    <div className="flex flex-wrap items-center gap-3 text-xs text-slate-600">
                      <span>
                        Loaded <span className="font-semibold">{uploadedSymbols.length.toLocaleString()}</span> tickers from <span className="font-semibold">{uploadFileName || 'uploaded file'}</span>.
                      </span>
                      <button
                        onClick={clearUploadedSymbols}
                        type="button"
                        className="px-2 py-1 text-xs text-slate-600 border border-slate-300 rounded hover:bg-slate-100 transition-colors"
                      >
                        Clear
                      </button>
                    </div>
                  )}
                  <p className="text-xs text-slate-500">
                    Provide a .txt file with one symbol per line. Lines starting with <code>#</code> are ignored.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>

      {/* Queue Overview */}
      <Card title="Historical Queue Overview">
        {!queueStatus ? (
          <div className="text-center py-6 text-slate-600">Loading queue status...</div>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-xs text-slate-600 uppercase tracking-wide">Queue Size</div>
                <div className="text-lg font-bold text-slate-900">{queueStatus.queue_size}</div>
              </div>
              <div>
                <div className="text-xs text-slate-600 uppercase tracking-wide">Pending Chunks</div>
                <div className="text-lg font-bold text-slate-900">
                  {queueStatus.db_summary?.pending_chunks ?? 0}
                </div>
              </div>
              <div>
                <div className="text-xs text-slate-600 uppercase tracking-wide">In Progress</div>
                <div className="text-lg font-bold text-slate-900">
                  {queueStatus.db_summary?.in_progress_chunks ?? 0}
                </div>
              </div>
              <div>
                <div className="text-xs text-slate-600 uppercase tracking-wide">Failed Chunks</div>
                <div className="text-lg font-bold text-rose-600">
                  {queueStatus.db_summary?.failed_chunks ?? 0}
                </div>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h3 className="text-xs font-semibold text-slate-600 uppercase tracking-wide mb-2">
                  Active Requests
                </h3>
                {queueStatus.active_requests.length === 0 ? (
                  <div className="text-sm text-slate-500 bg-slate-50 border border-slate-200 rounded-lg p-3">
                    None currently running.
                  </div>
                ) : (
                  <div className="space-y-2">
                    {queueStatus.active_requests.slice(0, 5).map((req) => (
                      <div
                        key={req.id}
                        className="border border-slate-200 rounded-lg p-3 bg-white shadow-sm"
                      >
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-semibold text-slate-900">
                            {req.symbol} · {req.bar_size}
                          </span>
                          <span className="text-xs uppercase font-semibold text-blue-600">
                            {req.status.replace('_', ' ')}
                          </span>
                        </div>
                        <div className="text-xs text-slate-500 mt-1">
                          {req.started_at
                            ? `Started ${new Date(req.started_at).toLocaleTimeString()}`
                            : 'Pending start'}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div>
                <h3 className="text-xs font-semibold text-slate-600 uppercase tracking-wide mb-2">
                  Recent Completions
                </h3>
                {queueStatus.recent_completions.length === 0 ? (
                  <div className="text-sm text-slate-500 bg-slate-50 border border-slate-200 rounded-lg p-3">
                    No recent activity.
                  </div>
                ) : (
                  <div className="space-y-2">
                    {queueStatus.recent_completions.slice().reverse().slice(0, 5).map((req) => (
                      <div
                        key={req.id}
                        className="border border-slate-200 rounded-lg p-3 bg-white shadow-sm"
                      >
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-semibold text-slate-900">
                            {req.symbol} · {req.bar_size}
                          </span>
                          <span
                            className={`text-xs uppercase font-semibold ${
                              req.status === 'failed' ? 'text-rose-600' :
                              req.status === 'skipped' ? 'text-amber-600' : 'text-emerald-600'
                            }`}
                          >
                            {req.status.replace('_', ' ')}
                          </span>
                        </div>
                        <div className="text-xs text-slate-500 mt-1">
                          {req.completed_at
                            ? new Date(req.completed_at).toLocaleTimeString()
                            : 'Time unavailable'}
                          {' · '}
                          {req.bars_count.toLocaleString()} bars
                          {req.error ? ` · ${req.error}` : ''}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* Historical Jobs */}
      <Card title={`Historical Jobs (${jobs.length})`}>
        {jobsLoading ? (
          <div className="text-center py-6 text-slate-600">Loading jobs...</div>
        ) : jobs.length === 0 ? (
          <div className="text-center py-6 text-slate-600">
            No jobs yet. Submit a request to start collecting data.
          </div>
        ) : (
          <div className="overflow-x-auto max-h-96 overflow-y-auto border border-gray-200 rounded-lg">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50 sticky top-0 z-10">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-600 uppercase tracking-wide">
                    Job
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-600 uppercase tracking-wide">
                    Symbol
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-600 uppercase tracking-wide">
                    Timeframe
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-600 uppercase tracking-wide">
                    Progress
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-600 uppercase tracking-wide">
                    Status
                  </th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-600 uppercase tracking-wide">
                    Updated
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-100">
                {jobs.map((job) => {
                  const progress = job.total_chunks > 0
                    ? Math.round((job.completed_chunks / job.total_chunks) * 100)
                    : 0;
                  return (
                    <tr key={job.id} className="hover:bg-gray-50 transition-colors">
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="text-sm font-semibold text-slate-900">#{job.id}</div>
                        <div className="text-xs text-slate-500">{job.job_key}</div>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-700">
                        {job.symbol}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-700">
                        {job.bar_size}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="text-sm text-slate-700">
                          {progress}% ({job.completed_chunks}/{job.total_chunks})
                        </div>
                        {job.failed_chunks > 0 && (
                          <div className="text-xs text-rose-600">
                            {job.failed_chunks} failed chunk{job.failed_chunks === 1 ? '' : 's'}
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span
                          className={`px-2 py-1 text-xs font-semibold rounded-full ${
                            job.status === 'completed'
                              ? 'bg-emerald-50 text-emerald-700'
                              : job.status === 'failed'
                              ? 'bg-rose-50 text-rose-700'
                              : job.status === 'pending'
                              ? 'bg-slate-50 text-slate-700'
                              : 'bg-blue-50 text-blue-700'
                          }`}
                        >
                          {job.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-500">
                        {job.updated_at ? new Date(job.updated_at).toLocaleString() : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
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
