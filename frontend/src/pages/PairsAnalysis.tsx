import { useState, useEffect, useMemo } from 'react';
import { Card } from '../components/Card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Brush, ScatterChart, Scatter, ReferenceLine } from 'recharts';
import { Activity, RefreshCw, AlertCircle, TrendingUp, BarChart2, Clock, Calendar, ZoomIn } from 'lucide-react';
import { api } from '../services/api';
import { BacktestSection } from '../components/BacktestSection';

interface AnalysisResult {
    symbol_a: string;
    symbol_b: string;
    timeframe: string;
    coint_pvalue: number;
    half_life_minutes: number;
    hedge_ratio: number;
    pair_sharpe: number;
    spread_mean: number;
    spread_std: number;
    price_data: {
        timestamps: string[];
        symbol_a: { close: number[] };
        symbol_b: { close: number[] };
    };
    spread_series?: number[];
    zscore_series?: number[];
}

interface Availability {
    available: boolean;
    start_date: string | null;
    end_date: string | null;
    count: number;
}

export function PairsAnalysis() {
    const [symbolA, setSymbolA] = useState('');
    const [symbolB, setSymbolB] = useState('');
    const [timeframe, setTimeframe] = useState('5 mins');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const [availA, setAvailA] = useState<Availability | null>(null);
    const [availB, setAvailB] = useState<Availability | null>(null);

    const [priceChartMode, setPriceChartMode] = useState<'raw' | 'pct'>('raw');

    // Fetch availability when symbols or timeframe change
    useEffect(() => {
        const fetchAvailability = async (sym: string, setAvail: (a: Availability | null) => void) => {
            if (!sym) {
                setAvail(null);
                return;
            }
            try {
                const data = await api.getResearchAvailability(sym, timeframe);
                setAvail(data);
            } catch (e) {
                console.error(`Failed to fetch availability for ${sym}`, e);
                setAvail(null);
            }
        };

        const timeoutA = setTimeout(() => fetchAvailability(symbolA, setAvailA), 500);
        const timeoutB = setTimeout(() => fetchAvailability(symbolB, setAvailB), 500);

        return () => {
            clearTimeout(timeoutA);
            clearTimeout(timeoutB);
        };
    }, [symbolA, symbolB, timeframe]);

    const handleAnalyze = async () => {
        setLoading(true);
        setError(null);
        try {
            const payload: any = { symbol_a: symbolA, symbol_b: symbolB, timeframe };
            if (startDate) payload.start_date = new Date(startDate).toISOString();
            if (endDate) payload.end_date = new Date(endDate).toISOString();

            const data = await api.analyzePair(payload);
            setResult(data);
        } catch (err: any) {
            setError(err.response?.data?.detail || err.message || 'Analysis failed');
        } finally {
            setLoading(false);
        }
    };

    // Prepare chart data with memoization and downsampling
    const chartData = useMemo(() => {
        if (!result?.price_data?.timestamps) return [];

        const rawData = result.price_data.timestamps.map((ts, i) => {
            const priceA = result.price_data.symbol_a.close[i];
            const priceB = result.price_data.symbol_b.close[i];

            // Calculate % change from start
            const startA = result.price_data.symbol_a.close[0];
            const startB = result.price_data.symbol_b.close[0];
            const pctA = startA ? ((priceA - startA) / startA) * 100 : 0;
            const pctB = startB ? ((priceB - startB) / startB) * 100 : 0;

            return {
                timestamp: new Date(ts).toLocaleString(),
                raw_ts: new Date(ts).getTime(), // For scatter plot
                [result.symbol_a]: priceA,
                [result.symbol_b]: priceB,
                [`${result.symbol_a} %`]: pctA,
                [`${result.symbol_b} %`]: pctB,
                zscore: result.zscore_series ? result.zscore_series[i] : null,
                spread: result.spread_series ? result.spread_series[i] : null,
            };
        });

        // Downsample if too many points to prevent UI freezing
        // Recharts struggles with > 2-3k points, especially with multiple charts
        const MAX_POINTS = 1000;
        if (rawData.length <= MAX_POINTS) return rawData;

        const step = Math.ceil(rawData.length / MAX_POINTS);
        return rawData.filter((_, i) => i % step === 0);
    }, [result]);

    // Prepare scatter data (Price A vs Price B)
    const scatterData = useMemo(() => {
        if (!chartData.length) return [];

        // Scatter plots are heavy, limit points even further if needed
        // But since chartData is already downsampled, we can just map it
        return chartData.map(d => ({
            x: d[result?.symbol_a || ''],
            y: d[result?.symbol_b || ''],
            z: d.zscore // Optional: color by z-score
        }));
    }, [chartData, result?.symbol_a, result?.symbol_b]);

    const currentZScore = result?.zscore_series ? result.zscore_series[result.zscore_series.length - 1] : null;

    const AvailabilityBadge = ({ avail, onApply }: { avail: Availability | null, onApply: () => void }) => {
        if (!avail) return null;
        if (!avail.available) return <span className="text-xs text-red-500 mt-1 block">No data available</span>;
        return (
            <div className="flex items-center justify-between mt-1">
                <div className="text-xs text-slate-500 flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    <span>
                        {new Date(avail.start_date!).toLocaleDateString()} - {new Date(avail.end_date!).toLocaleDateString()}
                    </span>
                </div>
                <button
                    onClick={onApply}
                    className="text-xs text-blue-600 hover:text-blue-800 font-medium px-2 py-0.5 rounded hover:bg-blue-50 transition-colors"
                    title="Use this date range"
                >
                    Use Range
                </button>
            </div>
        );
    };

    const applyRange = (avail: Availability | null) => {
        if (!avail || !avail.start_date || !avail.end_date) return;
        const format = (iso: string) => {
            const d = new Date(iso);
            return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
        };
        setStartDate(format(avail.start_date));
        setEndDate(format(avail.end_date));
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-2xl font-bold text-slate-900">Pairs Analysis</h1>
            </div>

            <Card title="Analysis Configuration">
                <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Ticker A</label>
                            <input
                                type="text"
                                value={symbolA}
                                onChange={(e) => setSymbolA(e.target.value.toUpperCase())}
                                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono"
                                placeholder="e.g. AAPL"
                            />
                            <AvailabilityBadge avail={availA} onApply={() => applyRange(availA)} />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Ticker B</label>
                            <input
                                type="text"
                                value={symbolB}
                                onChange={(e) => setSymbolB(e.target.value.toUpperCase())}
                                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono"
                                placeholder="e.g. MSFT"
                            />
                            <AvailabilityBadge avail={availB} onApply={() => applyRange(availB)} />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Timeframe</label>
                            <select
                                value={timeframe}
                                onChange={(e) => setTimeframe(e.target.value)}
                                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            >
                                <option value="5 secs">5 secs</option>
                                <option value="1 min">1 min</option>
                                <option value="5 mins">5 mins</option>
                                <option value="1 hour">1 hour</option>
                                <option value="1 day">1 day</option>
                            </select>
                        </div>
                    </div>

                    <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                        <div className="flex flex-col md:flex-row gap-4 items-end">
                            <div className="flex-1 w-full">
                                <label className="block text-sm font-medium text-slate-700 mb-1">Analysis Period</label>
                                <div className="flex items-center gap-2">
                                    <div className="relative flex-1">
                                        <input
                                            type="datetime-local"
                                            value={startDate}
                                            onChange={(e) => setStartDate(e.target.value)}
                                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                                        />
                                    </div>
                                    <span className="text-slate-400 font-medium">to</span>
                                    <div className="relative flex-1">
                                        <input
                                            type="datetime-local"
                                            value={endDate}
                                            onChange={(e) => setEndDate(e.target.value)}
                                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                                        />
                                    </div>
                                </div>
                            </div>
                            <button
                                onClick={handleAnalyze}
                                disabled={loading || !symbolA || !symbolB}
                                className={`flex items-center justify-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors min-w-[140px] h-[42px] ${loading || !symbolA || !symbolB ? 'opacity-50 cursor-not-allowed' : ''
                                    }`}
                            >
                                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
                                {loading ? 'Analyzing...' : 'Run Analysis'}
                            </button>
                        </div>
                        <p className="text-xs text-slate-500 mt-2">
                            Leave dates empty to use the default lookback period (smart auto-detection).
                        </p>
                    </div>
                </div>
            </Card>

            {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-2 text-red-700">
                    <AlertCircle className="w-5 h-5" />
                    <span>{error}</span>
                </div>
            )}

            {result && (
                <div className="space-y-6">
                    {/* Statistics Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                        <Card>
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-500 uppercase tracking-wide">Cointegration</p>
                                    <p className={`text-xl font-bold ${result.coint_pvalue < 0.05 ? 'text-emerald-600' : 'text-rose-600'}`}>
                                        {result.coint_pvalue?.toFixed(4) ?? 'N/A'}
                                    </p>
                                    <p className="text-xs text-slate-400">p-value</p>
                                </div>
                                <Activity className={`w-5 h-5 ${result.coint_pvalue < 0.05 ? 'text-emerald-500' : 'text-rose-500'}`} />
                            </div>
                        </Card>
                        <Card>
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-500 uppercase tracking-wide">Half-Life</p>
                                    <p className="text-xl font-bold text-slate-900">
                                        {result.half_life_minutes?.toFixed(1) ?? 'N/A'}
                                    </p>
                                    <p className="text-xs text-slate-400">minutes</p>
                                </div>
                                <Clock className="w-5 h-5 text-blue-500" />
                            </div>
                        </Card>
                        <Card>
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-500 uppercase tracking-wide">Hedge Ratio</p>
                                    <p className="text-xl font-bold text-slate-900">
                                        {result.hedge_ratio?.toFixed(4) ?? 'N/A'}
                                    </p>
                                    <p className="text-xs text-slate-400">beta</p>
                                </div>
                                <BarChart2 className="w-5 h-5 text-purple-500" />
                            </div>
                        </Card>
                        <Card>
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-500 uppercase tracking-wide">Sharpe Ratio</p>
                                    <p className={`text-xl font-bold ${result.pair_sharpe > 1 ? 'text-emerald-600' : 'text-slate-900'}`}>
                                        {result.pair_sharpe?.toFixed(2) ?? 'N/A'}
                                    </p>
                                    <p className="text-xs text-slate-400">annualized</p>
                                </div>
                                <TrendingUp className="w-5 h-5 text-slate-400" />
                            </div>
                        </Card>
                        <Card>
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-500 uppercase tracking-wide">Current Z-Score</p>
                                    <p className={`text-xl font-bold ${Math.abs(currentZScore || 0) > 2 ? 'text-amber-600' : 'text-slate-900'}`}>
                                        {currentZScore?.toFixed(2) ?? 'N/A'}
                                    </p>
                                    <p className="text-xs text-slate-400">std dev</p>
                                </div>
                                <ZoomIn className="w-5 h-5 text-amber-500" />
                            </div>
                        </Card>
                    </div>

                    {/* Charts Grid */}
                    <div className="space-y-6">
                        {/* Row 1: Spread Z-Score (Most Important) */}
                        <Card title="Spread Z-Score (Mean Reversion Signal)">
                            <div className="h-[350px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={chartData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                        <XAxis
                                            dataKey="timestamp"
                                            stroke="#64748b"
                                            tick={{ fill: '#64748b', fontSize: 11 }}
                                            tickFormatter={(val) => val.split(',')[0]}
                                            minTickGap={50}
                                        />
                                        <YAxis
                                            stroke="#64748b"
                                            tick={{ fill: '#64748b', fontSize: 11 }}
                                            domain={['auto', 'auto']}
                                        />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#fff', borderColor: '#e2e8f0', color: '#1e293b' }}
                                        />
                                        <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
                                        <ReferenceLine y={2} stroke="#ef4444" strokeDasharray="3 3" label={{ value: '+2σ', fill: '#ef4444', fontSize: 10 }} />
                                        <ReferenceLine y={-2} stroke="#10b981" strokeDasharray="3 3" label={{ value: '-2σ', fill: '#10b981', fontSize: 10 }} />
                                        <Line
                                            type="monotone"
                                            dataKey="zscore"
                                            stroke="#3b82f6"
                                            dot={false}
                                            strokeWidth={2}
                                            name="Z-Score"
                                        />
                                        <Brush dataKey="timestamp" height={30} stroke="#cbd5e1" />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </Card>

                        {/* Row 2: Price History & Performance */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {/* Price Chart (Dual Axis or Performance) */}
                            <Card
                                title={
                                    <div className="flex items-center justify-between w-full">
                                        <span>{priceChartMode === 'raw' ? 'Price History' : 'Performance Comparison (%)'}</span>
                                        <div className="flex bg-slate-100 rounded-lg p-1">
                                            <button
                                                onClick={() => setPriceChartMode('raw')}
                                                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${priceChartMode === 'raw' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                                                    }`}
                                            >
                                                Raw Price
                                            </button>
                                            <button
                                                onClick={() => setPriceChartMode('pct')}
                                                className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${priceChartMode === 'pct' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                                                    }`}
                                            >
                                                % Change
                                            </button>
                                        </div>
                                    </div>
                                }
                            >
                                <div className="h-[400px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={chartData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                            <XAxis
                                                dataKey="timestamp"
                                                stroke="#64748b"
                                                tick={{ fill: '#64748b', fontSize: 11 }}
                                                tickFormatter={(val) => val.split(',')[0]}
                                                minTickGap={50}
                                            />
                                            {priceChartMode === 'raw' ? (
                                                <>
                                                    <YAxis
                                                        yAxisId="left"
                                                        stroke="#2563eb"
                                                        tick={{ fill: '#2563eb', fontSize: 11 }}
                                                        domain={['auto', 'auto']}
                                                    />
                                                    <YAxis
                                                        yAxisId="right"
                                                        orientation="right"
                                                        stroke="#16a34a"
                                                        tick={{ fill: '#16a34a', fontSize: 11 }}
                                                        domain={['auto', 'auto']}
                                                    />
                                                    <Line
                                                        yAxisId="left"
                                                        type="monotone"
                                                        dataKey={result.symbol_a}
                                                        stroke="#2563eb"
                                                        dot={false}
                                                        strokeWidth={2}
                                                    />
                                                    <Line
                                                        yAxisId="right"
                                                        type="monotone"
                                                        dataKey={result.symbol_b}
                                                        stroke="#16a34a"
                                                        dot={false}
                                                        strokeWidth={2}
                                                    />
                                                </>
                                            ) : (
                                                <>
                                                    <YAxis
                                                        stroke="#64748b"
                                                        tick={{ fill: '#64748b', fontSize: 11 }}
                                                        unit="%"
                                                    />
                                                    <Line
                                                        type="monotone"
                                                        dataKey={`${result.symbol_a} %`}
                                                        name={result.symbol_a}
                                                        stroke="#2563eb"
                                                        dot={false}
                                                        strokeWidth={2}
                                                    />
                                                    <Line
                                                        type="monotone"
                                                        dataKey={`${result.symbol_b} %`}
                                                        name={result.symbol_b}
                                                        stroke="#16a34a"
                                                        dot={false}
                                                        strokeWidth={2}
                                                    />
                                                </>
                                            )}
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#fff', borderColor: '#e2e8f0', color: '#1e293b' }}
                                                formatter={(value: number) => [
                                                    priceChartMode === 'pct' ? `${value.toFixed(2)}%` : value.toFixed(2),
                                                    priceChartMode === 'pct' ? 'Change' : 'Price'
                                                ]}
                                            />
                                            <Legend />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </Card>

                            {/* Scatter Plot (Correlation) */}
                            <Card
                                title={
                                    <div className="flex items-center gap-2">
                                        <span>Correlation: {result.symbol_a} vs {result.symbol_b}</span>
                                        <div className="group relative flex items-center">
                                            <AlertCircle className="w-4 h-4 text-slate-400 cursor-help" />
                                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-2 bg-slate-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                                                <p className="font-semibold mb-1">How to interpret:</p>
                                                <ul className="list-disc pl-3 space-y-1">
                                                    <li>Tight diagonal line = High correlation (Good for pairs trading)</li>
                                                    <li>Scattered cloud = Low correlation</li>
                                                    <li>Curved line = Non-linear relationship</li>
                                                </ul>
                                                <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800"></div>
                                            </div>
                                        </div>
                                    </div>
                                }
                            >
                                <div className="h-[400px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis
                                                type="number"
                                                dataKey="x"
                                                name={result.symbol_a}
                                                domain={['auto', 'auto']}
                                                tick={{ fontSize: 11 }}
                                                label={{ value: result.symbol_a, position: 'bottom', offset: 0 }}
                                            />
                                            <YAxis
                                                type="number"
                                                dataKey="y"
                                                name={result.symbol_b}
                                                domain={['auto', 'auto']}
                                                tick={{ fontSize: 11 }}
                                                label={{ value: result.symbol_b, angle: -90, position: 'left' }}
                                            />
                                            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                            <Scatter name="Price Correlation" data={scatterData} fill="#8884d8" opacity={0.6} />
                                        </ScatterChart>
                                    </ResponsiveContainer>
                                </div>
                            </Card>
                        </div>
                    </div>
                </div>
            )}

            {/* Backtest Section */}
            {result && (
                <div className="mt-8">
                    <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                        <RefreshCw className="w-6 h-6 text-blue-600" />
                        Strategy Backtest
                    </h2>
                    <BacktestSection
                        symbolA={result.symbol_a}
                        symbolB={result.symbol_b}
                        timeframe={timeframe}
                        startDate={startDate || undefined}
                        endDate={endDate || undefined}
                    />
                </div>
            )}
        </div>
    );
}
