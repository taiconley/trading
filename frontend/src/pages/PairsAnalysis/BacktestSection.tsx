import React, { useState, useEffect } from 'react';
import { Card } from '../../components/Card';
import { api, Strategy } from '../../services/api';
import { Play, RefreshCw, AlertTriangle, CheckCircle, Terminal } from 'lucide-react';
import clsx from 'clsx';

interface BacktestSectionProps {
    symbolA: string;
    symbolB: string;
    timeframe: string;
    startDate?: string;
    endDate?: string;
}

export const BacktestSection: React.FC<BacktestSectionProps> = ({ symbolA, symbolB, timeframe, startDate, endDate }) => {
    const [strategies, setStrategies] = useState<Strategy[]>([]);
    const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
    const [params, setParams] = useState<string>('{}');
    const [isRunning, setIsRunning] = useState(false);
    const [result, setResult] = useState<any | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [logs, setLogs] = useState<string[]>([]);

    useEffect(() => {
        loadStrategies();
    }, []);

    const loadStrategies = async () => {
        try {
            const data = await api.getStrategies();
            setStrategies(data.strategies);
            // Default to Kalman if available
            const kalman = data.strategies.find((s: Strategy) => s.name.includes('Kalman'));
            if (kalman) {
                selectStrategy(kalman);
            }
        } catch (err) {
            console.error('Failed to load strategies', err);
        }
    };

    const selectStrategy = (strategy: Strategy) => {
        setSelectedStrategy(strategy);
        // Pre-fill params with current strategy params, but override pairs/symbols for this analysis
        const currentParams = { ...strategy.params };

        // Ensure we target the current pair
        // The structure depends on the strategy, but usually it's 'pairs' or 'symbols'
        // For Kalman/Pairs strategies, it's often 'pairs': [['A', 'B']]
        if (currentParams.pairs) {
            currentParams.pairs = [[symbolA, symbolB]];
        }
        if (currentParams.symbols) {
            // If it uses a flat list of symbols
            currentParams.symbols = [symbolA, symbolB];
        }

        setParams(JSON.stringify(currentParams, null, 2));
    };

    const handleRunBacktest = async () => {
        if (!selectedStrategy) return;

        setIsRunning(true);
        setResult(null);
        setError(null);
        setLogs(['Starting backtest...', `Strategy: ${selectedStrategy.name}`, `Pair: ${symbolA}/${symbolB}`]);

        try {
            let parsedParams = {};
            try {
                parsedParams = JSON.parse(params);
            } catch (e) {
                setError('Invalid JSON parameters');
                setIsRunning(false);
                return;
            }

            const response = await api.runBacktest({
                strategy_name: selectedStrategy.name,
                strategy_params: parsedParams,
                symbols: [symbolA, symbolB],
                timeframe: timeframe,
                start_date: startDate,
                end_date: endDate,
                initial_capital: 100000,
                lookback_periods: 20000 // Ensure enough data
            });

            setLogs(prev => [...prev, `Job started: ${response.job_id}`, 'Waiting for results...']);

            // Poll for status
            const pollInterval = setInterval(async () => {
                try {
                    const job = await api.getBacktestJob(response.job_id);

                    if (job.status === 'completed') {
                        clearInterval(pollInterval);
                        setIsRunning(false);
                        setResult(job.result);
                        setLogs(prev => [...prev, 'Backtest completed successfully!']);
                    } else if (job.status === 'failed') {
                        clearInterval(pollInterval);
                        setIsRunning(false);
                        setError(job.error || 'Backtest failed');
                        setLogs(prev => [...prev, `Error: ${job.error}`]);
                    }
                } catch (err) {
                    console.error('Polling error', err);
                }
            }, 1000);

        } catch (err: any) {
            setError(err.message || 'Failed to start backtest');
            setIsRunning(false);
        }
    };

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column: Configuration */}
                <div className="lg:col-span-1 space-y-6">
                    <Card title="Backtest Configuration" icon={<Terminal className="w-5 h-5 text-blue-500" />}>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Strategy</label>
                                <select
                                    className="w-full rounded-lg border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                                    value={selectedStrategy?.id || ''}
                                    onChange={(e) => {
                                        const s = strategies.find(s => s.id === e.target.value);
                                        if (s) selectStrategy(s);
                                    }}
                                >
                                    <option value="">Select Strategy...</option>
                                    {strategies.map(s => (
                                        <option key={s.id} value={s.id}>{s.name}</option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Parameters (JSON)</label>
                                <textarea
                                    className="w-full h-64 font-mono text-xs rounded-lg border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all"
                                    value={params}
                                    onChange={(e) => setParams(e.target.value)}
                                />
                            </div>

                            <button
                                onClick={handleRunBacktest}
                                disabled={isRunning || !selectedStrategy}
                                className={clsx(
                                    "w-full py-2 px-4 rounded-lg font-medium text-white transition-all flex items-center justify-center gap-2",
                                    isRunning || !selectedStrategy
                                        ? "bg-gray-400 cursor-not-allowed"
                                        : "bg-blue-600 hover:bg-blue-700 shadow-md hover:shadow-lg active:transform active:scale-95"
                                )}
                            >
                                {isRunning ? (
                                    <>
                                        <RefreshCw className="w-4 h-4 animate-spin" />
                                        Running...
                                    </>
                                ) : (
                                    <>
                                        <Play className="w-4 h-4" />
                                        Run Backtest
                                    </>
                                )}
                            </button>
                        </div>
                    </Card>
                </div>

                {/* Right Column: Results */}
                <div className="lg:col-span-2 space-y-6">
                    <Card title="Backtest Results" icon={<CheckCircle className="w-5 h-5 text-green-500" />}>
                        <div className="space-y-4">
                            {/* Logs / Status */}
                            <div className="bg-gray-900 text-gray-100 p-4 rounded-lg font-mono text-xs h-32 overflow-y-auto">
                                {logs.map((log, i) => (
                                    <div key={i}>{log}</div>
                                ))}
                                {logs.length === 0 && <div className="text-gray-500 italic">Ready to run backtest...</div>}
                            </div>

                            {error && (
                                <div className="p-4 bg-red-50 text-red-700 rounded-lg flex items-center gap-2">
                                    <AlertTriangle className="w-5 h-5" />
                                    {error}
                                </div>
                            )}

                            {result && (
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <StatBox label="Total P&L" value={`$${result.metrics.total_pnl?.toFixed(2)}`}
                                        color={result.metrics.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'} />
                                    <StatBox label="Sharpe Ratio" value={result.metrics.sharpe_ratio?.toFixed(2) || 'N/A'} />
                                    <StatBox label="Win Rate" value={`${result.metrics.win_rate?.toFixed(1)}%`} />
                                    <StatBox label="Max Drawdown" value={`${result.metrics.max_drawdown_pct?.toFixed(1)}%`} color="text-red-600" />
                                    <StatBox label="Total Trades" value={result.metrics.total_trades} />
                                    <StatBox label="Profit Factor" value={result.metrics.profit_factor?.toFixed(2) || 'N/A'} />
                                    <StatBox label="Avg Win" value={`$${result.metrics.avg_win?.toFixed(2)}`} className="text-green-600" />
                                    <StatBox label="Avg Loss" value={`$${result.metrics.avg_loss?.toFixed(2)}`} className="text-red-600" />
                                </div>
                            )}
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
};

const StatBox = ({ label, value, color = 'text-gray-900', className }: any) => (
    <div className={clsx("p-3 bg-gray-50 rounded-lg border border-gray-100", className)}>
        <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">{label}</div>
        <div className={clsx("text-lg font-bold", color)}>{value}</div>
    </div>
);
