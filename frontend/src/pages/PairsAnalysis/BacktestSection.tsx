import React, { useState } from 'react';
import { Card } from '../../components/Card';
import { api, Strategy } from '../../services/api';
import { Play, RefreshCw, AlertTriangle, CheckCircle } from 'lucide-react';
import clsx from 'clsx';
import { BacktestVisualization } from '../../components/BacktestVisualization';

interface BacktestSectionProps {
    symbolA: string;
    symbolB: string;
    timeframe: string;
    startDate?: string;
    endDate?: string;
    preSelectedStrategy?: Strategy | null;
    preSetParams?: string;
    analysisData?: any; // Pass through the analysis results for visualization
}

export const BacktestSection: React.FC<BacktestSectionProps> = ({
    symbolA,
    symbolB,
    timeframe,
    startDate,
    endDate,
    preSelectedStrategy = null,
    preSetParams = '{}',
    analysisData = null
}) => {
    const [isRunning, setIsRunning] = useState(false);
    const [result, setResult] = useState<any | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [logs, setLogs] = useState<string[]>([]);

    const handleRunBacktest = async () => {
        // If no strategy is selected, show error
        if (!preSelectedStrategy) {
            setError('Please select a strategy in the Analysis Configuration above');
            return;
        }

        setIsRunning(true);
        setResult(null);
        setError(null);
        setLogs(['Starting backtest...', `Strategy: ${preSelectedStrategy.name}`, `Pair: ${symbolA}/${symbolB}`]);

        try {
            let parsedParams: any = {};
            try {
                parsedParams = JSON.parse(preSetParams);
                // Override pairs/symbols for this analysis
                if (parsedParams.pairs) {
                    parsedParams.pairs = [[symbolA, symbolB]];
                }
                if (parsedParams.symbols) {
                    parsedParams.symbols = [symbolA, symbolB];
                }
            } catch (e) {
                setError('Invalid JSON parameters in configuration above');
                setIsRunning(false);
                return;
            }

            const response = await api.runBacktest({
                strategy_name: preSelectedStrategy.name,
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
            {preSelectedStrategy && (
                <div className="p-4 bg-gradient-to-r from-blue-50 to-cyan-50 border border-blue-200 rounded-lg flex items-start gap-3">
                    <CheckCircle className="w-6 h-6 text-blue-600 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                        <p className="text-sm font-semibold text-blue-900">
                            Ready to Run Backtest
                        </p>
                        <p className="text-xs text-blue-700 mt-1">
                            Using <strong>{preSelectedStrategy.name}</strong> with the parameters you configured above.
                            The backtest will use the same pair, timeframe, and date range from your analysis.
                        </p>
                    </div>
                </div>
            )}

            {!preSelectedStrategy && (
                <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg flex items-start gap-3">
                    <AlertTriangle className="w-6 h-6 text-amber-600 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                        <p className="text-sm font-semibold text-amber-900">
                            No Strategy Selected
                        </p>
                        <p className="text-xs text-amber-700 mt-1">
                            Please select a strategy in the Analysis Configuration section above to run a backtest.
                        </p>
                    </div>
                </div>
            )}

            <Card title="Run Backtest">
                <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg border border-slate-200">
                        <div className="flex-1">
                            <h4 className="text-sm font-medium text-slate-900">Configuration Summary</h4>
                            <div className="mt-2 space-y-1 text-xs text-slate-600">
                                <div className="flex items-center gap-2">
                                    <span className="font-medium">Pair:</span>
                                    <span className="font-mono">{symbolA} / {symbolB}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="font-medium">Timeframe:</span>
                                    <span>{timeframe}</span>
                                </div>
                                {preSelectedStrategy && (
                                    <div className="flex items-center gap-2">
                                        <span className="font-medium">Strategy:</span>
                                        <span className="text-blue-700 font-semibold">{preSelectedStrategy.name}</span>
                                    </div>
                                )}
                                {startDate && (
                                    <div className="flex items-center gap-2">
                                        <span className="font-medium">Period:</span>
                                        <span>{new Date(startDate).toLocaleDateString()} - {endDate ? new Date(endDate).toLocaleDateString() : 'now'}</span>
                                    </div>
                                )}
                            </div>
                        </div>
                        <button
                            onClick={handleRunBacktest}
                            disabled={isRunning || !preSelectedStrategy}
                            className={clsx(
                                "py-3 px-6 rounded-lg font-semibold text-white transition-all flex items-center gap-2 shadow-md",
                                isRunning || !preSelectedStrategy
                                    ? "bg-gray-400 cursor-not-allowed"
                                    : "bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 hover:shadow-lg active:transform active:scale-95"
                            )}
                        >
                            {isRunning ? (
                                <>
                                    <RefreshCw className="w-5 h-5 animate-spin" />
                                    Running Backtest...
                                </>
                            ) : (
                                <>
                                    <Play className="w-5 h-5" />
                                    Run Backtest
                                </>
                            )}
                        </button>
                    </div>

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
                        <>
                            <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg flex items-center gap-2">
                                <CheckCircle className="w-5 h-5 text-emerald-600" />
                                <div className="flex-1">
                                    <p className="text-sm font-medium text-emerald-900">Backtest Complete!</p>
                                    <p className="text-xs text-emerald-700 mt-0.5">
                                        These results reflect your configured parameters. Modify parameters above and re-run to test different configurations.
                                    </p>
                                </div>
                            </div>

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
                        </>
                    )}
                </div>
            </Card>

            {/* Detailed Visualization */}
            {result && result.run_id && (
                <BacktestVisualizationWrapper
                    runId={result.run_id}
                    analysisData={analysisData}
                />
            )}
        </div>
    );
};

const BacktestVisualizationWrapper: React.FC<{ runId: number, analysisData: any }> = ({ runId, analysisData }) => {
    const [trades, setTrades] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    React.useEffect(() => {
        const loadTrades = async () => {
            try {
                setLoading(true);
                const data = await api.getBacktestTrades(runId);

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
            } catch (err) {
                console.error('Failed to load trades', err);
            } finally {
                setLoading(false);
            }
        };

        loadTrades();
    }, [runId]);

    if (loading) return <div className="text-center py-8 text-gray-500">Loading trade data...</div>;

    return (
        <BacktestVisualization
            trades={trades}
            analysisData={analysisData}
        />
    );
};

const StatBox = ({ label, value, color = 'text-gray-900', className }: any) => (
    <div className={clsx("p-3 bg-gray-50 rounded-lg border border-gray-100", className)}>
        <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">{label}</div>
        <div className={clsx("text-lg font-bold", color)}>{value}</div>
    </div>
);
