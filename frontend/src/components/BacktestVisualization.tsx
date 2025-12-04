import React, { useState } from 'react';
import { Card } from './Card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { BarChart2, Table as TableIcon, AlertTriangle } from 'lucide-react';
import clsx from 'clsx';

interface BacktestVisualizationProps {
    trades: Trade[];
    analysisData?: any; // The original analysis result with spread/zscore data
}

export interface Trade {
    id: number;
    symbol: string;
    side: string;
    quantity: number;
    entry_time: string;
    entry_price: number;
    exit_time: string | null;
    exit_price: number | null;
    pnl: number | null;
    runningPnL?: number;
}

export const BacktestVisualization: React.FC<BacktestVisualizationProps> = ({
    trades,
    analysisData
}) => {
    const [activeTab, setActiveTab] = useState<'charts' | 'trades'>('charts');

    // Calculate equity curve from trades
    const equityCurve = React.useMemo(() => {
        if (!trades.length) return [];

        const points: Array<{ timestamp: string; equity: number; tradeType?: string }> = [];
        let runningEquity = 100000; // Initial capital

        // Add initial point
        points.push({
            timestamp: 'Start',
            equity: runningEquity
        });

        // Add points for each trade with valid P&L (don't require exit_time)
        trades.forEach((trade, idx) => {
            if (trade.pnl != null && !isNaN(trade.pnl)) {
                runningEquity += trade.pnl;
                points.push({
                    timestamp: `Trade ${idx + 1}`,
                    equity: runningEquity,
                    tradeType: trade.pnl > 0 ? 'win' : 'loss'
                });
            }
        });


        return points;
    }, [trades]);

    // Create trade markers for spread/z-score chart
    const tradeMarkers = React.useMemo(() => {
        if (!trades.length || !analysisData?.zscore_series) {
            return { entries: [], exits: [] };
        }

        const entries: Array<{ timestamp: string; zscore: number; type: string; trade: any }> = [];
        const exits: Array<{ timestamp: string; zscore: number; type: string; pnl: number }> = [];

        const timestamps = analysisData.price_data.timestamps;
        const zscores = analysisData.zscore_series;


        trades.forEach((trade) => {
            // Find closest timestamp for entry
            const entryTime = new Date(trade.entry_time).getTime();
            let closestEntryIdx = -1;
            let minEntryDiff = Infinity;

            timestamps.forEach((ts: string, i: number) => {
                const diff = Math.abs(new Date(ts).getTime() - entryTime);
                if (diff < minEntryDiff) {
                    minEntryDiff = diff;
                    closestEntryIdx = i;
                }
            });

            if (closestEntryIdx >= 0) {
                let zscore = zscores[closestEntryIdx];

                // If zscore is missing (e.g. during warmup), search nearby
                if (zscore === undefined || zscore === null) {
                    // Search up to 10 bars forward and backward
                    for (let offset = 1; offset <= 10; offset++) {
                        if (zscores[closestEntryIdx + offset] != null) {
                            zscore = zscores[closestEntryIdx + offset];
                            break;
                        }
                        if (zscores[closestEntryIdx - offset] != null) {
                            zscore = zscores[closestEntryIdx - offset];
                            break;
                        }
                    }
                }

                if (zscore != null) {
                    entries.push({
                        timestamp: timestamps[closestEntryIdx],
                        zscore: zscore,
                        type: trade.side,
                        trade: trade
                    });
                }
            }

            // Find closest timestamp for exit
            if (trade.exit_time) {
                const exitTime = new Date(trade.exit_time).getTime();
                let closestExitIdx = -1;
                let minExitDiff = Infinity;

                timestamps.forEach((ts: string, i: number) => {
                    const diff = Math.abs(new Date(ts).getTime() - exitTime);
                    if (diff < minExitDiff) {
                        minExitDiff = diff;
                        closestExitIdx = i;
                    }
                });

                if (closestExitIdx >= 0 && trade.pnl != null) {
                    let zscore = zscores[closestExitIdx];

                    if (zscore === undefined || zscore === null) {
                        for (let offset = 1; offset <= 10; offset++) {
                            if (zscores[closestExitIdx + offset] != null) {
                                zscore = zscores[closestExitIdx + offset];
                                break;
                            }
                            if (zscores[closestExitIdx - offset] != null) {
                                zscore = zscores[closestExitIdx - offset];
                                break;
                            }
                        }
                    }

                    if (zscore != null) {
                        exits.push({
                            timestamp: timestamps[closestExitIdx],
                            zscore: zscore,
                            type: trade.pnl > 0 ? 'profit' : 'loss',
                            pnl: trade.pnl
                        });
                    }
                }
            }
        });

        return { entries, exits };
    }, [trades, analysisData]);

    // Calculate running P&L for each trade
    const tradesWithRunningPnL = React.useMemo(() => {
        let runningPnL = 0;
        return trades.map(trade => {
            if (trade.pnl != null && !isNaN(trade.pnl)) {
                runningPnL += trade.pnl;
            }
            return {
                ...trade,
                runningPnL: runningPnL
            };
        });
    }, [trades]);



    // Check if trades have data quality issues
    const hasDateIssues = trades.some(t => !t.entry_time || isNaN(new Date(t.entry_time).getTime()));
    const hasMissingExitTimes = trades.some(t => t.pnl != null && !t.exit_time);

    return (
        <div className="space-y-6">
            {/* Data Quality Warning */}
            {(hasDateIssues || hasMissingExitTimes) && (
                <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg">
                    <div className="flex items-start gap-3">
                        <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
                        <div className="flex-1">
                            <p className="text-sm font-semibold text-amber-900">Trade Data Quality Issues Detected</p>
                            <ul className="text-xs text-amber-700 mt-2 space-y-1">
                                {hasDateIssues && <li>• Some trades have missing or invalid entry/exit timestamps</li>}
                                {hasMissingExitTimes && <li>• Some trades have P&L but no exit time (using trade sequence for chart)</li>}
                                <li>• Check the browser console (F12) for detailed trade data structure</li>
                            </ul>
                        </div>
                    </div>
                </div>
            )}
            {/* Tab Navigation */}
            <div className="flex items-center gap-4 border-b border-gray-200">
                <button
                    onClick={() => setActiveTab('charts')}
                    className={clsx(
                        "px-4 py-2 font-medium text-sm border-b-2 transition-colors",
                        activeTab === 'charts'
                            ? "border-blue-600 text-blue-600"
                            : "border-transparent text-gray-600 hover:text-gray-900"
                    )}
                >
                    <BarChart2 className="w-4 h-4 inline mr-2" />
                    Charts & Analysis
                </button>
                <button
                    onClick={() => setActiveTab('trades')}
                    className={clsx(
                        "px-4 py-2 font-medium text-sm border-b-2 transition-colors",
                        activeTab === 'trades'
                            ? "border-blue-600 text-blue-600"
                            : "border-transparent text-gray-600 hover:text-gray-900"
                    )}
                >
                    <TableIcon className="w-4 h-4 inline mr-2" />
                    Trade Log ({trades.length} trades)
                </button>
            </div>

            {/* Charts Tab */}
            {activeTab === 'charts' && (
                <div className="space-y-6">
                    {/* Equity Curve */}
                    <Card title="Equity Curve - Portfolio Value Over Time">
                        {equityCurve.length > 1 ? (
                            <>
                                <div className="h-[350px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={equityCurve}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                            <XAxis
                                                dataKey="timestamp"
                                                stroke="#64748b"
                                                tick={{ fill: '#64748b', fontSize: 10 }}
                                            />
                                            <YAxis
                                                stroke="#64748b"
                                                tick={{ fill: '#64748b', fontSize: 11 }}
                                                domain={['dataMin - 50', 'dataMax + 50']}
                                                tickFormatter={(val) => `$${val.toLocaleString()}`}
                                            />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#fff', borderColor: '#e2e8f0' }}
                                                formatter={(value: number) => {
                                                    const change = value - 100000;
                                                    return [
                                                        <>
                                                            ${value.toFixed(2)}
                                                            <span className={change >= 0 ? 'text-green-600 ml-2' : 'text-red-600 ml-2'}>
                                                                ({change >= 0 ? '+' : ''}${change.toFixed(2)})
                                                            </span>
                                                        </>,
                                                        'Equity'
                                                    ];
                                                }}
                                            />
                                            <ReferenceLine
                                                y={100000}
                                                stroke="#94a3b8"
                                                strokeDasharray="3 3"
                                                label={{ value: 'Break Even', fill: '#94a3b8', fontSize: 10 }}
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="equity"
                                                stroke="#3b82f6"
                                                strokeWidth={3}
                                                dot={{ r: 4, fill: '#3b82f6', strokeWidth: 2, stroke: '#fff' }}
                                                activeDot={{ r: 6 }}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </>
                        ) : (
                            <div className="h-[350px] flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200">
                                <div className="text-center text-gray-500">
                                    <p className="text-sm font-medium">No completed trades with P&L data</p>
                                    <p className="text-xs mt-1">Trades need exit times and P&L values to show on the equity curve</p>
                                </div>
                            </div>
                        )}
                        {equityCurve.length > 1 && (
                            <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
                                <div className="p-3 bg-blue-50 rounded">
                                    <div className="text-xs text-blue-600 font-medium">Starting Capital</div>
                                    <div className="text-lg font-bold text-blue-900">$100,000</div>
                                </div>
                                <div className="p-3 bg-green-50 rounded">
                                    <div className="text-xs text-green-600 font-medium">Ending Equity</div>
                                    <div className="text-lg font-bold text-green-900">
                                        ${equityCurve[equityCurve.length - 1].equity.toFixed(2)}
                                    </div>
                                </div>
                                <div className={clsx(
                                    "p-3 rounded",
                                    equityCurve[equityCurve.length - 1].equity >= 100000
                                        ? "bg-emerald-50"
                                        : "bg-red-50"
                                )}>
                                    <div className={clsx(
                                        "text-xs font-medium",
                                        equityCurve[equityCurve.length - 1].equity >= 100000
                                            ? "text-emerald-600"
                                            : "text-red-600"
                                    )}>Net Change</div>
                                    <div className={clsx(
                                        "text-lg font-bold",
                                        equityCurve[equityCurve.length - 1].equity >= 100000
                                            ? "text-emerald-900"
                                            : "text-red-900"
                                    )}>
                                        {equityCurve[equityCurve.length - 1].equity >= 100000 ? '+' : ''}
                                        ${(equityCurve[equityCurve.length - 1].equity - 100000).toFixed(2)}
                                    </div>
                                </div>
                            </div>
                        )}
                    </Card>

                    {/* Z-Score with Trade Markers (if analysis data available) */}
                    {analysisData?.zscore_series && (
                        <Card title="Z-Score with Trade Markers">
                            <div className="mb-4 flex items-center justify-between">
                                <div className="flex items-center gap-4 text-xs">
                                    <div className="flex items-center gap-2">
                                        <div className="w-3 h-3 rounded-full bg-green-500"></div>
                                        <span>Entry Points ({tradeMarkers.entries.length})</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                                        <span>Profitable Exits</span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="w-3 h-3 rounded-full bg-red-500"></div>
                                        <span>Loss Exits</span>
                                    </div>
                                </div>
                                {tradeMarkers.entries.length === 0 && tradeMarkers.exits.length === 0 && (
                                    <div className="text-xs text-amber-600 bg-amber-50 px-3 py-1 rounded">
                                        ⚠️ No trade markers (date range mismatch)
                                    </div>
                                )}
                            </div>
                            <div className="h-[350px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={(() => {
                                        const chartData = analysisData.price_data.timestamps.map((ts: string, i: number) => {
                                            const entry = tradeMarkers.entries.find(e => e.timestamp === ts);
                                            const exit = tradeMarkers.exits.find(e => e.timestamp === ts);

                                            return {
                                                timestamp: new Date(ts).toLocaleString(),
                                                rawTimestamp: ts,
                                                zscore: analysisData.zscore_series[i],
                                                // Add marker data
                                                entryMarker: entry?.zscore,
                                                exitProfitMarker: exit && exit.type === 'profit' ? exit.zscore : undefined,
                                                exitLossMarker: exit && exit.type === 'loss' ? exit.zscore : undefined
                                            };
                                        });


                                        return chartData;
                                    })()}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                        <XAxis
                                            dataKey="timestamp"
                                            stroke="#64748b"
                                            tick={{ fill: '#64748b', fontSize: 10 }}
                                            tickFormatter={(val) => val.split(',')[0]}
                                            minTickGap={50}
                                        />
                                        <YAxis
                                            stroke="#64748b"
                                            tick={{ fill: '#64748b', fontSize: 11 }}
                                        />
                                        <Tooltip />
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
                                        {/* Entry markers as green points - Rendered FIRST (behind) but LARGER to create a halo effect if overlapping */}
                                        <Line
                                            type="monotone"
                                            dataKey="entryMarker"
                                            stroke="none"
                                            dot={{ fill: '#10b981', r: 10, strokeWidth: 2, stroke: '#fff' }}
                                            name="Entry"
                                            isAnimationActive={false}
                                            connectNulls={false}
                                        />
                                        {/* Loss exit markers as red points - Rendered ON TOP but SMALLER */}
                                        <Line
                                            type="monotone"
                                            dataKey="exitLossMarker"
                                            stroke="none"
                                            dot={{ fill: '#ef4444', r: 6, strokeWidth: 2, stroke: '#fff' }}
                                            name="Loss Exit"
                                            isAnimationActive={false}
                                            connectNulls={false}
                                        />
                                        {/* Profit exit markers as blue points - Rendered ON TOP but SMALLER */}
                                        <Line
                                            type="monotone"
                                            dataKey="exitProfitMarker"
                                            stroke="none"
                                            dot={{ fill: '#3b82f6', r: 6, strokeWidth: 2, stroke: '#fff' }}
                                            name="Profit Exit"
                                            isAnimationActive={false}
                                            connectNulls={false}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="mt-4 p-3 bg-amber-50 rounded-lg border border-amber-200">
                                <p className="text-xs text-amber-800">
                                    <strong>📊 How to interpret:</strong> Green dots show where trades entered. Blue dots show profitable exits. Red dots show losing exits.
                                    {tradeMarkers.entries.length === 0 && tradeMarkers.exits.length === 0 && (
                                        <span className="block mt-2 text-amber-900 font-semibold">
                                            ⚠️ No markers visible: The backtest and analysis were run on different date ranges.
                                            Make sure to use the same date range for both.
                                        </span>
                                    )}
                                </p>
                            </div>
                        </Card>
                    )}
                </div>
            )}

            {/* Trades Tab */}
            {activeTab === 'trades' && (
                <Card title="Trade-by-Trade Breakdown">
                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">#</th>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Side</th>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Qty</th>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Entry</th>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Exit</th>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">P&L</th>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Running P&L</th>
                                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Duration</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {tradesWithRunningPnL.map((trade, idx) => {
                                    const duration = trade.exit_time
                                        ? Math.round((new Date(trade.exit_time).getTime() - new Date(trade.entry_time).getTime()) / (1000 * 60)) // minutes
                                        : null;

                                    return (
                                        <tr key={trade.id} className={clsx(
                                            "hover:bg-gray-50",
                                            trade.pnl !== null && trade.pnl < 0 && "bg-red-50"
                                        )}>
                                            <td className="px-3 py-2 text-sm text-gray-900">{idx + 1}</td>
                                            <td className="px-3 py-2 text-sm font-mono text-gray-900">{trade.symbol}</td>
                                            <td className="px-3 py-2 text-sm">
                                                <span className={clsx(
                                                    "px-2 py-1 rounded text-xs font-medium",
                                                    trade.side === 'BUY' ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"
                                                )}>
                                                    {trade.side}
                                                </span>
                                            </td>
                                            <td className="px-3 py-2 text-sm text-gray-900">{trade.quantity ?? '-'}</td>
                                            <td className="px-3 py-2 text-sm text-gray-600">
                                                <div>
                                                    {trade.entry_time ? (
                                                        (() => {
                                                            const date = new Date(trade.entry_time);
                                                            return isNaN(date.getTime()) ? 'Invalid Date' : date.toLocaleString();
                                                        })()
                                                    ) : 'N/A'}
                                                </div>
                                                <div className="text-xs font-mono">
                                                    {trade.entry_price != null ? `$${trade.entry_price.toFixed(2)}` : 'N/A'}
                                                </div>
                                            </td>
                                            <td className="px-3 py-2 text-sm text-gray-600">
                                                {trade.exit_time ? (
                                                    <>
                                                        <div>
                                                            {(() => {
                                                                const date = new Date(trade.exit_time);
                                                                return isNaN(date.getTime()) ? 'Invalid Date' : date.toLocaleString();
                                                            })()}
                                                        </div>
                                                        <div className="text-xs font-mono">
                                                            {trade.exit_price != null ? `$${trade.exit_price.toFixed(2)}` : 'N/A'}
                                                        </div>
                                                    </>
                                                ) : (
                                                    <span className="text-gray-400 italic">
                                                        {trade.pnl != null ? 'Completed (no exit time)' : 'Open'}
                                                    </span>
                                                )}
                                            </td>
                                            <td className="px-3 py-2 text-sm font-bold">
                                                {trade.pnl != null ? (
                                                    <span className={trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}>
                                                        {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                                                    </span>
                                                ) : (
                                                    <span className="text-gray-400">-</span>
                                                )}
                                            </td>
                                            <td className="px-3 py-2 text-sm font-medium">
                                                {trade.runningPnL != null ? (
                                                    <span className={trade.runningPnL >= 0 ? 'text-green-700' : 'text-red-700'}>
                                                        {trade.runningPnL >= 0 ? '+' : ''}${trade.runningPnL.toFixed(2)}
                                                    </span>
                                                ) : (
                                                    <span className="text-gray-400">-</span>
                                                )}
                                            </td>
                                            <td className="px-3 py-2 text-sm text-gray-600">
                                                {duration !== null ? `${duration} min` : '-'}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                    {trades.length === 0 && (
                        <div className="text-center py-8 text-gray-500">
                            No trades found for this backtest.
                        </div>
                    )}
                </Card>
            )}
        </div>
    );
};

