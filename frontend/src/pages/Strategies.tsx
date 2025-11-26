import React, { useEffect, useState } from 'react';
import { Card } from '../components/Card';
import { api, Strategy } from '../services/api';
import { Power, Edit, RefreshCw, Save, X, TrendingUp, TrendingDown, Activity, AlertCircle, CheckCircle, Clock, Download, ChevronDown, ChevronUp, Zap } from 'lucide-react';

export function Strategies() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editParams, setEditParams] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [backfilling, setBackfilling] = useState<Set<string>>(new Set());
  const [warmingUp, setWarmingUp] = useState<Set<string>>(new Set());
  const [expandedPairs, setExpandedPairs] = useState<Set<string>>(new Set());

  const fetchStrategies = async () => {
    try {
      setError(null);
      const data = await api.getStrategies();
      setStrategies(data.strategies || []);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStrategies();
    
    // Auto-refresh every 5 seconds if enabled
    let intervalId: number | null = null;
    if (autoRefresh) {
      intervalId = setInterval(fetchStrategies, 5000);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [autoRefresh]);

  const handleToggle = async (strategyId: string, currentlyEnabled: boolean) => {
    try {
      setError(null);
      await api.enableStrategy(strategyId, !currentlyEnabled);
      // Wait a moment for strategy service to reload, then fetch updated data
      await new Promise(resolve => setTimeout(resolve, 1000));
      await fetchStrategies();
      // Fetch again after a short delay to get state_details if strategy was enabled
      if (!currentlyEnabled) {
        setTimeout(async () => {
          await fetchStrategies();
        }, 2000);
      }
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleEdit = (strategy: Strategy) => {
    setEditingId(strategy.id);
    setEditParams(JSON.stringify(strategy.params, null, 2));
  };

  const handleSave = async (strategyId: string) => {
    try {
      setError(null);
      const params = JSON.parse(editParams);
      await api.updateStrategyParams(strategyId, params);
      setEditingId(null);
      await fetchStrategies();
    } catch (err: any) {
      setError('Invalid JSON or update failed: ' + err.message);
    }
  };

  const handleCancel = () => {
    setEditingId(null);
    setEditParams('');
  };

  const handleBackfill = async (strategyId: string) => {
    try {
      setError(null);
      setBackfilling(prev => new Set(prev).add(strategyId));
      await api.backfillStrategy(strategyId);
      // Show success briefly
      setTimeout(() => {
        setBackfilling(prev => {
          const next = new Set(prev);
          next.delete(strategyId);
          return next;
        });
      }, 3000);
    } catch (err: any) {
      setError(`Backfill failed for ${strategyId}: ` + err.message);
      setBackfilling(prev => {
        const next = new Set(prev);
        next.delete(strategyId);
        return next;
      });
    }
  };

  const handleWarmup = async (strategyId: string) => {
    try {
      setError(null);
      setWarmingUp(prev => new Set(prev).add(strategyId));
      await api.warmupStrategy(strategyId);
      // Show success briefly and refresh strategy data
      setTimeout(async () => {
        setWarmingUp(prev => {
          const next = new Set(prev);
          next.delete(strategyId);
          return next;
        });
        await fetchStrategies();
      }, 1000);
    } catch (err: any) {
      setError(`Warmup failed for ${strategyId}: ` + err.message);
      setWarmingUp(prev => {
        const next = new Set(prev);
        next.delete(strategyId);
        return next;
      });
    }
  };

  const togglePairDetails = (pairKey: string) => {
    setExpandedPairs(prev => {
      const next = new Set(prev);
      if (next.has(pairKey)) {
        next.delete(pairKey);
      } else {
        next.add(pairKey);
      }
      return next;
    });
  };

  const getStatusColor = (state?: string) => {
    switch (state) {
      case 'running': return 'text-green-600';
      case 'stopped': return 'text-red-600';
      case 'stopping': return 'text-yellow-600';
      default: return 'text-gray-600';
    }
  };

  const getProximityColor = (proximity?: number) => {
    if (proximity === undefined || proximity === null) return 'text-gray-400';
    if (proximity >= 1.0) return 'text-green-600';
    if (proximity >= 0.8) return 'text-yellow-600';
    return 'text-gray-400';
  };

  const getPositionBadge = (position: string) => {
    if (position === 'flat') {
      return <span className="px-2 py-1 text-xs rounded bg-gray-100 text-gray-700">Flat</span>;
    }
    if (position.includes('long_a')) {
      return <span className="px-2 py-1 text-xs rounded bg-green-100 text-green-700">Long A / Short B</span>;
    }
    if (position.includes('short_a')) {
      return <span className="px-2 py-1 text-xs rounded bg-red-100 text-red-700">Short A / Long B</span>;
    }
    return <span className="px-2 py-1 text-xs rounded bg-gray-100 text-gray-700">{position}</span>;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-600" />
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
        title="Strategies"
        action={
          <div className="flex items-center space-x-4">
            <label className="flex items-center space-x-2 text-sm">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded"
              />
              <span>Auto-refresh (5s)</span>
            </label>
            <button
              onClick={fetchStrategies}
              className="text-blue-600 hover:text-blue-700 text-sm font-medium flex items-center"
            >
              <RefreshCw className="w-4 h-4 mr-1" />
              Refresh
            </button>
          </div>
        }
      >
        {strategies.length > 0 ? (
          <div className="space-y-6">
            {strategies.map((strategy) => {
              const stateDetails = strategy.state_details;
              const pairsState = stateDetails?.pairs_state || {};
              const config = stateDetails?.config || {};
              const metrics = strategy.metrics || {};
              
              return (
                <div
                  key={strategy.id}
                  className="p-6 border border-gray-200 rounded-lg space-y-6 bg-white shadow-sm"
                >
                  {/* Strategy Header */}
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <h4 className={`text-xl font-semibold ${strategy.enabled ? 'text-gray-900' : 'text-gray-500'}`}>
                          {strategy.name}
                        </h4>
                        {!strategy.enabled && (
                          <span className="px-2 py-1 text-xs rounded-full bg-gray-200 text-gray-600 flex items-center">
                            Disabled
                          </span>
                        )}
                        {strategy.enabled && strategy.running && (
                          <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-700 flex items-center">
                            <Activity className="w-3 h-3 mr-1" />
                            Running
                          </span>
                        )}
                        {strategy.enabled && !strategy.running && strategy.state && strategy.state !== 'stopped' && (
                          <span className={`text-sm font-medium ${getStatusColor(strategy.state)}`}>
                            {strategy.state}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-500 mt-1">ID: {strategy.id}</p>
                      
                      {/* Metrics - only show if strategy is enabled */}
                      {strategy.enabled && metrics && (metrics.total_signals !== undefined || metrics.total_pnl !== undefined) && (
                        <div className="flex items-center space-x-6 mt-3">
                          {metrics.total_signals !== undefined && (
                            <div className="flex items-center space-x-1 text-sm">
                              <Activity className="w-4 h-4 text-gray-400" />
                              <span className="text-gray-600">Signals: <strong>{metrics.total_signals}</strong></span>
                            </div>
                          )}
                          {metrics.total_pnl !== undefined && (
                            <div className={`flex items-center space-x-1 text-sm ${metrics.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {metrics.total_pnl >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                              <span>P&L: <strong>${metrics.total_pnl.toFixed(2)}</strong></span>
                            </div>
                          )}
                          {metrics.win_rate !== undefined && (
                            <div className="flex items-center space-x-1 text-sm text-gray-600">
                              <span>Win Rate: <strong>{(metrics.win_rate * 100).toFixed(1)}%</strong></span>
                            </div>
                          )}
                        </div>
                      )}
                      {!strategy.enabled && (
                        <div className="mt-3 text-sm text-gray-500 italic">
                          Strategy is disabled. Enable it to start monitoring and trading.
                        </div>
                      )}
                    </div>

                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => handleEdit(strategy)}
                        disabled={editingId !== null}
                        className="p-2 text-gray-600 hover:text-blue-600 disabled:opacity-50"
                        title="Edit Parameters"
                      >
                        <Edit className="w-5 h-5" />
                      </button>

                      <button
                        onClick={() => handleBackfill(strategy.id)}
                        disabled={backfilling.has(strategy.id)}
                        className="flex items-center px-4 py-2 rounded-md font-medium text-sm bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Backfill Historical Data (works even when strategy is disabled)"
                      >
                        {backfilling.has(strategy.id) ? (
                          <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        ) : (
                          <Download className="w-4 h-4 mr-2" />
                        )}
                        Backfill
                      </button>

                      <button
                        onClick={() => handleWarmup(strategy.id)}
                        disabled={!strategy.enabled || warmingUp.has(strategy.id)}
                        className="flex items-center px-4 py-2 rounded-md font-medium text-sm bg-orange-600 text-white hover:bg-orange-700 disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Warmup strategy from cached historical data (requires strategy to be enabled)"
                      >
                        {warmingUp.has(strategy.id) ? (
                          <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        ) : (
                          <Zap className="w-4 h-4 mr-2" />
                        )}
                        Warmup
                      </button>

                      <button
                        onClick={() => handleToggle(strategy.id, strategy.enabled)}
                        className={`flex items-center px-4 py-2 rounded-md font-medium text-sm ${
                          strategy.enabled
                            ? 'bg-green-600 text-white hover:bg-green-700'
                            : 'bg-gray-300 text-gray-700 hover:bg-gray-400'
                        }`}
                      >
                        <Power className="w-4 h-4 mr-2" />
                        {strategy.enabled ? 'Enabled' : 'Disabled'}
                      </button>
                    </div>
                  </div>

                  {/* Detailed Pair Status for Pairs Trading */}
                  {strategy.enabled && !stateDetails && (
                    <div className="border-t pt-4">
                      <div className="flex items-center space-x-2 text-sm text-gray-500">
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Strategy is starting up... Pair status will appear shortly.</span>
                      </div>
                      <div className="mt-2 text-xs text-gray-400">
                        Note: Strategy needs to receive bar data from the market data service. Check that market data collection is enabled.
                      </div>
                    </div>
                  )}
                  {strategy.enabled && stateDetails && pairsState && Object.keys(pairsState).length === 0 && (
                    <div className="border-t pt-4">
                      <div className="text-sm text-yellow-600 bg-yellow-50 p-3 rounded">
                        <AlertCircle className="w-4 h-4 inline mr-2" />
                        Strategy is running but no pair data available yet. This may indicate:
                        <ul className="list-disc list-inside mt-2 ml-4 text-xs">
                          <li>Strategy is still initializing</li>
                          <li>Bar data is not being received from market data service</li>
                          <li>Check market data service logs for connection issues</li>
                        </ul>
                      </div>
                    </div>
                  )}
                  {strategy.enabled && stateDetails && pairsState && Object.keys(pairsState).length > 0 && (
                    <div className="border-t pt-4">
                      <h5 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                        <Activity className="w-4 h-4 mr-2" />
                        Pair Status ({stateDetails.num_pairs || Object.keys(pairsState).length} pairs)
                        {config.entry_threshold && config.exit_threshold && (
                          <span className="ml-3 text-xs font-normal text-gray-500">
                            Entry: {config.entry_threshold} | Exit: {config.exit_threshold}
                          </span>
                        )}
                      </h5>
                      
                      <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                          <thead className="bg-gray-50">
                            <tr>
                              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Details</th>
                              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Pair</th>
                              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Position</th>
                              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Z-Score</th>
                              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Entry %</th>
                              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Exit %</th>
                              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Bars in Trade</th>
                              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Spread Bars</th>
                              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-gray-200">
                            {Object.entries(pairsState)
                              .sort(([a], [b]) => a.localeCompare(b))
                              .map(([pairKey, pairState]) => {
                                const zscore = pairState.current_zscore;
                                const entryProx = pairState.entry_proximity;
                                const exitProx = pairState.exit_proximity;
                                const position = pairState.position;
                                const barsInTrade = pairState.bars_in_trade || 0;
                                const cooldown = pairState.cooldown_remaining || 0;
                                const hasSufficientData = pairState.has_sufficient_data;
                                const spreadHistoryLen = pairState.spread_history_length || 0;
                                const lookbackWindow = pairState.lookback_window || 222;
                                
                                let statusIcon = null;
                                let statusText = '';
                                
                                // Check if we have sufficient data first
                                if (!hasSufficientData) {
                                  statusText = `Warming (${spreadHistoryLen}/${lookbackWindow})`;
                                } else if (position !== 'flat') {
                                  statusIcon = <CheckCircle className="w-4 h-4 text-green-600" />;
                                  statusText = 'In Trade';
                                  if (exitProx !== undefined && exitProx < 0.5) {
                                    statusIcon = <AlertCircle className="w-4 h-4 text-yellow-600" />;
                                    statusText = 'Near Exit';
                                  }
                                } else if (cooldown > 0) {
                                  statusIcon = <Clock className="w-4 h-4 text-gray-400" />;
                                  statusText = `Cooldown (${cooldown})`;
                                } else if (entryProx !== undefined) {
                                  if (entryProx >= 1.0) {
                                    statusIcon = <CheckCircle className="w-4 h-4 text-green-600" />;
                                    statusText = 'Ready';
                                  } else if (entryProx >= 0.8) {
                                    statusIcon = <AlertCircle className="w-4 h-4 text-yellow-600" />;
                                    statusText = 'Close';
                                  } else {
                                    statusText = 'Far';
                                  }
                                } else {
                                  statusText = 'Warming';
                                }
                                
                                const isExpanded = expandedPairs.has(pairKey);
                                const blockingReasons = pairState.blocking_reasons || [];
                                const aggBuffer = pairState.aggregation_buffer;
                                
                                return (
                                  <React.Fragment key={pairKey}>
                                    <tr className="hover:bg-gray-50">
                                      <td className="px-3 py-2 whitespace-nowrap text-sm">
                                        <button
                                          onClick={() => togglePairDetails(pairKey)}
                                          className="text-gray-600 hover:text-gray-900"
                                        >
                                          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                        </button>
                                      </td>
                                      <td className="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-900">
                                        {pairKey}
                                      </td>
                                      <td className="px-3 py-2 whitespace-nowrap text-sm">
                                        {getPositionBadge(position)}
                                      </td>
                                      <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-900">
                                        {zscore !== undefined && zscore !== null ? zscore.toFixed(3) : 'N/A'}
                                      </td>
                                      <td className={`px-3 py-2 whitespace-nowrap text-sm font-medium ${getProximityColor(entryProx)}`}>
                                        {entryProx !== undefined && entryProx !== null 
                                          ? `${(entryProx * 100).toFixed(1)}%` 
                                          : 'N/A'}
                                      </td>
                                      <td className={`px-3 py-2 whitespace-nowrap text-sm font-medium ${getProximityColor(exitProx)}`}>
                                        {exitProx !== undefined && exitProx !== null 
                                          ? `${(exitProx * 100).toFixed(1)}%` 
                                          : 'N/A'}
                                      </td>
                                      <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-500">
                                        {barsInTrade}
                                      </td>
                                      <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-600">
                                        {spreadHistoryLen}
                                        {!hasSufficientData && (
                                          <span className="text-xs text-gray-400 ml-1">/ {lookbackWindow}</span>
                                        )}
                                      </td>
                                      <td className="px-3 py-2 whitespace-nowrap text-sm">
                                        <div className="flex items-center space-x-1">
                                          {statusIcon}
                                          <span>{statusText}</span>
                                        </div>
                                      </td>
                                    </tr>
                                    {isExpanded && (
                                      <tr className="bg-gray-50">
                                        <td colSpan={9} className="px-3 py-4">
                                          <div className="grid grid-cols-3 gap-4 text-xs">
                                            {/* Column 1: Data & Aggregation */}
                                            <div className="space-y-3">
                                              <h6 className="font-semibold text-gray-700 mb-2">Data Collection</h6>
                                              
                                              <div>
                                                <div className="text-gray-500 font-mono text-xs">lookback_window</div>
                                                <div className="font-medium">{spreadHistoryLen} / {lookbackWindow} spread bars</div>
                                                {config.stats_aggregation_seconds && (
                                                  <>
                                                    <div className="text-xs text-gray-400 font-mono">
                                                      stats_aggregation_seconds: {config.stats_aggregation_seconds}s 
                                                      {config.stats_aggregation_seconds >= 60 && ` (${(config.stats_aggregation_seconds / 60).toFixed(0)} min)`}
                                                    </div>
                                                    {!hasSufficientData && lookbackWindow && spreadHistoryLen != null && (
                                                      <div className="text-xs text-blue-600 font-medium">
                                                        ⏱️ Need {((lookbackWindow - spreadHistoryLen) * config.stats_aggregation_seconds / 60).toFixed(0)} more minutes
                                                      </div>
                                                    )}
                                                  </>
                                                )}
                                              </div>
                                              
                                              <div>
                                                <div className="text-gray-500 font-mono text-xs">price_history_length</div>
                                                <div className="font-medium">{pairState.price_history_length || 0} bars</div>
                                                {config.base_bar_seconds && (
                                                  <div className="text-xs text-gray-400 font-mono">bar_timeframe: {config.base_bar_seconds}s</div>
                                                )}
                                              </div>
                                              
                                              {aggBuffer && (
                                                <div>
                                                  <div className="text-gray-500 font-mono text-xs">stats_aggregation_bars</div>
                                                  <div className="font-medium">
                                                    {aggBuffer.count || 0} / {aggBuffer.target || 0} bars 
                                                    {aggBuffer.progress_pct != null && ` (${aggBuffer.progress_pct.toFixed(1)}%)`}
                                                  </div>
                                                  {aggBuffer.last_timestamp && (
                                                    <div className="text-xs text-gray-400">Last: {new Date(aggBuffer.last_timestamp).toLocaleTimeString()}</div>
                                                  )}
                                                  {config.base_bar_seconds && config.stats_aggregation_seconds && aggBuffer.target && (
                                                    <div className="text-xs text-gray-400 font-mono">
                                                      {aggBuffer.target} × {config.base_bar_seconds}s = {config.stats_aggregation_seconds}s/bar
                                                    </div>
                                                  )}
                                                </div>
                                              )}
                                              
                                              {pairState.last_processed_timestamp && (
                                                <div>
                                                  <div className="text-gray-500 font-mono text-xs">last_processed_timestamp</div>
                                                  <div className="font-medium">{new Date(pairState.last_processed_timestamp).toLocaleTimeString()}</div>
                                                </div>
                                              )}
                                            </div>
                                            
                                            {/* Column 2: Spread & Thresholds */}
                                            <div className="space-y-3">
                                              <h6 className="font-semibold text-gray-700 mb-2">Spread Analysis</h6>
                                              
                                              <div>
                                                <div className="text-gray-500 font-mono text-xs">current_spread</div>
                                                <div className="font-medium">{pairState.current_spread?.toFixed(4) || 'N/A'}</div>
                                                {pairState.current_zscore != null && (
                                                  <div className="text-xs text-gray-400 font-mono">current_zscore: {pairState.current_zscore.toFixed(2)}</div>
                                                )}
                                              </div>
                                              
                                              {pairState.spread_stats && (
                                                <>
                                                  <div>
                                                    <div className="text-gray-500 font-mono text-xs">mean_spread / std_spread</div>
                                                    <div className="font-medium font-mono">
                                                      {pairState.spread_stats.mean?.toFixed(4) || 'N/A'} / {pairState.spread_stats.std?.toFixed(4) || 'N/A'}
                                                    </div>
                                                  </div>
                                                  <div>
                                                    <div className="text-gray-500 font-mono text-xs">spread_min / spread_max</div>
                                                    <div className="font-medium font-mono">
                                                      {pairState.spread_stats.min?.toFixed(4) || 'N/A'} to {pairState.spread_stats.max?.toFixed(4) || 'N/A'}
                                                    </div>
                                                  </div>
                                                </>
                                              )}
                                              
                                              <div>
                                                <div className="text-gray-500 font-mono text-xs">entry_threshold</div>
                                                <div className="font-medium font-mono">
                                                  base: {pairState.base_entry_threshold?.toFixed(2) || 'N/A'}
                                                  {pairState.adjusted_entry_threshold != null && ` → adj: ${pairState.adjusted_entry_threshold.toFixed(2)}`}
                                                </div>
                                              </div>
                                              
                                              <div>
                                                <div className="text-gray-500 font-mono text-xs">exit_threshold</div>
                                                <div className="font-medium font-mono">
                                                  base: {pairState.base_exit_threshold?.toFixed(2) || 'N/A'}
                                                  {pairState.adjusted_exit_threshold != null && ` → adj: ${pairState.adjusted_exit_threshold.toFixed(2)}`}
                                                </div>
                                              </div>
                                              
                                              <div>
                                                <div className="text-gray-500 font-mono text-xs">volatility_ratio</div>
                                                <div className="font-medium">{pairState.volatility_ratio?.toFixed(3) || 'N/A'}</div>
                                              </div>
                                            </div>
                                            
                                            {/* Column 3: Model & Trade Readiness */}
                                            <div className="space-y-3">
                                              <h6 className="font-semibold text-gray-700 mb-2">Model & Readiness</h6>
                                              
                                              <div>
                                                <div className="text-gray-500 font-mono text-xs">hedge_ratio</div>
                                                <div className="font-medium">{pairState.hedge_ratio?.toFixed(4) || 'N/A'}</div>
                                                {pairState.hedge_intercept != null && (
                                                  <div className="text-xs text-gray-400 font-mono">intercept: {pairState.hedge_intercept.toFixed(4)}</div>
                                                )}
                                              </div>
                                              
                                              {pairState.kalman_state && (
                                                <div>
                                                  <div className="text-gray-500 font-mono text-xs">use_kalman (kalman_state)</div>
                                                  <div className="font-medium font-mono">β: {pairState.kalman_state.beta?.toFixed(4) || 'N/A'}</div>
                                                  <div className="text-xs text-gray-400 font-mono">α: {pairState.kalman_state.alpha?.toFixed(4) || 'N/A'}</div>
                                                </div>
                                              )}
                                              
                                              <div>
                                                <div className="text-gray-500 font-mono text-xs">half_life / max_halflife_bars</div>
                                                <div className="font-medium font-mono">
                                                  {pairState.half_life?.toFixed(1) || 'N/A'}
                                                  {pairState.max_halflife_bars ? ` / ${pairState.max_halflife_bars}` : ''}
                                                </div>
                                              </div>
                                              
                                              {config.stationarity_checks_enabled && (
                                                <>
                                                  <div>
                                                    <div className="text-gray-500 font-mono text-xs">
                                                      adf_pvalue {config.adf_pvalue_threshold != null && `(threshold: ${config.adf_pvalue_threshold})`}
                                                    </div>
                                                    <div className={`font-medium font-mono ${pairState.adf_pvalue != null && pairState.adf_pvalue <= (config.adf_pvalue_threshold || 0.05) ? 'text-green-600' : 'text-red-600'}`}>
                                                      {pairState.adf_pvalue?.toFixed(4) || 'N/A'}
                                                    </div>
                                                  </div>
                                                  
                                                  <div>
                                                    <div className="text-gray-500 font-mono text-xs">
                                                      cointegration_pvalue {config.cointegration_pvalue_threshold != null && `(threshold: ${config.cointegration_pvalue_threshold})`}
                                                    </div>
                                                    <div className={`font-medium font-mono ${pairState.cointegration_pvalue != null && pairState.cointegration_pvalue <= (config.cointegration_pvalue_threshold || 0.05) ? 'text-green-600' : 'text-red-600'}`}>
                                                      {pairState.cointegration_pvalue?.toFixed(4) || 'N/A'}
                                                    </div>
                                                  </div>
                                                </>
                                              )}
                                              
                                              {blockingReasons.length > 0 && (
                                                <div>
                                                  <div className="text-red-600 font-semibold font-mono text-xs">blocking_reasons</div>
                                                  <ul className="list-disc list-inside text-red-600 mt-1 text-xs">
                                                    {blockingReasons.map((reason: string, idx: number) => (
                                                      <li key={idx}>{reason}</li>
                                                    ))}
                                                  </ul>
                                                </div>
                                              )}
                                              
                                              {blockingReasons.length === 0 && pairState.can_trade && (
                                                <div>
                                                  <div className="text-gray-500 font-mono text-xs">can_trade</div>
                                                  <div className="text-green-600 font-semibold flex items-center">
                                                    <CheckCircle className="w-4 h-4 mr-1" />
                                                    Ready
                                                  </div>
                                                </div>
                                              )}
                                            </div>
                                          </div>
                                        </td>
                                      </tr>
                                    )}
                                  </React.Fragment>
                                );
                              })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Parameters Section */}
                  {editingId === strategy.id ? (
                    <div className="space-y-3 border-t pt-4">
                      <label className="block text-sm font-medium text-gray-700">
                        Parameters (JSON)
                      </label>
                      <textarea
                        value={editParams}
                        onChange={(e) => setEditParams(e.target.value)}
                        rows={8}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                      <div className="flex justify-end space-x-2">
                        <button
                          onClick={handleCancel}
                          className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 flex items-center text-sm"
                        >
                          <X className="w-4 h-4 mr-1" />
                          Cancel
                        </button>
                        <button
                          onClick={() => handleSave(strategy.id)}
                          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center text-sm"
                        >
                          <Save className="w-4 h-4 mr-1" />
                          Save
                        </button>
                      </div>
                    </div>
                  ) : (
                    <details className="border-t pt-4">
                      <summary className="text-sm font-medium text-gray-700 cursor-pointer hover:text-gray-900">
                        View Parameters
                      </summary>
                      <div className="bg-gray-50 p-3 rounded-md mt-2">
                        <pre className="text-sm text-gray-700 overflow-x-auto">
                          {JSON.stringify(strategy.params, null, 2)}
                        </pre>
                      </div>
                    </details>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">No strategies found</p>
        )}
      </Card>
    </div>
  );
}
