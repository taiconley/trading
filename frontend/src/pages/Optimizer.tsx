import React, { useEffect, useState } from 'react';
import { Card } from '../components/Card';
import { api, Optimization } from '../services/api';
import { RefreshCw, Eye, TrendingUp, BarChart3 } from 'lucide-react';

export function Optimizer() {
  const [optimizations, setOptimizations] = useState<Optimization[]>([]);
  const [selectedOpt, setSelectedOpt] = useState<any | null>(null);
  const [results, setResults] = useState<any[]>([]);
  const [analysis, setAnalysis] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchOptimizations = async () => {
    try {
      setError(null);
      const data = await api.getOptimizations(50);
      setOptimizations(data.optimizations || []);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOptimizations();
  }, []);

  const handleViewDetails = async (opt: Optimization) => {
    try {
      setError(null);
      const [optDetails, resultsData] = await Promise.all([
        api.getOptimization(opt.id),
        api.getOptimizationResults(opt.id, 20),
      ]);

      // Try to fetch sensitivity analysis if available
      let analysisData = null;
      try {
        analysisData = await api.getOptimizationAnalysis(opt.id);
      } catch {
        // Analysis might not be available
      }

      setSelectedOpt(optDetails);
      setResults(resultsData.results || []);
      setAnalysis(analysisData);
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

  if (selectedOpt) {
    return (
      <div className="space-y-6">
        <button
          onClick={() => {
            setSelectedOpt(null);
            setResults([]);
            setAnalysis(null);
          }}
          className="text-blue-600 hover:text-blue-700 font-medium"
        >
          ← Back to list
        </button>

        <Card title={`Optimization #${selectedOpt.id} - ${selectedOpt.strategy_name}`}>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-6">
            <div>
              <p className="text-sm text-gray-500">Algorithm</p>
              <p className="text-lg font-semibold text-gray-900 capitalize">
                {selectedOpt.algorithm}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Status</p>
              <p className="text-lg font-semibold text-gray-900 capitalize">
                {selectedOpt.status}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Progress</p>
              <p className="text-lg font-semibold text-gray-900">
                {selectedOpt.completed_combinations} / {selectedOpt.total_combinations}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Best Score</p>
              <p className="text-lg font-semibold text-green-600">
                {selectedOpt.best_score?.toFixed(4) || 'N/A'}
              </p>
            </div>
          </div>

          {selectedOpt.best_params && (
            <div className="bg-green-50 p-4 rounded-md border border-green-200">
              <h4 className="font-medium text-green-900 mb-2 flex items-center">
                <TrendingUp className="w-5 h-5 mr-2" />
                Best Parameters Found
              </h4>
              <pre className="text-sm text-green-800 overflow-x-auto">
                {JSON.stringify(selectedOpt.best_params, null, 2)}
              </pre>
            </div>
          )}
        </Card>

        {/* Sensitivity Analysis */}
        {analysis && analysis.sensitivity && (
          <Card
            title="Parameter Sensitivity Analysis"
            action={
              <BarChart3 className="w-5 h-5 text-gray-500" />
            }
          >
            <div className="space-y-3">
              {analysis.sensitivity.map((item: any, idx: number) => (
                <div key={idx} className="flex items-center">
                  <div className="w-1/4">
                    <p className="font-medium text-gray-900">{item.parameter}</p>
                    <p className="text-xs text-gray-500">Rank #{item.importance_rank}</p>
                  </div>
                  <div className="w-3/4">
                    <div className="flex items-center">
                      <div className="flex-1 bg-gray-200 rounded-full h-4 mr-3">
                        <div
                          className="bg-blue-600 h-4 rounded-full"
                          style={{
                            width: `${Math.min(100, Math.abs(item.sensitivity_score || 0) * 100)}%`,
                          }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-700 w-16 text-right">
                        {(item.sensitivity_score || 0).toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* Top Results */}
        <Card title="Top Parameter Combinations">
          {results.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead>
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Rank
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Parameters
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Score
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Sharpe
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Return %
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Trades
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {results.map((result, idx) => (
                    <tr key={idx} className={idx === 0 ? 'bg-green-50' : ''}>
                      <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                        #{idx + 1}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-700">
                        <details>
                          <summary className="cursor-pointer hover:text-blue-600">
                            View params
                          </summary>
                          <pre className="mt-2 text-xs bg-gray-50 p-2 rounded">
                            {JSON.stringify(result.params, null, 2)}
                          </pre>
                        </details>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap font-medium text-green-600">
                        {result.score?.toFixed(4) || 'N/A'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                        {result.sharpe_ratio?.toFixed(4) || 'N/A'}
                      </td>
                      <td
                        className={`px-6 py-4 whitespace-nowrap font-medium ${
                          (result.total_return || 0) >= 0
                            ? 'text-green-600'
                            : 'text-red-600'
                        }`}
                      >
                        {result.total_return?.toFixed(2)}%
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                        {result.total_trades || 0}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">No results available</p>
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
        title="Optimization Results"
        action={
          <button
            onClick={fetchOptimizations}
            className="text-blue-600 hover:text-blue-700 text-sm font-medium flex items-center"
          >
            <RefreshCw className="w-4 h-4 mr-1" />
            Refresh
          </button>
        }
      >
        {optimizations.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Strategy
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Algorithm
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Progress
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Best Score
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
                {optimizations.map((opt) => (
                  <tr key={opt.id}>
                    <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                      #{opt.id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-900">
                      {opt.strategy_name}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-600 capitalize">
                      {opt.algorithm}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded capitalize ${
                          opt.status === 'completed'
                            ? 'bg-green-100 text-green-800'
                            : opt.status === 'running'
                            ? 'bg-blue-100 text-blue-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {opt.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                      {opt.completed_combinations} / {opt.total_combinations}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap font-medium text-green-600">
                      {opt.best_score?.toFixed(4) || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(opt.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <button
                        onClick={() => handleViewDetails(opt)}
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
            No optimizations found. Run optimizations via CLI to see results here.
          </p>
        )}
      </Card>
    </div>
  );
}

