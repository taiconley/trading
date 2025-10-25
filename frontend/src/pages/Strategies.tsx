import React, { useEffect, useState } from 'react';
import { Card } from '../components/Card';
import { api, Strategy } from '../services/api';
import { Power, Edit, RefreshCw, Save, X } from 'lucide-react';

export function Strategies() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editParams, setEditParams] = useState('');

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
  }, []);

  const handleToggle = async (strategyId: string, currentlyEnabled: boolean) => {
    try {
      setError(null);
      await api.enableStrategy(strategyId, !currentlyEnabled);
      await fetchStrategies();
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
        title="Active Strategies"
        action={
          <button
            onClick={fetchStrategies}
            className="text-blue-600 hover:text-blue-700 text-sm font-medium flex items-center"
          >
            <RefreshCw className="w-4 h-4 mr-1" />
            Refresh
          </button>
        }
      >
        {strategies.length > 0 ? (
          <div className="space-y-4">
            {strategies.map((strategy) => (
              <div
                key={strategy.id}
                className="p-4 border border-gray-200 rounded-lg space-y-4"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-lg font-semibold text-gray-900">
                      {strategy.name}
                    </h4>
                    <p className="text-sm text-gray-500">ID: {strategy.id}</p>
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

                {editingId === strategy.id ? (
                  <div className="space-y-3">
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
                  <div className="bg-gray-50 p-3 rounded-md">
                    <pre className="text-sm text-gray-700 overflow-x-auto">
                      {JSON.stringify(strategy.params, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">No strategies found</p>
        )}
      </Card>
    </div>
  );
}

