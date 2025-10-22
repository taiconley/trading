import React, { useEffect, useState } from 'react';
import { Card } from '../components/Card';
import { api } from '../services/api';
import { Plus, X, RefreshCw } from 'lucide-react';

export function MarketData() {
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [newSymbol, setNewSymbol] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchWatchlist = async () => {
    try {
      const data = await api.getWatchlist();
      setWatchlist(data.symbols || []);
    } catch (err: any) {
      setError(err.message);
    }
  };

  useEffect(() => {
    fetchWatchlist();
    const interval = setInterval(fetchWatchlist, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleAddSymbol = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSymbol.trim()) return;

    try {
      setLoading(true);
      setError(null);
      await api.updateWatchlist('add', newSymbol.toUpperCase());
      setNewSymbol('');
      await fetchWatchlist();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveSymbol = async (symbol: string) => {
    try {
      setLoading(true);
      setError(null);
      await api.updateWatchlist('remove', symbol);
      await fetchWatchlist();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card
        title="Watchlist"
        action={
          <form onSubmit={handleAddSymbol} className="flex items-center space-x-2">
            <input
              type="text"
              value={newSymbol}
              onChange={(e) => setNewSymbol(e.target.value)}
              placeholder="Add symbol..."
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center text-sm font-medium disabled:opacity-50"
            >
              <Plus className="w-4 h-4 mr-1" />
              Add
            </button>
          </form>
        }
      >
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md text-red-800 text-sm">
            {error}
          </div>
        )}

        {watchlist.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {watchlist.map((symbol) => (
              <SymbolCard
                key={symbol}
                symbol={symbol}
                onRemove={handleRemoveSymbol}
              />
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">
            No symbols in watchlist. Add some to get started!
          </p>
        )}
      </Card>
    </div>
  );
}

interface SymbolCardProps {
  symbol: string;
  onRemove: (symbol: string) => void;
}

function SymbolCard({ symbol, onRemove }: SymbolCardProps) {
  const [ticks, setTicks] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    // Fetch initial tick data
    const fetchInitialTicks = async () => {
      try {
        setLoading(true);
        const data = await api.getTicks(symbol, 1);
        if (data.ticks && data.ticks.length > 0) {
          setTicks(data.ticks[0]);
        }
      } catch (err) {
        console.error('Failed to fetch initial ticks:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchInitialTicks();
  }, [symbol]);

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const wsUrl = 'ws://localhost:8002/ws';
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log(`WebSocket connected for market data`);
      setWsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        // Handle tick updates for this symbol
        if (message.type === 'tick_update' && message.data.symbol === symbol) {
          setTicks({
            symbol: message.data.symbol,
            bid: message.data.bid,
            ask: message.data.ask,
            last: message.data.last,
            bid_size: message.data.bid_size,
            ask_size: message.data.ask_size,
            last_size: message.data.last_size,
            timestamp: message.data.timestamp
          });
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setWsConnected(false);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWsConnected(false);
    };

    // Cleanup on unmount
    return () => {
      ws.close();
    };
  }, [symbol]);

  return (
    <div className="p-4 border border-gray-200 rounded-lg relative">
      <button
        onClick={() => onRemove(symbol)}
        className="absolute top-2 right-2 text-gray-400 hover:text-red-600"
      >
        <X className="w-4 h-4" />
      </button>

      <div className="flex items-center justify-between mb-2">
        <h4 className="text-lg font-bold text-gray-900">{symbol}</h4>
        <div className="flex items-center space-x-1">
          <div 
            className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`}
            title={wsConnected ? 'Live' : 'Disconnected'}
          />
        </div>
      </div>

      {loading && !ticks ? (
        <div className="flex items-center justify-center py-4">
          <RefreshCw className="w-5 h-5 animate-spin text-gray-400" />
        </div>
      ) : ticks ? (
        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-sm text-gray-500">Last:</span>
            <span className="text-sm font-medium text-gray-900">
              ${ticks.last?.toFixed(2) || 'N/A'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm text-gray-500">Bid:</span>
            <span className="text-sm font-medium text-gray-900">
              ${ticks.bid?.toFixed(2) || 'N/A'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-sm text-gray-500">Ask:</span>
            <span className="text-sm font-medium text-gray-900">
              ${ticks.ask?.toFixed(2) || 'N/A'}
            </span>
          </div>
          <div className="text-xs text-gray-400 mt-2">
            {new Date(ticks.timestamp).toLocaleTimeString()}
          </div>
        </div>
      ) : (
        <p className="text-sm text-gray-500 py-4">No data available</p>
      )}
    </div>
  );
}

