import React, { useEffect, useState } from 'react';
import { Card } from '../components/Card';
import { api, Order, Position } from '../services/api';
import { RefreshCw, TrendingUp, TrendingDown, AlertCircle, Activity, CheckCircle, XCircle, Clock, AlertTriangle, Shield } from 'lucide-react';

export function Overview() {
  const [loading, setLoading] = useState(true);
  const [account, setAccount] = useState<any>(null);
  const [orders, setOrders] = useState<Order[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [accountData, ordersData, healthData] = await Promise.all([
        api.getAccountStats().catch(() => null),
        api.getOrders({ limit: 10 }).catch(() => ({ orders: [] })),
        api.getAggregateHealth().catch(() => null),
      ]);

      setAccount(accountData);
      setOrders(ordersData.orders || ordersData);
      setHealth(healthData);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading && !account) {
    return (
      <div className="flex items-center justify-center h-full">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Error Alert */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center">
          <AlertCircle className="w-5 h-5 text-red-600 mr-3" />
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Account Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-sm font-medium text-slate-600 uppercase tracking-wide">Net Liquidation</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">
                ${account?.net_liquidation?.toLocaleString() || '0.00'}
              </p>
              <div className="flex items-center mt-3">
                <div className="flex items-center text-green-600 text-sm font-semibold">
                  <TrendingUp className="w-4 h-4 mr-1" />
                  <span>+2.4%</span>
                </div>
                <span className="text-xs text-slate-500 ml-2">vs last week</span>
              </div>
            </div>
            <div className="bg-green-100 p-4 rounded-xl">
              <TrendingUp className="w-8 h-8 text-green-600" />
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-sm font-medium text-slate-600 uppercase tracking-wide">Available Funds</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">
                ${account?.available_funds?.toLocaleString() || '0.00'}
              </p>
              <div className="flex items-center mt-3">
                <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden mr-2">
                  <div className="h-full bg-blue-600 rounded-full" style={{width: '68%'}}></div>
                </div>
                <span className="text-xs text-slate-600 font-medium">68%</span>
              </div>
            </div>
            <div className="bg-blue-100 p-4 rounded-xl">
              <TrendingUp className="w-8 h-8 text-blue-600" />
            </div>
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-sm font-medium text-slate-600 uppercase tracking-wide">Buying Power</p>
              <p className="text-3xl font-bold text-slate-900 mt-2">
                ${account?.buying_power?.toLocaleString() || '0.00'}
              </p>
              <div className="flex items-center mt-3">
                <span className="px-3 py-1 bg-purple-100 text-purple-700 text-xs font-semibold rounded-lg">
                  4x Leverage
                </span>
              </div>
            </div>
            <div className="bg-purple-100 p-4 rounded-xl">
              <TrendingUp className="w-8 h-8 text-purple-600" />
            </div>
          </div>
        </Card>
      </div>

      {/* Positions */}
      <Card title="Current Positions">
        {account?.positions && account.positions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-gray-200 bg-gray-50">
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Symbol
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Quantity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Avg Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Market Value
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Unrealized P&L
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {account.positions.map((position: Position, idx: number) => (
                  <tr key={idx} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="font-bold text-slate-900">{position.symbol}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-slate-700">
                      {position.qty}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-slate-700">
                      ${position.avg_price?.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-slate-700">
                      ${position.market_value?.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center px-3 py-1 rounded-lg text-sm font-semibold ${
                        (position.unrealized_pnl || 0) >= 0
                          ? 'bg-green-100 text-green-700'
                          : 'bg-red-100 text-red-700'
                      }`}>
                        {(position.unrealized_pnl || 0) >= 0 ? '+' : ''}${position.unrealized_pnl?.toFixed(2)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gray-100 rounded-full mb-4">
              <TrendingUp className="w-8 h-8 text-gray-400" />
            </div>
            <p className="text-gray-500 font-medium">No open positions</p>
            <p className="text-sm text-gray-400 mt-1">Your positions will appear here</p>
          </div>
        )}
      </Card>

      {/* Recent Orders */}
      <Card
        title="Recent Orders"
        action={
          <button
            onClick={fetchData}
            className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold rounded-lg shadow-sm transition-all duration-200"
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        }
      >
        {orders && orders.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-gray-200 bg-gray-50">
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Symbol
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Side
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Qty
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Time
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {orders.map((order) => (
                  <tr key={order.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="font-bold text-slate-900">{order.symbol}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`inline-flex items-center px-3 py-1 rounded-lg text-xs font-semibold ${
                          order.side === 'BUY'
                            ? 'bg-green-100 text-green-700'
                            : 'bg-red-100 text-red-700'
                        }`}
                      >
                        {order.side}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-slate-700">
                      {order.qty}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-slate-700">
                      {order.order_type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`inline-flex items-center px-3 py-1 rounded-lg text-xs font-semibold ${
                          order.status === 'Filled'
                            ? 'bg-green-100 text-green-700'
                            : order.status === 'Cancelled'
                            ? 'bg-gray-100 text-gray-700'
                            : 'bg-blue-100 text-blue-700'
                        }`}
                      >
                        {order.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">
                      {new Date(order.placed_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gray-100 rounded-full mb-4">
              <Activity className="w-8 h-8 text-gray-400" />
            </div>
            <p className="text-gray-500 font-medium">No recent orders</p>
            <p className="text-sm text-gray-400 mt-1">Your order history will appear here</p>
          </div>
        )}
      </Card>

      {/* System Health Overview */}
      {health?.status && (
        <div className={`rounded-xl p-4 flex items-center justify-between border-2 ${
          health.status === 'healthy' 
            ? 'bg-green-50 border-green-200' 
            : health.status === 'degraded'
            ? 'bg-yellow-50 border-yellow-200'
            : health.status === 'unhealthy'
            ? 'bg-red-50 border-red-200'
            : 'bg-gray-50 border-gray-200'
        }`}>
          <div className="flex items-center">
            <div className={`w-3 h-3 rounded-full mr-3 ${
              health.status === 'healthy' 
                ? 'bg-green-500' 
                : health.status === 'degraded'
                ? 'bg-yellow-500'
                : health.status === 'unhealthy'
                ? 'bg-red-500'
                : 'bg-gray-500'
            }`}></div>
            <div>
              <p className={`text-sm font-bold uppercase tracking-wide ${
                health.status === 'healthy' 
                  ? 'text-green-900' 
                  : health.status === 'degraded'
                  ? 'text-yellow-900'
                  : health.status === 'unhealthy'
                  ? 'text-red-900'
                  : 'text-gray-900'
              }`}>
                System Status: {health.status}
              </p>
              <p className={`text-sm ${
                health.status === 'healthy' 
                  ? 'text-green-700' 
                  : health.status === 'degraded'
                  ? 'text-yellow-700'
                  : health.status === 'unhealthy'
                  ? 'text-red-700'
                  : 'text-gray-700'
              }`}>
                {health.message || 'All services operational'}
              </p>
            </div>
          </div>
          {health.summary && (
            <div className="flex items-center space-x-6 text-sm">
              <div className="text-center">
                <p className="font-bold text-slate-900 text-lg">{health.summary.total}</p>
                <p className="text-slate-600 text-xs">Total</p>
              </div>
              <div className="text-center">
                <p className="font-bold text-green-600 text-lg">{health.summary.healthy}</p>
                <p className="text-slate-600 text-xs">Healthy</p>
              </div>
              {health.summary.stale > 0 && (
                <div className="text-center">
                  <p className="font-bold text-yellow-600 text-lg">{health.summary.stale}</p>
                  <p className="text-slate-600 text-xs">Stale</p>
                </div>
              )}
              {health.summary.unhealthy > 0 && (
                <div className="text-center">
                  <p className="font-bold text-red-600 text-lg">{health.summary.unhealthy}</p>
                  <p className="text-slate-600 text-xs">Unhealthy</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Service Health Details */}
      <Card title="Service Health">
        {health?.services && Array.isArray(health.services) ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {health.services.map((service: any) => {
              const isStale = service.is_stale;
              const isHealthy = service.is_healthy;
              const isCritical = service.is_critical;
              
              return (
                <div
                  key={service.service}
                  className={`p-4 border-2 rounded-lg transition-all duration-200 ${
                    isStale
                      ? 'border-yellow-300 bg-yellow-50 hover:border-yellow-400'
                      : isHealthy
                      ? 'border-green-200 bg-white hover:border-green-300'
                      : 'border-red-300 bg-red-50 hover:border-red-400'
                  }`}
                >
                  <div className="flex flex-col">
                    {/* Header with icon and badges */}
                    <div className="flex items-start justify-between mb-3">
                      <div className="relative">
                        <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                          isStale
                            ? 'bg-yellow-100'
                            : isHealthy 
                            ? 'bg-green-100' 
                            : 'bg-red-100'
                        }`}>
                          {isStale ? (
                            <Clock className="w-6 h-6 text-yellow-600" />
                          ) : isHealthy ? (
                            <CheckCircle className="w-6 h-6 text-green-600" />
                          ) : (
                            <XCircle className="w-6 h-6 text-red-600" />
                          )}
                        </div>
                        {isHealthy && !isStale && (
                          <div className="absolute inset-0 w-12 h-12 bg-green-500 rounded-full pulse-ring"></div>
                        )}
                      </div>
                      
                      {/* Critical badge */}
                      {isCritical && (
                        <span className="px-2 py-1 rounded-md text-xs font-bold bg-blue-100 text-blue-700 flex items-center">
                          <Shield className="w-3 h-3 mr-1" />
                          Critical
                        </span>
                      )}
                    </div>
                    
                    {/* Service name */}
                    <p className="font-bold text-slate-900 capitalize mb-2 text-lg">{service.service}</p>
                    
                    {/* Status badge */}
                    <div className="flex items-center justify-between mb-2">
                      <span className={`px-3 py-1 rounded-lg text-xs font-semibold ${
                        isStale
                          ? 'bg-yellow-200 text-yellow-800'
                          : isHealthy
                          ? 'bg-green-100 text-green-700'
                          : 'bg-red-100 text-red-700'
                      }`}>
                        {isStale ? 'Stale' : service.status}
                      </span>
                    </div>
                    
                    {/* Age indicator */}
                    <div className="flex items-center text-xs text-slate-600 mt-2">
                      <Clock className="w-3 h-3 mr-1" />
                      <span>Updated {service.age_seconds}s ago</span>
                    </div>
                    
                    {/* Stale warning */}
                    {isStale && (
                      <div className="mt-3 p-2 bg-yellow-100 rounded-md flex items-start">
                        <AlertTriangle className="w-4 h-4 text-yellow-600 mr-2 mt-0.5 flex-shrink-0" />
                        <p className="text-xs text-yellow-800">
                          Service hasn't updated in over 60 seconds
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : health?.services && typeof health.services === 'object' ? (
          // Fallback for old API format
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(health.services).map(([service, data]: [string, any]) => (
              <div
                key={service}
                className="p-4 border border-gray-200 rounded-lg bg-white hover:border-gray-300 hover:shadow-sm transition-all duration-200"
              >
                <div className="flex flex-col items-center text-center">
                  <div className="relative mb-3">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                      data.status === 'healthy' 
                        ? 'bg-green-100' 
                        : 'bg-red-100'
                    }`}>
                      {data.status === 'healthy' ? (
                        <CheckCircle className="w-6 h-6 text-green-600" />
                      ) : (
                        <XCircle className="w-6 h-6 text-red-600" />
                      )}
                    </div>
                  </div>
                  <p className="font-semibold text-slate-900 capitalize mb-1">{service}</p>
                  <span className={`px-3 py-1 rounded-lg text-xs font-semibold ${
                    data.status === 'healthy'
                      ? 'bg-green-100 text-green-700'
                      : 'bg-red-100 text-red-700'
                  }`}>
                    {data.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gray-100 rounded-full mb-4">
              <AlertCircle className="w-8 h-8 text-gray-400" />
            </div>
            <p className="text-slate-600 font-medium">Unable to fetch service health</p>
            <p className="text-sm text-slate-500 mt-1">Please check your connection</p>
          </div>
        )}
      </Card>
    </div>
  );
}

