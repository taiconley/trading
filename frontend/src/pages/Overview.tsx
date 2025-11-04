import { useEffect, useState } from 'react';
import { Card } from '../components/Card';
import { api, Order, Position } from '../services/api';
import { RefreshCw, TrendingUp, AlertCircle, Activity, CheckCircle, XCircle, Clock, Shield, DollarSign, TrendingDown } from 'lucide-react';

export function Overview() {
  const [loading, setLoading] = useState(true);
  const [account, setAccount] = useState<any>(null);
  const [orders, setOrders] = useState<Order[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      const [accountData, ordersData, healthData] = await Promise.all([
        api.getAccountStats().catch(() => null),
        api.getOrders({ limit: 50 }).catch(() => ({ orders: [] })),
        api.getAggregateHealth().catch(() => null),
      ]);
      
      setAccount(accountData);
      setOrders(ordersData.orders || ordersData);
      setHealth(healthData);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchData();
  }, []);

  // Poll for updates every 2 seconds
  useEffect(() => {
    const pollInterval = setInterval(() => {
      fetchData();
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(pollInterval);
  }, []);

  if (loading && !account) {
    return (
      <div className="flex items-center justify-center h-96">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <Card title="Error">
          <div className="flex items-center gap-2 text-red-600">
            <AlertCircle className="w-5 h-5" />
            <span>{error}</span>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-slate-900">Dashboard Overview</h1>
        <button
          onClick={fetchData}
          className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Account Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card title="Net Liquidation">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-3xl font-bold text-slate-900">
                ${account?.net_liquidation?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
              </p>
              <p className="text-sm text-slate-500 mt-1">Account Value</p>
            </div>
            <div className="p-3 bg-blue-100 rounded-lg">
              <TrendingUp className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </Card>

        <Card title="Available Funds">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-3xl font-bold text-slate-900">
                ${account?.available_funds?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
              </p>
              <p className="text-sm text-slate-500 mt-1">Cash Available</p>
            </div>
            <div className="p-3 bg-green-100 rounded-lg">
              <Activity className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </Card>

        <Card title="Buying Power">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-3xl font-bold text-slate-900">
                ${account?.buying_power?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
              </p>
              <p className="text-sm text-slate-500 mt-1">Margin Available</p>
            </div>
            <div className="p-3 bg-purple-100 rounded-lg">
              <Shield className="w-6 h-6 text-purple-600" />
            </div>
          </div>
        </Card>

        <Card title="Daily P&L">
          <div className="flex items-center justify-between">
            <div>
              <p className={`text-3xl font-bold ${
                account?.pnl?.daily_pnl && account.pnl.daily_pnl >= 0 
                  ? 'text-green-600' 
                  : account?.pnl?.daily_pnl && account.pnl.daily_pnl < 0 
                  ? 'text-red-600' 
                  : 'text-slate-900'
              }`}>
                {account?.pnl?.daily_pnl !== undefined 
                  ? `$${account.pnl.daily_pnl.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                  : '$0.00'}
              </p>
              <p className="text-sm text-slate-500 mt-1">Today's Profit/Loss</p>
            </div>
            <div className={`p-3 rounded-lg ${
              account?.pnl?.daily_pnl && account.pnl.daily_pnl >= 0 
                ? 'bg-green-100' 
                : account?.pnl?.daily_pnl && account.pnl.daily_pnl < 0 
                ? 'bg-red-100' 
                : 'bg-gray-100'
            }`}>
              {account?.pnl?.daily_pnl && account.pnl.daily_pnl >= 0 ? (
                <TrendingUp className={`w-6 h-6 ${account.pnl.daily_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`} />
              ) : (
                <TrendingDown className="w-6 h-6 text-red-600" />
              )}
            </div>
          </div>
        </Card>

        <Card title="Unrealized P&L">
          <div className="flex items-center justify-between">
            <div>
              <p className={`text-3xl font-bold ${
                account?.pnl?.unrealized_pnl && account.pnl.unrealized_pnl >= 0 
                  ? 'text-green-600' 
                  : account?.pnl?.unrealized_pnl && account.pnl.unrealized_pnl < 0 
                  ? 'text-red-600' 
                  : 'text-slate-900'
              }`}>
                {account?.pnl?.unrealized_pnl !== undefined 
                  ? `$${account.pnl.unrealized_pnl.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                  : '$0.00'}
              </p>
              <p className="text-sm text-slate-500 mt-1">Open Positions P&L</p>
            </div>
            <div className={`p-3 rounded-lg ${
              account?.pnl?.unrealized_pnl && account.pnl.unrealized_pnl >= 0 
                ? 'bg-green-100' 
                : account?.pnl?.unrealized_pnl && account.pnl.unrealized_pnl < 0 
                ? 'bg-red-100' 
                : 'bg-gray-100'
            }`}>
              {account?.pnl?.unrealized_pnl && account.pnl.unrealized_pnl >= 0 ? (
                <TrendingUp className="w-6 h-6 text-green-600" />
              ) : (
                <TrendingDown className="w-6 h-6 text-red-600" />
              )}
            </div>
          </div>
        </Card>

        <Card title="Realized P&L">
          <div className="flex items-center justify-between">
            <div>
              <p className={`text-3xl font-bold ${
                account?.pnl?.realized_pnl && account.pnl.realized_pnl >= 0 
                  ? 'text-green-600' 
                  : account?.pnl?.realized_pnl && account.pnl.realized_pnl < 0 
                  ? 'text-red-600' 
                  : 'text-slate-900'
              }`}>
                {account?.pnl?.realized_pnl !== undefined 
                  ? `$${account.pnl.realized_pnl.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                  : '$0.00'}
              </p>
              <p className="text-sm text-slate-500 mt-1">Closed Trades P&L</p>
            </div>
            <div className={`p-3 rounded-lg ${
              account?.pnl?.realized_pnl && account.pnl.realized_pnl >= 0 
                ? 'bg-green-100' 
                : account?.pnl?.realized_pnl && account.pnl.realized_pnl < 0 
                ? 'bg-red-100' 
                : 'bg-gray-100'
            }`}>
              <DollarSign className={`w-6 h-6 ${
                account?.pnl?.realized_pnl && account.pnl.realized_pnl >= 0 
                  ? 'text-green-600' 
                  : 'text-red-600'
              }`} />
            </div>
          </div>
        </Card>
      </div>

      {/* Current Positions */}
      <Card title="Current Positions">
        {account?.positions && account.positions.filter((p: Position) => p.qty !== 0).length > 0 ? (
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
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {account.positions.filter((p: Position) => p.qty !== 0).map((position: Position) => (
                  <tr key={position.symbol} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="font-bold text-slate-900">{position.symbol}</span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-slate-700">
                      {position.qty}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-slate-700">
                      ${position.avg_price.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-slate-700">
                      ${(position.qty * position.avg_price).toFixed(2)}
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
            <p className="text-gray-500 font-medium">No positions</p>
            <p className="text-sm text-gray-400 mt-1">Your open positions will appear here</p>
          </div>
        )}
      </Card>

      {/* Open Orders */}
      <Card title="Open Orders">
        {orders && orders.filter((order: Order) => ['PendingSubmit', 'PendingCancel', 'Submitted'].includes(order.status)).length > 0 ? (
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
                    Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Placed
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {orders.filter((order: Order) => ['PendingSubmit', 'PendingCancel', 'Submitted'].includes(order.status)).map((order) => (
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
                    <td className="px-6 py-4 whitespace-nowrap text-slate-700">
                      {order.limit_price ? `$${parseFloat(String(order.limit_price)).toFixed(2)}` : 'Market'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className="inline-flex items-center px-3 py-1 rounded-lg text-xs font-semibold bg-blue-100 text-blue-700"
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
            <p className="text-gray-500 font-medium">No open orders</p>
            <p className="text-sm text-gray-400 mt-1">Your open orders will appear here</p>
          </div>
        )}
      </Card>

      {/* Recent Orders (All) */}
      <Card title="Recent Orders (All)">
        {orders && orders.length > 0 ? (
          <div className="overflow-x-auto max-h-96 overflow-y-auto border border-gray-200 rounded-lg">
            <table className="min-w-full">
              <thead className="bg-gray-50 sticky top-0 z-10">
                <tr className="border-b border-gray-200">
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
                    Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Placed
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 bg-white">
                {[...orders].sort((a, b) => new Date(b.placed_at).getTime() - new Date(a.placed_at).getTime()).map((order) => (
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
                    <td className="px-6 py-4 whitespace-nowrap text-slate-700">
                      {order.limit_price ? `$${parseFloat(String(order.limit_price)).toFixed(2)}` : 'Market'}
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
              <Clock className="w-8 h-8 text-gray-400" />
            </div>
            <p className="text-gray-500 font-medium">No orders yet</p>
            <p className="text-sm text-gray-400 mt-1">Your order history will appear here</p>
          </div>
        )}
      </Card>

      {/* Service Health */}
      <Card title="Service Health">
        {health?.services && health.services.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {health.services.map((service: any) => (
              <div
                key={service.service}
                className={`p-4 rounded-lg border-2 ${
                  service.status === 'healthy'
                    ? 'border-green-200 bg-green-50'
                    : service.status === 'stale'
                    ? 'border-yellow-200 bg-yellow-50'
                    : 'border-red-200 bg-red-50'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-slate-900 capitalize">{service.service}</h3>
                  {service.status === 'healthy' ? (
                    <CheckCircle className="w-5 h-5 text-green-600" />
                  ) : service.status === 'stale' ? (
                    <AlertCircle className="w-5 h-5 text-yellow-600" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-600" />
                  )}
                </div>
                <p
                  className={`text-sm font-medium ${
                    service.status === 'healthy'
                      ? 'text-green-700'
                      : service.status === 'stale'
                      ? 'text-yellow-700'
                      : 'text-red-700'
                  }`}
                >
                  {service.status === 'healthy' ? 'Operational' : service.status === 'stale' ? 'Stale' : 'Unhealthy'}
                </p>
                {service.last_update && (
                  <p className="text-xs text-slate-500 mt-1">
                    Updated {new Date(service.last_update).toLocaleTimeString()}
                  </p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gray-100 rounded-full mb-4">
              <Activity className="w-8 h-8 text-gray-400" />
            </div>
            <p className="text-gray-500 font-medium">No health data available</p>
          </div>
        )}
      </Card>
    </div>
  );
}
