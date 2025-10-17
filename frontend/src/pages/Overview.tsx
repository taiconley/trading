import React, { useEffect, useState } from 'react';
import { Card } from '../components/Card';
import { api, Order, Position } from '../services/api';
import { RefreshCw, TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';

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
            <div>
              <p className="text-sm text-gray-500">Net Liquidation</p>
              <p className="text-2xl font-bold text-gray-900">
                ${account?.net_liquidation?.toLocaleString() || '0.00'}
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-green-600" />
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Available Funds</p>
              <p className="text-2xl font-bold text-gray-900">
                ${account?.available_funds?.toLocaleString() || '0.00'}
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-600" />
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Buying Power</p>
              <p className="text-2xl font-bold text-gray-900">
                ${account?.buying_power?.toLocaleString() || '0.00'}
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-purple-600" />
          </div>
        </Card>
      </div>

      {/* Positions */}
      <Card title="Current Positions">
        {account?.positions && account.positions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Symbol
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Quantity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Avg Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Market Value
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Unrealized P&L
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {account.positions.map((position: Position, idx: number) => (
                  <tr key={idx}>
                    <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                      {position.symbol}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                      {position.qty}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                      ${position.avg_price?.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                      ${position.market_value?.toLocaleString()}
                    </td>
                    <td
                      className={`px-6 py-4 whitespace-nowrap font-medium ${
                        (position.unrealized_pnl || 0) >= 0
                          ? 'text-green-600'
                          : 'text-red-600'
                      }`}
                    >
                      ${position.unrealized_pnl?.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">No open positions</p>
        )}
      </Card>

      {/* Recent Orders */}
      <Card
        title="Recent Orders"
        action={
          <button
            onClick={fetchData}
            className="text-blue-600 hover:text-blue-700 text-sm font-medium flex items-center"
          >
            <RefreshCw className="w-4 h-4 mr-1" />
            Refresh
          </button>
        }
      >
        {orders && orders.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Symbol
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Side
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Qty
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Time
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {orders.map((order) => (
                  <tr key={order.id}>
                    <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900">
                      {order.symbol}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded ${
                          order.side === 'BUY'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                        }`}
                      >
                        {order.side}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                      {order.qty}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-gray-600">
                      {order.order_type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded ${
                          order.status === 'Filled'
                            ? 'bg-green-100 text-green-800'
                            : order.status === 'Cancelled'
                            ? 'bg-gray-100 text-gray-800'
                            : 'bg-blue-100 text-blue-800'
                        }`}
                      >
                        {order.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(order.placed_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">No recent orders</p>
        )}
      </Card>

      {/* Service Health */}
      <Card title="Service Health">
        {health?.services ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(health.services).map(([service, data]: [string, any]) => (
              <div
                key={service}
                className="p-4 border border-gray-200 rounded-lg flex items-center justify-between"
              >
                <div>
                  <p className="font-medium text-gray-900 capitalize">{service}</p>
                  <p className="text-sm text-gray-500">{data.status}</p>
                </div>
                <div
                  className={`w-3 h-3 rounded-full ${
                    data.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                  }`}
                />
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">
            Unable to fetch service health
          </p>
        )}
      </Card>
    </div>
  );
}

