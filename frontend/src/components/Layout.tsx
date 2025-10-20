import React, { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  TrendingUp,
  Zap,
  Activity,
  BarChart3,
  Sparkles,
} from 'lucide-react';

interface LayoutProps {
  children: ReactNode;
}

const navigation = [
  { name: 'Overview', href: '/', icon: LayoutDashboard },
  { name: 'Market Data', href: '/market', icon: TrendingUp },
  { name: 'Strategies', href: '/strategies', icon: Zap },
  { name: 'Backtests', href: '/backtests', icon: Activity },
  { name: 'Optimizer', href: '/optimizer', icon: BarChart3 },
];

export function Layout({ children }: LayoutProps) {
  const location = useLocation();

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-72 bg-slate-900 border-r border-slate-800 flex flex-col shadow-xl">
        {/* Logo */}
        <div className="p-6 border-b border-slate-800">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 p-3 rounded-xl shadow-lg">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">TradingBot</h1>
              <p className="text-sm text-slate-400">Professional Trading</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2 custom-scrollbar overflow-y-auto">
          {navigation.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`group flex items-center px-4 py-3 rounded-lg transition-all duration-200 ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                }`}
              >
                <Icon className="w-5 h-5 mr-3" />
                <span className="font-medium">{item.name}</span>
                {isActive && (
                  <div className="ml-auto w-2 h-2 rounded-full bg-white" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="p-6 border-t border-slate-800">
          <div className="flex items-center space-x-3 p-3 rounded-lg bg-slate-800">
            <div className="relative">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <div className="absolute inset-0 w-3 h-3 bg-green-500 rounded-full pulse-ring"></div>
            </div>
            <div>
              <p className="text-sm font-medium text-white">System Online</p>
              <p className="text-xs text-slate-400">All services operational</p>
            </div>
          </div>
          <p className="text-xs text-slate-500 text-center mt-4">v1.0.0 • © 2025</p>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 shadow-sm">
          <div className="px-8 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-slate-900">
                  {navigation.find((item) => item.href === location.pathname)?.name ||
                    'Trading Dashboard'}
                </h2>
                <p className="text-sm text-slate-600 mt-1">
                  Real-time market insights and analytics
                </p>
              </div>
              <div className="flex items-center space-x-3">
                <div className="px-4 py-2 bg-blue-50 rounded-lg border border-blue-200">
                  <span className="text-sm font-semibold text-blue-700">Paper Trading</span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-8 custom-scrollbar">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

