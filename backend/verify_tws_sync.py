#!/usr/bin/env python3
"""
Verification script to check TWS websocket connections and database sync status.

Usage:
    docker compose exec backend-api python /app/verify_tws_sync.py
"""

import sys
import os
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, '/app/src')

from common.db import get_db_session
from common.models import Order, Position
from sqlalchemy import desc, func


async def check_trader_service_health():
    """Check trader service health and TWS connection"""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://backend-trader:8004/healthz")
            if response.status_code == 200:
                health = response.json()
                # Trader service returns details nested
                details = health.get("details", health)  # Fallback to root if no details
                return {
                    "service": "trader",
                    "status": health.get("status", "unknown"),
                    "tws_connected": details.get("tws_connected", False),
                    "client_id": details.get("client_id"),
                    "active_orders": details.get("active_orders", 0),
                    "trading_mode": details.get("trading_mode", "unknown")
                }
    except Exception as e:
        return {
            "service": "trader",
            "status": "error",
            "error": str(e)
        }


async def check_account_service_health():
    """Check account service health and TWS connection"""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://backend-account:8001/healthz")
            if response.status_code == 200:
                health = response.json()
                return {
                    "service": "account",
                    "status": health.get("status", "unknown"),
                    "tws_connected": health.get("tws_connected", False),
                    "account": health.get("account"),
                    "last_update": health.get("stats", {}).get("last_update")
                }
    except Exception as e:
        return {
            "service": "account",
            "status": "error",
            "error": str(e)
        }


def check_database_orders():
    """Check orders in database and their statuses"""
    with get_db_session() as session:
        # Get all orders
        all_orders = session.query(Order).order_by(desc(Order.placed_at)).limit(100).all()
        
        # Count by status
        status_counts = {}
        active_orders = []
        inactive_orders = []
        
        for order in all_orders:
            status = order.status
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if status in ['PendingSubmit', 'PendingCancel', 'Submitted']:
                active_orders.append({
                    "id": order.id,
                    "external_order_id": order.external_order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": float(order.qty),
                    "status": order.status,
                    "placed_at": order.placed_at.isoformat(),
                    "updated_at": order.updated_at.isoformat()
                })
            else:
                inactive_orders.append({
                    "id": order.id,
                    "external_order_id": order.external_order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": float(order.qty),
                    "status": order.status,
                    "placed_at": order.placed_at.isoformat(),
                    "updated_at": order.updated_at.isoformat()
                })
        
        # Get stale orders (PreSubmitted or Inactive that haven't updated in 5 minutes)
        stale_threshold = datetime.now(timezone.utc).timestamp() - 300  # 5 minutes
        stale_orders = []
        for order in all_orders:
            if order.status in ['PreSubmitted', 'Inactive']:
                if order.updated_at.timestamp() < stale_threshold:
                    stale_orders.append({
                        "id": order.id,
                        "external_order_id": order.external_order_id,
                        "symbol": order.symbol,
                        "status": order.status,
                        "updated_at": order.updated_at.isoformat(),
                        "age_seconds": int(datetime.now(timezone.utc).timestamp() - order.updated_at.timestamp())
                    })
        
        return {
            "total_orders": len(all_orders),
            "status_counts": status_counts,
            "active_orders_count": len(active_orders),
            "inactive_orders_count": len(inactive_orders),
            "stale_orders_count": len(stale_orders),
            "active_orders": active_orders[:10],  # Show first 10
            "stale_orders": stale_orders[:10]  # Show first 10
        }


def check_database_positions():
    """Check positions in database"""
    with get_db_session() as session:
        # Get latest position for each symbol using window function approach
        # First, get all positions ordered by symbol and timestamp
        all_positions = session.query(Position).order_by(
            Position.symbol, desc(Position.ts)
        ).all()
        
        # Group by symbol and take the latest for each
        positions_by_symbol = {}
        for pos in all_positions:
            if pos.symbol not in positions_by_symbol:
                positions_by_symbol[pos.symbol] = pos
        
        positions = []
        for symbol, pos in positions_by_symbol.items():
            positions.append({
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "avg_price": float(pos.avg_price),
                "last_update": pos.ts.isoformat()
            })
        
        # Sort by symbol
        positions.sort(key=lambda x: x['symbol'])
        
        return {
            "total_positions": len(positions),
            "positions": positions
        }


async def main():
    """Main verification function"""
    print("=" * 80)
    print("TWS Connection & Database Sync Verification")
    print("=" * 80)
    print()
    
    # Check trader service (orders)
    print("📋 Checking Trader Service (Orders)...")
    trader_health = await check_trader_service_health()
    print(f"   Status: {trader_health.get('status', 'unknown')}")
    print(f"   TWS Connected: {trader_health.get('tws_connected', False)}")
    if 'client_id' in trader_health:
        print(f"   Client ID: {trader_health.get('client_id')}")
    if 'active_orders' in trader_health:
        print(f"   Active Orders (in memory): {trader_health.get('active_orders', 0)}")
    if 'error' in trader_health:
        print(f"   ❌ Error: {trader_health['error']}")
    print()
    
    # Check account service (positions)
    print("💼 Checking Account Service (Positions)...")
    account_health = await check_account_service_health()
    print(f"   Status: {account_health.get('status', 'unknown')}")
    print(f"   TWS Connected: {account_health.get('tws_connected', False)}")
    if 'account' in account_health:
        print(f"   Account: {account_health.get('account')}")
    if 'error' in account_health:
        print(f"   ❌ Error: {account_health['error']}")
    print()
    
    # Check database orders
    print("📊 Checking Database Orders...")
    orders_info = check_database_orders()
    print(f"   Total Orders: {orders_info['total_orders']}")
    print(f"   Active Orders: {orders_info['active_orders_count']}")
    print(f"   Inactive Orders: {orders_info['inactive_orders_count']}")
    print(f"   Stale Orders (PreSubmitted/Inactive > 5min old): {orders_info['stale_orders_count']}")
    print(f"   Status Breakdown:")
    for status, count in orders_info['status_counts'].items():
        print(f"      {status}: {count}")
    
    if orders_info['active_orders']:
        print(f"\n   Active Orders (showing first {len(orders_info['active_orders'])}):")
        for order in orders_info['active_orders']:
            print(f"      [{order['id']}] {order['symbol']} {order['side']} {order['qty']} @ {order['status']} (ext_id: {order['external_order_id']})")
    
    if orders_info['stale_orders']:
        print(f"\n   ⚠️  Stale Orders (showing first {len(orders_info['stale_orders'])}):")
        for order in orders_info['stale_orders']:
            print(f"      [{order['id']}] {order['symbol']} @ {order['status']} (age: {order['age_seconds']}s, ext_id: {order['external_order_id']})")
            print(f"          Last updated: {order['updated_at']}")
    print()
    
    # Check database positions
    print("📈 Checking Database Positions...")
    positions_info = check_database_positions()
    print(f"   Total Positions: {positions_info['total_positions']}")
    if positions_info['positions']:
        print(f"   Positions:")
        for pos in positions_info['positions']:
            print(f"      {pos['symbol']}: {pos['qty']} @ ${pos['avg_price']:.2f} (updated: {pos['last_update']})")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    issues = []
    if not trader_health.get('tws_connected', False):
        issues.append("❌ Trader service not connected to TWS")
    if not account_health.get('tws_connected', False):
        issues.append("❌ Account service not connected to TWS")
    if orders_info['stale_orders_count'] > 0:
        issues.append(f"⚠️  {orders_info['stale_orders_count']} stale orders in database (may be out of sync with TWS)")
    
    if issues:
        print("Issues Found:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ All systems appear healthy!")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

