#!/bin/bash
# Real-time Strategy Monitoring Script

echo "========================================"
echo "   PAIRS TRADING STRATEGY MONITOR"
echo "========================================"
echo ""

echo "🔄 STRATEGY STATUS:"
docker compose exec backend-strategy curl -s http://localhost:8005/strategies | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data['strategies']:
    print(f\"  Strategy: {s['name']} ({s['strategy_id']})\")
    print(f\"  State: {s['state']}\")
    print(f\"  Enabled: {s['enabled']}\")
    print(f\"  Running: {s['running']}\")
    print(f\"  Symbols: {len(s['symbols'])}\")
    m = s['metrics']
    print(f\"  Total Signals: {m['total_signals']}\")
    print(f\"  Successful: {m['successful_signals']}\")
    print(f\"  Total P&L: \${m['total_pnl']:.2f}\")
"
echo ""

echo "📊 MARKET DATA STATUS:"
docker compose exec backend-marketdata curl -s http://localhost:8002/realtime-bars/subscriptions | python3 -c "
import sys, json
data = json.load(sys.stdin)
subs = data['subscriptions']
print(f\"  Active Subscriptions: {len(subs)}\")
print(f\"  Symbols: {', '.join(subs[:5])}... ({len(subs)-5} more)\")
"
echo ""

echo "⏰ LATEST DATA (AAPL sample):"
docker compose exec postgres psql -U bot -d trading -t -c "
SELECT 
    '  Last Bar: ' || ts || ' | Close: $' || close 
FROM candles 
WHERE tf='5 secs' AND symbol='AAPL' 
ORDER BY ts DESC LIMIT 1;"

echo ""

echo "📈 RECENT SIGNALS (last 10):"
docker compose exec postgres psql -U bot -d trading -c "
SELECT 
    strategy_id,
    symbol,
    signal_type,
    strength,
    ts
FROM signals 
ORDER BY ts DESC LIMIT 10;" 2>/dev/null || echo "  No signals yet"

echo ""

echo "🔍 STRATEGY EXECUTION (last 30 seconds):"
docker compose logs backend-strategy --tail 50 --since 30s | grep -E "(DEBUG exec_loop|DEBUG multi_symbol|Received.*signals)" | tail -5

echo ""
echo "========================================"
echo "Refresh this script to see updates"
echo "========================================"

