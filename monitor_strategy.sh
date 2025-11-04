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
    
    # Try to get strategy state if available
    if 'state_details' in s:
        print(f\"  State Details: {json.dumps(s['state_details'], indent=4)}\")
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
echo ""

echo "📊 BAR COUNTS PER SYMBOL (5 secs timeframe):"
docker compose exec postgres psql -U bot -d trading -c "
SELECT 
    symbol,
    COUNT(*) as bar_count,
    MIN(ts) as first_bar,
    MAX(ts) as last_bar,
    CASE 
        WHEN COUNT(*) >= 240 THEN '✅ Ready'
        WHEN COUNT(*) >= 120 THEN '⚠️  Warming (hedge)'
        ELSE '❌ Insufficient'
    END as status
FROM candles 
WHERE tf = '5 secs'
GROUP BY symbol
ORDER BY bar_count DESC
LIMIT 20;" 2>/dev/null || echo "  Error querying bar counts"

echo ""
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
echo ""

echo "🔍 STRATEGY EXECUTION LOGS (last 60 seconds):"
echo "--- Recent Activity ---"
docker compose logs backend-strategy --tail 100 --since 60s | grep -E "(pairs_trading|WARMING UP|Z-SCORE|ENTRY|EXIT|Failed|stationarity|ADF|cointegration|SKIP)" | tail -20

echo ""
echo "--- Processing Status ---"
docker compose logs backend-strategy --tail 50 --since 60s | grep -E "(DEBUG exec_loop|DEBUG multi_symbol|Received.*signals)" | tail -10

echo ""
echo ""

echo "📊 PAIR STATUS (from logs - last 5 minutes):"
docker compose logs backend-strategy --tail 200 --since 5m | grep -E "\[WARMING UP\]|\[Z-SCORE\]|\[STATIONARITY\]|\[SKIP\]" | tail -15 || echo "  No pair status logs found (may still be warming up)"

echo ""
echo ""

echo "⚠️  ERRORS/WARNINGS (last 5 minutes):"
docker compose logs backend-strategy --tail 200 --since 5m | grep -E "(ERROR|WARNING|Failed|Exception|Traceback)" | tail -10 || echo "  No errors found"

echo ""
echo "========================================"
echo "Refresh this script to see updates"
echo "========================================"
