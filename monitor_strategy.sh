#!/bin/bash
# Real-time Strategy Monitoring Script

echo "========================================"
echo "   PAIRS TRADING STRATEGY MONITOR"
echo "========================================"
echo ""

echo "🔄 STRATEGY STATUS:"
STRATEGY_DATA=$(docker compose exec -T backend-strategy curl -s http://localhost:8005/strategies)
echo "$STRATEGY_DATA" | python3 -c "
import sys, json
data = json.load(sys.stdin)
strategy_symbols = []
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
    
    # Collect symbols for filtering
    strategy_symbols.extend(s.get('symbols', []))
    
    # Try to get strategy state if available
    if 'state_details' in s:
        state = s['state_details']
        if 'pairs_state' in state:
            print(f\"  Pairs: {state.get('num_pairs', 0)}\")

# Output symbols as comma-separated for use in SQL
if strategy_symbols:
    symbols_str = ','.join([f\"'{s}'\" for s in strategy_symbols])
    print(f\"__SYMBOLS__:{symbols_str}\", file=sys.stderr)
"
STRATEGY_SYMBOLS=$(echo "$STRATEGY_DATA" | python3 -c "
import sys, json
data = json.load(sys.stdin)
symbols = []
for s in data['strategies']:
    symbols.extend(s.get('symbols', []))
if symbols:
    print(','.join([f\"'{s}'\" for s in symbols]))
")
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

echo "⏰ LATEST DATA (first strategy symbol):"
FIRST_SYMBOL=$(echo "$STRATEGY_DATA" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data['strategies']:
    symbols = s.get('symbols', [])
    if symbols:
        print(symbols[0])
        break
")
if [ -n "$FIRST_SYMBOL" ]; then
    docker compose exec postgres psql -U bot -d trading -t -c "
    SELECT 
        '  Last Bar: ' || ts || ' | Close: \$' || close 
    FROM candles 
    WHERE tf='5 secs' AND symbol='$FIRST_SYMBOL' 
    ORDER BY ts DESC LIMIT 1;" 2>/dev/null || echo "  No data for $FIRST_SYMBOL"
else
    echo "  No strategy symbols found"
fi

echo ""
echo ""

echo "📊 BAR COUNTS PER SYMBOL (5 secs timeframe - Strategy symbols only):"
if [ -n "$STRATEGY_SYMBOLS" ]; then
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
    WHERE tf = '5 secs' AND symbol IN ($STRATEGY_SYMBOLS)
    GROUP BY symbol
    ORDER BY symbol;" 2>/dev/null || echo "  Error querying bar counts"
else
    echo "  No strategy symbols found to filter"
fi

echo ""
echo ""

echo "📈 RECENT SIGNALS (last 10 - Strategy only, with Z-Scores):"
STRATEGY_ID=$(echo "$STRATEGY_DATA" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data['strategies']:
    print(s.get('strategy_id', ''))
    break
")
if [ -n "$STRATEGY_ID" ]; then
    docker compose exec postgres psql -U bot -d trading -c "
    SELECT 
        strategy_id,
        symbol,
        signal_type,
        strength,
        metadata->>'zscore' as zscore,
        metadata->>'pair' as pair,
        ts
    FROM signals 
    WHERE strategy_id = '$STRATEGY_ID'
    ORDER BY ts DESC LIMIT 10;" 2>/dev/null || echo "  No signals yet"
else
    echo "  No strategy ID found"
fi

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

echo "📊 DETAILED PAIR STATUS (Current Z-Scores & Positions):"
echo "$STRATEGY_DATA" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data['strategies']:
    if 'state_details' in s and 'pairs_state' in s['state_details']:
        pairs_state = s['state_details']['pairs_state']
        config = s['state_details'].get('config', {})
        entry_threshold = config.get('entry_threshold', 1.5)
        exit_threshold = config.get('exit_threshold', 0.4)
        
        print(f\"  Entry Threshold: {entry_threshold} | Exit Threshold: {exit_threshold}\")
        print(f\"  {'Pair':<20} {'Position':<20} {'Z-Score':<12} {'Entry%':<10} {'Exit%':<10} {'Bars':<8} {'Status'}\")
        print(f\"  {'-'*20} {'-'*20} {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*30}\")
        
        for pair_key, state in sorted(pairs_state.items()):
            position = state.get('position', 'flat')
            zscore = state.get('current_zscore')
            entry_prox = state.get('entry_proximity')
            exit_prox = state.get('exit_proximity')
            bars_in_trade = state.get('bars_in_trade', 0)
            cooldown = state.get('cooldown_remaining', 0)
            
            zscore_str = f\"{zscore:.3f}\" if zscore is not None else \"N/A\"
            entry_pct = f\"{entry_prox*100:.1f}%\" if entry_prox is not None else \"N/A\"
            exit_pct = f\"{exit_prox*100:.1f}%\" if exit_prox is not None else \"N/A\"
            
            # Determine status
            status = []
            if position != 'flat':
                status.append('IN TRADE')
                if exit_prox is not None and exit_prox < 0.5:
                    status.append('⚠️  NEAR EXIT')
            else:
                if cooldown > 0:
                    status.append(f'COOLDOWN({cooldown})')
                elif entry_prox is not None:
                    if entry_prox >= 1.0:
                        status.append('✅ READY')
                    elif entry_prox >= 0.8:
                        status.append('🟡 CLOSE')
                    else:
                        status.append('🟢 FAR')
                else:
                    status.append('WARMING')
            
            status_str = ' '.join(status)
            print(f\"  {pair_key:<20} {position:<20} {zscore_str:<12} {entry_pct:<10} {exit_pct:<10} {bars_in_trade:<8} {status_str}\")
        break
" || echo "  No detailed pair state available"

echo ""
echo ""

echo "⚠️  ERRORS/WARNINGS (last 5 minutes):"
docker compose logs backend-strategy --tail 200 --since 5m | grep -E "(ERROR|WARNING|Failed|Exception|Traceback)" | tail -10 || echo "  No errors found"

echo ""
echo "========================================"
echo "Refresh this script to see updates"
echo "========================================"
