#!/bin/bash
# Test script to verify order placement after market hours

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "   ORDER PLACEMENT TEST (After Hours)"
echo "=========================================="
echo ""

# Check current trading mode
echo "📊 Current Trading Configuration:"
docker compose exec backend-trader printenv | grep -E "DRY_RUN|ENABLE_LIVE|USE_PAPER" | sort
echo ""

# Get account ID
ACCOUNT_ID="DU7084660"  # Default paper account
echo "📋 Using Account: $ACCOUNT_ID"
echo ""

# Test 1: Health check
echo "1️⃣ Testing Trader Service Health..."
HEALTH_RESPONSE=$(docker compose exec -T backend-trader curl -s http://localhost:8004/healthz)
echo "   Response: $HEALTH_RESPONSE"
echo ""

# Test 2: Place a test order via API
echo "2️⃣ Testing Order Placement via API..."
echo "   Placing test order: BUY 10 shares of AAPL @ \$150.00 (LIMIT)"
echo ""

# Get a valid strategy_id from database
STRATEGY_ID=$(docker compose exec -T postgres psql -U bot -d trading -t -c "SELECT strategy_id FROM strategies LIMIT 1;" | xargs)
if [ -z "$STRATEGY_ID" ]; then
    STRATEGY_ID="pairs_trading"  # Fallback to known strategy
fi
echo "   Using strategy_id: $STRATEGY_ID"
echo ""

ORDER_RESPONSE=$(docker compose exec -T backend-trader curl -s -X POST http://localhost:8004/orders \
  -H "Content-Type: application/json" \
  -d "{
    \"symbol\": \"AAPL\",
    \"side\": \"BUY\",
    \"quantity\": 10,
    \"order_type\": \"LMT\",
    \"limit_price\": 150.00,
    \"time_in_force\": \"DAY\",
    \"strategy_id\": \"$STRATEGY_ID\"
  }")

echo "   Response:"
echo "$ORDER_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$ORDER_RESPONSE"
echo ""

# Check if order was created in database
if echo "$ORDER_RESPONSE" | grep -q '"id"'; then
    ORDER_ID=$(echo "$ORDER_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', 'N/A'))" 2>/dev/null || echo "N/A")
    echo "✅ Order created successfully!"
    echo "   Order ID: $ORDER_ID"
    echo ""
    
    # Test 3: Query order from database
    echo "3️⃣ Verifying Order in Database..."
    docker compose exec postgres psql -U bot -d trading -c "
    SELECT 
        id,
        strategy_id,
        symbol,
        side,
        qty,
        order_type,
        limit_price,
        status,
        external_order_id,
        placed_at
    FROM orders 
    WHERE id = $ORDER_ID
    ORDER BY placed_at DESC 
    LIMIT 1;" 2>/dev/null || echo "   Could not query database"
    echo ""
    
    # Test 4: Check order status
    echo "4️⃣ Checking Order Status..."
    STATUS_RESPONSE=$(docker compose exec -T backend-trader curl -s http://localhost:8004/orders/$ORDER_ID)
    echo "$STATUS_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$STATUS_RESPONSE"
    echo ""
    
else
    echo "❌ Order placement failed!"
    echo ""
    
    # Check recent logs for errors
    echo "5️⃣ Checking Recent Error Logs..."
    docker compose logs backend-trader --tail 20 --since 2m | grep -E "(ERROR|Failed|Exception)" | tail -5
    echo ""
fi

# Test 5: Check recent orders in database
echo "6️⃣ Recent Orders in Database (last 5):"
docker compose exec postgres psql -U bot -d trading -c "
SELECT 
    id,
    strategy_id,
    symbol,
    side,
    qty,
    order_type,
    status,
    placed_at
FROM orders 
ORDER BY placed_at DESC 
LIMIT 5;" 2>/dev/null || echo "   Could not query database"

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="

