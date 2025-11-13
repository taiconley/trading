#!/bin/bash
# Add 20 tickers (10 pairs) to watchlist

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

echo "Adding 20 tickers to watchlist..."

TICKERS=(
  "AAPL" "MSFT"    # Pair 1: Tech Large Cap
  "JPM" "BAC"      # Pair 3: Banks
  "GS" "MS"        # Pair 4: Investment Banks
  "XOM" "CVX"      # Pair 5: Energy
  "V" "MA"         # Pair 6: Payments
  "KO" "PEP"       # Pair 7: Beverages
  "WMT" "TGT"      # Pair 8: Retail
  "PFE" "MRK"      # Pair 9: Pharma
  "DIS" "NFLX"     # Pair 10: Media
)

for symbol in "${TICKERS[@]}"; do
  echo "Adding $symbol..."
  curl -s -X POST http://localhost:8000/api/watchlist \
    -H "Content-Type: application/json" \
    -d "{\"symbol\": \"$symbol\", \"action\": \"add\"}" | jq -r '.message // .error // "Added"'
  sleep 0.2  # Small delay between requests
done

echo ""
echo "✅ All 20 tickers added to watchlist!"
echo ""
echo "Verify with:"
echo "curl -s http://localhost:8000/api/watchlist | jq"

