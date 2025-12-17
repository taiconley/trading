#!/bin/bash
# Add 48 tickers (24 pairs) to watchlist for Kalman pairs trading strategy
# Each ticker appears exactly once (no duplicates)

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

echo "Adding 48 tickers (24 pairs) to watchlist..."

TICKERS=(
  "PG" "MUSA"        # Pair 1: Sharpe 4.03
  "AMN" "CORT"       # Pair 2: Sharpe 4.03
  "ACA" "KTB"        # Pair 3: Sharpe 3.74
  "FITB" "TMHC"      # Pair 4: Sharpe 3.67
  "D" "SPXC"         # Pair 5: Sharpe 3.27
  "RTX" "AEP"        # Pair 6: Sharpe 4.40
  "ANDE" "LYEL"      # Pair 7: Sharpe 3.87
  "ABR" "GRPN"       # Pair 8: Sharpe 3.68
  "AHCO" "MGY"       # Pair 9: Sharpe 3.56
  "AIN" "ESE"        # Pair 10: Sharpe 3.64
  "ADPT" "IBEX"      # Pair 11: Sharpe 3.50
  "ACT" "DUK"        # Pair 12: Sharpe 4.08
  "ADC" "AFL"        # Pair 13: Sharpe 3.76
  "AMZN" "PFBC"      # Pair 14: Sharpe 3.65
  "ADEA" "PTCT"      # Pair 15: Sharpe 3.79
  "DE" "ENPH"        # Pair 16: Sharpe 3.89
  "ALRM" "NNOX"      # Pair 17: Sharpe 3.71
  "WELL" "PBI"       # Pair 18: Sharpe 3.64
  "ACCO" "WERN"      # Pair 19: Sharpe 3.05
  "ALHC" "ATKR"      # Pair 20: Sharpe 3.95
  "ANIP" "UAA"       # Pair 21: Sharpe 3.83
  "O" "CTRA"         # Pair 22: Sharpe 3.67
  "ETSY" "SHLS"      # Pair 23: Sharpe 3.65
  "MSCI" "PECO"      # Pair 24: Sharpe 3.58
)

for symbol in "${TICKERS[@]}"; do
  echo "Adding $symbol..."
  curl -s -X POST http://localhost:8000/api/watchlist \
    -H "Content-Type: application/json" \
    -d "{\"symbol\": \"$symbol\", \"action\": \"add\"}" | jq -r '.message // .error // "Added"'
  sleep 0.2  # Small delay between requests
done

echo ""
echo "✅ All 48 tickers (24 pairs) added to watchlist!"
echo ""
echo "Pairs for Kalman Filter strategy with NO duplicate tickers:"
echo "  PG/MUSA, AMN/CORT, ACA/KTB, FITB/TMHC, D/SPXC, RTX/AEP,"
echo "  ANDE/LYEL, ABR/GRPN, AHCO/MGY, AIN/ESE, ADPT/IBEX, ACT/DUK,"
echo "  ADC/AFL, AMZN/PFBC, ADEA/PTCT, DE/ENPH, ALRM/NNOX, WELL/PBI,"
echo "  ACCO/WERN, ALHC/ATKR, ANIP/UAA, O/CTRA, ETSY/SHLS, MSCI/PECO"
echo ""
echo "Verify with:"
echo "curl -s http://localhost:8000/api/watchlist | jq"

