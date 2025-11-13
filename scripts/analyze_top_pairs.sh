#!/bin/bash
# Analyze top pairs from the potential_pairs table
#
# Usage:
#   ./scripts/analyze_top_pairs.sh [OPTIONS]
#
# Options:
#   -n, --limit N        Number of top pairs to show (default: 10)
#   -o, --output FILE    Output results to CSV file
#   -q, --quiet          Show only summary table (no detailed output)
#   -h, --help           Show this help message

# Change to project root directory (where compose.yaml is located)
cd "$(dirname "$0")/.." || exit 1

# Default values
LIMIT=100
OUTPUT_CSV="potential_pairs.csv"
QUIET=""
MIN_TRADES=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        -q|--quiet)
            QUIET="--quiet"
            shift
            ;;
        -m|--min-trades)
            MIN_TRADES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Analyze and rank potential pairs from the potential_pairs table"
            echo ""
            echo "Options:"
            echo "  -n, --limit N        Number of top pairs to show (default: 10)"
            echo "  -o, --output FILE    Output results to CSV file"
            echo "  -q, --quiet          Show only summary table (no detailed output)"
            echo "  -m, --min-trades N   Minimum number of trades required (default: 5)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --limit 20 --output top_pairs.csv"
            echo "  $0 -n 15 -o results.csv --quiet"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the command
TEMP_CSV_PATH=""
FINAL_OUTPUT=""
if [ -n "$OUTPUT_CSV" ]; then
    # Determine the final output path on the host
    if [[ "$OUTPUT_CSV" != /* ]]; then
        # Relative path - make it relative to project root
        FINAL_OUTPUT="$(pwd)/$OUTPUT_CSV"
    else
        # Absolute path
        FINAL_OUTPUT="$OUTPUT_CSV"
    fi
    
    # Use a temporary path inside the container
    TEMP_CSV_PATH="/tmp/top_pairs_$$.csv"
    CMD="docker compose exec backend-api python -m src.research.top_pairs_analyzer --limit $LIMIT --min-trades $MIN_TRADES --csv $TEMP_CSV_PATH"
else
    CMD="docker compose exec backend-api python -m src.research.top_pairs_analyzer --limit $LIMIT --min-trades $MIN_TRADES"
fi

# Add quiet flag if specified
if [ -n "$QUIET" ]; then
    CMD="$CMD $QUIET"
fi

# Execute the command
echo "Analyzing top $LIMIT pairs..."
if [ -n "$OUTPUT_CSV" ]; then
    echo "Output will be saved to: $FINAL_OUTPUT"
fi
echo ""

eval $CMD

# Copy CSV from container to host if needed
if [ -n "$OUTPUT_CSV" ] && [ -n "$TEMP_CSV_PATH" ]; then
    echo ""
    echo "Copying CSV file from container..."
    docker cp "trading-api:$TEMP_CSV_PATH" "$FINAL_OUTPUT"
    if [ $? -eq 0 ]; then
        echo "✅ CSV file saved to: $FINAL_OUTPUT"
        # Clean up temp file in container
        docker compose exec backend-api rm -f "$TEMP_CSV_PATH" 2>/dev/null
    else
        echo "❌ Failed to copy CSV file from container"
        exit 1
    fi
fi

