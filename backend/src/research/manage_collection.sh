#!/bin/bash
# Historical Data Collection Management Script
# 
# This script provides easy commands for managing historical data collection jobs
# Run this from the backend directory: ./src/research/manage_collection.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Change to the backend directory
cd "$(dirname "$0")/../.."

# Default values
SYMBOLS_FILE="src/research/sample_symbols.txt"
BASE_URL="http://localhost:8003"

echo -e "${BLUE}Historical Data Collection Manager${NC}"
echo "=================================="

# Check if Python script exists
if [ ! -f "src/research/historical_data_collector.py" ]; then
    echo -e "${RED}Error: historical_data_collector.py not found${NC}"
    echo "Please run this script from the backend directory"
    exit 1
fi

# Check if historical service is running
echo -e "${YELLOW}Checking if historical data service is running...${NC}"
if ! curl -s "$BASE_URL/healthz" > /dev/null; then
    echo -e "${RED}Error: Historical data service not running at $BASE_URL${NC}"
    echo "Please start your trading bot services first:"
    echo "  docker compose up -d"
    exit 1
fi
echo -e "${GREEN}✓ Historical data service is running${NC}"

# Function to show help
show_help() {
    echo -e "${YELLOW}Available Commands:${NC}"
    echo ""
    echo "1. Create a new job:"
    echo "   ./src/research/manage_collection.sh create-job"
    echo ""
    echo "2. Run/resume a job:"
    echo "   ./src/research/manage_collection.sh run-job JOB_ID"
    echo ""
    echo "3. Pause a job:"
    echo "   ./src/research/manage_collection.sh pause-job JOB_ID"
    echo ""
    echo "4. Resume a paused job:"
    echo "   ./src/research/manage_collection.sh resume-job JOB_ID"
    echo ""
    echo "5. Check job status:"
    echo "   ./src/research/manage_collection.sh status JOB_ID"
    echo ""
    echo "6. List all jobs:"
    echo "   ./src/research/manage_collection.sh list-jobs"
    echo ""
    echo "7. Show this help:"
    echo "   ./src/research/manage_collection.sh help"
}

# Function to create a job interactively
create_job_interactive() {
    echo -e "${YELLOW}Creating a new data collection job...${NC}"
    echo ""
    
    read -p "Job name: " JOB_NAME
    read -p "Symbols file (default: $SYMBOLS_FILE): " INPUT_SYMBOLS_FILE
    SYMBOLS_FILE=${INPUT_SYMBOLS_FILE:-$SYMBOLS_FILE}
    
    if [ ! -f "$SYMBOLS_FILE" ]; then
        echo -e "${RED}Error: Symbols file $SYMBOLS_FILE not found${NC}"
        exit 1
    fi
    
    # Count symbols
    SYMBOL_COUNT=$(grep -v '^#' "$SYMBOLS_FILE" | grep -v '^$' | wc -l)
    echo -e "${BLUE}Found $SYMBOL_COUNT symbols in $SYMBOLS_FILE${NC}"
    
    read -p "Start date (YYYY-MM-DD): " START_DATE
    read -p "End date (YYYY-MM-DD): " END_DATE
    read -p "Bar size (default: 5 secs): " BAR_SIZE
    BAR_SIZE=${BAR_SIZE:-"5 secs"}
    read -p "Description (optional): " DESCRIPTION
    
    # Calculate estimates
    START_DATETIME="${START_DATE}T00:00:00Z"
    END_DATETIME="${END_DATE}T00:00:00Z"
    
    # Calculate days
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        DAYS=$(( ($(date -j -f "%Y-%m-%d" "$END_DATE" +%s) - $(date -j -f "%Y-%m-%d" "$START_DATE" +%s)) / 86400 ))
    else
        # Linux
        DAYS=$(( ($(date -d "$END_DATE" +%s) - $(date -d "$START_DATE" +%s)) / 86400 ))
    fi
    
    REQUESTS_PER_SYMBOL=$((DAYS * 24))  # 1 request per hour
    TOTAL_REQUESTS=$((SYMBOL_COUNT * REQUESTS_PER_SYMBOL))
    ESTIMATED_HOURS=$((TOTAL_REQUESTS / 30 / 60))  # 30 requests per minute
    
    echo ""
    echo -e "${YELLOW}Job Configuration:${NC}"
    echo "  Name: $JOB_NAME"
    echo "  Symbols: $SYMBOL_COUNT"
    echo "  Date range: $START_DATE to $END_DATE ($DAYS days)"
    echo "  Bar size: $BAR_SIZE"
    echo "  Total requests: $TOTAL_REQUESTS"
    echo "  Estimated time: $ESTIMATED_HOURS hours"
    echo ""
    
    read -p "Create this job? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi
    
    # Create the job
        python3 src/research/historical_data_collector.py create-job \
        --name "$JOB_NAME" \
        --symbols-file "$SYMBOLS_FILE" \
        --start-date "$START_DATETIME" \
        --end-date "$END_DATETIME" \
        --bar-size "$BAR_SIZE" \
        --description "$DESCRIPTION"
    
    echo -e "${GREEN}Job created successfully!${NC}"
}

# Main command handling
case "$1" in
    "create-job")
        create_job_interactive
        ;;
    "run-job")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Job ID required${NC}"
            echo "Usage: $0 run-job JOB_ID"
            exit 1
        fi
        echo -e "${YELLOW}Running job $2...${NC}"
        python3 src/research/historical_data_collector.py run-job --job-id "$2"
        ;;
    "pause-job")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Job ID required${NC}"
            echo "Usage: $0 pause-job JOB_ID"
            exit 1
        fi
        echo -e "${YELLOW}Pausing job $2...${NC}"
        python3 src/research/historical_data_collector.py pause-job --job-id "$2"
        ;;
    "resume-job")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Job ID required${NC}"
            echo "Usage: $0 resume-job JOB_ID"
            exit 1
        fi
        echo -e "${YELLOW}Resuming job $2...${NC}"
        python3 src/research/historical_data_collector.py run-job --job-id "$2"
        ;;
    "status")
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Job ID required${NC}"
            echo "Usage: $0 status JOB_ID"
            exit 1
        fi
        echo -e "${YELLOW}Job $2 status:${NC}"
        python3 src/research/historical_data_collector.py status --job-id "$2"
        ;;
    "list-jobs")
        echo -e "${YELLOW}All jobs:${NC}"
        python3 src/research/historical_data_collector.py list-jobs
        ;;
    "help"|"")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
