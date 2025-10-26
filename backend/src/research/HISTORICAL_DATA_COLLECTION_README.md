# Historical Data Collection System

This system manages large-scale historical data collection with database persistence, pause/resume functionality, and progress tracking. It uses your trading bot's historical data service for actual data collection.

## Key Features

- **Database Persistence**: All progress is stored in the database
- **Pause/Resume**: Stop and restart collection jobs at any time
- **Progress Tracking**: Real-time monitoring of collection progress
- **Batch Processing**: Handle thousands of symbols efficiently
- **Error Handling**: Comprehensive error handling and retry logic
- **Fixed Time Frames**: Ensure all batches collect the exact same time period

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Symbol List   │───▶│  Database        │───▶│  Collection     │
│   (File/API)    │    │  (Jobs/Symbols)  │    │  Job Manager    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Progress        │    │  Historical     │
                       │  Tracking        │    │  Data Service   │
                       └─────────────────┘    └─────────────────┘
```

## Database Schema

### `data_collection_jobs`
- Job metadata (name, description, date range, parameters)
- Overall progress tracking (total/completed/failed symbols and requests)
- Status management (pending, running, paused, completed, failed)

### `data_collection_symbols`
- Individual symbol tracking within a job
- Symbol-specific progress and error handling
- Links to individual requests

### `data_collection_requests`
- Individual request tracking (1-hour chunks)
- Request status (pending, queued, completed, failed)
- Historical service request ID linking

## How Pause/Resume Works

### When You Stop the Process:

1. **Graceful Stop (Ctrl+C)**:
   - Current symbol processing completes
   - Job status remains 'running'
   - All progress is saved to database
   - Next restart will resume from where it left off

2. **Pause Command**:
   - Job status changes to 'paused'
   - All progress is saved
   - Can be resumed later

3. **Ungraceful Stop (kill, crash)**:
   - Job status remains 'running'
   - Database contains all progress up to crash point
   - Resume will continue from last completed symbol

### When You Restart:

1. **Resume Command**:
   - Finds job with status 'running' or 'paused'
   - Loads all progress from database
   - Continues from next incomplete symbol
   - No duplicate requests (database prevents this)

2. **Automatic Resume**:
   - Script automatically detects incomplete jobs
   - Resumes from last checkpoint

## Quick Start

### 1. Create a Collection Job

```bash
# Interactive job creation
./manage_collection.sh create-job

# Or direct command
python3 historical_data_collector.py create-job \
    --name "SP500_Jan2024" \
    --symbols-file "sp500_symbols.txt" \
    --start-date "2024-01-01T00:00:00Z" \
    --end-date "2024-01-11T00:00:00Z" \
    --bar-size "5 secs" \
    --description "SP500 10-day collection"
```

### 2. Run the Collection

```bash
# Run job (will resume if paused)
./manage_collection.sh run-job 1

# Or direct command
python3 historical_data_collector.py run-job --job-id 1
```

### 3. Monitor Progress

```bash
# Check job status
./manage_collection.sh status 1

# List all jobs
./manage_collection.sh list-jobs
```

### 4. Pause/Resume

```bash
# Pause a running job
./manage_collection.sh pause-job 1

# Resume a paused job
./manage_collection.sh run-job 1
```

## Command Reference

### Management Script (`manage_collection.sh`)

```bash
./manage_collection.sh create-job          # Interactive job creation
./manage_collection.sh run-job JOB_ID      # Run/resume a job
./manage_collection.sh pause-job JOB_ID    # Pause a job
./manage_collection.sh status JOB_ID       # Check job status
./manage_collection.sh list-jobs           # List all jobs
./manage_collection.sh help                # Show help
```

### Direct Python Commands

```bash
# Create job
python3 historical_data_collector.py create-job \
    --name "Job Name" \
    --symbols-file "symbols.txt" \
    --start-date "2024-01-01T00:00:00Z" \
    --end-date "2024-01-11T00:00:00Z" \
    --bar-size "5 secs" \
    --description "Optional description"

# Run job
python3 historical_data_collector.py run-job --job-id 1

# Pause job
python3 historical_data_collector.py pause-job --job-id 1

# Check status
python3 historical_data_collector.py status --job-id 1

# List jobs
python3 historical_data_collector.py list-jobs
```

## Example Workflow

### Large Collection (e.g., 2000 symbols, 10 days)

1. **Create Job**:
   ```bash
   ./manage_collection.sh create-job
   # Name: SP500_Jan2024
   # Symbols: sp500_symbols.txt (500 symbols)
   # Dates: 2024-01-01 to 2024-01-11
   ```

2. **Start Collection**:
   ```bash
   ./manage_collection.sh run-job 1
   ```

3. **Monitor Progress**:
   ```bash
   ./manage_collection.sh status 1
   ```

4. **Pause for Maintenance**:
   ```bash
   ./manage_collection.sh pause-job 1
   ```

5. **Resume Later**:
   ```bash
   ./manage_collection.sh run-job 1
   ```

6. **Check Final Status**:
   ```bash
   ./manage_collection.sh status 1
   ```

## File Structure

```
trading/
├── historical_data_collector.py    # Main collection script
├── manage_collection.sh            # Management script
├── sample_symbols.txt              # Sample symbols file
├── historical_data_collection.log  # Collection logs
└── backend/
    ├── migrations/versions/
    │   └── data_collection_tracking_add_data_collection_tables.py
    └── src/common/models.py        # Database models
```

## Prerequisites

1. **Trading Bot Services Running**:
   ```bash
   docker compose up -d
   ```

2. **Database Migration Applied**:
   ```bash
   cd backend
   alembic upgrade head
   ```

3. **Python Dependencies**:
   ```bash
   pip install aiohttp sqlalchemy
   ```

## Configuration

### Symbol File Format

Create a text file with one symbol per line:
```
AAPL
MSFT
GOOGL
AMZN
TSLA
# Comments are ignored
META
NVDA
```

### Date Format

Use ISO 8601 format with timezone:
- `2024-01-01T00:00:00Z` (UTC)
- `2024-01-01T09:30:00Z` (9:30 AM ET = 14:30 UTC)

### Bar Sizes

Supported bar sizes (TWS limits apply):
- `5 secs` (max 1 hour per request)
- `10 secs` (max 4 hours per request)
- `1 min` (max 1 week per request)
- `5 mins` (max 1 month per request)

## Monitoring and Troubleshooting

### Progress Monitoring

The system provides real-time progress updates:

```
Processing symbol 25/500: AAPL
  Request 1: queued
  Request 2: queued
  ...
✅ Completed symbol AAPL

Overall Progress:
  Completed symbols: 25/500
  Completed requests: 6,000/120,000
  Current rate: 2.2 requests/minute
```

### Status Codes

**Job Status**:
- `pending`: Created but not started
- `running`: Currently processing
- `paused`: Manually paused
- `completed`: All symbols processed
- `failed`: Job failed

**Symbol Status**:
- `pending`: Not started
- `running`: Currently processing
- `completed`: All requests completed
- `failed`: Symbol failed

**Request Status**:
- `pending`: Not started
- `queued`: Sent to historical service
- `completed`: Historical service completed
- `failed`: Request failed

### Common Issues

1. **Service Not Running**:
   ```bash
   docker compose ps
   docker compose up -d
   ```

2. **Database Connection Issues**:
   ```bash
   cd backend
   alembic upgrade head
   ```

3. **Pacing Warnings**:
   - Normal behavior - historical service handles pacing
   - Collection will continue automatically

4. **Memory Issues**:
   - Use smaller symbol batches
   - Monitor system resources
   - Pause/resume as needed

## Performance Considerations

### Large Collections

For collections with thousands of symbols:

1. **Batch Processing**: Process symbols in smaller batches
2. **Resource Monitoring**: Monitor CPU, memory, and disk usage
3. **Pause/Resume**: Use pause/resume for maintenance windows
4. **Progress Tracking**: Regular status checks

### Time Estimates

- **5-second bars**: ~240 requests per symbol per day
- **Pacing**: 30 requests per minute (historical service limit)
- **Large collection**: 2000 symbols × 10 days = ~48,000 requests = ~27 hours

### Database Performance

- Indexes on job_id, status, and timestamps
- Automatic cleanup of old completed requests
- Efficient querying for resume functionality

This system ensures reliable, resumable data collection for any scale of historical data needs.
