# Morning Trading Automation

This automation system handles your daily morning trading routine, eliminating manual steps and ensuring consistency every trading day.

## Overview

The automation performs the following steps automatically:

| Time | Step | Description |
|------|------|-------------|
| 6:25 AM | Open TWS | Launches TWS and signs in (Paper Trading mode) |
| 6:27 AM | Start Docker | Runs `docker compose up` to start all services |
| 6:28 AM | Health Check | Verifies all 4 services are running properly |
| 6:29 AM | Enable Strategy | Ensures "Pairs_Trading_Adaptive_Kalman" is enabled |
| 6:30 AM | Warmup | Triggers warmup to load data into cache |
| 6:35 AM | Ready to Trade | Enables "Ready to Trade" flag |

## Quick Start

### 1. Setup (One-time)

```bash
cd /home/taiconley/Desktop/Projects/trading
./scripts/setup_automation.sh
```

The setup script will:
- Verify your Python virtual environment
- Install required dependencies (requests)
- Make scripts executable
- Configure scheduling (systemd or cron)

### 2. Choose Your Scheduling Method

During setup, you'll choose between:

#### Option 1: Systemd Timer (Recommended)
- Modern, reliable scheduling
- Better logging and monitoring
- Automatic restart on failure (if configured)

#### Option 2: Crontab
- Traditional Unix scheduling
- Simple and widely compatible
- Runs as your user

#### Option 3: Manual
- Skip automatic scheduling
- Run manually when needed

## Usage

### Manual Execution

To run the automation manually:

```bash
# Using the wrapper script (easiest)
./scripts/run_morning_automation.sh

# Or directly with virtual environment
source env/bin/activate
./scripts/morning_automation.py
```

### Monitoring

#### View Logs

```bash
# Tail the automation log in real-time
tail -f logs/morning_automation.log

# View today's automation runs
grep "$(date +%Y-%m-%d)" logs/morning_automation.log

# View TWS startup logs
tail -f logs/tws_startup.log
```

#### Check Systemd Status (if using systemd)

```bash
# Check timer status
systemctl --user status trading-morning.timer

# Check service status
systemctl --user status trading-morning.service

# View service logs
journalctl --user -u trading-morning.service

# List all timers
systemctl --user list-timers
```

#### Check Crontab (if using cron)

```bash
# View your crontab
crontab -l

# Edit your crontab
crontab -e
```

## Configuration

### Timing Adjustments

Edit `scripts/morning_automation.py` to adjust wait times:

```python
# Timing configuration (in seconds)
WAIT_AFTER_TWS = 120      # 2 minutes
WAIT_AFTER_DOCKER = 60    # 1 minute
WAIT_AFTER_HEALTH = 60    # 1 minute
WAIT_AFTER_ENABLE = 60    # 1 minute
WAIT_AFTER_WARMUP = 300   # 5 minutes
```

### Strategy Configuration

To change the strategy being automated, edit:

```python
STRATEGY_ID = "Pairs_Trading_Adaptive_Kalman"
```

### TWS Configuration

Edit `scripts/start_tws.sh` to configure TWS:

1. Update `TWS_DIR` to point to your TWS installation
2. Adjust Java memory settings if needed
3. Configure auto-login (optional)

## Systemd Management

### Enable/Disable Timer

```bash
# Enable timer (runs automatically)
systemctl --user enable trading-morning.timer

# Disable timer (stop automatic runs)
systemctl --user disable trading-morning.timer

# Start timer now
systemctl --user start trading-morning.timer

# Stop timer
systemctl --user stop trading-morning.timer
```

### Modify Schedule

Edit `scripts/trading-morning.timer`:

```ini
# Run at 6:25 AM every weekday
OnCalendar=Mon..Fri *-*-* 06:25:00

# Or run at 6:25 AM every day
OnCalendar=*-*-* 06:25:00
```

After editing, reload systemd:

```bash
systemctl --user daemon-reload
systemctl --user restart trading-morning.timer
```

## Crontab Management

### Edit Cron Schedule

```bash
crontab -e
```

The cron job format:
```
25 6 * * 1-5 /path/to/env/bin/python3 /path/to/morning_automation.py >> /path/to/logs/morning_automation.log 2>&1
```

- `25 6 * * 1-5` - 6:25 AM, Monday through Friday
- Change to `25 6 * * *` to run every day (including weekends)

### Remove Cron Job

```bash
crontab -e
# Delete the line containing morning_automation.py
# Save and exit
```

## Troubleshooting

### TWS Not Starting

1. Check TWS installation path in `scripts/start_tws.sh`
2. Verify TWS_DIR points to correct location
3. Check TWS logs: `tail -f logs/tws_startup.log`
4. Ensure X server is available (for GUI)

### Docker Not Starting

1. Check Docker is installed: `docker --version`
2. Verify Docker daemon is running: `sudo systemctl status docker`
3. Check user has Docker permissions: `groups` (should include 'docker')
4. View Docker logs: `docker compose logs`

### Services Not Healthy

1. Check individual service logs:
   ```bash
   docker logs trading-api
   docker logs trading-strategy
   docker logs trading-marketdata
   docker logs trading-historical
   ```

2. Verify TWS connection:
   - TWS is running
   - TWS is on correct port (7497 for paper trading)
   - TWS API is enabled

### Warmup Fails

1. Check strategy service logs:
   ```bash
   docker logs trading-strategy
   ```

2. Verify historical data exists:
   - Check database for candle data
   - Run historical data collection if needed

3. Check warmup timeout in script (may need to increase)

### Strategy Not Found

1. Verify strategy name matches database:
   ```bash
   # From frontend: Go to Strategies tab and check the strategy ID
   # Or query database directly
   ```

2. Update STRATEGY_ID in `morning_automation.py` if needed

## File Structure

```
/home/taiconley/Desktop/Projects/trading/
├── scripts/
│   ├── morning_automation.py          # Main automation script
│   ├── run_morning_automation.sh      # Quick run wrapper
│   ├── start_tws.sh                   # TWS startup script
│   ├── setup_automation.sh            # Setup/installation script
│   ├── trading-morning.service        # Systemd service file
│   └── trading-morning.timer          # Systemd timer file
├── logs/
│   ├── morning_automation.log         # Automation execution logs
│   └── tws_startup.log                # TWS startup logs
└── env/                               # Python virtual environment
    └── bin/
        └── python3                    # Python interpreter
```

## Dependencies

- **Python 3**: Required for automation scripts
- **Virtual Environment**: `env/` directory with activated venv
- **requests**: Python library for HTTP requests
- **Docker & Docker Compose**: For running services
- **TWS/IB Gateway**: Interactive Brokers trading platform
- **systemd** or **cron**: For scheduling (Linux)

## API Endpoints Used

The automation interacts with these backend endpoints:

- `GET /api/health` - Check service health
- `GET /api/strategies` - List strategies
- `POST /api/strategies/{id}/enable` - Enable strategy
- `POST /api/strategies/{id}/warmup` - Trigger warmup
- `POST /api/strategies/{id}/ready-to-trade` - Enable trading

## Safety Features

1. **Health Checks**: Verifies all services are running before proceeding
2. **Strategy Verification**: Confirms strategy exists before enabling
3. **Gradual Warmup**: Waits for warmup completion before enabling trading
4. **Comprehensive Logging**: All steps logged with timestamps
5. **Error Handling**: Catches and logs errors without crashing
6. **Weekday Only**: Default schedule runs Monday-Friday only

## Testing

### Dry Run Test

Before scheduling, test the automation:

```bash
# Run the full automation (takes ~10 minutes)
./scripts/run_morning_automation.sh

# Monitor progress in another terminal
tail -f logs/morning_automation.log
```

### Test Individual Components

```bash
# Test TWS startup
./scripts/start_tws.sh

# Test Docker startup
docker compose up -d

# Test API connectivity
curl http://localhost:8000/api/health
```

## Uninstalling

### Remove Systemd Timer

```bash
systemctl --user stop trading-morning.timer
systemctl --user disable trading-morning.timer
rm ~/.config/systemd/user/trading-morning.service
rm ~/.config/systemd/user/trading-morning.timer
systemctl --user daemon-reload
```

### Remove Cron Job

```bash
crontab -e
# Delete the line containing morning_automation.py
```

### Remove Scripts

```bash
# Optionally remove automation scripts
rm scripts/morning_automation.py
rm scripts/start_tws.sh
rm scripts/run_morning_automation.sh
rm scripts/setup_automation.sh
rm scripts/trading-morning.*
```

## Support

For issues or questions:
1. Check logs: `logs/morning_automation.log`
2. Review service health: Frontend Overview tab
3. Check Docker status: `docker compose ps`
4. Verify TWS connection: TWS interface

## Future Enhancements

Potential improvements:
- [ ] SMS/Email notifications on completion or failure
- [ ] Integration with system wake-up (wake computer at 6:20 AM)
- [ ] Multi-strategy support
- [ ] Web dashboard for automation status
- [ ] Automatic retry on failure
- [ ] Pre-flight checks before market open
- [ ] Post-market shutdown automation

---

**Happy Automated Trading!** 🚀📈

