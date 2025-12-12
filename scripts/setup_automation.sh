#!/bin/bash
# Setup script for Morning Trading Automation
# 
# This script installs dependencies and configures the automation to run at 6:25 AM PST

set -e  # Exit on error

echo "=========================================="
echo "Morning Trading Automation Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root (needed for systemd installation)
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}Please do not run this script as root.${NC}"
    echo "It will prompt for sudo password when needed."
    exit 1
fi

# Project directory
PROJECT_DIR="/home/taiconley/Desktop/Projects/trading"
cd "$PROJECT_DIR" || exit 1

echo "Step 1: Checking dependencies..."
echo "-------------------------------------------"

# Check if virtual environment exists
VENV_DIR="$PROJECT_DIR/env"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}Virtual environment not found at $VENV_DIR${NC}"
    echo "Please create it first with: python3 -m venv env"
    exit 1
else
    echo -e "${GREEN}✓${NC} Virtual environment found at $VENV_DIR"
fi

# Check if virtual environment has Python
VENV_PYTHON="$VENV_DIR/bin/python3"
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}Python not found in virtual environment${NC}"
    exit 1
else
    echo -e "${GREEN}✓${NC} Python 3 in venv: $($VENV_PYTHON --version)"
fi

# Check pip in virtual environment
VENV_PIP="$VENV_DIR/bin/pip3"
if [ ! -f "$VENV_PIP" ]; then
    echo -e "${RED}pip3 not found in virtual environment${NC}"
    exit 1
else
    echo -e "${GREEN}✓${NC} pip3 in venv: available"
fi

# Install Python dependencies in virtual environment
echo ""
echo "Step 2: Installing Python dependencies in venv..."
echo "-------------------------------------------"
$VENV_PIP install requests || {
    echo -e "${YELLOW}Note: requests package may already be installed${NC}"
}

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠${NC} Docker is not installed or not in PATH"
    echo "   Docker is required for this automation"
else
    echo -e "${GREEN}✓${NC} Docker is installed: $(docker --version)"
fi

# Check Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo -e "${YELLOW}⚠${NC} Docker Compose is not available"
    echo "   Docker Compose is required for this automation"
else
    echo -e "${GREEN}✓${NC} Docker Compose is available"
fi

# Create logs directory
echo ""
echo "Step 3: Creating log directory..."
echo "-------------------------------------------"
mkdir -p "$PROJECT_DIR/logs"
echo -e "${GREEN}✓${NC} Log directory created at $PROJECT_DIR/logs"

# Make scripts executable
echo ""
echo "Step 4: Making scripts executable..."
echo "-------------------------------------------"
chmod +x "$PROJECT_DIR/scripts/morning_automation.py"
chmod +x "$PROJECT_DIR/scripts/start_tws.sh"
echo -e "${GREEN}✓${NC} Scripts are now executable"

# Configuration method selection
echo ""
echo "=========================================="
echo "Step 5: Choose scheduling method"
echo "=========================================="
echo ""
echo "Select how you want to schedule the automation:"
echo "  1) Systemd timer (recommended for modern Linux)"
echo "  2) Crontab (traditional method)"
echo "  3) Skip automation setup (configure manually later)"
echo ""
read -p "Enter your choice (1-3): " scheduling_choice

case $scheduling_choice in
    1)
        echo ""
        echo "Setting up Systemd timer..."
        echo "-------------------------------------------"
        
        # Copy systemd files to user systemd directory
        USER_SYSTEMD_DIR="$HOME/.config/systemd/user"
        mkdir -p "$USER_SYSTEMD_DIR"
        
        cp "$PROJECT_DIR/scripts/trading-morning.service" "$USER_SYSTEMD_DIR/"
        cp "$PROJECT_DIR/scripts/trading-morning.timer" "$USER_SYSTEMD_DIR/"
        
        # Reload systemd
        systemctl --user daemon-reload
        
        # Enable and start the timer
        systemctl --user enable trading-morning.timer
        systemctl --user start trading-morning.timer
        
        echo -e "${GREEN}✓${NC} Systemd timer installed and enabled"
        echo ""
        echo "Useful systemd commands:"
        echo "  - Check timer status: systemctl --user status trading-morning.timer"
        echo "  - Check service logs: journalctl --user -u trading-morning.service"
        echo "  - List all timers: systemctl --user list-timers"
        echo "  - Disable timer: systemctl --user disable trading-morning.timer"
        echo "  - Stop timer: systemctl --user stop trading-morning.timer"
        echo ""
        
        # Show timer info
        echo "Current timer configuration:"
        systemctl --user list-timers trading-morning.timer
        ;;
        
    2)
        echo ""
        echo "Setting up Crontab..."
        echo "-------------------------------------------"
        
        # Create cron job using virtual environment Python
        CRON_JOB="25 6 * * 1-5 $PROJECT_DIR/env/bin/python3 $PROJECT_DIR/scripts/morning_automation.py >> $PROJECT_DIR/logs/morning_automation.log 2>&1"
        
        # Check if cron job already exists
        if crontab -l 2>/dev/null | grep -q "morning_automation.py"; then
            echo -e "${YELLOW}⚠${NC} Cron job already exists. Skipping..."
        else
            # Add cron job
            (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
            echo -e "${GREEN}✓${NC} Cron job added"
        fi
        
        echo ""
        echo "Current crontab entries:"
        crontab -l | grep morning_automation || echo "(none found)"
        echo ""
        echo "Useful cron commands:"
        echo "  - View crontab: crontab -l"
        echo "  - Edit crontab: crontab -e"
        echo "  - Remove all crontab entries: crontab -r"
        echo ""
        ;;
        
    3)
        echo ""
        echo "Skipping automation setup."
        echo "You can manually configure the automation later using:"
        echo "  - Systemd: See $PROJECT_DIR/scripts/trading-morning.service"
        echo "  - Cron: Add a crontab entry to run morning_automation.py"
        echo ""
        ;;
        
    *)
        echo -e "${RED}Invalid choice. Skipping automation setup.${NC}"
        ;;
esac

# TWS Configuration
echo ""
echo "=========================================="
echo "Step 6: TWS Configuration"
echo "=========================================="
echo ""
echo "The automation script will attempt to start TWS automatically."
echo "Please ensure:"
echo "  1. TWS is installed at ~/Desktop/tws/"
echo "  2. TWS is set to Paper Trading mode (already configured)"
echo ""

# Credentials setup
echo "TWS Credentials Setup:"
echo "-------------------------------------------"
CRED_FILE="$PROJECT_DIR/scripts/tws_credentials.env"

if [ -f "$CRED_FILE" ]; then
    echo -e "${GREEN}✓${NC} Credentials file already exists"
    echo "   Location: $CRED_FILE"
else
    echo "For automatic login, you need to set up credentials."
    echo ""
    read -p "Would you like to set up TWS credentials now? (y/n): " setup_creds
    
    if [ "$setup_creds" = "y" ]; then
        "$PROJECT_DIR/scripts/setup_credentials.sh"
        echo ""
        echo "Please edit the credentials file with your IB username and password:"
        echo "  nano $CRED_FILE"
        echo ""
        read -p "Press Enter when you've added your credentials..."
    else
        echo -e "${YELLOW}⚠${NC} Skipping credentials setup"
        echo "   TWS will require manual login"
        echo "   You can set up credentials later by running:"
        echo "   ./scripts/setup_credentials.sh"
    fi
fi

# Test run option
echo ""
echo "=========================================="
echo "Step 7: Test Run (Optional)"
echo "=========================================="
echo ""
read -p "Would you like to do a test run now? (y/n): " do_test

if [ "$do_test" = "y" ]; then
    echo ""
    echo "Running test automation..."
    echo "This will execute all steps (may take 10+ minutes)..."
    echo "-------------------------------------------"
    
    # Run with virtual environment Python
    "$VENV_PYTHON" "$PROJECT_DIR/scripts/morning_automation.py"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Test run completed successfully!"
    else
        echo -e "${RED}✗${NC} Test run failed. Check logs at:"
        echo "    $PROJECT_DIR/logs/morning_automation.log"
    fi
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Scripts installed and configured"
echo "  ✓ Automation scheduled (if selected)"
echo "  ✓ Logs will be written to: $PROJECT_DIR/logs/"
echo ""
echo "The automation will run every weekday at 6:25 AM PST and will:"
echo "  1. Open TWS and sign in (6:25 AM)"
echo "  2. Start Docker containers (6:27 AM)"
echo "  3. Check service health (6:28 AM)"
echo "  4. Ensure strategy is enabled (6:29 AM)"
echo "  5. Trigger warmup (6:30 AM)"
echo "  6. Enable 'Ready to Trade' (6:35 AM)"
echo ""
echo "Manual execution:"
echo "  $PROJECT_DIR/scripts/morning_automation.py"
echo ""
echo "View logs:"
echo "  tail -f $PROJECT_DIR/logs/morning_automation.log"
echo ""
echo -e "${GREEN}Happy Trading!${NC}"
echo ""

