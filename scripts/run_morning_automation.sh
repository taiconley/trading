#!/bin/bash
# Quick wrapper script to run morning automation with virtual environment
# This makes it easier to manually trigger the automation

PROJECT_DIR="/home/taiconley/Desktop/Projects/trading"
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment and run the script
source "$PROJECT_DIR/env/bin/activate"
python3 "$PROJECT_DIR/scripts/morning_automation.py"
exit_code=$?

deactivate
exit $exit_code

