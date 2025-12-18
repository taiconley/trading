#!/bin/bash
# TWS Startup Script
# 
# This script launches Interactive Brokers TWS (Trader Workstation) in Paper Trading mode
# and automatically logs in using credentials from tws_credentials.env

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TWS_DIR="$HOME/Desktop/tws"  # TWS installation directory
TWS_LAUNCHER="$TWS_DIR/tws"  # TWS launcher script
TWS_DATA_DIR="$HOME/Jts"     # TWS data and configuration directory
TWS_LOG_DIR="$HOME/Desktop/Projects/trading/logs"
TWS_LOG_FILE="$TWS_LOG_DIR/tws_startup.log"
CREDENTIALS_FILE="$SCRIPT_DIR/tws_credentials.env"

# Function to automate TWS login using xdotool
auto_login_tws() {
    # Wait for TWS login window to appear (longer wait for GUI)
    echo "  Waiting for TWS login window to load..."
    sleep 10
    
    # Try to find the TWS window
    local max_attempts=20
    local attempt=0
    local window_id=""
    
    while [ $attempt -lt $max_attempts ]; do
        # Try different window title patterns
        window_id=$(xdotool search --name "Login" 2>/dev/null | head -1)
        if [ -z "$window_id" ]; then
            window_id=$(xdotool search --name "IB Gateway" 2>/dev/null | head -1)
        fi
        if [ -z "$window_id" ]; then
            window_id=$(xdotool search --name "Trader Workstation" 2>/dev/null | head -1)
        fi
        if [ -z "$window_id" ]; then
            window_id=$(xdotool search --name "TWS" 2>/dev/null | head -1)
        fi
        
        if [ -n "$window_id" ]; then
            echo "  Found TWS window (ID: $window_id)"
            break
        fi
        
        echo "  Still waiting for login window... (attempt $((attempt+1))/$max_attempts)"
        sleep 3
        ((attempt++))
    done
    
    if [ -z "$window_id" ]; then
        echo "  Could not find TWS login window after $max_attempts attempts"
        echo "  The window may have a different title or may require manual login"
        return 1
    fi
    
    # Wait a bit more for window to be fully ready
    echo "  Waiting for window to be ready..."
    sleep 3
    
    # Get window position and size
    eval $(xdotool getwindowgeometry --shell "$window_id")
    
    # Calculate absolute coordinates for the username field
    # Typically username field is around 200px from left, 150px from top of window
    username_x=$((X + 200))
    username_y=$((Y + 150))
    password_y=$((Y + 190))  # Password field is usually ~40px below username
    
    # Bring window to front and activate it
    xdotool windowactivate --sync "$window_id" 2>/dev/null
    sleep 2
    
    echo "  Clicking username field at absolute position ($username_x, $username_y)..."
    # Use absolute coordinates (works better with Java apps)
    xdotool mousemove "$username_x" "$username_y"
    sleep 0.3
    xdotool click 1
    sleep 0.5
    
    # Triple-click to select all text in field
    xdotool click 1
    sleep 0.1
    xdotool click 1
    sleep 0.3
    
    echo "  Typing username..."
    # Type without window flag (types at current focus, which should be username field)
    xdotool type --delay 150 "$TWS_USERNAME"
    sleep 1
    
    echo "  Moving to password field..."
    # Use Tab key to move to password field (more reliable than clicking)
    xdotool key Tab
    sleep 1
    
    echo "  Typing password..."
    # Type password
    xdotool type --delay 150 "$TWS_PASSWORD"
    sleep 1
    
    echo "  Submitting login..."
    # Press Enter to submit
    xdotool key Return
    
    echo "  Login credentials submitted - waiting for authentication..."
    
    # Wait for login to process
    sleep 5
    
    # Handle Paper Trading confirmation dialog
    echo "  Checking for Paper Trading confirmation dialog..."
    local confirm_attempts=0
    local confirm_max_attempts=15
    local paper_dialog_id=""
    
    while [ $confirm_attempts -lt $confirm_max_attempts ]; do
        # Look for the paper trading confirmation dialog
        paper_dialog_id=$(xdotool search --name "paper" 2>/dev/null | head -1)
        if [ -z "$paper_dialog_id" ]; then
            paper_dialog_id=$(xdotool search --name "brokerage account" 2>/dev/null | head -1)
        fi
        if [ -z "$paper_dialog_id" ]; then
            paper_dialog_id=$(xdotool search --name "simulated trading" 2>/dev/null | head -1)
        fi
        
        if [ -n "$paper_dialog_id" ]; then
            echo "  Found Paper Trading confirmation dialog (ID: $paper_dialog_id)"
            
            # Activate the dialog
            xdotool windowactivate --sync "$paper_dialog_id" 2>/dev/null
            sleep 1
            
            echo "  Accepting Paper Trading terms..."
            
            # Method 1: Try Tab to focus button, then Enter
            xdotool key Tab
            sleep 0.5
            xdotool key Return
            sleep 1
            
            # Check if dialog is still there
            if xdotool search --name "paper" >/dev/null 2>&1; then
                # Method 2: Try Space key instead
                echo "  Trying alternate method..."
                xdotool windowactivate --sync "$paper_dialog_id" 2>/dev/null
                sleep 0.5
                xdotool key Tab
                sleep 0.3
                xdotool key space
                sleep 1
            fi
            
            # Check again
            if xdotool search --name "paper" >/dev/null 2>&1; then
                # Method 3: Click at multiple common button positions
                echo "  Trying click method..."
                eval $(xdotool getwindowgeometry --shell "$paper_dialog_id")
                
                # Try clicking at bottom center
                button_x=$((X + WIDTH / 2))
                button_y=$((Y + HEIGHT - 40))
                
                xdotool mousemove "$button_x" "$button_y"
                sleep 0.3
                xdotool click 1
                sleep 1
                
                # If still there, try bottom right area
                if xdotool search --name "paper" >/dev/null 2>&1; then
                    button_x=$((X + WIDTH - 120))
                    xdotool mousemove "$button_x" "$button_y"
                    sleep 0.3
                    xdotool click 1
                    sleep 1
                fi
            fi
            
            # Final verification
            if ! xdotool search --name "paper" >/dev/null 2>&1; then
                echo "  ✓ Paper Trading confirmation accepted"
            else
                echo "  ⚠ Paper Trading dialog still present - may need manual click"
            fi
            
            break
        fi
        
        sleep 2
        ((confirm_attempts++))
    done
    
    if [ -z "$paper_dialog_id" ]; then
        echo "  No Paper Trading confirmation dialog found (may have already been accepted)"
    fi
    
    return 0
}

# Create log directory if it doesn't exist
mkdir -p "$TWS_LOG_DIR"

# Log start time
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting TWS..." >> "$TWS_LOG_FILE"

# Check X11 display availability
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DISPLAY not set, defaulting to :1" >> "$TWS_LOG_FILE"
fi

# Verify X server is accessible
if ! xdpyinfo >/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Cannot connect to X server at $DISPLAY" >> "$TWS_LOG_FILE"
    echo "ERROR: Cannot connect to X server at $DISPLAY"
    echo "Make sure you are logged in to a graphical session and X server is running."
    echo "Try running: xhost +local: (to allow local connections)"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] X server connection verified (DISPLAY=$DISPLAY)" >> "$TWS_LOG_FILE"

# Set XAUTHORITY if not set
if [ -z "$XAUTHORITY" ]; then
    # Try GDM location first (most common on modern systems)
    if [ -f "/run/user/$(id -u)/gdm/Xauthority" ]; then
        export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
    elif [ -f "$HOME/.Xauthority" ]; then
        export XAUTHORITY="$HOME/.Xauthority"
    else
        export XAUTHORITY="$HOME/.Xauthority"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] XAUTHORITY not set, using $XAUTHORITY" >> "$TWS_LOG_FILE"
fi

# Load credentials if available
if [ -f "$CREDENTIALS_FILE" ]; then
    source "$CREDENTIALS_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loaded credentials from $CREDENTIALS_FILE" >> "$TWS_LOG_FILE"
    AUTO_LOGIN=true
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No credentials file found at $CREDENTIALS_FILE" >> "$TWS_LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] TWS will require manual login" >> "$TWS_LOG_FILE"
    AUTO_LOGIN=false
fi

# Check if TWS is already running (look for the Java process)
if pgrep -f "java.*trader.*workstation\|java.*Desktop/tws" > /dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] TWS is already running" >> "$TWS_LOG_FILE"
    echo "TWS is already running"
    exit 0
fi

# Check if TWS launcher exists
if [ ! -f "$TWS_LAUNCHER" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: TWS launcher not found at $TWS_LAUNCHER" >> "$TWS_LOG_FILE"
    echo "Please update the TWS_DIR variable in this script to point to your TWS installation"
    exit 1
fi

# Launch TWS
# The launcher script handles Java settings and configuration
cd "$TWS_DIR" || exit 1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching TWS from $TWS_DIR..." >> "$TWS_LOG_FILE"
echo "Launching TWS..."

# Start TWS (it will use settings from tws.vmoptions and jts.ini in $TWS_DATA_DIR)
# Paper Trading mode is configured in $TWS_DATA_DIR/jts.ini
# Make sure you have Paper Trading selected in TWS before running this automation
# Use setsid to fully detach TWS from the parent process tree
setsid nohup "$TWS_LAUNCHER" >> "$TWS_LOG_FILE" 2>&1 < /dev/null &

LAUNCHER_PID=$!

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TWS launcher started with PID: $LAUNCHER_PID" >> "$TWS_LOG_FILE"

# Wait for TWS Java process to start (the launcher spawns a Java process)
echo "Waiting for TWS to start..."
for i in {1..10}; do
    sleep 2
    if pgrep -f "java.*Desktop/tws" > /dev/null 2>&1; then
        TWS_PID=$(pgrep -f "java.*Desktop/tws" | head -1)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] TWS Java process started with PID: $TWS_PID" >> "$TWS_LOG_FILE"
        echo "✓ TWS started successfully! (PID: $TWS_PID)"
        
        # Attempt automatic login if credentials are available
        if [ "$AUTO_LOGIN" = true ] && [ -n "$TWS_USERNAME" ] && [ -n "$TWS_PASSWORD" ]; then
            echo "  Attempting automatic login..."
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempting automatic login" >> "$TWS_LOG_FILE"
            
            # Call the login automation function
            if auto_login_tws; then
                echo "✓ Automatic login completed"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Automatic login completed" >> "$TWS_LOG_FILE"
            else
                echo "⚠ Automatic login may have failed - please check TWS window"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Automatic login may have failed" >> "$TWS_LOG_FILE"
            fi
        else
            echo "  Please log in to Paper Trading manually..."
        fi
        
        exit 0
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Could not verify TWS Java process, but launcher started" >> "$TWS_LOG_FILE"
echo "TWS launcher started, but Java process not detected yet. Check GUI..."
exit 0

# Optional: Auto-login using xdotool (if credentials are saved in TWS)
# This section can automate the login process if TWS is set to remember credentials
# Uncomment and adjust if needed:

# sleep 10  # Wait for TWS GUI to load
# 
# # Check if xdotool is installed
# if command -v xdotool &> /dev/null; then
#     echo "[$(date '+%Y-%m-%d %H:%M:%S')] Attempting auto-login..." >> "$TWS_LOG_FILE"
#     
#     # Find the TWS window
#     WINDOW_ID=$(xdotool search --name "IB Gateway" 2>/dev/null || xdotool search --name "TWS" 2>/dev/null)
#     
#     if [ -n "$WINDOW_ID" ]; then
#         # Activate the window
#         xdotool windowactivate "$WINDOW_ID"
#         sleep 1
#         
#         # If credentials are saved, just press Enter or click Login
#         # Adjust these commands based on your TWS version and configuration
#         xdotool key Return
#         
#         echo "[$(date '+%Y-%m-%d %H:%M:%S')] Login sequence sent" >> "$TWS_LOG_FILE"
#     else
#         echo "[$(date '+%Y-%m-%d %H:%M:%S')] Could not find TWS window" >> "$TWS_LOG_FILE"
#     fi
# fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TWS startup script completed" >> "$TWS_LOG_FILE"
exit 0

