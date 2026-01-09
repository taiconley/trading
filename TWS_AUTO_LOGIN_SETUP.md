# TWS Auto-Login Configuration Guide

## Problem Identified (Dec 18, 2025)

The automation script was using `xdotool` to automatically click and type into TWS for login. This caused two issues:

1. **Display Issues**: xdotool moving the mouse and clicking while the monitor was sleeping caused graphics glitches (monitor "off color")
2. **TWS Shutdown**: TWS shut down when the automation script completed at 6:36 AM

## Solution

**Disable xdotool automation** and use TWS's built-in auto-login feature instead.

## How to Configure TWS Auto-Login

### Option 1: TWS Remember Credentials (Recommended)

1. **Open TWS Manually**
   ```bash
   /home/taiconley/Desktop/tws/tws
   ```

2. **On the Login Screen:**
   - Enter your username
   - Enter your password
   - **Check the box: "Save Settings"** or "Remember Me"
   - Log in

3. **Verify Auto-Login Works:**
   - Close TWS completely
   - Launch TWS again
   - It should auto-login without prompting

### Option 2: Configure TWS Settings (If Option 1 doesn't work)

1. **While logged into TWS:**
   - Go to: **File → Global Configuration → Lock and Exit**
   - Or: **Configure → Settings → Startup**
   
2. **Look for:**
   - "Auto login" option
   - "Remember credentials" option
   - Enable these settings

3. **Test:**
   - Exit TWS (not just close window)
   - Restart TWS
   - Should login automatically

### Option 3: TWS Configuration File (Advanced)

The TWS configuration is stored in: `~/Jts/jts.ini`

You can verify/edit login persistence settings there, but use GUI options above first.

## Testing the Fix

After configuring TWS auto-login:

```bash
# 1. Stop any running TWS
pkill -f "java.*Desktop/tws"

# 2. Launch TWS via our startup script
/home/taiconley/Desktop/Projects/trading/scripts/start_tws.sh

# 3. Watch the TWS window - it should:
#    - Open the login screen
#    - Automatically fill credentials  
#    - Automatically log in (no xdotool needed!)
```

## Why This is Better

| xdotool Automation | TWS Built-in Auto-Login |
|-------------------|------------------------|
| ❌ Moves mouse while screen sleeping | ✅ No screen interaction |
| ❌ Causes display glitches | ✅ Clean, native behavior |
| ❌ Fragile (window position changes break it) | ✅ Robust |
| ❌ Timing-dependent | ✅ Reliable |
| ✅ Works from CLI | ✅ Works from CLI |

## Remaining Issue: TWS Shutting Down

Even with auto-login fixed, TWS still shut down when the automation script ended. This needs further investigation.

### Current Fixes Applied:
1. ✅ `KillMode=process` in systemd service
2. ✅ `start_new_session=True` in Python subprocess
3. ✅ `setsid nohup` in TWS launch script

### To Test Tomorrow:

After you configure TWS auto-login, tomorrow morning:

1. **Leave computer on** (but screen can sleep)
2. **Stay logged in** to your user session
3. **At 6:25 AM** - automation will run
4. **At 6:36 AM** - automation completes
5. **Check at 7:00 AM** - Verify TWS is still running:
   ```bash
   pgrep -f "java.*Desktop/tws" && echo "✓ TWS running" || echo "✗ TWS shut down"
   ```

### If TWS Still Shuts Down Tomorrow:

We may need to launch TWS via a systemd service instead of directly from the automation script. This would make TWS completely independent.

## Alternative: Separate TWS Startup Service

If TWS keeps shutting down, we can create a dedicated systemd service for TWS:

**File: `~/.config/systemd/user/tws-startup.service`**

```ini
[Unit]
Description=TWS (Trader Workstation) Launch Service
After=graphical-session.target xhost-local.service

[Service]
Type=forking
ExecStart=/home/taiconley/Desktop/Projects/trading/scripts/start_tws.sh
Restart=no
RemainAfterExit=yes

Environment="DISPLAY=:1"
Environment="XAUTHORITY=/run/user/1000/gdm/Xauthority"
Environment="XDG_RUNTIME_DIR=/run/user/1000"

[Install]
WantedBy=graphical-session.target
```

Then the morning automation would just check if TWS is running, not launch it.

## Summary

**Immediate Action Required:**
1. Configure TWS to remember credentials and auto-login
2. Test manually that TWS auto-login works
3. Tomorrow morning, verify TWS stays running after 6:36 AM

**Changes Made:**
- ✅ Disabled xdotool automation (no more display glitches)
- ✅ Added better process isolation for TWS
- ⏳ Waiting to verify TWS persistence tomorrow

**Files Modified:**
- `scripts/start_tws.sh` - Disabled xdotool automation
- `scripts/morning_automation.py` - Added `start_new_session=True`
- `~/.config/systemd/user/trading-morning.service` - Added `KillMode=process`

