#!/home/taiconley/Desktop/Projects/trading/env/bin/python3
"""
Morning Trading Automation Script

Automates the daily morning routine:
1. Opens TWS and signs in (Paper Trading mode) at 6:25 AM
2. Runs docker compose up at 6:27 AM
3. Waits 1 minute and checks service health at 6:28 AM
4. Ensures "Pairs_Trading_Adaptive_Kalman" is enabled at 6:29 AM
5. Triggers "Warmup" at 6:30 AM (takes over a minute)
6. Waits 5 minutes, then enables "Ready to Trade" at 6:35 AM

Usage:
    # Activate virtual environment first (if running manually)
    source /home/taiconley/Desktop/Projects/trading/env/bin/activate
    ./morning_automation.py
    
    Or schedule with cron at 6:25 AM PST:
    25 6 * * * /home/taiconley/Desktop/Projects/trading/env/bin/python3 /home/taiconley/Desktop/Projects/trading/scripts/morning_automation.py >> /home/taiconley/Desktop/Projects/trading/logs/morning_automation.log 2>&1
"""

import os
import sys
import time
import logging
import subprocess
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Configuration
PROJECT_DIR = Path("/home/taiconley/Desktop/Projects/trading")
API_BASE_URL = "http://localhost:8000"
STRATEGY_ID = "pairs_trading_kalman_v1"  # The actual strategy_id in the database
LOG_FILE = PROJECT_DIR / "logs" / "morning_automation.log"

# Timing configuration (in seconds)
WAIT_AFTER_TWS = 60  # 2 minutes: 6:25 -> 6:27. set to 60 because tws startup time is slow
WAIT_AFTER_DOCKER = 60  # 1 minute: 6:27 -> 6:28
WAIT_AFTER_HEALTH = 60  # 1 minute: 6:28 -> 6:29
WAIT_AFTER_ENABLE = 60  # 1 minute: 6:29 -> 6:30
WAIT_AFTER_WARMUP = 300  # 5 minutes: 6:30 -> 6:35

# Setup logging
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TradingAutomation:
    """Handles the automated morning trading routine."""
    
    def __init__(self):
        self.project_dir = PROJECT_DIR
        self.api_base_url = API_BASE_URL
        self.strategy_id = STRATEGY_ID
        
    def log_step(self, step: str, message: str):
        """Log a step with timestamp."""
        logger.info(f"[{step}] {message}")
        
    def run(self):
        """Execute the complete morning automation routine."""
        try:
            logger.info("=" * 80)
            logger.info("Starting Morning Trading Automation")
            logger.info(f"Timestamp: {datetime.now().isoformat()}")
            logger.info("=" * 80)
            
            # Step 1: Open TWS and sign in (6:25 AM)
            tws_was_started = self.step1_open_tws()
            
            # Wait 2 minutes for TWS to fully initialize (only if we just started it)
            if tws_was_started:
                self.log_step("WAIT", f"Waiting {WAIT_AFTER_TWS} seconds for TWS to initialize...")
                time.sleep(WAIT_AFTER_TWS)
            else:
                self.log_step("WAIT", "TWS already running, skipping initialization wait")
            
            # Step 2: Run docker compose up (6:27 AM)
            self.step2_start_docker()
            
            # Wait 1 minute for containers to start
            self.log_step("WAIT", f"Waiting {WAIT_AFTER_DOCKER} seconds for services to start...")
            time.sleep(WAIT_AFTER_DOCKER)
            
            # Step 3: Check service health (6:28 AM)
            self.step3_check_health()
            
            # Wait 1 minute
            self.log_step("WAIT", f"Waiting {WAIT_AFTER_HEALTH} seconds before checking strategy...")
            time.sleep(WAIT_AFTER_HEALTH)
            
            # Step 4: Ensure strategy is enabled (6:29 AM)
            self.step4_ensure_strategy_enabled()
            
            # Wait 1 minute
            self.log_step("WAIT", f"Waiting {WAIT_AFTER_ENABLE} seconds before warmup...")
            time.sleep(WAIT_AFTER_ENABLE)
            
            # Step 5: Click "Warmup" (6:30 AM)
            self.step5_trigger_warmup()
            
            # Wait 5 minutes for warmup to complete
            self.log_step("WAIT", f"Waiting {WAIT_AFTER_WARMUP} seconds for warmup to complete...")
            time.sleep(WAIT_AFTER_WARMUP)
            
            # Step 6: Turn on "Ready to Trade" (6:35 AM)
            self.step6_enable_ready_to_trade()
            
            logger.info("=" * 80)
            logger.info("Morning Trading Automation COMPLETED SUCCESSFULLY")
            logger.info(f"Timestamp: {datetime.now().isoformat()}")
            logger.info("=" * 80)
            
            return 0
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR in automation: {e}", exc_info=True)
            return 1
    
    def step1_open_tws(self):
        """Step 1: Open TWS and sign in (Paper Trading mode)."""
        self.log_step("STEP 1", "Opening TWS and signing in...")
        
        try:
            # Check if TWS Java process is already running (more specific check)
            result = subprocess.run(
                ["pgrep", "-f", "java.*Desktop/tws"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log_step("STEP 1", "TWS is already running")
                return False  # Indicate TWS was already running
            
            # Launch TWS
            tws_script = self.project_dir / "scripts" / "start_tws.sh"
            
            if tws_script.exists():
                # Run the script and wait for it to complete
                result = subprocess.run(
                    [str(tws_script)],
                    capture_output=True,
                    text=True,
                    cwd=self.project_dir
                )
                
                if result.returncode == 0:
                    self.log_step("STEP 1", "TWS launched and logged in successfully")
                    return True  # Indicate we just started TWS
                else:
                    logger.error(f"TWS startup script failed: {result.stderr}")
                    logger.warning("Please manually ensure TWS is running in Paper Trading mode")
                    return False
            else:
                logger.warning(f"TWS startup script not found at {tws_script}")
                logger.warning("Please manually ensure TWS is running in Paper Trading mode")
                return False
                
        except Exception as e:
            logger.error(f"Error opening TWS: {e}")
            logger.warning("Please manually ensure TWS is running in Paper Trading mode")
            return False
    
    def step2_start_docker(self):
        """Step 2: Run docker compose up."""
        self.log_step("STEP 2", "Starting Docker containers...")
        
        try:
            # Check if containers are already running
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                cwd=self.project_dir
            )
            
            running_containers = result.stdout.strip().split('\n')
            if 'trading-api' in running_containers:
                self.log_step("STEP 2", "Docker containers are already running")
                return
            
            # Start docker compose
            subprocess.run(
                ["docker", "compose", "up", "-d"],
                cwd=self.project_dir,
                check=True
            )
            
            self.log_step("STEP 2", "Docker containers started successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error starting Docker containers: {e}")
            raise
    
    def step3_check_health(self):
        """Step 3: Check service health."""
        self.log_step("STEP 3", "Checking service health...")
        
        max_retries = 5
        retry_delay = 10  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.api_base_url}/api/health", timeout=10)
                
                if response.status_code == 200:
                    health_data = response.json()
                    status = health_data.get('status', 'unknown')
                    services = health_data.get('services', {})
                    
                    self.log_step("STEP 3", f"System Status: {status}")
                    
                    # Handle services as either dict or list
                    if isinstance(services, dict):
                        # Log individual service statuses
                        for service_name, service_info in services.items():
                            service_status = service_info.get('status', 'unknown')
                            self.log_step("STEP 3", f"  - {service_name}: {service_status}")
                        
                        # Check if all critical services are healthy
                        critical_services = ['account', 'marketdata', 'historical', 'strategy']
                        all_healthy = all(
                            services.get(svc, {}).get('status') == 'healthy' 
                            for svc in critical_services
                        )
                    elif isinstance(services, list):
                        # Services returned as list - try different key names
                        service_count = len(services)
                        healthy_count = 0
                        
                        for service_info in services:
                            # Try common key names for service identification
                            service_name = service_info.get('service_name') or service_info.get('name') or service_info.get('service', 'service')
                            service_status = service_info.get('status', 'unknown')
                            
                            if service_status == 'healthy':
                                healthy_count += 1
                            
                            self.log_step("STEP 3", f"  - {service_name}: {service_status}")
                        
                        # If we have at least 4 healthy services and overall status is healthy, consider it good
                        all_healthy = (status == 'healthy' and healthy_count >= 4)
                    else:
                        logger.warning("Unexpected services format")
                        all_healthy = status == 'healthy'
                    
                    if all_healthy:
                        self.log_step("STEP 3", "All services are healthy ✓")
                        return
                    else:
                        logger.warning(f"Some services are not healthy yet (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        continue
                else:
                    logger.warning(f"Health check returned status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Health check failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
        
        logger.warning("Not all services are healthy, but continuing with automation...")
    
    def step4_ensure_strategy_enabled(self):
        """Step 4: Ensure strategy is enabled."""
        self.log_step("STEP 4", f"Checking if {self.strategy_id} is enabled...")
        
        try:
            # Get current strategy status
            response = requests.get(f"{self.api_base_url}/api/strategies", timeout=10)
            
            if response.status_code != 200:
                raise Exception(f"Failed to get strategies: {response.status_code}")
            
            strategies_data = response.json()
            strategies = strategies_data.get('strategies', [])
            
            # Find our strategy
            strategy = next(
                (s for s in strategies if s['id'] == self.strategy_id),
                None
            )
            
            if not strategy:
                raise Exception(f"Strategy {self.strategy_id} not found")
            
            is_enabled = strategy.get('enabled', False)
            
            if is_enabled:
                self.log_step("STEP 4", f"{self.strategy_id} is already enabled ✓")
            else:
                # Enable the strategy
                self.log_step("STEP 4", f"Enabling {self.strategy_id}...")
                enable_response = requests.post(
                    f"{self.api_base_url}/api/strategies/{self.strategy_id}/enable",
                    json={"enabled": True},
                    timeout=10
                )
                
                if enable_response.status_code == 200:
                    self.log_step("STEP 4", f"{self.strategy_id} enabled successfully ✓")
                else:
                    raise Exception(f"Failed to enable strategy: {enable_response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error ensuring strategy is enabled: {e}")
            raise
    
    def step5_trigger_warmup(self):
        """Step 5: Trigger strategy warmup."""
        self.log_step("STEP 5", f"Triggering warmup for {self.strategy_id}...")
        
        try:
            response = requests.post(
                f"{self.api_base_url}/api/strategies/{self.strategy_id}/warmup",
                timeout=180  # Warmup can take up to 3 minutes
            )
            
            if response.status_code == 200:
                warmup_data = response.json()
                self.log_step("STEP 5", f"Warmup triggered successfully ✓")
                self.log_step("STEP 5", f"Response: {warmup_data.get('message', '')}")
            elif response.status_code == 503:
                # 503 often means the service is processing - this is OK
                self.log_step("STEP 5", "Warmup is processing (service busy) - continuing...")
                logger.info("Warmup request accepted but service is busy processing. This is normal.")
            else:
                logger.warning(f"Warmup returned status {response.status_code}, but continuing anyway")
                self.log_step("STEP 5", f"Warmup may still be processing (status: {response.status_code})")
                
        except requests.exceptions.Timeout:
            logger.warning("Warmup request timed out after 3 minutes")
            logger.info("Warmup is likely still processing in the background - continuing with automation")
            self.log_step("STEP 5", "Warmup initiated (processing in background)")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Warmup request error: {e}")
            logger.info("Warmup may still be processing - continuing with automation")
            self.log_step("STEP 5", "Warmup initiated (may be processing)")
    
    def step6_enable_ready_to_trade(self):
        """Step 6: Enable 'Ready to Trade' status."""
        self.log_step("STEP 6", f"Enabling 'Ready to Trade' for {self.strategy_id}...")
        
        try:
            response = requests.post(
                f"{self.api_base_url}/api/strategies/{self.strategy_id}/ready-to-trade",
                json={"ready_to_trade": True},
                timeout=10
            )
            
            if response.status_code == 200:
                rtrade_data = response.json()
                self.log_step("STEP 6", f"Ready to Trade enabled successfully ✓")
                self.log_step("STEP 6", f"Response: {rtrade_data.get('message', '')}")
                self.log_step("STEP 6", "🚀 System is now ready for live trading!")
            else:
                raise Exception(f"Ready to Trade request failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error enabling Ready to Trade: {e}")
            raise


def main():
    """Main entry point."""
    automation = TradingAutomation()
    return automation.run()


if __name__ == "__main__":
    sys.exit(main())

