#!/usr/bin/env python3
"""
Main entry point for the log analytics application
"""
import os
import sys
import time
import signal
import threading
import argparse
import logging
from multiprocessing import Process

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from log_analytics.config.config import LOG_DIR
from log_analytics.log_monitor import start_monitoring
from log_analytics.api import start_api

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Log Analytics Application")
    parser.add_argument("--api-only", action="store_true", help="Run only the API server")
    parser.add_argument("--monitor-only", action="store_true", help="Run only the log monitor")
    parser.add_argument("--api-port", type=int, default=5000, help="Port for the API server")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    return parser.parse_args()

def run_api(port, debug):
    """Run the API server"""
    from log_analytics.api import start_api
    logger.info(f"Starting API server on port {port}")
    start_api(host="0.0.0.0", port=port, debug=debug)

def run_monitor():
    """Run the log monitor"""
    from log_analytics.log_monitor import start_monitoring
    logger.info(f"Starting log monitor for directory: {LOG_DIR}")
    start_monitoring()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Store processes to manage
    processes = []
    
    try:
        # Start API server if requested or both services requested
        if not args.monitor_only:
            api_process = Process(target=run_api, args=(args.api_port, args.debug))
            api_process.start()
            processes.append(api_process)
            logger.info(f"API server started on port {args.api_port}")
            
        # Start log monitor if requested or both services requested
        if not args.api_only:
            monitor_process = Process(target=run_monitor)
            monitor_process.start()
            processes.append(monitor_process)
            logger.info("Log monitor started")
            
        # If no processes started, print help
        if not processes:
            logger.error("No services started. Use --help for options.")
            return 1
            
        # Wait for all processes to complete (which they won't unless terminated)
        for process in processes:
            process.join()
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        
        # Terminate all processes
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                
        logger.info("All processes terminated")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())