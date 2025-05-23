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
import multiprocessing
from multiprocessing import Process

# Set multiprocessing start method to 'spawn'
multiprocessing.set_start_method("spawn", force=True)

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from log_analytics.config.config import LOG_DIR
from log_analytics.log_monitor import start_monitoring
from log_analytics.api import start_api
from log_analytics.elastic_handler import ElasticsearchHandler

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
    
    # Always generate a new log file at startup
    log_gen_cmd = "python3 generate_ecommerce_logs.py --count 200 --anomaly-rate 0.1"
    logger.info(f"Generating new log file with: {log_gen_cmd}")
    os.system(log_gen_cmd)
    
    # Initialize ElasticsearchHandler
    es_handler = ElasticsearchHandler()

    # Create and start an anomaly detection job
    job_id = "log_anomaly_detection"
    index_pattern = "logs_vector_index"
    field_name = "anomaly_score"
    bucket_span = "5m"

    # es_handler.create_anomaly_detection_job(job_id, index_pattern, field_name, bucket_span)
    
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

    

# def stop_server(server_thread):
#     time.sleep(10)
#     print("Stopping server")
#     server_thread.interrupt()
#     server_thread.join(timeout=3)
#     if server_thread.is_alive():
#         print("Server thread did not terminate in time!")

if __name__ == "__main__":
    sys.exit(main())
    # server_thread = threading.Thread(target=main, daemon=True)
    # server_thread.start()
    # stop_server(server_thread)

