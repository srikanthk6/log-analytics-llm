"""
Configuration file for the log analytics application
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Elasticsearch Configuration
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", 9200))
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "logs_vector_index")

# LogAI Configuration
LOG_DIR = os.getenv("LOG_DIR", "/home/ubuntu/simplellm/log_analytics/logs")
WATCH_INTERVAL = float(os.getenv("WATCH_INTERVAL", 1000000.0))  # seconds

# Anomaly Detection Configuration
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", 0.8))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 100))