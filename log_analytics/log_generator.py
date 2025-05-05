"""
Log generator for testing purposes
"""
import os
import time
import random
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Sample log patterns for various applications
LOG_PATTERNS = {
    "web_server": [
        "{timestamp} INFO Request received: GET /api/users from {ip}",
        "{timestamp} INFO Request completed: GET /api/users - 200 OK in {duration}ms",
        "{timestamp} WARNING Slow request: GET /api/products/{id} in {slow_duration}ms",
        "{timestamp} ERROR Failed to connect to database: Connection timeout",
        "{timestamp} ERROR 500 Internal Server Error: {error_message}",
        "{timestamp} INFO User {user_id} logged in successfully",
        "{timestamp} WARNING Rate limit exceeded for IP {ip}",
        "{timestamp} INFO Cache hit for key: {cache_key}",
        "{timestamp} INFO Cache miss for key: {cache_key}",
        "{timestamp} CRITICAL Service unavailable: Redis connection failed"
    ],
    "application": [
        "{timestamp} INFO Application started with environment: {environment}",
        "{timestamp} INFO User {user_id} performed action: {action}",
        "{timestamp} WARNING High memory usage detected: {memory_usage}%",
        "{timestamp} ERROR Exception in module {module}: {error_message}",
        "{timestamp} INFO Successfully processed task {task_id}",
        "{timestamp} WARNING Task {task_id} took longer than expected: {duration}s",
        "{timestamp} ERROR Failed to process message: {error_message}",
        "{timestamp} INFO Scheduled job {job_name} started",
        "{timestamp} INFO Scheduled job {job_name} completed in {duration}s",
        "{timestamp} CRITICAL System error: {error_message}"
    ],
    "database": [
        "{timestamp} INFO Query executed: {query_type} in {duration}ms",
        "{timestamp} WARNING Slow query detected: {query_type} in {slow_duration}ms",
        "{timestamp} ERROR Query failed: {error_message}",
        "{timestamp} INFO Connection pool size: {pool_size}",
        "{timestamp} WARNING Connection pool reaching limit: {pool_size}/{max_pool_size}",
        "{timestamp} INFO Database backup started",
        "{timestamp} INFO Database backup completed successfully",
        "{timestamp} ERROR Database backup failed: {error_message}",
        "{timestamp} WARNING Index {index_name} is missing",
        "{timestamp} CRITICAL Database disk usage at {disk_usage}%"
    ]
}

# Sample values for variable placeholders
SAMPLE_VALUES = {
    "ip": ["192.168.1.1", "10.0.0.5", "172.16.0.10", "8.8.8.8", "1.1.1.1"],
    "duration": [50, 120, 80, 30, 60, 20, 40],
    "slow_duration": [1500, 2000, 3000, 5000, 8000],
    "error_message": [
        "NullReferenceException", 
        "Connection refused", 
        "Timeout exceeded", 
        "Invalid input", 
        "Permission denied",
        "Out of memory",
        "File not found"
    ],
    "user_id": ["user123", "admin", "guest", "john_doe", "jane_smith"],
    "environment": ["production", "staging", "development", "testing"],
    "action": ["create", "update", "delete", "view", "export", "import"],
    "memory_usage": [75, 80, 85, 90, 95],
    "module": ["auth", "api", "database", "cache", "scheduler", "processor"],
    "task_id": ["task-1234", "task-5678", "batch-process-1", "email-sender-2", "data-cleanup"],
    "job_name": ["daily-cleanup", "user-report", "data-aggregation", "metrics-calculator"],
    "query_type": ["SELECT", "INSERT", "UPDATE", "DELETE", "JOIN", "AGGREGATE"],
    "pool_size": [5, 8, 10, 15, 20],
    "max_pool_size": [20, 25, 30],
    "index_name": ["users_idx", "products_idx", "orders_idx", "audit_log_idx"],
    "disk_usage": [80, 85, 90, 95, 97],
    "id": [1001, 2002, 3003, 4004, 5005],
    "cache_key": ["user:1001", "product:2002", "settings:site", "config:api"]
}

# Anomalous log patterns
ANOMALY_PATTERNS = [
    "{timestamp} ERROR Unexpected token < in JSON at position 0",
    "{timestamp} CRITICAL System crash detected: segmentation fault",
    "{timestamp} ERROR Database corruption detected in table {table_name}",
    "{timestamp} WARNING Potential security breach from IP {ip}",
    "{timestamp} ERROR API rate limit exceeded by client {client_id}",
    "{timestamp} CRITICAL Disk full on server: {server_name}",
    "{timestamp} ERROR Memory leak detected in module {module}",
    "{timestamp} WARNING Unusual network traffic pattern detected",
    "{timestamp} ERROR Failed to rotate logs: permission denied",
    "{timestamp} CRITICAL Unauthorized access attempt for user {user_id}"
]

# Additional sample values for anomalies
ANOMALY_VALUES = {
    "table_name": ["users", "transactions", "products", "orders", "audit_logs"],
    "client_id": ["mobile-app", "web-client", "external-api", "data-importer"],
    "server_name": ["app-server-01", "db-server-02", "cache-server-01", "worker-node-03"]
}

# Merge regular and anomaly values
ALL_VALUES = {**SAMPLE_VALUES, **ANOMALY_VALUES}

def generate_timestamp():
    """Generate current timestamp in standard log format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

def fill_template(template, values_dict):
    """Fill a log template with random values"""
    # Use a dictionary containing all potential placeholders for the format call
    format_args = {'timestamp': generate_timestamp()}
    
    # Find all placeholders in the template
    import re
    placeholders = re.findall(r'\{([^}]+)\}', template)
    
    # Add random values for all placeholders found in the template
    for key in placeholders:
        if key != 'timestamp' and key in values_dict:
            format_args[key] = random.choice(values_dict[key])
    
    # Format the template with all the collected values
    try:
        return template.format(**format_args)
    except KeyError as e:
        # If any placeholders are missing, provide a default value
        missing_key = str(e).strip("'")
        format_args[missing_key] = f"[missing_{missing_key}]"
        return template.format(**format_args)

def generate_logs(log_file_path, log_type="application", count=100, interval=0.1, anomaly_probability=0.05):
    """
    Generate random logs and write them to a file
    
    Args:
        log_file_path: Path to the log file
        log_type: Type of logs to generate (web_server, application, database)
        count: Number of logs to generate
        interval: Interval between log generation in seconds
        anomaly_probability: Probability of generating an anomalous log
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    patterns = LOG_PATTERNS.get(log_type, LOG_PATTERNS["application"])
    print(f"Generating {count} logs to {log_file_path}...")

    # --- Accurate anomaly rate logic ---
    num_anomalies = int(count * anomaly_probability)
    anomaly_indices = set(random.sample(range(count), num_anomalies)) if num_anomalies > 0 else set()

    with open(log_file_path, "a") as log_file:
        for i in range(count):
            if i in anomaly_indices:
                log_template = random.choice(ANOMALY_PATTERNS)
                log_line = fill_template(log_template, ALL_VALUES)
            else:
                log_template = random.choice(patterns)
                log_line = fill_template(log_template, SAMPLE_VALUES)
            log_file.write(log_line + "\n")
            # --- Batch sleep for speed ---
            if interval > 0 and (i + 1) % 1000 == 0:
                time.sleep(interval)
    print(f"Log generation completed. {count} logs written to {log_file_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate sample logs for testing")
    parser.add_argument("--type", choices=["web_server", "application", "database", "mixed"], 
                        default="mixed", help="Type of logs to generate")
    parser.add_argument("--count", type=int, default=100, help="Number of logs to generate")
    parser.add_argument("--interval", type=float, default=0.1, help="Interval between logs in seconds")
    parser.add_argument("--anomaly-rate", type=float, default=0.05, help="Probability of anomaly logs (0-1)")
    parser.add_argument("--output-dir", default=None, help="Directory to write log files")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "logs")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.type == "mixed":
        # Generate logs for all types
        for log_type in ["web_server", "application", "database"]:
            log_file_path = os.path.join(args.output_dir, f"{log_type}.log")
            generate_logs(
                log_file_path, 
                log_type, 
                count=args.count // 3,  # Split count across types
                interval=args.interval,
                anomaly_probability=args.anomaly_rate
            )
    else:
        # Generate logs for specific type
        log_file_path = os.path.join(args.output_dir, f"{args.type}.log")
        generate_logs(
            log_file_path, 
            args.type, 
            count=args.count,
            interval=args.interval,
            anomaly_probability=args.anomaly_rate
        )

if __name__ == "__main__":
    main()