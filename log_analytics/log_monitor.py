"""
Real-time log monitoring and anomaly detection system
"""
import os
import time
import logging
from datetime import datetime
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

from log_analytics.config.config import (
    ELASTICSEARCH_HOST,
    ELASTICSEARCH_PORT,
    ELASTICSEARCH_INDEX,
    LOG_DIR,
    WATCH_INTERVAL,
    ANOMALY_THRESHOLD,
    WINDOW_SIZE
)
from log_analytics.elastic_handler import ElasticsearchHandler
from log_analytics.logai_handler import LogAIHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LogFileHandler(FileSystemEventHandler):
    """
    Handler for log file events
    
    Monitors new log files and changes to existing log files
    """
    def __init__(self, es_handler, logai_handler):
        self.es_handler = es_handler
        self.logai_handler = logai_handler
        self.file_positions = {}  # Track file positions to read only new content
        
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            self._process_log_file(event.src_path)
            
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            logger.info(f"New log file detected: {event.src_path}")
            self._process_log_file(event.src_path)
    
    def _process_log_file(self, file_path):
        """Process new content in log file"""
        try:
            # Skip non-log files
            if not self._is_log_file(file_path):
                return

            # Get last known position or start from beginning
            last_pos = self.file_positions.get(file_path, 0)

            with open(file_path, 'r') as f:
                # Seek to last known position
                f.seek(last_pos)

                # Read new lines
                new_content = f.read()

                # Update position
                self.file_positions[file_path] = f.tell()

                # Batch process new lines for LLM efficiency
                if new_content:
                    lines = [line for line in new_content.splitlines() if line.strip()]
                    if lines:
                        source = os.path.basename(file_path)
                        # Parallelize batch processing for large log batches
                        processed_logs = []
                        with ThreadPoolExecutor(max_workers=8) as executor:
                            futures = [executor.submit(self._process_log_line_parallel, line, source) for line in lines]
                            for future in as_completed(futures):
                                result = future.result()
                                if result:
                                    processed_logs.append(result)
                        for processed_log in processed_logs:
                            rag_template = self.logai_handler.get_llm_template_with_rag(
                                processed_log["message"], self.es_handler, context_count=10)
                            processed_log["semantic_template"] = rag_template
                            self.es_handler.index_log(processed_log)
                            logger.debug(f"Log: {processed_log['message']} | Anomaly Score: {processed_log['anomaly_score']}")
                            if processed_log['anomaly_score'] > ANOMALY_THRESHOLD:
                                logger.warning(f"ANOMALY DETECTED: {processed_log['message']} (score: {processed_log['anomaly_score']:.4f})")
                                self._collect_user_feedback(processed_log)
                        # After processing all logs, generate grouped LLM reports
                        self._generate_grouped_llm_reports(processed_logs)
        except Exception as e:
            logger.error(f"Error processing log file {file_path}: {e}")
    
    def _process_log_line_parallel(self, line, source):
        try:
            return self.logai_handler.process_log(line, source)
        except Exception as e:
            logger.error(f"Parallel log processing error: {e}")
            return None
    
    def _collect_user_feedback(self, processed_log):
        """
        Append detected anomaly log and its score to a feedback file for future fine-tuning or in-context learning.
        """
        feedback_file = os.path.join(LOG_DIR, "anomaly_feedback.csv")
        import csv
        file_exists = os.path.isfile(feedback_file)
        with open(feedback_file, mode="a", newline="") as csvfile:
            fieldnames = ["timestamp", "message", "source", "level", "anomaly_score", "semantic_template", "is_anomaly"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            row = {k: processed_log.get(k, "") for k in fieldnames}
            row["is_anomaly"] = 1  # Default to 1 (anomaly detected)
            writer.writerow(row)
    
    def _generate_grouped_llm_reports(self, processed_logs):
        """
        Generate a single LLM report per application group (source) for logs in the last 15 minutes.
        """
        import requests
        import json
        from datetime import datetime, timedelta
        report_file = os.path.join(LOG_DIR, "anomaly_reports.txt")
        logger.info(f"Writing anomaly group reports to: {os.path.abspath(report_file)}")
        now = datetime.utcnow()
        # Group logs by source (application) and filter for last 15 minutes
        grouped = {}

        logger.info(f"processed_logs::: {processed_logs}")
        for log in processed_logs:
            try:
                log_time = datetime.fromisoformat(log.get('timestamp'))
            except Exception:
                continue
            if (now - log_time).total_seconds() > 900:
                continue
            source = log.get('source', 'unknown')
            grouped.setdefault(source, []).append(log)
        for source, logs in grouped.items():
            logger.info(f"logs::: {logs}")
            
            if not logs:
                continue
            # Aggregate log details for the group
            messages = '\n'.join(f"- [{l['timestamp']}] {l['level']}: {l['message']} (score: {l['anomaly_score']})" for l in logs)
            prompt = (
                f"You are an expert SRE and log analyst.\n"
                f"Analyze the following group of anomalous logs from application: {source} in the last 15 minutes.\n"
                f"Summarize the main issues, possible root causes, and provide actionable suggestions.\n"
                f"Logs:\n{messages}\n"
            )
            logger.info(f"Sending LLM request for source '{source}' with {len(logs)} logs.")
            logger.debug(f"LLM prompt: {prompt}")
            try:
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': 'llama3.1',
                        'prompt': prompt,
                        'stream': False
                    },
                    timeout=120
                )
                logger.info(f"LLM HTTP status: {response.status_code}")
                logger.debug(f"LLM raw response: {response.text}")
                data = response.json()
                report = data.get('response', '').strip()
                logger.info(f"LLM report received for source '{source}', length: {len(report)}")
                with open(report_file, 'a') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Application: {source}\n")
                    f.write(f"Time Window: {now - timedelta(minutes=15)} to {now}\n")
                    f.write(f"Log Count: {len(logs)}\n")
                    f.write(f"LLM Grouped Report:\n{report}\n")
            except Exception as e:
                logger.error(f"Failed to generate grouped LLM report for {source}: {e}")
    
    def _is_log_file(self, file_path):
        """Check if file is a log file based on extension or name"""
        log_extensions = ['.log', '.txt']
        path = Path(file_path)
        return (
            path.suffix.lower() in log_extensions or 
            'log' in path.name.lower() or
            path.suffix == ''  # Include files with no extension
        )
    
    def _process_log_line(self, line, file_path):
        """Process a single log line"""
        try:
            # Extract source from file path
            source = os.path.basename(file_path)
            
            # Process log with LogAI
            processed_log = self.logai_handler.process_log(line, source)
            
            # Store in Elasticsearch
            self.es_handler.index_log(processed_log)
            
            # Check if anomalous
            if processed_log['anomaly_score'] > ANOMALY_THRESHOLD:
                logger.warning(f"ANOMALY DETECTED: {processed_log['message']} (score: {processed_log['anomaly_score']:.4f})")
                
        except Exception as e:
            logger.error(f"Error processing log line: {e}")

    def scan_existing_files(self):
        """Scan existing log files in the directory"""
        log_dir = Path(LOG_DIR)
        if log_dir.exists():
            for file_path in log_dir.glob("**/*"):
                if file_path.is_file() and self._is_log_file(str(file_path)):
                    logger.info(f"Scanning existing log file: {file_path}")
                    self._process_log_file(str(file_path))
        else:
            logger.warning(f"Log directory {LOG_DIR} does not exist")


class LLMQueryHandler:
    """
    Handler for LLM queries on log data
    
    Allows querying the log data using LLMs for insights and resolution suggestions
    """
    def __init__(self, es_handler, logai_handler=None):
        self.es_handler = es_handler
        self.logai_handler = logai_handler or LogAIHandler()
        
    def analyze_anomalies(self, max_anomalies=5):
        """
        Analyze recent anomalies and provide insights
        
        In a real implementation, this would send the logs to an LLM for analysis
        """
        anomalies = self.es_handler.get_anomalies(threshold=ANOMALY_THRESHOLD, size=max_anomalies)
        
        if not anomalies:
            return "No anomalies detected."
            
        analysis = f"Found {len(anomalies)} anomalies in the logs:\n\n"
        
        for idx, anomaly in enumerate(anomalies, 1):
            log_data = anomaly["_source"]
            analysis += f"{idx}. [{log_data['level']}] {log_data['message']}\n"
            analysis += f"   Score: {log_data['anomaly_score']:.4f}, Source: {log_data['source']}\n"
            analysis += f"   Timestamp: {log_data['timestamp']}\n"
            
            # Add placeholder for LLM analysis
            analysis += "   Analysis: [This would contain LLM-generated analysis in production]\n\n"
            
        return analysis
    
    def query_logs(self, query_text, limit=10):
        """
        Query logs with natural language
        
        In a real implementation, this would:
        1. Convert the query to a vector
        2. Search for similar logs
        3. Send logs + query to an LLM for analysis
        """
        # In a production system, this would use a proper embedding model
        query_vector = self.logai_handler._generate_vector_embedding(query_text).tolist()
        
        # Find similar logs
        similar_logs = self.es_handler.search_similar_logs(query_vector, top_k=limit)
        
        if not similar_logs:
            return "No matching logs found."
            
        result = f"Found {len(similar_logs)} logs matching your query:\n\n"
        
        for idx, log in enumerate(similar_logs, 1):
            log_data = log["_source"]
            result += f"{idx}. [{log_data['level']}] {log_data['message']}\n"
            result += f"   Source: {log_data['source']}, Timestamp: {log_data['timestamp']}\n"
            
        # Add placeholder for LLM response
        result += "\nLLM Analysis: [This would contain LLM-generated analysis in production]\n"
        
        return result


def start_monitoring():
    """Start monitoring log files"""
    try:
        logger.info("Starting log monitoring service")
        
        # Initialize handlers
        es_handler = ElasticsearchHandler(
            host=ELASTICSEARCH_HOST,
            port=ELASTICSEARCH_PORT,
            index_name=ELASTICSEARCH_INDEX
        )
        
        logai_handler = LogAIHandler(window_size=WINDOW_SIZE)
        
        # Create log file handler
        log_file_handler = LogFileHandler(es_handler, logai_handler)
        
        # Initialize LLM query handler
        llm_handler = LLMQueryHandler(es_handler, logai_handler)
        
        # Set up directory observer
        observer = Observer()
        observer.schedule(log_file_handler, LOG_DIR, recursive=True)
        observer.start()
        
        # Scan existing files
        logger.info("Scanning existing log files")
        log_file_handler.scan_existing_files()
        
        logger.info(f"Monitoring log directory: {LOG_DIR}")
        logger.info(f"Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                time.sleep(WATCH_INTERVAL)
        except KeyboardInterrupt:
            observer.stop()
            
        observer.join()
        
    except Exception as e:
        logger.error(f"Error in log monitoring: {e}")
        raise


if __name__ == "__main__":
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Start monitoring in the main thread
    start_monitoring()