"""
Log parsing and anomaly detection using LogAI
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Update imports with correct class names from LogAI
from logai.dataloader.data_model import LogRecordObject
from logai.preprocess.preprocessor import Preprocessor, PreprocessorConfig
from logai.algorithms.parsing_algo.drain import Drain, DrainParams
from logai.algorithms.vectorization_algo.tfidf import TfIdf, TfIdfParams
# Import LogAI's native IsolationForestDetector and its params class
from logai.algorithms.anomaly_detection_algo.isolation_forest import IsolationForestDetector, IsolationForestParams

logger = logging.getLogger(__name__)

class LogAIHandler:
    def __init__(self, window_size: int = 100):
        """Initialize LogAI components"""
        self.window_size = window_size
        self.template_cache = {}
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        self._setup_components()
        
    def _setup_components(self) -> None:
        """Set up LogAI components for parsing and anomaly detection"""
        # Set up LogAI parser (Drain) with DrainParams
        drain_params = DrainParams(
            depth=4,
            sim_th=0.4,
            max_children=100,
            max_clusters=None
        )
        self.parser = Drain(params=drain_params)
        
        # Set up feature extractor (TF-IDF) with TfIdfParams
        tfidf_params = TfIdfParams(
            max_features=100,
            ngram_range=(1, 1)
        )
        self.feature_extractor = TfIdf(params=tfidf_params)
        
        # Set up preprocessor with PreprocessorConfig
        preprocessor_config = PreprocessorConfig()
        self.preprocessor = Preprocessor(config=preprocessor_config)
        
        # Create IsolationForestParams object for LogAI's detector
        # Make sure all parameters have the correct type - warm_start must be boolean
        isolation_forest_params = IsolationForestParams(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            warm_start=False,  # Explicitly set as boolean, not integer
            bootstrap=False,
            n_jobs=None,
            verbose=False
        )
        # Pass params object to LogAI's IsolationForestDetector
        self.anomaly_detector = IsolationForestDetector(params=isolation_forest_params)
        
        # Storage for log data
        self.logs_buffer = []
        
    def get_llm_template(self, message: str) -> str:
        """
        Use LLM to generate a semantic template for a log message, with caching.
        """
        if message in self.template_cache:
            return self.template_cache[message]
        import requests
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'codellama:instruct',
                    'prompt': f"Extract a log template for the following log message. Replace all variable parts (numbers, IDs, paths, etc.) with <*>. Only output the template string.\nLog: {message}",
                    'stream': False
                },
                timeout=10
            )
            data = response.json()
            template = data['response'].strip().replace('\n', ' ')
            # Basic cleanup: remove quotes if present
            if template.startswith('"') and template.endswith('"'):
                template = template[1:-1]
            self.template_cache[message] = template
            return template
        except Exception as e:
            logger.warning(f"LLM template extraction failed, using raw message: {e}")
            self.template_cache[message] = message
            return message

    def get_recent_logs_for_context(self, es_handler, count=10):
        """
        Retrieve recent logs from Elasticsearch for use as context in LLM prompts.
        """
        try:
            logs = es_handler.get_recent_logs(size=count)
            # Extract just the message or template for context
            return [log['_source']['message'] for log in logs if '_source' in log and 'message' in log['_source']]
        except Exception as e:
            logger.warning(f"Failed to retrieve recent logs for RAG context: {e}")
            return []

    def get_llm_template_with_rag(self, message: str, es_handler, context_count=10) -> str:
        """
        Use LLM to generate a semantic template for a log message, with RAG context and caching.
        """
        if message in self.template_cache:
            return self.template_cache[message]
        import requests
        # Retrieve recent logs for context
        context_logs = self.get_recent_logs_for_context(es_handler, count=context_count)
        context_str = "\n".join(context_logs)
        prompt = (
            f"You are an expert log analyst. Here are some recent log messages for context:\n"
            f"{context_str}\n"
            f"Now, extract a log template for the following log message. Replace all variable parts (numbers, IDs, paths, etc.) with <*>. Only output the template string.\nLog: {message}"
        )
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'codellama:instruct',
                    'prompt': prompt,
                    'stream': False
                },
                timeout=120
            )
            data = response.json()
            template = data['response'].strip().replace('\n', ' ')
            if template.startswith('"') and template.endswith('"'):
                template = template[1:-1]
            self.template_cache[message] = template
            return template
        except Exception as e:
            logger.warning(f"LLM template extraction with RAG failed, using raw message: {e}")
            self.template_cache[message] = message
            return message
        
    def process_log(self, log_line: str, source: str = "application") -> Dict[str, Any]:
        """
        Process a single log line
        
        Args:
            log_line: Raw log line text
            source: Source of the log
            
        Returns:
            Dict with processed log data including anomaly score and vector embedding
        """
        try:
            # Parse timestamp and level if present in the log line
            timestamp, level, message = self._extract_log_metadata(log_line)
            
            # Create log record object with correct OpenTelemetry compatible parameters
            # Create DataFrames for each parameter as required by LogRecordObject
            timestamp_df = pd.DataFrame({'timestamp': [timestamp]})
            body_df = pd.DataFrame({'body': [message]})
            resource_df = pd.DataFrame({'source': [source]})
            severity_text_df = pd.DataFrame({'severity_text': [level]})
            
            log_record = LogRecordObject(
                timestamp=timestamp_df,
                body=body_df,
                resource=resource_df,
                severity_text=severity_text_df
            )
            
            # Add to buffer
            self.logs_buffer.append(log_record)
            
            # If buffer is large enough, perform anomaly detection
            anomaly_score = 0.0
            log_vector = self._generate_vector_embedding(message)
            semantic_template = self.get_llm_template(message)
            
            if len(self.logs_buffer) >= self.window_size:
                anomaly_score = self._detect_anomalies()
                # Maintain buffer size
                if len(self.logs_buffer) > self.window_size:
                    self.logs_buffer = self.logs_buffer[-self.window_size:]
            
            return {
                "timestamp": timestamp,
                "message": message,
                "source": source,
                "level": level,
                "log_vector": log_vector.tolist(),
                "anomaly_score": float(anomaly_score),
                "semantic_template": semantic_template
            }
            
        except Exception as e:
            logger.error(f"Error processing log: {e}")
            # Return minimal data if processing fails
            return {
                "timestamp": datetime.now().isoformat(),
                "message": log_line,
                "source": source,
                "level": "ERROR",
                "log_vector": np.zeros(384).tolist(),  # Empty vector
                "anomaly_score": 0.0,
                "semantic_template": log_line
            }
    
    def _extract_log_metadata(self, log_line: str) -> Tuple[str, str, str]:
        """
        Extract timestamp and level from log line
        
        Simple extraction based on common log formats
        """
        timestamp = datetime.now().isoformat()
        level = "INFO"
        message = log_line
        
        # Try to extract timestamp and level
        # This is a simplified approach - in production, use more sophisticated parsing
        try:
            parts = log_line.split(" ", 3)
            if len(parts) >= 3:
                # Check if first part might be timestamp
                try:
                    # Try different timestamp formats
                    ts_formats = [
                        "%Y-%m-%d %H:%M:%S,%f",
                        "%Y-%m-%d %H:%M:%S.%f",
                        "%Y-%m-%dT%H:%M:%S.%fZ"
                    ]
                    
                    for fmt in ts_formats:
                        try:
                            dt = datetime.strptime(" ".join(parts[0:2]), fmt)
                            timestamp = dt.isoformat()
                            # If successful, remaining parts are level and message
                            if len(parts) > 2:
                                level = parts[2]
                                message = parts[3] if len(parts) > 3 else ""
                            break
                        except ValueError:
                            continue
                except Exception:
                    # If timestamp parsing fails, check for log level
                    level_keywords = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                    for i, part in enumerate(parts):
                        if part.upper() in level_keywords:
                            level = part.upper()
                            message = " ".join(parts[i+1:]) if i < len(parts) - 1 else ""
                            break
        except Exception:
            # If extraction fails, use default values
            pass
            
        return timestamp, level, message
    
    def _generate_vector_embedding(self, message: str) -> np.ndarray:
        """
        Generate vector embedding for the log message using all-MiniLM-L6-v2.
        """
        try:
            embedding = self.embedding_model.encode(message, show_progress_bar=False, normalize_embeddings=True)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Falling back to hash-based embedding due to embedding model error: {e}")
            vector = np.zeros(self.embedding_dim)
            for i, char in enumerate(message):
                vector[i % self.embedding_dim] += ord(char) / 255.0
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            return vector
    
    def _detect_anomalies(self) -> float:
        """
        Detect anomalies in the current log buffer using LogAI's IsolationForestDetector.

        Returns:
            Anomaly score for the latest log (0 for normal, 1 for anomaly).
        """
        try:
            if len(self.logs_buffer) < 10:  # Need minimum data for detection
                return 0.0

            # Convert buffer to DataFrame
            df = pd.DataFrame([{
                'timestamp': log_record.timestamp.iloc[0, 0] if not log_record.timestamp.empty else "",
                'message': log_record.body.iloc[0, 0] if not log_record.body.empty else "",
                'source': log_record.resource.iloc[0, 0] if not log_record.resource.empty else "",
                'level': log_record.severity_text.iloc[0, 0] if not log_record.severity_text.empty else ""
            } for log_record in self.logs_buffer])

            # Parse logs with Drain
            templates = []
            for message in df['message']:
                cluster = self.parser.match(message)
                templates.append(cluster.template if cluster else message)

            # Convert templates list to a Pandas Series before passing to transform
            templates_series = pd.Series(templates)

            # Create feature vectors using LogAI's TF-IDF
            self.feature_extractor.fit(templates_series)
            X = self.feature_extractor.transform(templates_series)

            # Convert to format expected by LogAI's detector
            if hasattr(X, 'toarray'):
                X_array = X.toarray()
            else:
                X_array = np.array(X)
            
            # Ensure we have a 2D array with shape (n_samples, n_features)
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)

            # Fix: If each element is a numpy array, stack them
            if X_array.shape[1] == 1 and isinstance(X_array[0, 0], np.ndarray):
                logger.debug("Stacking feature vectors from array of arrays to 2D matrix.")
                X_array = np.vstack(X_array[:, 0])

            # Remove detailed element logging for cleaner logs

            # Check for sequences in the array (keep this for safety, but only log if found)
            for i in range(X_array.shape[0]):
                for j in range(X_array.shape[1]):
                    if isinstance(X_array[i, j], (list, np.ndarray, dict)):
                        logger.error(f"Problematic element at [{i},{j}]: type={type(X_array[i, j])}, value={X_array[i, j]}")

            # Explicitly convert to float to ensure numeric dtype
            try:
                X_array = X_array.astype(float)
            except Exception as e:
                logger.error(f"Failed to convert feature array to float: {e}")
                return 0.0

            # Create a proper DataFrame with numeric data only
            feature_df = pd.DataFrame(X_array)

            # Log DataFrame head and dtypes for debugging
            logger.debug(f"Feature DataFrame head:\n{feature_df.head()}")
            logger.debug(f"Feature DataFrame dtypes:\n{feature_df.dtypes}")

            # Check for object dtype or any cell with a sequence
            if any(feature_df.dtypes == 'object'):
                logger.error("Feature DataFrame contains non-numeric columns. Aborting anomaly detection.")
                return 0.0
            for col in feature_df.columns:
                if feature_df[col].apply(lambda x: isinstance(x, (list, np.ndarray, dict))).any():
                    logger.error(f"Feature DataFrame column {col} contains sequences. Aborting anomaly detection.")
                    return 0.0

            # Use LogAI's detector with proper error handling
            self.anomaly_detector.fit(feature_df)
            result = self.anomaly_detector.predict(feature_df)
            
            # Get prediction for the latest log entry
            if len(result) > 0:
                # Get the prediction for the latest log
                if isinstance(result, pd.DataFrame):
                    latest_prediction = result.iloc[-1]
                    # Check if it's a Series with multiple items
                    if isinstance(latest_prediction, pd.Series) and len(latest_prediction) > 0:
                        anomaly_score = 1.0 if latest_prediction.iloc[0] < 0 else 0.0
                    else:
                        # Direct access if it's a simple value
                        anomaly_score = 1.0 if latest_prediction < 0 else 0.0
                else:
                    # Handle numpy array case
                    latest_prediction = result[-1]
                    anomaly_score = 1.0 if latest_prediction < 0 else 0.0
                    
                return anomaly_score

        except Exception as e:
            # Log the full traceback for better debugging
            logger.exception(f"Error in anomaly detection: {e}")

        return 0.0

    def batch_process_logs(self, log_lines: list, source: str = "application") -> list:
        """
        Batch process log lines: group by template, call LLM once per group, cache result.
        Returns a list of processed log dicts.
        """
        from collections import defaultdict
        processed_logs = []
        template_groups = defaultdict(list)
        # Step 1: Group logs by template hash (using Drain or simple hash)
        for log_line in log_lines:
            # Use Drain parser to get template (or fallback to message)
            try:
                _, _, message = self._extract_log_metadata(log_line)
                cluster = self.parser.match(message)
                template = cluster.template if cluster else message
            except Exception:
                template = log_line
            template_hash = hash(template)
            template_groups[template_hash].append((log_line, template))
        # Step 2: For each group, call LLM once and cache result
        for group in template_groups.values():
            # Use the first log's message as representative
            log_line, template = group[0]
            semantic_template = self.get_llm_template(template)
            # Step 3: Process all logs in the group with the cached template
            for log_line, _ in group:
                result = self.process_log(log_line, source)
                result["semantic_template"] = semantic_template  # Overwrite with batched/cached template
                processed_logs.append(result)
        return processed_logs