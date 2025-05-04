"""
Log parsing and anomaly detection using LogAI
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Update imports with correct class names from LogAI and include parameter classes
from logai.dataloader.data_model import LogRecordObject
from logai.preprocess.preprocessor import Preprocessor, PreprocessorConfig
from logai.algorithms.parsing_algo.drain import Drain, DrainParams
from logai.algorithms.vectorization_algo.tfidf import TfIdf, TfIdfParams
# Import scikit-learn's IsolationForest directly for our custom detector
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# Custom detector class to bypass LogAI's IsolationForestDetector issue
class CustomIsolationForestDetector:
    """Simple wrapper for scikit-learn's IsolationForest for anomaly detection"""
    def __init__(self, n_estimators=100, contamination=0.05):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            warm_start=False,  # Explicitly set to False
            n_jobs=-1  # Use all available cores
        )
        self.is_fitted = False
    
    def fit(self, X):
        """Fit the model with the data X"""
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict if observations are anomalies or not"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def decision_function(self, X):
        """Compute the anomaly score for each observation"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.decision_function(X)

class LogAIHandler:
    def __init__(self, window_size: int = 100):
        """Initialize LogAI components"""
        self.window_size = window_size
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
        
        # Use our custom detector instead of LogAI's IsolationForestDetector
        self.anomaly_detector = CustomIsolationForestDetector(
            n_estimators=100,
            contamination=0.05
        )
        
        # Storage for log data
        self.logs_buffer = []
        
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
                "anomaly_score": float(anomaly_score)
            }
            
        except Exception as e:
            logger.error(f"Error processing log: {e}")
            # Return minimal data if processing fails
            return {
                "timestamp": datetime.now().isoformat(),
                "message": log_line,
                "source": source,
                "level": "ERROR",
                "log_vector": np.zeros(768).tolist(),  # Empty vector
                "anomaly_score": 0.0
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
        Generate vector embedding for the log message
        
        In a real implementation, this would use a more sophisticated embedding model
        For simplicity, we're using a basic approach here
        """
        # Simple hashing-based vector as placeholder
        # In production, use proper embedding models like transformers
        vector_size = 768  # Common embedding dimension
        vector = np.zeros(vector_size)
        
        # Simple hash-based encoding (for demonstration only)
        for i, char in enumerate(message):
            vector[i % vector_size] += ord(char) / 255.0
            
        # Normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def _detect_anomalies(self) -> float:
        """
        Detect anomalies in the current log buffer using our custom detector.

        Returns:
            Anomaly score for the latest log (0 for normal, 1 for anomaly).
        """
        try:
            if len(self.logs_buffer) < 10:  # Need minimum data for detection
                return 0.0

            # Convert buffer to DataFrame for LogAI
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

            # Convert to numpy array for scikit-learn
            # First check if X is a DataFrame or Series
            if isinstance(X, (pd.DataFrame, pd.Series)):
                # Handle case where X might contain lists or arrays
                try:
                    # Try direct numpy conversion
                    X_array = np.vstack([np.array(x).flatten() for x in X])
                except (ValueError, TypeError):
                    # If that fails, try to extract values one by one
                    rows = []
                    for x in X:
                        if isinstance(x, (list, np.ndarray)):
                            rows.append(np.array(x).flatten())
                        else:
                            # If a single scalar value, make it a 1D array
                            rows.append(np.array([x]))
                    
                    # Stack rows of potentially different lengths
                    # Pad with zeros if necessary
                    max_len = max(len(row) for row in rows)
                    X_array = np.zeros((len(rows), max_len))
                    for i, row in enumerate(rows):
                        X_array[i, :len(row)] = row
            
            # If X is a scipy sparse matrix, convert to dense
            elif hasattr(X, 'toarray'):
                X_array = X.toarray()
            # If X is already a numpy array
            elif isinstance(X, np.ndarray):
                X_array = X
            else:
                # As a fallback, try direct conversion
                X_array = np.array(X)
            
            # Ensure we have a 2D array with shape (n_samples, n_features)
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)

            # Check if we have valid data before proceeding
            if X_array.size == 0 or np.any(np.isnan(X_array)):
                logger.warning("Invalid data for anomaly detection. Using default score.")
                return 0.0

            # Detect anomalies with our custom detector
            self.anomaly_detector.fit(X_array)
            predictions = self.anomaly_detector.predict(X_array)

            # Get prediction for the latest log entry (last in the buffer)
            if predictions is not None and len(predictions) > 0:
                latest_prediction = predictions[-1]
                # Convert prediction label to a score (1 for anomaly, 0 for normal)
                anomaly_score = 1.0 if latest_prediction == -1 else 0.0
                return anomaly_score

        except Exception as e:
            # Log the full traceback for better debugging
            logger.exception(f"Error in anomaly detection: {e}")

        return 0.0