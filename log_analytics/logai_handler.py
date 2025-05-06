"""
Log parsing and anomaly detection using LogAI
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime
import os
import re
from difflib import SequenceMatcher
import json

# Update imports with correct class names from LogAI
from logai.dataloader.data_model import LogRecordObject
from logai.preprocess.preprocessor import Preprocessor, PreprocessorConfig
from logai.algorithms.parsing_algo.drain import Drain, DrainParams
from logai.algorithms.vectorization_algo.tfidf import TfIdf, TfIdfParams
# Import LogAI's native IsolationForestDetector and its params class
from logai.algorithms.anomaly_detection_algo.isolation_forest import IsolationForestDetector, IsolationForestParams

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(os.path.dirname(__file__), 'llm_template_cache.json')

class LogAIHandler:
    ABBREVIATION_MAP = {
        "conn": "connection",
        "svc": "service",
        "auth": "authentication",
        "db": "database",
        "cfg": "configuration",
        "msg": "message",
        "usr": "user",
        "pwd": "password",
        "err": "error",
        "req": "request",
        "resp": "response",
        "init": "initialize",
        "proc": "process",
        "agg": "aggregate",
        "num": "number",
        "mem": "memory",
        "perf": "performance",
        "temp": "temporary",
        "dest": "destination",
        "src": "source",
        "info": "information",
        "cfg": "configuration",
        "upd": "update",
        "del": "delete",
        "add": "add",
        "rm": "remove",
        "stat": "status",
        "env": "environment",
        "prod": "production",
        "dev": "development",
        "test": "testing",
        "qa": "quality_assurance"
    }

    # Canonical field mapping for normalization
    FIELD_MAP = {
        # Order/order number
        'orderNumber': 'order_number',
        'order_number': 'order_number',
        'order-id': 'order_number',
        'orderid': 'order_number',
        'ordernum': 'order_number',
        'orderNo': 'order_number',
        'order_no': 'order_number',
        # Customer/customer id
        'customerId': 'customer_id',
        'customer_id': 'customer_id',
        'custId': 'customer_id',
        'cust_id': 'customer_id',
        'userId': 'customer_id',
        'user_id': 'customer_id',
        'buyerId': 'customer_id',
        'buyer_id': 'customer_id',
        # Status/order status
        'status': 'order_status',
        'order_status': 'order_status',
        'orderState': 'order_status',
        'order_state': 'order_status',
        # Hold reason
        'holdReason': 'hold_reason',
        'hold_reason': 'hold_reason',
        'reason': 'hold_reason',
        'holdreason': 'hold_reason',
        # Trace/trace id
        'traceId': 'trace_id',
        'trace_id': 'trace_id',
        'traceID': 'trace_id',
        'correlationId': 'trace_id',
        'correlation_id': 'trace_id',
        # Shipping
        'shippingProvider': 'shipping_provider',
        'shipping_provider': 'shipping_provider',
        'shipper': 'shipping_provider',
        'carrier': 'shipping_provider',
        'trackingNumber': 'tracking_number',
        'tracking_number': 'tracking_number',
        'trackingNo': 'tracking_number',
        'tracking_no': 'tracking_number',
        # Order lines/items
        'lines': 'order_lines',
        'order_lines': 'order_lines',
        'items': 'order_lines',
        'orderItems': 'order_lines',
        'order_items': 'order_lines',
        'products': 'order_lines',
        # Amount/price/total
        'amount': 'amount',
        'total': 'amount',
        'orderTotal': 'amount',
        'order_total': 'amount',
        'price': 'amount',
        'grandTotal': 'amount',
        'grand_total': 'amount',
        # Application/service names
        'application_name': 'application_name',
        'app': 'application_name',
        'service': 'application_name',
        'svc': 'application_name',
        'component': 'application_name',
        'module': 'application_name',
        # Level/severity
        'level': 'level',
        'severity': 'level',
        'logLevel': 'level',
        'log_level': 'level',
        # Timestamp
        'timestamp': 'timestamp',
        'time': 'timestamp',
        'datetime': 'timestamp',
        'createdAt': 'timestamp',
        'created_at': 'timestamp',
        # Message
        'message': 'message',
        'msg': 'message',
        'log': 'message',
        'description': 'message',
        'details': 'message',
        # Payment
        'paymentId': 'payment_id',
        'payment_id': 'payment_id',
        'transactionId': 'payment_id',
        'transaction_id': 'payment_id',
        # Address
        'address': 'address',
        'shippingAddress': 'address',
        'shipping_address': 'address',
        'deliveryAddress': 'address',
        'delivery_address': 'address',
        # Product
        'productId': 'product_id',
        'product_id': 'product_id',
        'sku': 'product_id',
        # Quantity
        'qty': 'quantity',
        'quantity': 'quantity',
        'count': 'quantity',
        # Misc
        'user': 'customer_id',
        'buyer': 'customer_id',
        'seller': 'seller_id',
        'sellerId': 'seller_id',
        'seller_id': 'seller_id',
    }

    # List of regex patterns for common log formats
    LOG_PATTERNS = [
        # log4j2-style key-value
        re.compile(r'^(?P<timestamp>\S+ \S+),\d+ (?P<level>\w+) (?P<application_name>\S+)(?P<kv>.*) msg="(?P<message>.*)"$'),
        # Syslog
        re.compile(r'^(?P<timestamp>\w{3} +\d+ \d{2}:\d{2}:\d{2}) (?P<host>\S+) (?P<application_name>\S+): (?P<message>.*)$'),
        # Apache/Nginx access log
        re.compile(r'^(?P<ip>\S+) - - \[(?P<timestamp>[^\]]+)\] "(?P<request>[^"]+)" (?P<status>\d+) (?P<size>\d+)(?: "(?P<referrer>[^"]*)" "(?P<agent>[^"]*)")?'),
    ]

    def __init__(self, window_size: int = 100, embedding_model_name: str = None, llm_model_name: str = None):
        """Initialize LogAI components"""
        self.window_size = window_size
        self.template_cache = {}
        self.load_template_cache()
        from sentence_transformers import SentenceTransformer
        embedding_model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.llm_model_name = llm_model_name or os.getenv("LLM_MODEL", "llama3.1")
        self._setup_components()
        
    def load_template_cache(self):
        """Load LLM template cache from file if it exists."""
        if os.path.isfile(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    self.template_cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load LLM template cache: {e}")

    def save_template_cache(self):
        """Save LLM template cache to file."""
        try:
            # Write a static copy to avoid 'dictionary changed size during iteration'
            cache_copy = dict(self.template_cache)
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache_copy, f)
        except Exception as e:
            logger.warning(f"Failed to save LLM template cache: {e}")

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
        
    def normalize_abbreviations(self, text: str) -> str:
        """Expand common abbreviations in a string."""
        def repl(match):
            word = match.group(0)
            return self.ABBREVIATION_MAP.get(word.lower(), word)
        return re.sub(r'\b(' + '|'.join(self.ABBREVIATION_MAP.keys()) + r')\b', repl, text, flags=re.IGNORECASE)

    def normalize_fields(self, d):
        """Normalize dictionary keys to canonical field names. Handles dynamic changes safely."""
        # Defensive: if d is not a dict, return as is
        if not isinstance(d, dict):
            return d
        # Build a new dictionary to avoid modifying during iteration
        out = {}
        for k, v in list(d.items()):
            # If value is a dict, normalize recursively on a copy
            if isinstance(v, dict):
                v = self.normalize_fields(dict(v))
            # If value is a list of dicts, normalize each on a copy
            elif isinstance(v, list):
                v = [self.normalize_fields(dict(i)) if isinstance(i, dict) else i for i in v]
            out[self.FIELD_MAP.get(k, k)] = v
        return out

    def are_strings_similar(self, s1: str, s2: str, threshold: float = 0.85) -> bool:
        """Return True if two strings are similar above a threshold (using SequenceMatcher)."""
        s1_norm = self.normalize_abbreviations(s1.lower())
        s2_norm = self.normalize_abbreviations(s2.lower())
        ratio = SequenceMatcher(None, s1_norm, s2_norm).ratio()
        return ratio >= threshold

    def parse_log_line(self, log_line: str) -> dict:
        """
        Try to parse a log line using multiple strategies: JSON, regex patterns, key-value, fallback to plain text.
        """
        try:
            # Try JSON
            if log_line.strip().startswith('{'):
                try:
                    d = json.loads(log_line)
                    return self.normalize_fields(d)
                except Exception:
                    pass
            # Try regex patterns
            for pattern in self.LOG_PATTERNS:
                m = pattern.match(log_line)
                if m:
                    d = m.groupdict()
                    # If log4j2-style, parse key-value pairs in 'kv'
                    if 'kv' in d and d['kv']:
                        kv_pattern = re.compile(r'(\w+)=((?:\".*?\")|(?:\[.*?\])|(?:\S+))')
                        new_fields = {}
                        for match in kv_pattern.finditer(d['kv']):
                            key = match.group(1)
                            val = match.group(2)
                            if val.startswith('"') and val.endswith('"'):
                                val = val[1:-1]
                            new_fields[key] = val
                        try:
                            d.update(new_fields)
                            del d['kv']
                        except RuntimeError as e:
                            if 'dictionary changed size during iteration' in str(e):
                                logger.info(f"Old keys: {list(d.keys())}, New keys: {list(new_fields.keys())}")
                                logger.error(f"Error processing log: {e}")
                                raise
                            else:
                                raise
                    return self.normalize_fields(d)
            # Try key-value pairs
            if '=' in log_line:
                d = self.parse_key_value_log(log_line)
                return self.normalize_fields(d)
            # Fallback: treat as plain text
            return {'message': log_line}
        except Exception as e:
            if 'dictionary changed size during iteration' in str(e):
                logger.error(f"Error processing log: {e}")
            raise

    def fast_semantic_template(self, message: str) -> str:
        """
        Quickly generate a semantic template using Drain3 (rule-based log parser).
        """
        try:
            from drain3 import TemplateMiner
            if not hasattr(self, '_drain3_template_miner'):
                # Initialize Drain3 TemplateMiner only once
                self._drain3_template_miner = TemplateMiner()
            result = self._drain3_template_miner.add_log_message(message)
            if result and result['template_mined']:
                return result['template_mined']
            elif result and result['template']:
                return result['template']
            else:
                return message
        except Exception as e:
            logger.warning(f"Drain3 template extraction failed: {e}")
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
        Use Drain3 to quickly generate a semantic template for a log message.
        """
        if message in self.template_cache:
            return self.template_cache[message]
        template = self.fast_semantic_template(message)
        self.template_cache[message] = template
        return template

    def parse_key_value_log(self, log_line: str) -> dict:
        """
        Parse a log4j2-style key-value log line into a dictionary.
        Handles quoted values and arrays.
        """
        result = {}
        # Extract timestamp, level, and app name (assume at start)
        m = re.match(r'^(\S+ \S+),\d+ (\w+) (\S+)', log_line)
        if m:
            result['timestamp'] = m.group(1)
            result['level'] = m.group(2)
            result['application_name'] = m.group(3)
            rest = log_line[m.end():].strip()
        else:
            rest = log_line
        # Find key=value pairs (handles quoted values and arrays)
        pattern = re.compile(r'(\w+)=((?:\".*?\")|(?:\[.*?\])|(?:\S+))')
        for match in pattern.finditer(rest):
            key = match.group(1)
            val = match.group(2)
            # Remove quotes if present
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            result[key] = val
        # Extract msg if present at the end
        msg_match = re.search(r' msg="(.*)"$', log_line)
        if msg_match:
            result['message'] = msg_match.group(1)
        return result

    def to_iso8601(self, ts):
        """Convert a timestamp string or datetime to ISO8601 format (YYYY-MM-DDTHH:MM:SS[.ffffff][Z])"""
        if isinstance(ts, datetime):
            return ts.isoformat()
        if isinstance(ts, str):
            # Try common formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S,%f",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
            ]:
                try:
                    dt = datetime.strptime(ts, fmt)
                    return dt.isoformat()
                except Exception:
                    continue
        # If all fails, fallback to now
        return datetime.now().isoformat()

    def process_log(self, log_line: str, source: str = "application") -> Dict[str, Any]:
        """
        Process a single log line and return a dictionary with only the fixed set of fields.
        """
        try:
            log_data = self.parse_log_line(log_line)
            # Always normalize fields again to ensure trace_id is present
            log_data = self.normalize_fields(log_data)
            message = log_data.get("message", log_data.get("raw", ""))
            application_name = log_data.get("application_name", source)
            trace_id = log_data.get("trace_id", None)
            order_number = log_data.get("order_number", None)
            customer_id = log_data.get("customer_id", None)
            order_status = log_data.get("order_status", None)
            amount = log_data.get("amount", None)
            order_lines = log_data.get("order_lines", None)
            hold_reason = log_data.get("hold_reason", None)
            shipping_provider = log_data.get("shipping_provider", None)
            tracking_number = log_data.get("tracking_number", None)
            timestamp = self.to_iso8601(log_data.get("timestamp", datetime.now().isoformat()))
            level = log_data.get("level", "INFO")
            timestamp_df = pd.DataFrame({'timestamp': [timestamp]})
            body_df = pd.DataFrame({'body': [message]})
            resource_df = pd.DataFrame({'source': [application_name]})
            severity_text_df = pd.DataFrame({'severity_text': [level]})
            log_record = LogRecordObject(
                timestamp=timestamp_df,
                body=body_df,
                resource=resource_df,
                severity_text=severity_text_df
            )
            self.logs_buffer.append(log_record)
            anomaly_score = 0.0
            log_vector = self._generate_vector_embedding(message)
            if np.linalg.norm(log_vector) == 0:
                logger.warning("Zero-magnitude vector detected, skipping log indexing.")
            if len(self.logs_buffer) >= self.window_size:
                anomaly_score = self._detect_anomalies()
                if len(self.logs_buffer) > self.window_size:
                    self.logs_buffer = self.logs_buffer[-self.window_size:]
            # Always return a new dictionary with only the fixed set of fields
            return {
                "timestamp": timestamp,
                "message": message,
                "source": application_name,
                "level": level,
                "log_vector": log_vector.tolist(),
                "anomaly_score": float(anomaly_score),
                "trace_id": trace_id,
                "order_number": order_number,
                "customer_id": customer_id,
                "order_status": order_status,
                "amount": amount,
                "order_lines": order_lines,
                "hold_reason": hold_reason,
                "shipping_provider": shipping_provider,
                "tracking_number": tracking_number
            }
        except Exception as e:
            logger.error(f"Error processing log: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "message": log_line,
                "source": source,
                "level": "ERROR",
                "log_vector": np.zeros(self.embedding_dim).tolist(),
                "anomaly_score": 0.0,
                "semantic_template": log_line,
                "trace_id": None,
                "order_number": None,
                "customer_id": None,
                "order_status": None,
                "amount": None,
                "order_lines": None,
                "hold_reason": None,
                "shipping_provider": None,
                "tracking_number": None
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
        Generate vector embedding for the log message using the configured embedding model.
        Prevent zero-magnitude vectors from being returned.
        """
        cached = self.es_handler.get_cached_embedding(message) if hasattr(self, 'es_handler') else None
        if cached:
            arr = np.array(cached, dtype=np.float32)
            if np.linalg.norm(arr) == 0:
                logger.warning("Cached embedding is zero vector, skipping.")
                return arr
            return arr
        try:
            embedding = self.embedding_model.encode(message, show_progress_bar=False, normalize_embeddings=True)
            arr = np.array(embedding, dtype=np.float32)
            if np.linalg.norm(arr) == 0:
                logger.warning("Embedding model returned zero vector, skipping.")
            if hasattr(self, 'es_handler'):
                self.es_handler.cache_embedding(message, arr.tolist())
            return arr
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
        try:
            if getattr(self, '_just_fitted', False):
                self._just_fitted = False
                logger.info("Skipping prediction for batch immediately after fit to avoid offset_ error.")
                return 0.0
            if len(self.logs_buffer) < 10:
                return 0.0
            df = pd.DataFrame([
                {
                    'timestamp': log_record.timestamp.iloc[0, 0] if not log_record.timestamp.empty else "",
                    'message': log_record.body.iloc[0, 0] if not log_record.body.empty else "",
                    'source': log_record.resource.iloc[0, 0] if not log_record.resource.empty else "",
                    'level': log_record.severity_text.iloc[0, 0] if not log_record.severity_text.empty else ""
                } for log_record in self.logs_buffer
            ])
            templates = []
            for message in df['message']:
                cluster = self.parser.match(message)
                templates.append(cluster.template if cluster else message)
            templates = [t for t in templates if isinstance(t, str) and t.strip() and t.strip().lower() not in {"", "the", "a", "an", "and", "or", "is", "are", "was", "were"}]
            if not templates:
                logger.warning("No valid templates for TF-IDF. Skipping anomaly detection for this batch.")
                return 0.0
            templates_series = pd.Series(templates)
            # Only fit TF-IDF and IsolationForest if not already fitted and feature count is stable
            if not hasattr(self, '_tfidf_fitted') or not self._tfidf_fitted:
                if len(templates_series) == 0:
                    logger.warning("No templates to fit TF-IDF. Skipping anomaly detection.")
                    return 0.0
                self.feature_extractor.fit(templates_series)
                X = self.feature_extractor.transform(templates_series)
                if len(X.shape) < 2 or X.shape[1] == 0:
                    logger.error("TF-IDF transform returned no features. Skipping anomaly detection.")
                    # logger.info(f"Templates for TF-IDF: {templates}")
                    return 0.0
                self._tfidf_fitted = True
                self._anomaly_feature_count = X.shape[1]
            X = self.feature_extractor.transform(templates_series)
            if hasattr(X, 'toarray'):
                X_array = X.toarray()
            else:
                X_array = np.array(X)
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
            if X_array.shape[1] == 1 and isinstance(X_array[0, 0], np.ndarray):
                X_array = np.vstack(X_array[:, 0])
            try:
                X_array = X_array.astype(float)
            except Exception as e:
                logger.error(f"Failed to convert feature array to float: {e}")
                return 0.0
            feature_df = pd.DataFrame(X_array)
            if feature_df.empty or feature_df.shape[1] == 0:
                logger.error("Feature DataFrame is empty. Skipping anomaly detection.")
                return 0.0
            feature_count = feature_df.shape[1]
            # Only re-fit if feature count changes
            if not hasattr(self, '_anomaly_feature_count') or feature_count != self._anomaly_feature_count:
                isolation_forest_params = IsolationForestParams(
                    n_estimators=100,
                    contamination=0.05,
                    random_state=42,
                    warm_start=False,
                    bootstrap=False,
                    n_jobs=None,
                    verbose=False
                )
                self.anomaly_detector = IsolationForestDetector(params=isolation_forest_params)
                self.anomaly_detector.fit(feature_df)
                self._anomaly_feature_count = feature_count
                self._anomaly_fitted = True
                self._just_fitted = True
                logger.info(f"Re-fitted IsolationForest with {feature_count} features. Skipping prediction for this batch.")
                return 0.0
            # Only predict if model is fitted and feature count matches
            if getattr(self, '_anomaly_fitted', False) and feature_count == self._anomaly_feature_count:
                if hasattr(self.anomaly_detector.model, "offset_"):
                    try:
                        result = self.anomaly_detector.predict(feature_df)
                        if len(result) > 0:
                            if isinstance(result, pd.DataFrame):
                                latest_prediction = result.iloc[-1]
                                if isinstance(latest_prediction, pd.Series) and len(latest_prediction) > 0:
                                    anomaly_score = 1.0 if latest_prediction.iloc[0] < 0 else 0.0
                                else:
                                    anomaly_score = 1.0 if latest_prediction < 0 else 0.0
                            else:
                                latest_prediction = result[-1]
                                anomaly_score = 1.0 if latest_prediction < 0 else 0.0
                            return anomaly_score
                    except Exception as e:
                        logger.exception(f"Error in anomaly detection predict: {e}")
                        return 0.0
                else:
                    logger.info("IsolationForest model not ready (offset_ missing), skipping prediction for this batch.")
                    return 0.0
        except Exception as e:
            logger.exception(f"Error in anomaly detection: {e}")
        return 0.0
