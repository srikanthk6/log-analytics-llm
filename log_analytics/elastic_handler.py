"""
Elasticsearch handler for storing and retrieving log data
"""
from elasticsearch import Elasticsearch
import json
import logging
from typing import Dict, List, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ElasticsearchHandler:
    def __init__(self, host: str = "localhost", port: int = 9200, index_name: str = "logs_vector_index", template_index_name: str = "semantic_templates_index"):
        """Initialize Elasticsearch connection"""
        self.host = host
        self.port = port
        self.index_name = index_name
        self.template_index_name = template_index_name
        self.es = None
        self.connect()

    def connect(self) -> None:
        """Connect to Elasticsearch"""
        try:
            self.es = Elasticsearch([f"http://{self.host}:{self.port}"], verify_certs=False)
            if not self.es.ping():
                logger.error("Could not connect to Elasticsearch")
                raise ConnectionError("Could not connect to Elasticsearch")
            logger.info("Connected to Elasticsearch")
            self._create_index_if_not_exists()
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            raise

    def _create_index_if_not_exists(self) -> None:
        """Create the index with vector mapping if it doesn't exist"""
        if not self.es.indices.exists(index=self.index_name):
            # Create index with mapping for vector fields and log data
            mapping = {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date", "format": "yyyy-MM-dd'T'HH:mm:ss.SSSZ"},
                        "message": {"type": "text"},
                        "log_vector": {
                            "type": "dense_vector",
                            "dims": 384,  # Updated for all-MiniLM-L6-v2
                            "index": True,
                            "similarity": "cosine"
                        },
                        "source": {"type": "keyword"},
                        "level": {"type": "keyword"},
                        "anomaly_score": {"type": "float"},
                        "semantic_template": {"type": "text"},
                        "template": {"type": "text"},
                        "llm_report": {"type": "text"},
                        "application_name": {"type": "keyword"},
                        "trace_id": {"type": "keyword"},
                        "order_number": {"type": "keyword"}
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created index: {self.index_name}")

    def _format_es_timestamp(self, dt):
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            except Exception:
                dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"

    def index_log(self, log_data: Dict[str, Any]) -> None:
        """Index a log entry with vector embedding"""
        try:
            # Ensure timestamp is in correct format
            if 'timestamp' in log_data:
                log_data['timestamp'] = self._format_es_timestamp(log_data['timestamp'])
            else:
                log_data['timestamp'] = self._format_es_timestamp(datetime.utcnow())
            res = self.es.index(index=self.index_name, document=log_data)
            # Only print a sample of the log indexing message occasionally
            import random # For reproducibility
            if random.random() < 0.01:
                logger.info(f"Sampling Data POST http://{self.host}:{self.port}/{self.index_name}/_doc [status:{res['result']} duration:~s]")
            return res
        except Exception as e:
            logger.error(f"Failed to index log: {e}")
            return None

    def search_similar_logs(self, vector, top_k=5, min_dt=None, max_dt=None) -> List[Dict[str, Any]]:
        """Search for logs similar to the given vector (Elasticsearch 8.x dense_vector compatible), with optional date range filter."""
        # Build the base query
        base_query = {"match_all": {}}
        # Add date range filter if present
        if min_dt and max_dt:
            min_dt_str = self._format_es_timestamp(min_dt)
            max_dt_str = self._format_es_timestamp(max_dt)
            base_query = {
                "bool": {
                    "filter": [
                        {"range": {"timestamp": {"gte": min_dt_str, "lte": max_dt_str}}}
                    ]
                }
            }
        query = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": base_query,
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'log_vector') + 1.0",
                        "params": {"query_vector": vector}
                    }
                }
            }
        }
        try:
            response = self.es.search(index=self.index_name, body=query)
            return response["hits"]["hits"]
        except Exception as e:
            logger.error(f"Error searching similar logs: {e}")
            return []

    def get_recent_logs(self, size=100) -> List[Dict[str, Any]]:
        """Get the most recent logs"""
        query = {
            "sort": [{"timestamp": {"order": "desc"}}],
            "size": size
        }
        try:
            response = self.es.search(index=self.index_name, **query)
            return response["hits"]["hits"]
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            return []

    def get_anomalies(self, threshold=0.8, size=100) -> List[Dict[str, Any]]:
        """Get logs with anomaly scores above the threshold"""
        query = {
            "query": {
                "range": {
                    "anomaly_score": {
                        "gte": threshold
                    }
                }
            },
            "sort": [{"timestamp": {"order": "desc"}}],
            "size": size
        }
        try:
            response = self.es.search(index=self.index_name, **query)
            return response["hits"]["hits"]
        except Exception as e:
            logger.error(f"Error getting anomalies: {e}")
            return []

    def get_cached_template(self, template: str) -> str:
        """Retrieve cached LLM template from Elasticsearch if exists."""
        try:
            query = {"query": {"term": {"semantic_template.keyword": template}}}
            resp = self.es.search(index=self.template_index_name, body=query, size=1)
            hits = resp.get("hits", {}).get("hits", [])
            if hits:
                return hits[0]["_source"].get("semantic_template", "")
        except Exception as e:
            logger.warning(f"Failed to retrieve cached template: {e}")
        return None

    def cache_template(self, template: str, semantic_template: str):
        """Store LLM template in Elasticsearch for future reuse."""
        try:
            doc = {"semantic_template": semantic_template, "template": template, "timestamp": self._format_es_timestamp(datetime.utcnow())}
            self.es.index(index=self.template_index_name, document=doc)
        except Exception as e:
            logger.warning(f"Failed to cache template: {e}")

    def get_cached_embedding(self, message: str):
        """Retrieve cached embedding from Elasticsearch if exists."""
        try:
            query = {"query": {"term": {"message.keyword": message}}}
            resp = self.es.search(index=self.index_name, body=query, size=1)
            hits = resp.get("hits", {}).get("hits", [])
            if hits:
                return hits[0]["_source"].get("log_vector", None)
        except Exception as e:
            logger.warning(f"Failed to retrieve cached embedding: {e}")
        return None

    def cache_embedding(self, message: str, log_vector):
        """Store embedding in Elasticsearch for future reuse."""
        try:
            doc = {"message": message, "log_vector": log_vector, "timestamp": self._format_es_timestamp(datetime.utcnow())}
            self.es.index(index=self.index_name, document=doc)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    def get_cached_llm_report(self, message: str):
        """Retrieve cached LLM anomaly report from Elasticsearch if exists."""
        try:
            query = {"query": {"term": {"message.keyword": message}}}
            resp = self.es.search(index=self.index_name, body=query, size=1)
            hits = resp.get("hits", {}).get("hits", [])
            if hits:
                return hits[0]["_source"].get("llm_report", None)
        except Exception as e:
            logger.warning(f"Failed to retrieve cached LLM report: {e}")
        return None

    def cache_llm_report(self, message: str, llm_report: str):
        """Store LLM anomaly report in Elasticsearch for future reuse."""
        try:
            doc = {"message": message, "llm_report": llm_report, "timestamp": self._format_es_timestamp(datetime.utcnow())}
            self.es.index(index=self.index_name, document=doc)
        except Exception as e:
            logger.warning(f"Failed to cache LLM report: {e}")

    def create_anomaly_detection_job(self, job_id, index_pattern, field_name, bucket_span="5m"):
        """Create an anomaly detection job in Elasticsearch."""
        job_config = {
            "description": f"Anomaly detection for {field_name} in {index_pattern}",
            "analysis_config": {
                "bucket_span": bucket_span,
                "detectors": [
                    {
                        "function": "mean",
                        "field_name": field_name
                    }
                ]
            },
            "data_description": {
                "time_field": "@timestamp"
            }
        }

        # Check if the job already exists
        response = self.es.transport.perform_request(
            "GET",
            f"/_ml/anomaly_detectors/{job_id}"
        )
        if hasattr(response, 'status_code') and response.status_code == 200:
            logger.info(f"Anomaly detection job {job_id} already exists.")
            return

        # Create the job
        response = self.es.transport.perform_request(
            "PUT",
            f"/_ml/anomaly_detectors/{job_id}",
            headers={"Content-Type": "application/json"},
            body=job_config
        )
        logger.info(f"Created anomaly detection job {job_id}: {response}")

        # Start the job
        response = self.es.transport.perform_request(
            "POST",
            f"/_ml/anomaly_detectors/{job_id}/_open",
            headers={"Content-Type": "application/json"}
        )
        logger.info(f"Started anomaly detection job {job_id}: {response}")