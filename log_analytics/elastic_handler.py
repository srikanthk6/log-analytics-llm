"""
Elasticsearch handler for storing and retrieving log data
"""
from elasticsearch import Elasticsearch
import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ElasticsearchHandler:
    def __init__(self, host: str = "localhost", port: int = 9200, index_name: str = "logs_vector_index"):
        """Initialize Elasticsearch connection"""
        self.host = host
        self.port = port
        self.index_name = index_name
        self.es = None
        self.connect()

    def connect(self) -> None:
        """Connect to Elasticsearch"""
        try:
            self.es = Elasticsearch([f"http://{self.host}:{self.port}"])
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
                        "timestamp": {"type": "date"},
                        "message": {"type": "text"},
                        "log_vector": {
                            "type": "dense_vector",
                            "dims": 384,  # Updated for all-MiniLM-L6-v2
                            "index": True,
                            "similarity": "cosine"
                        },
                        "source": {"type": "keyword"},
                        "level": {"type": "keyword"},
                        "anomaly_score": {"type": "float"}
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created index: {self.index_name}")

    def index_log(self, log_data: Dict[str, Any]) -> None:
        """Index a log entry with vector embedding"""
        try:
            self.es.index(index=self.index_name, document=log_data)
        except Exception as e:
            logger.error(f"Error indexing log: {e}")

    def search_similar_logs(self, vector, top_k=5) -> List[Dict[str, Any]]:
        """Search for logs similar to the given vector"""
        query = {
            "knn": {
                "field": "log_vector",
                "query_vector": vector,
                "k": top_k,
                "num_candidates": 100
            }
        }
        try:
            response = self.es.search(index=self.index_name, query=query)
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