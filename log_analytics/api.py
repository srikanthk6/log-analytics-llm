"""
REST API for log analytics system
"""
import json
import logging
from flask import Flask, request, jsonify

from log_analytics.config.config import (
    ELASTICSEARCH_HOST,
    ELASTICSEARCH_PORT,
    ELASTICSEARCH_INDEX,
    ANOMALY_THRESHOLD
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

app = Flask(__name__)

# Initialize handlers
es_handler = ElasticsearchHandler(
    host=ELASTICSEARCH_HOST,
    port=ELASTICSEARCH_PORT, 
    index_name=ELASTICSEARCH_INDEX
)
logai_handler = LogAIHandler()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/logs/recent', methods=['GET'])
def get_recent_logs():
    """Get recent logs endpoint"""
    try:
        size = int(request.args.get('size', 100))
        logs = es_handler.get_recent_logs(size=size)
        return jsonify({
            "status": "success",
            "count": len(logs),
            "logs": logs
        })
    except Exception as e:
        logger.error(f"Error getting recent logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/logs/anomalies', methods=['GET'])
def get_anomalies():
    """Get anomaly logs endpoint"""
    try:
        threshold = float(request.args.get('threshold', ANOMALY_THRESHOLD))
        size = int(request.args.get('size', 100))
        
        anomalies = es_handler.get_anomalies(threshold=threshold, size=size)
        
        return jsonify({
            "status": "success",
            "count": len(anomalies),
            "anomalies": anomalies
        })
    except Exception as e:
        logger.error(f"Error getting anomalies: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/logs/query', methods=['POST'])
def query_logs():
    """Search logs with vector similarity"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"status": "error", "message": "Query parameter is required"}), 400
            
        query = data['query']
        size = int(data.get('size', 10))
        
        # Generate vector from query
        query_vector = logai_handler._generate_vector_embedding(query).tolist()
        
        # Search similar logs
        similar_logs = es_handler.search_similar_logs(query_vector, top_k=size)
        
        return jsonify({
            "status": "success",
            "count": len(similar_logs),
            "results": similar_logs
        })
    except Exception as e:
        logger.error(f"Error querying logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/logs/analyze', methods=['POST'])
def analyze_logs():
    """Analyze logs with LLM (placeholder)"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"status": "error", "message": "Query parameter is required"}), 400
            
        query = data['query']
        size = int(data.get('size', 5))
        
        # In a production environment, this would:
        # 1. Find relevant logs based on the query
        # 2. Send them to an LLM API for analysis
        # 3. Return the LLM's response
        
        # Placeholder response
        return jsonify({
            "status": "success",
            "analysis": f"Analysis for query: '{query}' would be performed by an LLM in production.",
            "query": query,
            "size": size
        })
    except Exception as e:
        logger.error(f"Error analyzing logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def start_api(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask API server"""
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    start_api(debug=True)