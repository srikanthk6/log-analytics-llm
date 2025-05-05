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
from log_analytics.human_format import format_log_query_response

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
        human = request.args.get('human', 'false').lower() == 'true'
        # Generate vector from query
        query_vector = logai_handler._generate_vector_embedding(query).tolist()
        # Search similar logs
        similar_logs = es_handler.search_similar_logs(query_vector, top_k=size)
        if human:
            api_response = {"results": similar_logs}
            return format_log_query_response(api_response), 200, {'Content-Type': 'text/plain; charset=utf-8'}
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
    """Analyze logs with LLM (llama3.1 via Ollama)"""
    import requests
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"status": "error", "message": "Query parameter is required"}), 400
        query = data['query']
        size = int(data.get('size', 5))
        human = request.args.get('human', 'false').lower() == 'true'
        # Step 1: Find relevant logs using vector search
        query_vector = logai_handler._generate_vector_embedding(query).tolist()
        similar_logs = es_handler.search_similar_logs(query_vector, top_k=size)
        if not similar_logs:
            if human:
                return "No relevant logs found.", 200, {'Content-Type': 'text/plain; charset=utf-8'}
            return jsonify({"status": "success", "analysis": "No relevant logs found.", "logs": []})
        # Step 2: Format logs for LLM prompt
        log_context = "\n".join([
            f"[{log['_source'].get('timestamp', '?')}] [{log['_source'].get('level', '?')}] {log['_source'].get('message', '')}"
            for log in similar_logs
        ])
        prompt = f"""
You are an expert SRE and log analyst. Analyze the following logs in relation to this query: '{query}'\n\nLOGS:\n{log_context}\n\nProvide a detailed analysis explaining:\n1. What issues are present in these logs\n2. Potential root causes\n3. Recommended actions to resolve any problems\n"""
        # Step 3: Send prompt to Ollama llama3.1
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.1',
                'prompt': prompt,
                'stream': False
            },
            timeout=120
        )
        if response.status_code == 200:
            llm_result = response.json().get('response', '').strip()
        else:
            llm_result = f"LLM error: {response.status_code} {response.text}"
        if human:
            lines = ["LLM Analysis:\n" + llm_result, "\nRelevant Logs:"]
            api_response = {"results": similar_logs}
            lines.append(format_log_query_response(api_response))
            return "\n\n".join(lines), 200, {'Content-Type': 'text/plain; charset=utf-8'}
        return jsonify({
            "status": "success",
            "analysis": llm_result,
            "logs": [log['_source'] for log in similar_logs]
        })
    except Exception as e:
        logger.error(f"Error analyzing logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def start_api(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask API server"""
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    start_api(debug=True)