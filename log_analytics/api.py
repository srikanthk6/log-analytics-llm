"""
REST API for log analytics system
"""
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

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
CORS(app)

# Initialize handlers
es_handler = ElasticsearchHandler(
    host=ELASTICSEARCH_HOST,
    port=ELASTICSEARCH_PORT, 
    index_name=ELASTICSEARCH_INDEX
)
logai_handler = LogAIHandler(window_size=1)

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
    """Search logs with vector similarity and/or string match for IDs/keywords"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"status": "error", "message": "Query parameter is required"}), 400
        query = data['query']
        size = int(data.get('size', 10))
        human = request.args.get('human', 'false').lower() == 'true'

        # Extract possible ID/keyword tokens from the query (e.g., UUID, ORD, CUST, etc.)
        import re
        tokens = re.findall(r"[a-f0-9\-]{36}|ORD\d+|CUST\d+|\w{6,}", query, re.IGNORECASE)
        should_clauses = []
        for token in tokens:
            should_clauses.append({"match_phrase": {"message": token}})
            should_clauses.append({"match_phrase": {"trace_id": token}})
            should_clauses.append({"match_phrase": {"order_number": token}})
            should_clauses.append({"match_phrase": {"customer_id": token}})
        # String/OR search if any tokens found
        string_hits = []
        if should_clauses:
            es_query = {
                "size": size,
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                }
            }
            try:
                response = es_handler.es.search(index=es_handler.index_name, body=es_query)
                string_hits = response["hits"]["hits"]
            except Exception as e:
                logger.error(f"Error in string/OR search: {e}")
                string_hits = []
        # Vector search as well
        query_vector = logai_handler._generate_vector_embedding(query).tolist()
        vector_hits = es_handler.search_similar_logs(query_vector, top_k=size)
        # Combine results, avoiding duplicates (by _id), then take top N unique logs
        all_hits = {}
        for hit in string_hits + vector_hits:
            all_hits[hit['_id']] = hit
        combined_hits = list(all_hits.values())[:size]
        # Remove 'semantic_template' and 'log_vector' from log data for LLM context, and remove duplicates
        def clean_log_source(src):
            src = dict(src)
            src.pop('semantic_template', None)
            src.pop('log_vector', None)
            return src
        # Remove duplicate logs based on their cleaned JSON string
        seen_logs = set()
        unique_logs = []
        for log in combined_hits:
            cleaned = clean_log_source(log['_source'])
            jstr = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
            if jstr not in seen_logs:
                seen_logs.add(jstr)
                unique_logs.append(log)
        log_context = "\n".join([
            json.dumps(clean_log_source(log['_source']), ensure_ascii=False)
            for log in unique_logs
        ])
        if human:
            api_response = {"results": unique_logs}
            return format_log_query_response(api_response), 200, {'Content-Type': 'text/plain; charset=utf-8'}
        return jsonify({
            "status": "success",
            "count": len(unique_logs),
            "results": unique_logs,
            "log_context": log_context  # For dev/debug only
        })
    except Exception as e:
        logger.error(f"Error querying logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/logs/analyze', methods=['POST'])
def analyze_logs():
    """Analyze logs with LLM (llama3.1 via Ollama), using both string/OR and vector search results"""
    import requests
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"status": "error", "message": "Query parameter is required"}), 400
        query = data['query']
        size = int(data.get('size', 5))
        human = request.args.get('human', 'false').lower() == 'true'
        # Extract possible ID/keyword tokens from the query (e.g., UUID, ORD, CUST, etc.)
        import re
        tokens = re.findall(r"[a-f0-9\-]{36}|ORD\d+|CUST\d+|\w{6,}", query, re.IGNORECASE)
        should_clauses = []
        for token in tokens:
            should_clauses.append({"match_phrase": {"message": token}})
            should_clauses.append({"match_phrase": {"trace_id": token}})
            should_clauses.append({"match_phrase": {"order_number": token}})
            should_clauses.append({"match_phrase": {"customer_id": token}})
        # String/OR search if any tokens found
        string_hits = []
        if should_clauses:
            es_query = {
                "size": size,
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                }
            }
            try:
                response = es_handler.es.search(index=es_handler.index_name, body=es_query)
                string_hits = response["hits"]["hits"]
            except Exception as e:
                logger.error(f"Error in string/OR search: {e}")
                string_hits = []
        # Vector search as well
        query_vector = logai_handler._generate_vector_embedding(query).tolist()
        vector_hits = es_handler.search_similar_logs(query_vector, top_k=size)
        # Combine results, avoiding duplicates (by _id), then take top N unique logs
        all_hits = {}
        for hit in string_hits + vector_hits:
            all_hits[hit['_id']] = hit
        combined_hits = list(all_hits.values())[:size]
        if not combined_hits:
            if human:
                return "No relevant logs found.", 200, {'Content-Type': 'text/plain; charset=utf-8'}
            return jsonify({"status": "success", "analysis": "No relevant logs found.", "logs": []})
        # Step 2: Format logs for LLM prompt
        # Remove 'semantic_template' and 'log_vector' from log data for LLM context, and remove duplicates
        def clean_log_source(src):
            src = dict(src)
            src.pop('semantic_template', None)
            src.pop('log_vector', None)
            return src
        # Remove duplicate logs based on their cleaned JSON string
        seen_logs = set()
        unique_logs = []
        for log in combined_hits:
            cleaned = clean_log_source(log['_source'])
            jstr = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
            if jstr not in seen_logs:
                seen_logs.add(jstr)
                unique_logs.append(log)
        log_context = "\n".join([
            json.dumps(clean_log_source(log['_source']), ensure_ascii=False)
            for log in unique_logs
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
            api_response = {"results": unique_logs}
            lines.append(format_log_query_response(api_response))
            return "\n\n".join(lines), 200, {'Content-Type': 'text/plain; charset=utf-8'}
        return jsonify({
            "status": "success",
            "analysis": llm_result,
            "logs": [log['_source'] for log in unique_logs]
        })
    except Exception as e:
        logger.error(f"Error analyzing logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Conversational Log Analytics Chatbot Endpoint"""
    import requests
    import re
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"status": "error", "message": "Message parameter is required"}), 400
        user_message = data['message']
        history = data.get('history', [])  # List of {"role": "user"|"assistant", "content": ...}
        size = int(data.get('size', 50))
        # Extract possible ID/keyword tokens from the user message (e.g., UUID, ORD, CUST, etc.)
        tokens = re.findall(r"[a-f0-9\-]{36}|ORD\d+|CUST\d+|\w{6,}", user_message, re.IGNORECASE)
        should_clauses = []
        for token in tokens:
            should_clauses.append({"match_phrase": {"message": token}})
            should_clauses.append({"match_phrase": {"trace_id": token}})
            should_clauses.append({"match_phrase": {"order_number": token}})
            should_clauses.append({"match_phrase": {"customer_id": token}})
        # String/OR search if any tokens found
        string_hits = []
        if should_clauses:
            es_query = {
                "size": size,
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                }
            }
            try:
                response = es_handler.es.search(index=es_handler.index_name, body=es_query)
                string_hits = response["hits"]["hits"]
            except Exception as e:
                logger.error(f"Error in string/OR search: {e}")
                string_hits = []
        # Vector search as well
        query_vector = logai_handler._generate_vector_embedding(user_message).tolist()
        vector_hits = es_handler.search_similar_logs(query_vector, top_k=size)
        # Combine results, avoiding duplicates (by _id), then take top N unique logs
        all_hits = {}
        for hit in string_hits + vector_hits:
            all_hits[hit['_id']] = hit
        combined_hits = list(all_hits.values())[:size]

        # Remove 'semantic_template' and 'log_vector' from log data for LLM context, and remove duplicates
        def clean_log_source(src):
            src = dict(src)
            src.pop('semantic_template', None)
            src.pop('log_vector', None)
            return src
        # Remove duplicate logs based on their cleaned JSON string
        seen_logs = set()
        unique_logs = []
        for log in combined_hits:
            cleaned = clean_log_source(log['_source'])
            jstr = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
            if jstr not in seen_logs:
                seen_logs.add(jstr)
                unique_logs.append(log)
        log_context = "\n".join([
            json.dumps(clean_log_source(log['_source']), ensure_ascii=False)
            for log in unique_logs
        ])
        # Build conversation prompt
        conversation = "".join([
            f"{turn['role'].capitalize()}: {turn['content']}\n" for turn in history
        ])
        # Avoid repeating the latest user message if it's already in history
        if not history or history[-1].get('role') != 'user' or history[-1].get('content') != user_message:
            conversation += f"User: {user_message}\n"
        prompt = f"""
You are a helpful log analytics assistant. Use the following logs to answer the user's question or help troubleshoot issues.\n\nRelevant logs:\n{log_context}\n\nConversation so far:\n{conversation}Assistant: """
        # Call LLM (Ollama or other backend)
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
            llm_reply = response.json().get('response', '').strip()
        else:
            llm_reply = f"LLM error: {response.status_code} {response.text}"
        return jsonify({
            "status": "success",
            "reply": llm_reply,
            "logs": [clean_log_source(log['_source']) for log in unique_logs],
            "prompt": prompt  # For dev/debug only
        })
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def start_api(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask API server"""
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    start_api(debug=True)