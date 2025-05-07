"""
REST API for log analytics system
"""
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dateutil import parser as date_parser

from log_analytics.config.config import (
    ELASTICSEARCH_HOST,
    ELASTICSEARCH_PORT,
    ELASTICSEARCH_INDEX,
    ELASTICSEARCH_TEMPLATE_INDEX,  # <-- add this
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
    index_name=ELASTICSEARCH_INDEX,
    template_index_name=ELASTICSEARCH_TEMPLATE_INDEX  # <-- add this
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

# Helper to extract tokens (LLM or regex) and possible date filters
def extract_tokens_and_dates(query, llm_tokens=None):
    import re
    tokens = llm_tokens if llm_tokens else re.findall(r"[a-f0-9\-]{36}|ORD\d+|CUST\d+|\w{6,}", query, re.IGNORECASE)
    date_ranges = []
    for token in tokens:
        try:
            dt = date_parser.parse(token, fuzzy=False)
            date_ranges.append(dt)
        except Exception:
            continue
    return tokens, date_ranges

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
        llm_tokens = data.get('tokens')
        tokens, date_ranges = extract_tokens_and_dates(query, llm_tokens)
        # Use tokens with OR condition in ES search
        should_clauses = [
            {"multi_match": {
                "query": token,
                "fields": ["message", "trace_id", "order_number", "customer_id"],
                "type": "phrase"
            }} for token in tokens
        ]
        # String/OR search if any tokens found
        string_hits = []
        es_query = {
            "size": 100,
            "query": {
                "bool": {
                    # Use filter for date range to strictly enforce it
                    "filter": [],
                    "must": []
                }
            }
        }
        if should_clauses:
            es_query["query"]["bool"]["must"].append({
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            })
        if date_ranges:
            # Use min/max as range
            min_dt = min(date_ranges)
            max_dt = max(date_ranges)
            # Format as yyyy-MM-dd'T'HH:mm:ss.SSSZ
            min_dt_str = min_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
            max_dt_str = max_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
            es_query["query"]["bool"]["filter"].append({
                "range": {"timestamp": {"gte": min_dt_str, "lte": max_dt_str}}
            })
        try:
            response = es_handler.es.search(index=es_handler.index_name, body=es_query)
            string_hits = response["hits"]["hits"]
        except Exception as e:
            logger.error(f"Error in string/OR search: {e}")
            string_hits = []
        # If no tokens but date range is present, still perform ES search within date range
        if not tokens and date_ranges:
            logger.info("No tokens found from LLM, but date range present. Searching within date range only.")
            min_dt = min(date_ranges)
            max_dt = max(date_ranges)
            min_dt_str = min_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
            max_dt_str = max_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
            es_query = {
                "size": 100,
                "query": {
                    "bool": {
                        "filter": [
                            {"range": {"timestamp": {"gte": min_dt_str, "lte": max_dt_str}}}
                        ]
                    }
                }
            }
            logger.info(f"Elasticsearch Query (date range only): {json.dumps(es_query, indent=2)}")
            try:
                response = es_handler.es.search(index=es_handler.index_name, body=es_query)
                string_hits = response["hits"]["hits"]
            except Exception as e:
                logger.error(f"Error in date range only search: {e}")
                string_hits = []
            vector_hits = []
            all_hits = {}
            for hit in string_hits + vector_hits:
                all_hits[hit['_id']] = hit
            combined_hits = list(all_hits.values())
            def clean_log_source(src):
                src = dict(src)
                src.pop('semantic_template', None)
                src.pop('log_vector', None)
                return src
            seen_logs = set()
            unique_logs = []
            for log in combined_hits:
                cleaned = clean_log_source(log['_source'])
                jstr = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
                if jstr not in seen_logs:
                    seen_logs.add(jstr)
                    unique_logs.append(log)
            logs = [clean_log_source(log['_source']) for log in unique_logs]
            log_context = "\n".join([
                json.dumps(log, ensure_ascii=False) for log in logs
            ])
        # Vector search as well
        query_vector = logai_handler._generate_vector_embedding(query).tolist()
        vector_hits = es_handler.search_similar_logs(query_vector, top_k=100)
        # Combine results, avoiding duplicates (by _id), then take top N unique logs
        all_hits = {}
        for hit in string_hits + vector_hits:
            all_hits[hit['_id']] = hit
        combined_hits = list(all_hits.values())
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
        # Use tokens with OR condition in ES search
        should_clauses = [
            {"multi_match": {
                "query": token,
                "fields": ["message", "trace_id", "order_number", "customer_id"],
                "type": "phrase"
            }} for token in tokens
        ]
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
    """Conversational Log Analytics Chatbot Endpoint (improved: skip search if not needed, rewrite second prompt based on first LLM)"""
    import requests
    import re
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"status": "error", "message": "Message parameter is required"}), 400
        user_message = data['message']
        history = data.get('history', [])
        size = 100

        # 1. Use LLM to determine if search is needed and to rephrase/understand the user question
        llm_query_prompt = f"""
You are a helpful log analytics assistant. Given the following user message, decide if it requires searching logs. If it does, do the following:
1. Extract or rephrase it into a concise, effective search query for log retrieval (focus on keywords, error codes, IDs, or relevant context).
2. Identify and output a list of key tokens for Elasticsearch search. These tokens should include only key identifiers (such as order numbers, tracking numbers, trace IDs, error codes, etc.) and should exclude generic or irrelevant words.
3. If the user refers to a relative date (like 'yesterday', 'last week', 'today', etc.), convert it to an explicit date or date range (e.g., 'yesterday' → '2025-05-06' to '2025-05-06').

Respond in the following JSON format:
{{
  "NO_SEARCH_NEEDED": true // Only include if no search is needed
  "search_query": "...your concise search query...",
  "tokens": ["token1", "token2", ...],
  "date_range": {{"from": "yyyy-MM-dd'T'HH:mm:ss.SSSZ", "to": "yyyy-MM-dd'T'HH:mm:ss.SSSZ"}} // Only include if a date or range is detected
}}
If the message does NOT require a log search (e.g., it's a greeting, general question, or meta question), respond with ONLY the phrase NO_SEARCH_NEEDED.

Current date: 2025-05-07
User message: {user_message}

JSON response or NO_SEARCH_NEEDED:"""
        logger.info(f"LLM Query Prompt: {llm_query_prompt}")
        llm_query_resp = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.1',
                'prompt': llm_query_prompt,
                'stream': False
            },
            timeout=60
        )
        if llm_query_resp.status_code == 200:
            llm_query_result = llm_query_resp.json().get('response', '').strip()
        else:
            llm_query_result = user_message
        logger.info(f"LLM Query Response: {llm_query_result}")

        # Try to parse as JSON and check NO_SEARCH_NEEDED field
        no_search_needed = False
        llm_json = None
        try:
            llm_json = json.loads(llm_query_result)
            if isinstance(llm_json, dict) and llm_json.get("NO_SEARCH_NEEDED", False):
                no_search_needed = True
        except Exception:
            # If not JSON, check for plain NO_SEARCH_NEEDED string
            no_search_needed = llm_query_result.strip().upper() == 'NO_SEARCH_NEEDED'

        search_query = None if no_search_needed else llm_query_result
        logs = []
        log_context = ""
        # Parse LLM response for search_query, tokens, and date_range
        if not no_search_needed:
            if not llm_json:
                try:
                    llm_json = json.loads(llm_query_result)
                except Exception:
                    logger.error("LLM did not return valid JSON, falling back to string parsing.")
            tokens = []
            date_ranges = []
            if llm_json and isinstance(llm_json, dict):
                tokens = llm_json.get('tokens', [])
                if not isinstance(tokens, list):
                    tokens = []
                date_range = llm_json.get('date_range', {})
                if isinstance(date_range, dict) and 'from' in date_range and 'to' in date_range:
                    try:
                        from_dt = date_parser.parse(date_range['from'], fuzzy=False)
                        to_dt = date_parser.parse(date_range['to'], fuzzy=False)
                        date_ranges = [from_dt, to_dt]
                    except Exception:
                        date_ranges = []
                search_query = llm_json.get('search_query', '')
            # If no tokens but date range is present, still perform ES search within date range
            if not tokens and date_ranges:
                logger.info("No tokens found from LLM, but date range present. Searching within date range only.")
                min_dt = min(date_ranges)
                max_dt = max(date_ranges)
                min_dt_str = min_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
                max_dt_str = max_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
                es_query = {
                    "size": size,
                    "query": {
                        "bool": {
                            "filter": [
                                {"range": {"timestamp": {"gte": min_dt_str, "lte": max_dt_str}}}
                            ]
                        }
                    }
                }
                logger.info(f"Elasticsearch Query (date range only): {json.dumps(es_query, indent=2)}")
                try:
                    response = es_handler.es.search(index=es_handler.index_name, body=es_query)
                    string_hits = response["hits"]["hits"]
                except Exception as e:
                    logger.error(f"Error in date range only search: {e}")
                    string_hits = []
                vector_hits = []
                all_hits = {}
                for hit in string_hits + vector_hits:
                    all_hits[hit['_id']] = hit
                combined_hits = list(all_hits.values())
                def clean_log_source(src):
                    src = dict(src)
                    src.pop('semantic_template', None)
                    src.pop('log_vector', None)
                    return src
                seen_logs = set()
                unique_logs = []
                for log in combined_hits:
                    cleaned = clean_log_source(log['_source'])
                    jstr = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
                    if jstr not in seen_logs:
                        seen_logs.add(jstr)
                        unique_logs.append(log)
                logs = [clean_log_source(log['_source']) for log in unique_logs]
                log_context = "\n".join([
                    json.dumps(log, ensure_ascii=False) for log in logs
                ])
            else:
                logger.info(f"Tokens for ES search: {tokens}")
                should_clauses = [
                    {"multi_match": {
                        "query": token,
                        "fields": ["message", "trace_id", "order_number", "customer_id"],
                        "type": "phrase"
                    }} for token in tokens
                ]
                string_hits = []
                es_query = {
                    "size": size,
                    "query": {
                        "bool": {
                            "filter": [],
                            "must": []
                        }
                    }
                }
                if should_clauses:
                    es_query["query"]["bool"]["must"].append({
                        "bool": {
                            "should": should_clauses,
                            "minimum_should_match": 1
                        }
                    })
                if date_ranges:
                    min_dt = min(date_ranges)
                    max_dt = max(date_ranges)
                    # Format as yyyy-MM-dd'T'HH:mm:ss.SSSZ
                    min_dt_str = min_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
                    max_dt_str = max_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
                    es_query["query"]["bool"]["filter"].append({
                        "range": {"timestamp": {"gte": min_dt_str, "lte": max_dt_str}}
                    })
                logger.info(f"Elasticsearch Query: {json.dumps(es_query, indent=2)}")
                try:
                    response = es_handler.es.search(index=es_handler.index_name, body=es_query)
                    string_hits = response["hits"]["hits"]
                except Exception as e:
                    logger.error(f"Error in string/OR search: {e}")
                    string_hits = []
                if not string_hits:
                    query_vector = logai_handler._generate_vector_embedding(search_query).tolist()
                    if date_ranges:
                        min_dt = min(date_ranges)
                        max_dt = max(date_ranges)
                        vector_hits = es_handler.search_similar_logs(query_vector, top_k=size, min_dt=min_dt, max_dt=max_dt)
                    else:
                        vector_hits = es_handler.search_similar_logs(query_vector, top_k=size)
                else:
                    vector_hits = []
                all_hits = {}
                for hit in string_hits + vector_hits:
                    all_hits[hit['_id']] = hit
                combined_hits = list(all_hits.values())
                def clean_log_source(src):
                    src = dict(src)
                    src.pop('semantic_template', None)
                    src.pop('log_vector', None)
                    return src
                seen_logs = set()
                unique_logs = []
                for log in combined_hits:
                    cleaned = clean_log_source(log['_source'])
                    jstr = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
                    if jstr not in seen_logs:
                        seen_logs.add(jstr)
                        unique_logs.append(log)
                logs = [clean_log_source(log['_source']) for log in unique_logs]
                log_context = "\n".join([
                    json.dumps(log, ensure_ascii=False) for log in logs
                ])
        # 3. Use LLM again to generate a fine-tuned response, prompt depends on search or not
        conversation = "".join([
            f"{turn['role'].capitalize()}: {turn['content']}\n" for turn in history
        ])
        if not history or history[-1].get('role') != 'user' or history[-1].get('content') != user_message:
            conversation += f"User: {user_message}\n"
        if no_search_needed:
            llm_final_prompt = f"""
You are a helpful log analytics assistant. The user asked: '{user_message}'.

Respond helpfully and concisely. If relevant, use your knowledge of log analytics, but do NOT hallucinate or make up log details. Do not reference or summarize any logs or search results unless they are explicitly provided. If no logs are available, do not assume or invent any log content.

Conversation so far:
{conversation}Assistant: """
        else:
            llm_final_prompt = f"""
You are a helpful log analytics assistant. The user asked: '{user_message}'.
The search query generated was: '{search_query}'.
Relevant logs (with timestamps):
{log_context}

Respond helpfully and concisely. ONLY use the logs provided above for your answer. Do NOT hallucinate, assume, or invent any log details that are not present in the provided logs. If the logs do not contain enough information, clearly state that based on the available logs.

Conversation so far:
{conversation}Assistant: """
        logger.info(f"LLM Final Prompt: {llm_final_prompt}")
        llm_final_resp = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.1',
                'prompt': llm_final_prompt,
                'stream': False
            },
            timeout=120
        )
        if llm_final_resp.status_code == 200:
            llm_reply = llm_final_resp.json().get('response', '').strip()
        else:
            llm_reply = f"LLM error: {llm_final_resp.status_code} {llm_final_resp.text}"
        logger.info(f"LLM Final Response: {llm_reply}")
        return jsonify({
            "status": "success",
            "reply": llm_reply,
            "logs": logs,
            "search_query": search_query if search_query else None
        })
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def start_api(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask API server"""
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    start_api(debug=True)