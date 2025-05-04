# SimpleLLM - Log Analytics with LogAI and LLM Integration

A flexible log analytics system that uses LogAI for log parsing and anomaly detection with support for LLM-based log analysis and insights.

## Project Overview

SimpleLLM combines the power of LogAI (for log parsing and anomaly detection) with integration hooks for LLMs (Large Language Models) to provide comprehensive log analytics capabilities:

- **Real-time log monitoring**: Watches for changes in log files and processes new log entries
- **Anomaly detection**: Uses LogAI's clustering and isolation forest techniques to detect unusual log patterns
- **Vector embeddings**: Creates embeddings of log messages for similarity search
- **REST API**: Provides endpoints for querying logs and retrieving anomalies
- **LLM integration**: Hooks for analyzing logs and anomalies using Large Language Models

## System Requirements

- Python 3.8+
- Elasticsearch 7.x+ (for log storage and vector search)
- LogAI library
- Docker (optional, for containerized deployment)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/simplellm.git
cd simplellm
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Set up Elasticsearch:

Elasticsearch is used for storing log data and performing vector similarity searches. You can:

- Install locally: https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html
- Use Docker:

```bash
docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:7.17.0
```

## Configuration

Create a `.env` file in the project root with your configuration settings:

```
# Elasticsearch settings
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_INDEX=logs_vector_index

# Log monitoring settings
LOG_DIR=/path/to/your/logs
WATCH_INTERVAL=1.0

# Anomaly detection settings
ANOMALY_THRESHOLD=0.8
WINDOW_SIZE=100
```

The default configuration is stored in `log_analytics/config/config.py`.

## Usage

### Starting the application

You can run both the log monitor and API server:

```bash
python main.py
```

Or run them separately:

```bash
# Run only the API server
python main.py --api-only --api-port=5000

# Run only the log monitor
python main.py --monitor-only
```

### Generating test logs

The system includes a log generator for testing:

```bash
python -m log_analytics.log_generator --type mixed --count 100 --anomaly-rate 0.1
```

Options:
- `--type`: Log type (`web_server`, `application`, `database`, or `mixed`)
- `--count`: Number of logs to generate
- `--interval`: Time between logs in seconds
- `--anomaly-rate`: Probability of generating anomalous logs (0.0-1.0)
- `--output-dir`: Directory for output log files

### Using the REST API

The system provides several REST API endpoints:

1. Health check:
```
GET /health
```

2. Get recent logs:
```
GET /logs/recent?size=100
```

3. Get detected anomalies:
```
GET /logs/anomalies?threshold=0.8&size=50
```

4. Search logs with vector similarity:
```
POST /logs/query
Content-Type: application/json

{
  "query": "database connection failure",
  "size": 10
}
```

5. Analyze logs with LLM (placeholder):
```
POST /logs/analyze
Content-Type: application/json

{
  "query": "What caused the system crash?",
  "size": 5
}
```

## Using LogAI and LLM Features

### LogAI Integration

SimpleLLM uses LogAI for log parsing and anomaly detection:

1. **Log Parsing**: The system uses LogAI's Drain algorithm to parse log messages and extract templates.
2. **Feature Extraction**: TF-IDF vectorization creates feature vectors from log templates.
3. **Anomaly Detection**: A custom Isolation Forest implementation detects anomalies in the log data.

### Setting Up LLM Integration

The current implementation includes placeholder methods for LLM integration. To fully enable LLM capabilities:

1. Choose an LLM provider (OpenAI, HuggingFace, etc.)
2. Add your API credentials to the `.env` file:

```
LLM_API_KEY=your_api_key_here
LLM_MODEL=preferred_model_name
```

3. Implement the LLM integration in the `analyze_logs` method in `api.py`. A sample implementation would look like:

```python
from openai import OpenAI  # or your preferred LLM library

# Initialize LLM client
client = OpenAI(api_key=os.getenv("LLM_API_KEY"))

@app.route('/logs/analyze', methods=['POST'])
def analyze_logs():
    try:
        data = request.get_json()
        query = data['query']
        size = int(data.get('size', 5))
        
        # Find relevant logs based on the query
        query_vector = logai_handler._generate_vector_embedding(query).tolist()
        similar_logs = es_handler.search_similar_logs(query_vector, top_k=size)
        
        # Prepare logs for LLM
        log_context = "\n".join([
            f"{log['_source']['timestamp']} [{log['_source']['level']}] {log['_source']['message']}"
            for log in similar_logs
        ])
        
        # Send to LLM for analysis
        prompt = f"""
        Analyze the following logs in relation to this query: "{query}"
        
        LOGS:
        {log_context}
        
        Provide a detailed analysis explaining:
        1. What issues are present in these logs
        2. Potential root causes
        3. Recommended actions to resolve any problems
        """
        
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4"),
            messages=[{"role": "user", "content": prompt}]
        )
        
        analysis = response.choices[0].message.content
        
        return jsonify({
            "status": "success",
            "analysis": analysis,
            "logs": [log['_source'] for log in similar_logs]
        })
    except Exception as e:
        logger.error(f"Error analyzing logs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
```

## Project Structure

```
main.py                 # Main entry point
requirements.txt        # Python dependencies
log_analytics/          # Main package
├── api.py              # REST API implementation
├── elastic_handler.py  # Elasticsearch integration
├── log_generator.py    # Test log generator
├── log_monitor.py      # Log file monitoring
├── logai_handler.py    # LogAI integration
├── config/             # Configuration
│   └── config.py       # Configuration settings
└── logs/               # Default log directory
    ├── application.log # Sample application logs
    ├── database.log    # Sample database logs
    └── web_server.log  # Sample web server logs
```

## Development and Testing

### Running Tests

```bash
pytest
```

### Adding Custom Log Sources

To add new log sources:

1. Create a new log file in the configured `LOG_DIR`
2. The log monitor will automatically detect and process the file
3. Customize log pattern detection in `logai_handler.py` if needed

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.