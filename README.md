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
- Elasticsearch 8.x+ (for log storage and vector search)
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

4. Set up Elasticsearch and Kibana with Docker Compose:

Elasticsearch is used for storing log data and performing vector similarity searches. Kibana provides a web UI for exploring and visualizing your logs.

You can use Docker Compose to run both services easily. A sample `docker-compose.yaml` is provided in this repository.

To start Elasticsearch and Kibana:

```bash
docker compose up -d
```

- Access Kibana at: [http://localhost:5601](http://localhost:5601)
- Access Elasticsearch at: [http://localhost:9200](http://localhost:9200)

> **Note:**  
> For vector search support (such as `knn` queries), Elasticsearch 8.x or later is required and is used in this setup.

To stop the services:

```bash
docker compose down
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

## Using LLMs in Log Analytics

This project can be enhanced by integrating a local LLM, such as Ollama's `llama3.1:8b-instruct-fp16`, in several ways:

1. **Log Message Embeddings**
   - Use the LLM to generate semantic embeddings for log messages. These embeddings capture deeper meaning and context than traditional TF-IDF, improving anomaly detection and log clustering.
   - Example: Replace or augment the `_generate_vector_embedding` method in `logai_handler.py` to call the Ollama model and use its output as the log vector.

2. **Anomaly Explanation**
   - When an anomaly is detected, use the LLM to generate a human-readable explanation of why the log is unusual compared to recent logs.
   - Example: In `log_monitor.py`, after detecting an anomaly, send the anomalous log and recent normal logs to the LLM and display its explanation.

3. **Log Summarization**
   - Use the LLM to summarize large volumes of logs, highlighting key events, trends, or issues.
   - Example: Periodically send batches of logs to the LLM and store/display its summary.

4. **Root Cause Analysis**
   - Use the LLM to analyze sequences of logs and suggest possible root causes for detected anomalies or errors.

5. **Natural Language Querying**
   - Allow users to query logs using natural language (e.g., “Show me all errors from the payment service in the last hour”), and use the LLM to translate these queries into structured searches.

6. **Log Classification and Tagging**
   - Use the LLM to automatically classify log messages (e.g., error, warning, info) or tag them with relevant categories.

### Integration Example with Ollama

- Install Ollama and run the model locally:
  ```sh
  ollama run llama3.1:8b-instruct-fp16
  ```
- In your Python code, use an HTTP client (e.g., `requests`) to send prompts to the Ollama server and receive responses.
- Example usage in `logai_handler.py`:
  ```python
  import requests
  def get_llm_embedding(log_message):
      response = requests.post(
          'http://localhost:11434/api/generate',
          json={
              'model': 'llama3.1:8b-instruct-fp16',
              'prompt': f"Generate a vector embedding for this log: {log_message}",
              'stream': False
          }
      )
      return response.json()['response']
  ```
- Replace the vector generation or add LLM-based features as needed.

These enhancements can make the log analytics system more powerful, interpretable, and user-friendly.

## Setting up Node.js and React Frontend

If you do not have a compatible Node.js version, you can use nvm (Node Version Manager) to install the latest version and set up the React frontend:

```sh
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash
nvm install v22.13.1
npx create-react-app .
npm install
npm start
```

This will:
- Install nvm (Node Version Manager)
- Install Node.js v22.13.1
- Create a new React app in the current directory
- Install dependencies
- Start the React development server

If you want to enable markdown and table formatting in the chatbot UI, install the following dependency in your frontend directory:

```
npm install react-markdown
```

## Frontend Chatbot UI (React)

To enable a conversational log analytics chatbot, you can use the following React component:

Create a new file `frontend/src/ChatbotUI.jsx`:

```jsx
import React, { useState } from 'react';
import axios from 'axios';

function ChatbotUI() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { role: 'user', content: input };
    setMessages([...messages, userMsg]);
    setInput('');
    setLoading(true);
    try {
      const res = await axios.post('/chat', {
        message: input,
        history: messages,
        size: 5
      });
      const reply = res.data.reply;
      setMessages([...messages, userMsg, { role: 'assistant', content: reply }]);
    } catch (err) {
      setMessages([...messages, userMsg, { role: 'assistant', content: 'Error: Could not get response.' }]);
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: 'auto', padding: 20 }}>
      <h2>Log Analytics Chatbot</h2>
      <div style={{ border: '1px solid #ccc', padding: 10, minHeight: 300, marginBottom: 10, background: '#fafafa' }}>
        {messages.map((msg, idx) => (
          <div key={idx} style={{ textAlign: msg.role === 'user' ? 'right' : 'left' }}>
            <b>{msg.role === 'user' ? 'You' : 'Assistant'}:</b> {msg.content}
          </div>
        ))}
        {loading && <div>Assistant is typing...</div>}
      </div>
      <input
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => e.key === 'Enter' ? sendMessage() : null}
        style={{ width: '80%' }}
        placeholder="Ask about your logs..."
        disabled={loading}
      />
      <button onClick={sendMessage} disabled={loading || !input.trim()}>Send</button>
    </div>
  );
}

export default ChatbotUI;
```

### Setup Instructions

1. Create a new folder `frontend` and initialize a React app:
   ```sh
   npx create-react-app frontend
   cd frontend
   npm install axios
   ```
2. Add the above `ChatbotUI.jsx` to `src/` and import it in `App.js`.
3. Make sure your Flask backend allows CORS (see requirements.txt and Flask-CORS usage).
4. Start the React app with `npm start`.
5. The React app should proxy API requests to your Flask backend (set up `proxy` in `frontend/package.json` if needed).

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