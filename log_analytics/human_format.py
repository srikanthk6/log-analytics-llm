def format_log_query_response(api_response):
    """
    Convert the API response from /logs/query to a human-friendly summary string.
    """
    if not api_response or "results" not in api_response:
        return "No results found."
    lines = []
    for idx, hit in enumerate(api_response["results"], 1):
        src = hit.get("_source", {})
        ts = src.get("timestamp", "?")
        lvl = src.get("level", "?")
        msg = src.get("message", "?")
        anomaly = src.get("anomaly_score", 0.0)
        score = hit.get("_score", 0.0)
        lines.append(f"{idx}. [{ts}] {lvl} | Score: {score:.2f} | Anomaly: {'YES' if anomaly >= 0.8 else 'no'}\n   {msg}")
    return "\n".join(lines)