"""
Deep Anomaly Detection using PyOD (AutoEncoder)
"""
import os
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Path to the most recent processed log file
data_path = os.path.join(
    os.path.dirname(__file__),
    "log_analytics",
    "processed",
    "ecommerce_20250507_181815.log"
)

# Load log file (assume one log message per line)
def load_logs(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({"message": lines})

def main():
    print(f"Loading logs from {data_path} ...")
    df = load_logs(data_path)
    print(f"Loaded {len(df)} log entries.")

    # Feature extraction: TF-IDF on messages
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df["message"]).toarray()

    # Train AutoEncoder
    print("Training PyOD AutoEncoder...")
    clf = AutoEncoder()
    clf.fit(X)

    # Get anomaly scores
    print("Scoring anomalies...")
    df["anomaly_score"] = clf.decision_function(X)
    df["anomaly_label"] = clf.predict(X)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "pyod_anomaly_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Anomaly results saved to {out_path}")
    print(df.head())

if __name__ == "__main__":
    main()
