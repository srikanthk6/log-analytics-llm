"""
Deep Anomaly Detection using LogAI (DeepLog)
"""
import os
import pandas as pd
from logai.analysis.anomaly_detector import DeepAnomalyDetectorConfig, DeepAnomalyDetector
from logai.dataloader.log_loader import LogLoaderConfig, LogLoader
from logai.preprocess import PreprocessorConfig, Preprocessor
from logai.utils import split_train_test

# Path to the log file (update if needed)
LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "test.log")
LOG_FORMAT = "<Date> <Time> <Level> <Component>: <Content>"

# Load logs
def load_logs():
    log_loader_config = LogLoaderConfig(log_path=LOG_FILE, log_format=LOG_FORMAT)
    log_loader = LogLoader(log_loader_config=log_loader_config)
    log_df = log_loader.load()
    return log_df

# Preprocess logs
def preprocess_logs(log_df):
    preprocessor_config = PreprocessorConfig()
    preprocessor = Preprocessor(preprocessor_config=preprocessor_config)
    processed_log = preprocessor.preprocess(log_df)
    return processed_log

# Main deep anomaly detection workflow
def main():
    print("Loading logs...")
    log_df = load_logs()
    print(f"Loaded {len(log_df)} log entries.")

    print("Preprocessing logs...")
    processed_log = preprocess_logs(log_df)
    print(f"Processed {len(processed_log)} log entries.")

    print("Splitting train/test...")
    train_df, test_df = split_train_test(processed_log, test_size=0.2)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    print("Configuring DeepAnomalyDetector (DeepLog)...")
    deep_ad_config = DeepAnomalyDetectorConfig(model_name="DeepLog", epochs=5)
    deep_ad = DeepAnomalyDetector(config=deep_ad_config)

    print("Training DeepLog model...")
    deep_ad.fit(train_df)

    print("Detecting anomalies on test set...")
    anomaly_results = deep_ad.predict(test_df)
    print(anomaly_results.head())

    # Save results to CSV
    out_path = os.path.join(os.path.dirname(__file__), "deep_anomaly_results.csv")
    anomaly_results.to_csv(out_path, index=False)
    print(f"Anomaly results saved to {out_path}")

if __name__ == "__main__":
    main()
