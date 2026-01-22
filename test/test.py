import joblib
import csv
from sentence_transformers import SentenceTransformer

model_embedding = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
model_classification = joblib.load("../models/log_classifier.joblib")


def classify_with_bert(log_message):
    embeddings = model_embedding.encode([log_message])
    probabilities = model_classification.predict_proba(embeddings)[0]
    if max(probabilities) < 0.5:
        return "Unclassified"
    predicted_label = model_classification.predict(embeddings)[0]
    
    return predicted_label


if __name__ == "__main__":
    with open("test_dataset.csv", encoding="utf-8") as f:
        next(f)
        logs = csv.DictReader(f=f, fieldnames=["log_message", "target_label"])
        for log in logs:
            label = classify_with_bert(log["log_message"])
            print(log["log_message"], "->", label)

