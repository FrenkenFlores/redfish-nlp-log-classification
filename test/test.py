import joblib
import csv
from sentence_transformers import SentenceTransformer

model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
model_classification = joblib.load("../models/log_classifier.joblib")


def classify_with_bert(message: str) -> str:
    """
    Classify the log with BERT model.

    Args:
        message (str): the Redfish log entry message

    Returns:
        str: the class that the log belongs to
    """
    embeddings = model_embedding.encode([message])
    probabilities = next(iter(model_classification.predict_proba(embeddings)))
    if max(probabilities) < 0.5:
        return "NULL"
    return next(iter(model_classification.predict(embeddings)))


if __name__ == "__main__":
    with open("test_dataset.csv", encoding="utf-8") as f:
        next(f) # skip header
        logs = csv.DictReader(f=f, fieldnames=["message", "target_label"])
        for log in logs:
            label = classify_with_bert(log["message"])
            print(log["message"], "->", label)

