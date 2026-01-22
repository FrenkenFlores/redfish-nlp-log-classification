from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
import re


# def classify_with_regex(log_message):
#     regex_patterns = {
#         r"User User\d+ logged (in|out).": "audit",
#         r"Backup (started|ended) at .*": "sel",
#         r"Backup completed successfully.": "sel",
#         r"System updated to version .*": "sel",
#         r"File .* uploaded successfully by user .*": "sel",
#         r"Disk cleanup completed successfully.": "sel",
#         r"System reboot initiated by user .*": "sel",
#         r"Account with ID .* created by .*": "audit"
#     }
#     for pattern, label in regex_patterns.items():
#         if re.search(pattern, log_message):
#             return label
#     return None


if __name__ == "__main__":
    df = pd.read_csv("dataset/redfish_logs.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
    embeddings = model.encode(df['log_message'].tolist())
    clustering = DBSCAN(eps=0.2, min_samples=1, metric='cosine').fit(embeddings)
    df['cluster'] = clustering.labels_
    # df['regex_label'] = df['log_message'].apply(lambda x: classify_with_regex(x))
    # df_non_regex = df[df['regex_label'].isnull()].copy()
    df_non_regex = df.copy()
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
    embeddings_filtered = model.encode(df_non_regex['log_message'].tolist())
    X = embeddings_filtered
    y = df_non_regex['target_label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)
    joblib.dump(clf, '../models/log_classifier.joblib')
