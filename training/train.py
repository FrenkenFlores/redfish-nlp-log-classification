from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib
import re


if __name__ == "__main__":
    df = pd.read_csv("dataset/dataset.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['message'].tolist())
    clustering = DBSCAN(eps=0.2, min_samples=1, metric='cosine').fit(embeddings)
    df['cluster'] = clustering.labels_
    X = embeddings
    y = df['class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)
    joblib.dump(clf, '../models/log_classifier.joblib')
