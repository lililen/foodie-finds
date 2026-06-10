import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


class RFTaggerModel:
    def __init__(self, max_features=20_000, n_estimators=200, n_jobs=-1):
        self.tfidf = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
        self.clf = OneVsRestClassifier(
            RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=42)
        )
        self.tag_names = None

    def fit(self, texts, Y, tag_names):
        self.tag_names = tag_names
        X = self.tfidf.fit_transform(texts)
        start = time.time()
        self.clf.fit(X, Y)
        self.train_time_ = time.time() - start
        return self

    def predict(self, texts) -> np.ndarray:
        X = self.tfidf.transform(texts)
        return self.clf.predict(X)

    def predict_proba(self, texts) -> np.ndarray:
        X = self.tfidf.transform(texts)
        return self.clf.predict_proba(X)

    def timed_predict(self, texts):
        start = time.time()
        preds = self.predict(texts)
        elapsed = time.time() - start
        return preds, elapsed
