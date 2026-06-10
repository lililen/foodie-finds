import time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV


class SVMTaggerModel:
    def __init__(self, max_features=30_000, ngram_range=(1, 2), C=1.0):
        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)
        self.clf = OneVsRestClassifier(
            CalibratedClassifierCV(LinearSVC(C=C, max_iter=2000))
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
