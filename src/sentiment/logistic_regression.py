import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class LRSentimentModel:
    """Logistic Regression on TF-IDF features for sentiment classification."""

    def __init__(self, max_features=50_000, ngram_range=(1, 2), C=1.0, max_iter=1000):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
                min_df=3,
            )),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", multi_class="multinomial")),
        ])
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, texts, labels):
        y = self.label_encoder.fit_transform(labels)
        self.classes_ = self.label_encoder.classes_
        start = time.time()
        self.pipeline.fit(texts, y)
        self.train_time_ = time.time() - start
        return self

    def predict(self, texts) -> list[str]:
        y_pred = self.pipeline.predict(texts)
        return self.label_encoder.inverse_transform(y_pred).tolist()

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def timed_predict(self, texts):
        start = time.time()
        preds = self.predict(texts)
        elapsed = time.time() - start
        return preds, elapsed

    def get_top_tfidf_tokens(self, n=20) -> dict:
        """Return top n TF-IDF tokens per class by LR coefficient."""
        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]
        feature_names = np.array(tfidf.get_feature_names_out())
        result = {}
        for i, cls in enumerate(self.classes_):
            coefs = clf.coef_[i] if len(clf.coef_) > 1 else clf.coef_[0]
            top_idx = np.argsort(coefs)[-n:][::-1]
            result[cls] = feature_names[top_idx].tolist()
        return result
