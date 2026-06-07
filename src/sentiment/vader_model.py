import time
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VADERSentimentModel:
    """Rule-based sentiment baseline using VADER compound score."""

    def __init__(self, positive_thresh=0.05, negative_thresh=-0.05):
        self.analyzer = SentimentIntensityAnalyzer()
        self.positive_thresh = positive_thresh
        self.negative_thresh = negative_thresh

    def predict_one(self, text: str) -> str:
        score = self.analyzer.polarity_scores(text)["compound"]
        if score >= self.positive_thresh:
            return "Positive"
        elif score <= self.negative_thresh:
            return "Negative"
        return "Neutral"

    def predict(self, texts) -> list[str]:
        return [self.predict_one(t) for t in texts]

    def predict_proba(self, texts) -> np.ndarray:
        """Return [neg, neu, pos] probability-like scores."""
        results = []
        for t in texts:
            s = self.analyzer.polarity_scores(t)
            results.append([s["neg"], s["neu"], s["pos"]])
        return np.array(results)

    def fit(self, texts, labels):
        """No-op — VADER is rule-based."""
        return self

    def timed_predict(self, texts):
        start = time.time()
        preds = self.predict(texts)
        elapsed = time.time() - start
        return preds, elapsed
