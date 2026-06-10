import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import KeyedVectors
from gensim.downloader import load as gensim_load


def load_word_embeddings(source="glove-wiki-gigaword-100"):
    print(f"loading word embeddings: {source}")
    return gensim_load(source)


def texts_to_embedding_matrix(texts, wv, dim=100, max_len=256):
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = text.lower().split()[:max_len]
        vecs = [wv[t] for t in tokens if t in wv]
        if vecs:
            X[i] = np.mean(vecs, axis=0)
    return X


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.4):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPTaggerModel:
    def __init__(
        self,
        wv=None,
        embed_source="glove-wiki-gigaword-100",
        embed_dim=100,
        hidden_dims=(256, 128),
        max_len=256,
        batch_size=128,
        epochs=10,
        lr=1e-3,
        threshold=0.5,
        device=None,
    ):
        self.wv = wv
        self.embed_source = embed_source
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tag_names = None

    def _ensure_wv(self):
        if self.wv is None:
            self.wv = load_word_embeddings(self.embed_source)

    def fit(self, texts, Y, tag_names):
        self._ensure_wv()
        self.tag_names = tag_names
        num_classes = Y.shape[1]

        X = texts_to_embedding_matrix(texts, self.wv, self.embed_dim, self.max_len)
        X_t = torch.tensor(X)
        Y_t = torch.tensor(Y, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=self.batch_size, shuffle=True)

        self.model = MLP(self.embed_dim, self.hidden_dims, num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        train_start = time.time()
        self.history = []
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            self.history.append({"epoch": epoch + 1, "loss": avg_loss})
            print(f"  Epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}")
        self.train_time_ = time.time() - train_start
        return self

    def _transform(self, texts) -> torch.Tensor:
        X = texts_to_embedding_matrix(texts, self.wv, self.embed_dim, self.max_len)
        return torch.tensor(X)

    def predict(self, texts) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._transform(texts).to(self.device)).cpu()
        return (torch.sigmoid(logits).numpy() >= self.threshold).astype(int)

    def predict_proba(self, texts) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._transform(texts).to(self.device)).cpu()
        return torch.sigmoid(logits).numpy()

    def timed_predict(self, texts):
        start = time.time()
        preds = self.predict(texts)
        elapsed = time.time() - start
        return preds, elapsed
