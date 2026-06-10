import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sentiment.lstm_model import Vocab


class TagDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.encodings = [vocab.encode(t, max_len) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.long), self.labels[idx]


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)
        pooled = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        out = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(out))



class CNNTaggerModel:
    def __init__(
        self,
        embed_dim=128,
        num_filters=128,
        filter_sizes=(2, 3, 4),
        max_len=256,
        batch_size=64,
        epochs=5,
        lr=1e-3, #small number no need to have massive change or else takes away meaning of its training 
        threshold=0.5,
        device=None,
    ):
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = Vocab()
        self.model = None
        self.tag_names = None

    def fit(self, texts, Y, tag_names):
        self.tag_names = tag_names
        self.vocab.build(texts)
        num_classes = Y.shape[1]

        dataset = TagDataset(texts, Y, self.vocab, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.model = TextCNN(
            len(self.vocab), self.embed_dim, self.num_filters,
            self.filter_sizes, num_classes,
        ).to(self.device)
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

    def _predict_logits(self, texts):
        self.model.eval()
        all_logits = []
        dataset = TagDataset(texts, np.zeros((len(texts), 1)), self.vocab, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False, num_workers=0)
        with torch.no_grad():
            for xb, _ in loader:
                all_logits.append(self.model(xb.to(self.device)).cpu())
        return torch.cat(all_logits, dim=0)

    def predict(self, texts) -> np.ndarray:
        logits = self._predict_logits(texts)
        return (torch.sigmoid(logits).numpy() >= self.threshold).astype(int)

    def predict_proba(self, texts) -> np.ndarray:
        logits = self._predict_logits(texts)
        return torch.sigmoid(logits).numpy()

    def timed_predict(self, texts):
        start = time.time()
        preds = self.predict(texts)
        elapsed = time.time() - start
        return preds, elapsed

    def get_filter_activations(self, text: str) -> dict:
        self.model.eval()
        enc = torch.tensor([self.vocab.encode(text, self.max_len)], dtype=torch.long).to(self.device)
        emb = self.model.embedding(enc).permute(0, 2, 1)
        result = {}
        with torch.no_grad():
            for fs, conv in zip(self.filter_sizes, self.model.convs):
                activated = torch.relu(conv(emb)).squeeze(0)
                result[f"filter_size_{fs}"] = activated.max(dim=1).values.cpu().numpy()
        return result
