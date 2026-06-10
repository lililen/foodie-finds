import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.preprocessing import LabelEncoder


class Vocab:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = []

    def build(self, texts):
        counts = Counter()
        for t in texts:
            counts.update(t.lower().split())
        self.idx2word = [self.PAD, self.UNK]
        for word, cnt in counts.items():
            if cnt >= self.min_freq:
                self.idx2word.append(word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        return self

    def encode(self, text, max_len=256):
        tokens = text.lower().split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.idx2word)


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.encodings = [vocab.encode(t, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (h, _) = self.lstm(emb)
        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h_cat))


class LSTMSentimentModel:
    def __init__(self, embed_dim=128, hidden_dim=256, num_layers=2,
                 max_len=256, batch_size=128, epochs=5, lr=1e-3, device=None):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = Vocab()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.classes_ = None

    def fit(self, texts, labels, val_texts=None, val_labels=None):
        y = self.label_encoder.fit_transform(labels)
        self.classes_ = self.label_encoder.classes_
        self.vocab.build(texts)

        dataset = ReviewDataset(texts, y.tolist(), self.vocab, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        self.model = BiLSTM(
            len(self.vocab), self.embed_dim, self.hidden_dim,
            self.num_layers, len(self.classes_),
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, steps_per_epoch=len(loader), epochs=self.epochs
        )
        criterion = nn.CrossEntropyLoss()

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
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            self.history.append({"epoch": epoch + 1, "loss": avg_loss})
            print(f"  Epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}")
        self.train_time_ = time.time() - train_start
        return self

    def _predict_raw(self, texts):
        self.model.eval()
        all_logits = []
        dataset = ReviewDataset(texts, [0] * len(texts), self.vocab, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False, num_workers=0)
        with torch.no_grad():
            for xb, _ in loader:
                logits = self.model(xb.to(self.device))
                all_logits.append(logits.cpu())
        return torch.cat(all_logits, dim=0)

    def predict(self, texts) -> list[str]:
        logits = self._predict_raw(texts)
        y_pred = logits.argmax(dim=1).numpy()
        return self.label_encoder.inverse_transform(y_pred).tolist()

    def predict_proba(self, texts) -> np.ndarray:
        logits = self._predict_raw(texts)
        return torch.softmax(logits, dim=1).numpy()

    def timed_predict(self, texts):
        start = time.time()
        preds = self.predict(texts)
        elapsed = time.time() - start
        return preds, elapsed
