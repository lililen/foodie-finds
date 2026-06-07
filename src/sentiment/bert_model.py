import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder


class BERTReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class BERTSentimentModel:
    """BERT fine-tuned for sentiment classification."""

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        max_len=128,
        batch_size=32,
        epochs=3,
        lr=2e-5,
        device=None,
    ):
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder = LabelEncoder()
        self.tokenizer = None
        self.model = None
        self.classes_ = None

    def fit(self, texts, labels, val_texts=None, val_labels=None):
        y = self.label_encoder.fit_transform(labels)
        self.classes_ = self.label_encoder.classes_
        num_labels = len(self.classes_)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        ).to(self.device)

        dataset = BERTReviewDataset(texts, y.tolist(), self.tokenizer, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )

        train_start = time.time()
        self.history = []
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                outputs.loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += outputs.loss.item()
            avg_loss = total_loss / len(loader)
            self.history.append({"epoch": epoch + 1, "loss": avg_loss})
            print(f"  Epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}")
        self.train_time_ = time.time() - train_start
        return self

    def _encode_and_predict(self, texts):
        self.model.eval()
        all_logits = []
        dataset = BERTReviewDataset(texts, [0] * len(texts), self.tokenizer, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                all_logits.append(logits.cpu())
        return torch.cat(all_logits, dim=0)

    def predict(self, texts) -> list[str]:
        logits = self._encode_and_predict(texts)
        y_pred = logits.argmax(dim=1).numpy()
        return self.label_encoder.inverse_transform(y_pred).tolist()

    def predict_proba(self, texts) -> np.ndarray:
        logits = self._encode_and_predict(texts)
        return torch.softmax(logits, dim=1).numpy()

    def timed_predict(self, texts):
        start = time.time()
        preds = self.predict(texts)
        elapsed = time.time() - start
        return preds, elapsed

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str, num_labels: int):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels).to(self.device)
