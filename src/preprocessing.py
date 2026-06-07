import json
import re
import os
import numpy as np
import pandas as pd
import nltk
import spacy
from pathlib import Path
from tqdm import tqdm

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ---------------------------------------------------------------------------
# Tag keyword lexicon
# ---------------------------------------------------------------------------
TAG_KEYWORDS = {
    "Quiet": [
        "quiet", "peaceful", "calm", "serene", "tranquil", "low-key",
        "relaxed", "mellow", "intimate", "hushed",
    ],
    "Brunch": [
        "brunch", "breakfast", "mimosa", "eggs benedict", "pancake",
        "waffle", "morning", "weekend brunch", "brunch spot",
    ],
    "Date Night": [
        "date night", "romantic", "date spot", "anniversary", "intimate",
        "candles", "mood lighting", "couples", "perfect for a date",
    ],
    "Hidden Gem": [
        "hidden gem", "underrated", "off the beaten path", "secret spot",
        "overlooked", "under the radar", "best kept secret", "undiscovered",
    ],
    "Aesthetic": [
        "aesthetic", "instagrammable", "beautiful decor", "stunning",
        "gorgeous", "photogenic", "cozy atmosphere", "ambiance", "vibes",
    ],
    "Family-Friendly": [
        "family friendly", "kid friendly", "kids menu", "families",
        "children", "highchair", "stroller", "family atmosphere",
    ],
    "Late Night": [
        "late night", "open late", "after hours", "midnight", "night owl",
        "2am", "1am", "late night eats", "bar vibes",
    ],
    "Trendy": [
        "trendy", "hip", "cool", "popular", "buzzing", "hot spot",
        "instagram worthy", "happening", "stylish", "modern",
    ],
}

SENTIMENT_MAP = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}


def load_yelp_reviews(reviews_path: str, businesses_path: str, max_reviews: int = 500_000) -> pd.DataFrame:
    """Load Yelp JSON files and merge reviews with restaurant business metadata."""
    print("Loading businesses...")
    businesses = []
    with open(businesses_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            b = json.loads(line)
            cats = b.get("categories") or ""
            if any(c.strip().lower() in ("restaurants", "food") for c in cats.split(",")):
                businesses.append({
                    "business_id": b["business_id"],
                    "name": b["name"],
                    "city": b.get("city", ""),
                    "state": b.get("state", ""),
                    "stars_business": b.get("stars", 0),
                    "review_count": b.get("review_count", 0),
                    "categories": cats,
                })
    biz_df = pd.DataFrame(businesses)
    biz_ids = set(biz_df["business_id"])
    print(f"  {len(biz_df):,} restaurant businesses loaded.")

    print("Loading reviews...")
    reviews = []
    with open(reviews_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            if len(reviews) >= max_reviews:
                break
            r = json.loads(line)
            if r["business_id"] in biz_ids and len(r.get("text", "")) >= 10:
                reviews.append({
                    "review_id": r["review_id"],
                    "business_id": r["business_id"],
                    "stars": r["stars"],
                    "text": r["text"],
                    "useful": r.get("useful", 0),
                })
    reviews_df = pd.DataFrame(reviews)
    print(f"  {len(reviews_df):,} reviews loaded.")

    df = reviews_df.merge(biz_df, on="business_id", how="left")
    return df


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s'.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def assign_sentiment_label(stars: int) -> str:
    return SENTIMENT_MAP.get(int(stars), "Neutral")


def assign_tags(text: str) -> list[str]:
    text_lower = text.lower()
    tags = []
    for tag, keywords in TAG_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)
    return tags


def preprocess_dataframe(df: pd.DataFrame, nlp=None) -> pd.DataFrame:
    """Clean text, assign sentiment labels and multi-hot tag columns."""
    print("Cleaning text...")
    df = df.copy()
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.split().str.len() >= 10].reset_index(drop=True)

    print("Assigning sentiment labels...")
    df["sentiment"] = df["stars"].apply(assign_sentiment_label)

    print("Assigning tags...")
    df["tags"] = df["text"].apply(assign_tags)

    all_tags = list(TAG_KEYWORDS.keys())
    for tag in all_tags:
        df[f"tag_{tag.lower().replace(' ', '_').replace('-', '_')}"] = df["tags"].apply(
            lambda t: 1 if tag in t else 0
        )

    return df


def get_tag_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("tag_")]


def train_val_test_split(df: pd.DataFrame, val_frac=0.1, test_frac=0.1, seed=42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = df.iloc[:n_test]
    val = df.iloc[n_test: n_test + n_val]
    train = df.iloc[n_test + n_val:]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_restaurant_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-restaurant sentiment and tag statistics."""
    tag_cols = get_tag_columns(df)
    agg = {
        "stars": "mean",
        "review_count": "first",
        "name": "first",
        "city": "first",
        "state": "first",
        "categories": "first",
    }
    agg.update({c: "mean" for c in tag_cols})

    profiles = df.groupby("business_id").agg(agg).reset_index()
    profiles.rename(columns={"stars": "avg_stars"}, inplace=True)

    sent_counts = (
        df.groupby(["business_id", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"Positive": "n_positive", "Neutral": "n_neutral", "Negative": "n_negative"})
    )
    for col in ("n_positive", "n_neutral", "n_negative"):
        if col not in sent_counts.columns:
            sent_counts[col] = 0
    profiles = profiles.merge(sent_counts, on="business_id", how="left")
    total = profiles[["n_positive", "n_neutral", "n_negative"]].sum(axis=1).clip(lower=1)
    profiles["positive_ratio"] = profiles["n_positive"] / total
    return profiles


def make_sample_dataset(n=2000, seed=42):
    """Generate a tiny synthetic dataset for quick smoke-testing without the full Yelp files."""
    rng = np.random.default_rng(seed)
    tag_keys = list(TAG_KEYWORDS.keys())
    texts = []
    stars_list = []
    for _ in range(n):
        star = rng.integers(1, 6)
        chosen_tags = rng.choice(tag_keys, size=rng.integers(0, 4), replace=False).tolist()
        snippets = []
        if star >= 4:
            snippets.append("great food amazing experience loved it")
        elif star == 3:
            snippets.append("it was okay nothing special average")
        else:
            snippets.append("terrible service bad food disappointed")
        for t in chosen_tags:
            snippets.append(rng.choice(TAG_KEYWORDS[t]))
        texts.append(" ".join(snippets))
        stars_list.append(star)

    biz_ids = [f"biz_{i % 200:04d}" for i in range(n)]
    names = [f"Restaurant {i % 200}" for i in range(n)]
    df = pd.DataFrame({
        "review_id": [f"rev_{i}" for i in range(n)],
        "business_id": biz_ids,
        "name": names,
        "city": rng.choice(["San Francisco", "Las Vegas", "Phoenix", "Charlotte"], size=n).tolist(),
        "state": "CA",
        "stars": stars_list,
        "review_count": rng.integers(10, 500, size=n).tolist(),
        "categories": "Restaurants",
        "text": texts,
        "useful": rng.integers(0, 10, size=n).tolist(),
        "stars_business": stars_list,
    })
    return df
