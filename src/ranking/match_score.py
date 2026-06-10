import re
import numpy as np
import pandas as pd
from typing import Optional

TAG_NAMES = [
    "Quiet", "Brunch", "Date Night", "Hidden Gem",
    "Aesthetic", "Family-Friendly", "Late Night", "Trendy",
]
TAG_COL_MAP = {
    tag: f"tag_{tag.lower().replace(' ', '_').replace('-', '_')}"
    for tag in TAG_NAMES
}
QUERY_TAG_SYNONYMS = {
    "quiet": "Quiet",
    "peaceful": "Quiet",
    "calm": "Quiet",
    "brunch": "Brunch",
    "breakfast": "Brunch",
    "date":"Date Night",
    "date night":"Date Night",
    "romantic":  "Date Night",
    "hidden":"Hidden Gem",
    "hidden gem":"Hidden Gem",
    "underrated":"Hidden Gem",
    "aesthetic":"Aesthetic",
    "beautiful":"Aesthetic",
    "photogenic":"Aesthetic",
    "instagrammable": "Aesthetic",
    "family":"Family-Friendly",
    "kids":"Family-Friendly",
    "late night":"Late Night",
    "late": "Late Night",
    "trendy": "Trendy",
    "hip":"Trendy",
    "cool":"Trendy",
}


def parse_query(query: str) -> dict:
    q = query.lower()

    detected_tags = set()
    for phrase, tag in QUERY_TAG_SYNONYMS.items():
        if phrase in q:
            detected_tags.add(tag)

    city = None
    #cities didnt works somehow. 
    city_patterns = [
        "san francisco", "las vegas", "new york", "los angeles", "chicago",
        "seattle", "austin", "portland", "phoenix", "charlotte", "nashville",
        "denver", "miami", "boston", "atlanta", "philadelphia", "tampa",
        "indianapolis", "new orleans", "tucson", "reno", "edmonton",
        "boise", "st. louis", "saint louis",
    ]
    for c in city_patterns:
        if c in q:
            city = c.title()
            break

    cuisine_keywords = []
    cuisine_list = [
        "vegan", "vegetarian", "italian", "mexican", "chinese", "japanese",
        "thai", "indian", "french", "korean", "mediterranean", "american",
        "sushi", "pizza", "burger", "bbq", "seafood", "ramen",
    ]
    for kw in cuisine_list:
        if kw in q:
            cuisine_keywords.append(kw)

    return {
        "raw_query": query,
        "tags": list(detected_tags),
        "city": city,
        "cuisine_keywords": cuisine_keywords,
    }
def compute_match_scores(
    profiles: pd.DataFrame,
    parsed_query: dict,
    weights: Optional[dict] = None,
    top_k: int = 20,
) -> pd.DataFrame:
    if weights is None:
        weights = {
            "sentiment": 0.30,
            "tag_match": 0.30,
            "stars":     0.20,
            "volume":    0.10,
            "cuisine":   0.10,
        }

    df = profiles.copy()
    if "positive_ratio" not in df.columns:
        df["positive_ratio"] = 0.5
    sentiment_score = df["positive_ratio"].fillna(0.5)

    query_tags = parsed_query.get("tags", [])
    if query_tags:
        tag_scores = pd.Series(np.zeros(len(df)), index=df.index)
        for tag in query_tags:
            col = TAG_COL_MAP.get(tag)
            if col and col in df.columns:
                tag_scores += df[col].fillna(0)
        tag_match_score = (tag_scores / len(query_tags)).clip(0, 1)
    else:
        tag_match_score = pd.Series(np.ones(len(df)) * 0.5, index=df.index)
    star_col = "avg_stars" if "avg_stars" in df.columns else "stars_business"
    stars_score = ((df[star_col].fillna(3) - 1) / 4).clip(0, 1)

    vol = df["review_count"].fillna(1).clip(lower=1)
    log_vol = np.log1p(vol)
    volume_score = (log_vol / log_vol.max()).clip(0, 1)

    cuisine_kws = parsed_query.get("cuisine_keywords", [])
    if cuisine_kws and "categories" in df.columns:
        def cuisine_hit(cats):
            cats_lower = str(cats).lower()
            return sum(1 for kw in cuisine_kws if kw in cats_lower) / len(cuisine_kws)
        cuisine_score = df["categories"].apply(cuisine_hit)
    else:
        cuisine_score = pd.Series(np.zeros(len(df)), index=df.index)

    if cuisine_kws and "categories" in df.columns:
        cuisine_mask = df["categories"].apply(
            lambda cats: any(kw in str(cats).lower() for kw in cuisine_kws)
        )
        df = df[cuisine_mask].copy()
        sentiment_score = sentiment_score[cuisine_mask]
        tag_match_score = tag_match_score[cuisine_mask]
        stars_score = stars_score[cuisine_mask]
        volume_score = volume_score[cuisine_mask]
        cuisine_score = cuisine_score[cuisine_mask]
    query_city = parsed_query.get("city")
    if query_city and "city" in df.columns:
        city_mask = df["city"].str.lower().str.contains(query_city.lower(), na=False)
        df = df[city_mask].copy()
        sentiment_score = sentiment_score[city_mask]
        tag_match_score = tag_match_score[city_mask]
        stars_score = stars_score[city_mask]
        volume_score = volume_score[city_mask]
        cuisine_score = cuisine_score[city_mask]
    raw_score = (
        weights["sentiment"] * sentiment_score + weights["tag_match"] * tag_match_score
        + weights["stars"] * stars_score + weights["volume"] * volume_score
        + weights["cuisine"] * cuisine_score
    )
    df = df.copy()
    df["match_score"] = (raw_score * 10).round(2)
    df["score_sentiment"] = (weights["sentiment"] * sentiment_score * 10).round(2)
    df["score_tag_match"] = (weights["tag_match"] * tag_match_score * 10).round(2)
    df["score_stars"] = (weights["stars"] * stars_score * 10).round(2)
    df["score_volume"] = (weights["volume"] * volume_score * 10).round(2)
    df["score_cuisine"] = (weights["cuisine"] * cuisine_score * 10).round(2)

    result = df.sort_values("match_score", ascending=False).head(top_k)
    return result.reset_index(drop=True)


def ablation_study(profiles: pd.DataFrame, parsed_query: dict) -> pd.DataFrame:
    base_weights = {
        "sentiment": 0.30,
        "tag_match": 0.30,
        "stars":     0.20,
        "volume":    0.10,
        "cuisine":   0.10,
    }
    components = list(base_weights.keys())
    records = []
    for drop in components:
        w = {k: (0.0 if k == drop else v) for k, v in base_weights.items()}
        total = sum(w.values()) or 1e-9
        w = {k: v / total for k, v in w.items()}
        top = compute_match_scores(profiles, parsed_query, weights=w, top_k=5)
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            records.append({
                "ablation": f"no_{drop}",
                "rank": rank,
                "name": row.get("name", row.get("business_id", "")),
                "match_score": row["match_score"],
            })
    base_top = compute_match_scores(profiles, parsed_query, weights=base_weights, top_k=5)
    for rank, (_, row) in enumerate(base_top.iterrows(), 1):
        records.append({
            "ablation": "full",
            "rank": rank,
            "name": row.get("name", row.get("business_id", "")),
            "match_score": row["match_score"],
        })
    return pd.DataFrame(records)
