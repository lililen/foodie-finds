import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from preprocessing import build_restaurant_profiles, make_sample_dataset, preprocess_dataframe
from ranking.match_score import parse_query, compute_match_scores
from evaluation.metrics import plot_match_score_distribution, plot_score_breakdown

st.set_page_config(page_title="Restaurant Recommender", page_icon="🍽️", layout="wide")

st.title("🍽️ Restaurant Recommender")
st.caption("Natural-language search powered by sentiment, tags, and Yelp signals")

@st.cache_data(show_spinner="Building restaurant profiles…")
def load_profiles():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    parquet = os.path.join(data_dir, "restaurant_profiles.parquet")
    reviews = os.path.join(data_dir, "yelp_academic_dataset_review.json")
    businesses = os.path.join(data_dir, "yelp_academic_dataset_business.json")

    if os.path.exists(parquet):
        return pd.read_parquet(parquet)
    elif os.path.exists(reviews) and os.path.exists(businesses):
        from preprocessing import load_yelp_reviews
        df = load_yelp_reviews(reviews, businesses)
        profiles = build_restaurant_profiles(df)
        profiles.to_parquet(parquet, index=False)
        return profiles
    else:
        df = preprocess_dataframe(make_sample_dataset(n=3000))
        return build_restaurant_profiles(df)

profiles = load_profiles()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Results to show", 5, 30, 10)
    st.markdown("---")
    st.subheader("Score weights")
    w_sentiment = st.slider("Sentiment", 0.0, 1.0, 0.30, 0.05)
    w_tag       = st.slider("Tag match",  0.0, 1.0, 0.30, 0.05)
    w_stars     = st.slider("Stars",      0.0, 1.0, 0.20, 0.05)
    w_volume    = st.slider("Volume",     0.0, 1.0, 0.10, 0.05)
    w_cuisine   = st.slider("Cuisine",   0.0, 1.0, 0.10, 0.05)
    total = w_sentiment + w_tag + w_stars + w_volume + w_cuisine
    if total == 0:
        st.warning("All weights are 0 — results will be blank.")

query = st.text_input(
    "Describe what you're looking for",
    placeholder="e.g. Vegan brunch spots with a quiet aesthetic vibe in San Francisco",
)

example_queries = [
    "Vegan brunch spots with a quiet, aesthetic vibe in San Francisco",
    "Late night trendy Korean BBQ spot in Las Vegas",
    "Romantic date night Italian restaurant in Charlotte",
    "Family-friendly hidden gem in Phoenix",
]
st.caption("Try: " + " · ".join(f"`{q}`" for q in example_queries))

if query:
    parsed = parse_query(query)

    col1, col2, col3 = st.columns(3)
    col1.metric("Tags detected", ", ".join(parsed["tags"]) or "none")
    col2.metric("City", parsed["city"] or "any")
    col3.metric("Cuisine", ", ".join(parsed["cuisine_keywords"]) or "any")

    weights = dict(sentiment=w_sentiment, tag_match=w_tag, stars=w_stars,
                   volume=w_volume, cuisine=w_cuisine)
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    results = compute_match_scores(profiles, parsed, weights=weights, top_k=top_k)

    if results.empty:
        st.warning("No restaurants found. Try removing the city or broadening your search.")
    else:
        st.subheader(f"Top {len(results)} matches")

        tab1, tab2 = st.tabs(["Table", "Charts"])

        with tab1:
            display_cols = ["name", "city", "categories", "match_score",
                            "score_sentiment", "score_tag_match", "score_stars"]
            display_cols = [c for c in display_cols if c in results.columns]
            st.dataframe(
                results[display_cols].style.background_gradient(subset=["match_score"], cmap="Blues"),
                use_container_width=True,
            )

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                fig1 = plot_match_score_distribution(results.head(15), query=query)
                st.pyplot(fig1)
            with col_b:
                if not results.empty:
                    top = results.iloc[0]
                    fig2 = plot_score_breakdown(top)
                    st.pyplot(fig2)

        with st.expander("Score breakdown for all results"):
            score_cols = [c for c in results.columns if c.startswith("score_")]
            if score_cols:
                st.bar_chart(results.set_index("name")[score_cols] if "name" in results.columns
                             else results[score_cols])
