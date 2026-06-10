import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from preprocessing import build_restaurant_profiles, make_sample_dataset, preprocess_dataframe
from ranking.match_score import parse_query, compute_match_scores

st.set_page_config(page_title="Foodie Finds", page_icon="🍽️", layout="wide")

st.markdown("""
<style>
  /* page background */
  [data-testid="stAppViewContainer"] { background: #f0f0eb; }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stSidebar"] { display: none; }
  /* kill default top padding */
  .block-container { padding-top: 1.4rem !important; padding-bottom: 1rem !important; }

  /* ── header row ── */
  .ff-logo {
    font-size: 2.1rem;
    font-weight: 900;
    color: #3a6b32;
    font-family: sans-serif;
    line-height: 1;
    white-space: nowrap;
  }

  /* style the native text_input to look like the mockup */
  div[data-testid="stTextInput"] > div {
    border-radius: 999px !important;
    background: #ffffff !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12) !important;
    padding: 0 6px !important;
  }
  div[data-testid="stTextInput"] input {
    font-size: 15px !important;
    color: #333 !important;
  }

  /* search button */
  div[data-testid="stButton"] button {
    border-radius: 999px !important;
    background: #3a6b32 !important;
    color: white !important;
    border: none !important;
    padding: 0.55rem 1.1rem !important;
    font-size: 18px !important;
    line-height: 1 !important;
    cursor: pointer !important;
  }
  div[data-testid="stButton"] button:hover {
    background: #2d5427 !important;
  }

  /* ── left info panel ── */
  .info-box {
    background: #e0e0da;
    border-radius: 14px;
    padding: 28px 22px;
    font-family: sans-serif;
    min-height: 340px;
  }
  .info-line {
    font-size: 15px;
    font-weight: 700;
    color: #3a6b32;
    margin-bottom: 18px;
    line-height: 1.35;
  }
  .info-cta {
    font-size: 15px;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 28px;
  }

  /* ── results table ── */
  .ff-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 14px;
    overflow: hidden;
    font-family: sans-serif;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  }
  .ff-table thead th {
    background: #4a7c42;
    color: #ffffff;
    padding: 14px 18px;
    font-size: 15px;
    font-weight: 600;
    text-align: left;
    white-space: nowrap;
  }
  .ff-table thead th.center { text-align: center; }
  .ff-table tbody td {
    background: #ffffff;
    padding: 12px 18px;
    border-bottom: 1px solid #e8e8e2;
    font-size: 14px;
    vertical-align: middle;
  }
  .ff-table tbody tr:last-child td { border-bottom: none; }
  .ff-table tbody tr:hover td { background: #f7faf6; }

  /* call link */
  .call-link {
    font-size: 11px;
    color: #6655cc;
    font-weight: 600;
    text-decoration: underline;
    cursor: pointer;
    white-space: nowrap;
  }
  /* name */
  .rest-name { font-weight: 500; color: #111; }
  /* cuisine */
  .cuisine { font-weight: 700; color: #111; }
  /* stars */
  .stars-filled { color: #e8b400; font-size: 17px; letter-spacing: 1px; }
  .stars-empty  { color: #d0d0d0; font-size: 17px; letter-spacing: 1px; }
  /* tags */
  .tag-text { color: #444; font-size: 13px; }
  /* match */
  .match-pct { font-weight: 700; font-size: 15px; color: #111; text-align: center; }
  .match-td { text-align: center; }
</style>
""", unsafe_allow_html=True)

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
        df = preprocess_dataframe(df)
        profiles = build_restaurant_profiles(df)
        profiles.to_parquet(parquet, index=False)
        return profiles
    else:
        df = preprocess_dataframe(make_sample_dataset(n=3000))
        return build_restaurant_profiles(df)

profiles = load_profiles()

TAG_NAMES = ["Quiet", "Brunch", "Date Night", "Hidden Gem",
             "Aesthetic", "Family-Friendly", "Late Night", "Trendy"]
TAG_DISPLAY = {
    "Quiet": "Quiet", "Brunch": "Brunch", "Date Night": "Date Night",
    "Hidden Gem": "Hidden Gem", "Aesthetic": "Aesthetic",
    "Family-Friendly": "Family", "Late Night": "Late Night", "Trendy": "Trendy",
}
DEFAULT_WEIGHTS = {"sentiment": 0.30, "tag_match": 0.30,
                   "stars": 0.20, "volume": 0.10, "cuisine": 0.10}

CUISINE_LIST = [
    "Chinese", "Dim Sum", "Cantonese", "Italian", "Mexican", "Japanese", "Thai", "Indian", "French", "Korean", "Mediterranean", "American",
    "Sushi", "Pizza", "Burgers", "BBQ", "Seafood", "Ramen", "Vietnamese", "Greek", "Middle Eastern", "Taiwanese", "Spanish",
]

def cuisine_label(cats: str, preferred: list[str] = None) -> str:
    cats_lower = str(cats).lower()
    if preferred:
        for kw in preferred:
            if kw in cats_lower:
                return kw.title()
    for c in CUISINE_LIST:
        if c.lower() in cats_lower:
            return c
    parts = [p.strip() for p in str(cats).split(",")
             if p.strip().lower() not in ("restaurants", "food", "nan", "")]
    return parts[0] if parts else "Restaurant"

def stars_html(avg: float) -> str:
    n = max(0, min(5, round(float(avg))))
    return (f'<span class="stars-filled">{"★" * n}</span>'
            f'<span class="stars-empty">{"★" * (5 - n)}</span>')

def top_tags_str(row: pd.Series, n: int = 3) -> str:
    scores = {}
    for tag in TAG_NAMES:
        col = f"tag_{tag.lower().replace(' ', '_').replace('-', '_')}"
        if col in row.index:
            scores[tag] = float(row[col])
    top = [TAG_DISPLAY[t] for t, v in
           sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
           if v > 0.05]
    return ", ".join(top) if top else "—"

def build_table(results: pd.DataFrame, cuisine_kws: list[str] = None) -> str:
    rows = ""
    for _, row in results.iterrows():
        name    = str(row.get("name", "Unknown"))[:40]
        cuisine = cuisine_label(row.get("categories", ""), preferred=cuisine_kws)
        stars   = stars_html(row.get("avg_stars", 3))
        tags    = top_tags_str(row)
        pct     = int(round(row["match_score"] * 10))
        rows += f"""
        <tr>
          <td><span class="call-link">Call</span></td>
          <td class="rest-name">{name}</td>
          <td class="cuisine">{cuisine}</td>
          <td>{stars}</td>
          <td class="tag-text">{tags}</td>
          <td class="match-td"><span class="match-pct">{pct}%</span></td>
        </tr>"""
    return f"""
    <table class="ff-table">
      <thead>
        <tr>
          <th></th>
          <th>Recommended</th>
          <th>Cuisine</th>
          <th class="center">&#9733;</th>
          <th>tags</th>
          <th class="center">Match %</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""

col_logo, col_input, col_btn = st.columns([1, 5, 0.6])

with col_logo:
    st.markdown('<div class="ff-logo">Foodie Finds</div>', unsafe_allow_html=True)

with col_input:
    query = st.text_input(
        label="search",
        label_visibility="collapsed",
        placeholder="Chinese family oriented spot with good food and is popular for lunch",
        key="query_input",
    )

with col_btn:
    search_clicked = st.button("🔍")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

col_info, col_results = st.columns([1, 3.2])

with col_info:
    st.markdown("""
    <div class="info-box">
      <div class="info-line">Don't know what to eat?<br>Where to eat?</div>
      <div class="info-line">Tasked with picking<br>a spot?</div>
      <div class="info-line">Looking for a date spot<br>with little to no<br>information?</div>
      <div class="info-cta">Hit the search bar.</div>
    </div>
    """, unsafe_allow_html=True)

with col_results:
    if query:
        parsed  = parse_query(query)
        results = compute_match_scores(profiles, parsed, weights=DEFAULT_WEIGHTS, top_k=10)
        if results.empty:
            st.warning("No restaurants found. Try removing the city or broadening your search.")
        else:
            st.markdown(build_table(results, cuisine_kws=parsed.get("cuisine_keywords")), unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='color:#888;font-family:sans-serif;padding-top:120px;"
            "text-align:center;font-size:16px;'>Enter a query above to find your next spot.</div>",
            unsafe_allow_html=True,
        )
