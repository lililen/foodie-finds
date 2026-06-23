# Foodie Finds Project Summary

## Non-Technical Summary

Foodie Finds is a restaurant recommendation app. You type a natural language description of what you're looking for (e.g. "a quiet Chinese family spot popular for lunch") and it returns a ranked list of restaurants with a match percentage showing how well each one fits the query.

The app processes Yelp review data to build a profile for each restaurant, capturing things like sentiment from reviews and tags like whether it's good for a date night, family-friendly, or a hidden gem. Those profiles are compared against the query to generate the rankings. The results show restaurant name, cuisine type, star rating, top vibe tags, and match score. There is also a call button on each result.

If Yelp data is present it loads from that; otherwise it falls back to a generated sample dataset.

## Technical Summary

### Project Structure

```
foodie-finds/
├── app.py                  # Streamlit app (UI + orchestration)
├── download_data.py        # Yelp dataset downloader
├── requirements.txt        # Python dependencies
├── data/                   # Yelp JSON data + cached parquet profiles
├── notebooks/              # Jupyter notebooks (exploration and development)
└── src/
    ├── preprocessing.py    # Data loading, cleaning, and profile building
    └── ranking/
        └── match_score.py  # Query parsing and scoring engine
```

### Stack and Dependencies

| Library | Role |
|---|---|
| `streamlit` | Web UI framework |
| `pandas` | Data manipulation and profile storage |
| `numpy` | Numerical operations |
| `scikit-learn` | ML utilities (likely TF-IDF or similarity) |
| `nltk` + `spacy` | NLP for query parsing |
| `vaderSentiment` | Sentiment analysis on review text |
| `matplotlib` + `seaborn` | Visualization, used in notebooks |
| `tqdm` | Progress bars during data processing |

### Architecture

#### Data Layer (`data/`, `download_data.py`, `preprocessing.py`)

- Loads raw Yelp JSON (reviews + businesses) with `load_yelp_reviews()`
- Cleans and normalizes restaurant data with `preprocess_dataframe()`
- Builds per-restaurant profiles with `build_restaurant_profiles()`, which aggregates VADER sentiment scores and tag scores (e.g. `tag_quiet`, `tag_date_night`, `tag_hidden_gem`) across all reviews for each restaurant
- Caches profiles as a `.parquet` file to avoid reprocessing on startup
- Falls back to `make_sample_dataset(n=3000)` if no Yelp data is available

#### Ranking Engine (`src/ranking/match_score.py`)

- `parse_query(query)` extracts cuisine keywords, vibe/tag signals, and sentiment signals from the input string using NLP
- `compute_match_scores(profiles, parsed, weights, top_k)` scores all restaurant profiles using a weighted composite:

| Signal | Default Weight |
|---|---|
| Sentiment (review positivity) | 30% |
| Tag match | 30% |
| Star rating | 20% |
| Review volume | 10% |
| Cuisine match | 10% |

Returns the top-K results sorted by score, displayed as a 0-100% match.

#### UI Layer (`app.py`)

- Built with Streamlit with fully custom CSS overriding all default component styles
- Layout: 3-column header (logo, search bar, search button) and 2-column body (info panel + results table)
- Results rendered as a custom HTML table with star ratings as filled/empty glyphs, top 3 vibe tags per restaurant, and match percentage
- Profiles loaded once on startup and cached with `@st.cache_data`
- Cuisine label logic parses multi-category Yelp strings (e.g. "Chinese, Dim Sum, Cantonese") and maps to a single display label, preferring any cuisine keyword found in the query

#### Tag System

Eight tags are tracked per restaurant, each with a continuous score from 0 to 1 inferred from review language: Quiet, Brunch, Date Night, Hidden Gem, Aesthetic, Family-Friendly, Late Night, Trendy. The top 3 scoring tags above a 0.05 threshold are shown in the results.

### Data Flow

```
Yelp JSON (reviews + businesses)
        |
  preprocess_dataframe()
        |
  build_restaurant_profiles()     <- VADER sentiment + tag scoring per restaurant
        |
  restaurant_profiles.parquet     <- cached on disk
        |
  User submits query
        |
  parse_query()                   <- extract cuisine + vibe + sentiment signals
        |
  compute_match_scores()          <- weighted scoring across all profiles
        |
  Top 10 results rendered as HTML table
```

### Notes

- The repo is 90% Jupyter Notebook by language, meaning most of the NLP and scoring logic was developed and iterated in notebooks before being refactored into `src/`
- Parquet caching keeps startup fast when working with the full Yelp dataset
- The synthetic fallback dataset makes the app runnable without downloading the Yelp data