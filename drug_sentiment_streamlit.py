import re
import subprocess
import sys

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
try:
    from ucimlrepo import fetch_ucirepo
except ModuleNotFoundError:
    # Streamlit can run with a different interpreter than notebook.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo"])
    from ucimlrepo import fetch_ucirepo


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def label_rating(r: float) -> int:
    if r >= 7:
        return 2
    if r >= 5:
        return 1
    return 0


def label_for_ui(label_3class: int) -> int:
    # Conservative policy for end users:
    # neutral and negative are grouped into "raczej nie".
    return 1 if int(label_3class) == 2 else 0


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data = fetch_ucirepo(id=461)
    df = pd.concat([data.data.features, data.data.targets], axis=1)

    df["full_review"] = (
        df["benefitsReview"].fillna("")
        + " "
        + df["sideEffectsReview"].fillna("")
        + " "
        + df["commentsReview"].fillna("")
    )
    df["clean_review"] = df["full_review"].apply(clean_text)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()
    df["label"] = df["rating"].apply(label_rating)
    df["label_ui"] = df["label"].apply(label_for_ui)
    return df[df["clean_review"] != ""].drop_duplicates(subset=["clean_review"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    X_train, _, y_train, _ = train_test_split(
        df["clean_review"], df["label_ui"], test_size=0.2, random_state=42, stratify=df["label_ui"]
    )
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)
    return vectorizer, model


st.set_page_config(page_title="Drug Review Dashboard", layout="wide")
st.title("Drug Review Dashboard")
st.caption("Dla bezpieczeństwa: neutralne opinie traktujemy jako 'raczej nie'.")

try:
    df = load_data()
    vectorizer, model = train_model(df)
except Exception as e:
    st.error(f"Nie udało się uruchomić aplikacji: {e}")
    st.stop()

label_map = {0: "raczej nie", 1: "godny uwagi"}
drug_col = "urlDrugName" if "urlDrugName" in df.columns else "drugName"

all_drugs = sorted(df[drug_col].dropna().astype(str).unique())
selected_drug = st.selectbox("Wybierz lek:", all_drugs)

drug_reviews = df[df[drug_col].astype(str) == selected_drug].copy()
review_count = len(drug_reviews)

majority_label = int(drug_reviews["label_ui"].mode().iloc[0]) if review_count > 0 else None
final_sentiment = label_map.get(majority_label, "brak")

if "sideEffects" in drug_reviews.columns and drug_reviews["sideEffects"].notna().sum() > 0:
    top_side_effect = str(drug_reviews["sideEffects"].mode().iloc[0])
else:
    top_side_effect = "brak danych"

c1, c2, c3 = st.columns(3)
c1.metric("Liczba recenzji", review_count)
c2.metric("Czy lek godny uwagi?", final_sentiment)
c3.metric("Dominujące skutki uboczne", top_side_effect)

st.subheader("Rozkład decyzji (2 klasy)")
sentiment_counts = drug_reviews["label_ui"].map(label_map).value_counts()
st.bar_chart(sentiment_counts)

st.subheader("Przykładowe recenzje")
review_table_cols = [drug_col, "clean_review"]
if "sideEffects" in drug_reviews.columns:
    review_table_cols.append("sideEffects")
review_preview = drug_reviews[review_table_cols].head(10).copy()
st.dataframe(review_preview, use_container_width=True)

csv_data = review_preview.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Pobierz podgląd recenzji (CSV)",
    data=csv_data,
    file_name=f"{selected_drug}_reviews_preview.csv",
    mime="text/csv",
)

st.subheader("Predykcja własnej recenzji")
custom = st.text_area("Wpisz własną recenzję:")
if custom.strip():
    pred = model.predict(vectorizer.transform([clean_text(custom)]))[0]
    st.success(f"Rekomendacja modelu: {label_map[int(pred)]}")
