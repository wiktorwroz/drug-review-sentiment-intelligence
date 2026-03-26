import re
import subprocess
import sys

import numpy as np
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


def calibrate_threshold(y_true: pd.Series, prob_pos: np.ndarray):
    # Tune decision threshold on validation data instead of fixed 0.5.
    thresholds = np.linspace(0.2, 0.8, 25)
    rows = []

    for thr in thresholds:
        pred = (prob_pos >= thr).astype(int)
        tp = int(((pred == 1) & (y_true.values == 1)).sum())
        fp = int(((pred == 1) & (y_true.values == 0)).sum())
        fn = int(((pred == 0) & (y_true.values == 1)).sum())
        tn = int(((pred == 0) & (y_true.values == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        bal_acc = (recall + specificity) / 2.0

        rows.append(
            {
                "threshold": float(thr),
                "f1_positive": float(f1),
                "balanced_accuracy": float(bal_acc),
                "precision_positive": float(precision),
                "recall_positive": float(recall),
            }
        )

    calibration_df = pd.DataFrame(rows)
    best_row = calibration_df.sort_values(
        by=["f1_positive", "balanced_accuracy", "precision_positive"],
        ascending=False,
    ).iloc[0]
    return float(best_row["threshold"]), calibration_df


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
    X_train, X_cal, y_train, y_cal = train_test_split(
        df["clean_review"], df["label_ui"], test_size=0.2, random_state=42, stratify=df["label_ui"]
    )
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_cal_tfidf = vectorizer.transform(X_cal)
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)
    cal_prob = model.predict_proba(X_cal_tfidf)[:, 1]
    threshold, calibration_df = calibrate_threshold(y_cal.reset_index(drop=True), cal_prob)
    return vectorizer, model, threshold, calibration_df


st.set_page_config(page_title="Drug Review Dashboard", layout="wide")
st.title("Drug Review Dashboard")
st.caption("Dla bezpieczeństwa: neutralne opinie traktujemy jako 'raczej nie', a decyzja modelu ma kalibrowany próg.")

try:
    df = load_data()
    vectorizer, model, threshold, calibration_df = train_model(df)
except Exception as e:
    st.error(f"Nie udało się uruchomić aplikacji: {e}")
    st.stop()

label_map = {0: "raczej nie", 1: "godny uwagi"}
drug_col = "urlDrugName" if "urlDrugName" in df.columns else "drugName"

all_drugs = sorted(df[drug_col].dropna().astype(str).unique())
selected_drug = st.selectbox("Wybierz lek:", all_drugs)

drug_reviews = df[df[drug_col].astype(str) == selected_drug].copy()
review_count = len(drug_reviews)

if review_count > 0:
    X_drug = vectorizer.transform(drug_reviews["clean_review"].fillna("").astype(str))
    prob_drug = model.predict_proba(X_drug)[:, 1]
    pred_drug = (prob_drug >= threshold).astype(int)
    pred_series = pd.Series(pred_drug)
    majority_label = int(pred_series.mode().iloc[0])
    final_sentiment = label_map.get(majority_label, "brak")
    decision_conf = float(pred_series.value_counts(normalize=True).max())
    sentiment_counts = pred_series.map(label_map).value_counts()
else:
    final_sentiment = "brak"
    decision_conf = 0.0
    sentiment_counts = pd.Series(dtype=int)

if "sideEffects" in drug_reviews.columns and drug_reviews["sideEffects"].notna().sum() > 0:
    top_side_effect = str(drug_reviews["sideEffects"].mode().iloc[0])
else:
    top_side_effect = "brak danych"

c1, c2, c3 = st.columns(3)
c1.metric("Liczba recenzji", review_count)
c2.metric("Czy lek godny uwagi?", final_sentiment)
c3.metric("Pewność decyzji", f"{decision_conf:.0%}")

c4, c5 = st.columns(2)
c4.metric("Dominujące skutki uboczne", top_side_effect)
c5.metric("Kalibrowany próg", f"{threshold:.2f}")

st.subheader("Rozkład decyzji (2 klasy)")
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
    prob = float(model.predict_proba(vectorizer.transform([clean_text(custom)]))[0, 1])
    pred = int(prob >= threshold)
    st.success(f"Rekomendacja modelu: {label_map[pred]} (p={prob:.2f}, próg={threshold:.2f})")

with st.expander("Pokaż tabelę kalibracji progu", expanded=False):
    st.dataframe(calibration_df, use_container_width=True)
