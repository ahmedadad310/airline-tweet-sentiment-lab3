"""End-to-end Twitter US Airline Sentiment analysis.

Run from this directory with:
    python analysis.py

Expected input: Tweets.csv in the same directory.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Tweets.csv"
OUTPUT_DIR = ROOT / "analysis_outputs"


def clean_text(text: str) -> str:
    """Lowercase text and remove URLs and @mentions."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return text.strip()


def remove_stopwords(text: str, stop_words: set[str]) -> str:
    """Remove common English stopwords but keep not/no because they affect sentiment."""
    return " ".join(word for word in text.split() if word not in stop_words)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.name}. Download it from "
            "https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    nltk.download("stopwords", quiet=True)

    df = pd.read_csv(DATA_PATH)
    required_columns = {"text", "airline_sentiment"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Tweets.csv is missing required columns: {sorted(missing)}")

    print(f"Loaded {len(df):,} rows from {DATA_PATH.name}")
    print("\nSentiment counts:")
    print(df["airline_sentiment"].value_counts())

    df["clean_text"] = df["text"].apply(clean_text)
    stop_words = set(stopwords.words("english")) - {"not", "no"}
    df["clean_text"] = df["clean_text"].apply(
        lambda text: remove_stopwords(text, stop_words)
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["airline_sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTF-IDF matrix shape: {X.shape}")
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    ax = df["airline_sentiment"].value_counts().sort_values().plot(
        kind="barh", color=["#f19b91", "#f0d783", "#9ce4c5"]
    )
    ax.set_title("Twitter US Airline Sentiment Distribution")
    ax.set_xlabel("Number of tweets")
    ax.set_ylabel("Sentiment")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution.png", dpi=160)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="GnBu")
    plt.title("Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    sample = "I had a great flight and the crew was amazing!"
    sample_clean = remove_stopwords(clean_text(sample), stop_words)
    sample_prediction = model.predict(vectorizer.transform([sample_clean]))[0]
    print(f"Sample prediction: {sample_prediction}")
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
