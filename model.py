"""Core ML pipeline for the AI Fake News Detector demo.

This module builds a small scikit-learn text classification pipeline
(TF-IDF vectorizer + Logistic Regression) trained on a bundled toy
dataset of synthetic headlines (see data/headlines.csv).

IMPORTANT: This is a portfolio / demo project. The dataset is small and
entirely synthetic/author-written, so this is NOT a production-grade
fake news detector. It exists to demonstrate a full, working ML
pipeline (data -> vectorizer -> classifier -> CLI) end to end.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "headlines.csv")

LABELS = {0: "real", 1: "fake"}


def load_dataset(path: str = DATA_PATH) -> tuple[list[str], list[int]]:
    """Load the bundled CSV dataset into parallel lists of texts and labels."""
    df = pd.read_csv(path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def build_pipeline() -> Pipeline:
    """Construct an untrained TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )


@dataclass
class TrainResult:
    pipeline: Pipeline
    accuracy: float
    n_train: int
    n_test: int


def train_model(
    texts: list[str] | None = None,
    labels: list[int] | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> TrainResult:
    """Train the pipeline on the dataset and evaluate on a held-out split."""
    if texts is None or labels is None:
        texts, labels = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return TrainResult(
        pipeline=pipeline,
        accuracy=accuracy,
        n_train=len(x_train),
        n_test=len(x_test),
    )


def predict(pipeline: Pipeline, text: str) -> tuple[str, float]:
    """Predict the label ('real'/'fake') and confidence for a piece of text."""
    proba = pipeline.predict_proba([text])[0]
    pred_idx = int(proba.argmax())
    label = LABELS[pred_idx]
    confidence = float(proba[pred_idx])
    return label, confidence
