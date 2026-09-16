#!/usr/bin/env python3
"""CLI entry point for the AI Fake News Detector demo.

Usage:
    python detect.py "some headline text"

Trains the TF-IDF + Logistic Regression pipeline on the bundled toy
dataset (data/headlines.csv) and prints a prediction (real/fake) with
a confidence score for the given text.

Reminder: this is a demo pipeline trained on a small synthetic dataset,
not a production fake-news detector.
"""

import sys

from model import predict, train_model


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print('Usage: python detect.py "some headline text"')
        return 1

    text = " ".join(argv[1:])

    result = train_model()
    label, confidence = predict(result.pipeline, text)

    print(f'Input: "{text}"')
    print(f"Prediction: {label.upper()} (confidence: {confidence:.2%})")
    print(f"[demo model — held-out test accuracy this run: {result.accuracy:.2%}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
