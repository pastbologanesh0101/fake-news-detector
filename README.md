# AI Fake News Detector

A small, self-contained portfolio project that demonstrates a full text
classification pipeline: a bundled toy dataset, a scikit-learn
TF-IDF + Logistic Regression model, a CLI, unit tests, and CI.

## Important disclaimer

**This is a demo / portfolio project, not a production fake-news
detector.** The dataset in `data/headlines.csv` (~290 rows) is
entirely **synthetic and author-written** for this project — short
template-generated "real" headlines (city council, weather, earnings,
research announcements, etc.) and "fake" headlines (sensational
clickbait, conspiracy phrasing, miracle-cure language, etc.). It is
not sourced from any real news outlet, fact-checking database, or
external API, and no data is downloaded at runtime.

The goal is to show a working, testable ML pipeline end to end, not to
make claims about detecting real-world misinformation. A model this
small, trained on this little synthetic data, will not generalize to
real news text.

## What it does

- Loads a labeled dataset of short headline/snippet text
  (`real = 0`, `fake = 1`).
- Splits it into train/test sets.
- Trains a scikit-learn `Pipeline` combining:
  - `TfidfVectorizer` (unigrams + bigrams, English stop words removed)
  - `LogisticRegression` classifier
- Evaluates held-out accuracy.
- Exposes a CLI (`detect.py`) that trains the pipeline and predicts
  `real` or `fake` (with a confidence score) for any input text.

## Project layout

```
fake-news-detector/
├── data/
│   └── headlines.csv        # ~290 rows of synthetic, labeled headlines
├── model.py                  # dataset loading + pipeline build/train/predict
├── detect.py                  # CLI entry point
├── tests/
│   └── test_model.py         # unit tests
├── requirements.txt
├── .github/workflows/tests.yml
├── LICENSE
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python detect.py "SHOCKING: This one weird trick cures cancer overnight, doctors furious"
```

Example output:

```
Input: "SHOCKING: This one weird trick cures cancer overnight, doctors furious"
Prediction: FAKE (confidence: 77.17%)
[demo model — held-out test accuracy this run: 100.00%]
```

```bash
python detect.py "City council approves budget for road repairs after public hearing"
```

```
Input: "City council approves budget for road repairs after public hearing"
Prediction: REAL (confidence: 77.36%)
[demo model — held-out test accuracy this run: 100.00%]
```

Each CLI run retrains the pipeline from scratch on the bundled dataset
(training is fast — well under a second on this small dataset) and
reports that run's held-out test accuracy alongside the prediction.

## Running the tests

```bash
pip install pytest
pytest -v
```

The test suite covers:

- The dataset loads correctly and stays within the expected size/label
  balance.
- `build_pipeline()` returns a pipeline with the expected steps.
- `train_model()` produces a sane train/test split.
- Held-out test accuracy exceeds a sanity threshold (`> 0.7`).
- `predict()` correctly classifies clearly "fake"-style and
  clearly "real"-style example text with reasonable confidence.

## Continuous integration

`.github/workflows/tests.yml` runs the test suite on every push and
pull request against `main`, on Python 3.11 and 3.12, via
`actions/checkout@v4` and `actions/setup-python@v5`.

## License

MIT — see [LICENSE](LICENSE).
