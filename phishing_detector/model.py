"""Model definition: TF-IDF text features + hand-crafted URL/keyword features."""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from .features import EmailFeatureExtractor

LABELS = ["Safe", "Phishing"]

CLASSIFIERS = {
    "logreg": lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "naive_bayes": lambda: ComplementNB(),
}


def build_pipeline(classifier: str = "logreg") -> Pipeline:
    """Combine word/char TF-IDF of the body with scaled hand-crafted features."""
    features = FeatureUnion([
        ("words", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, stop_words="english")),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                  sublinear_tf=True, max_features=20000)),
        ("handcrafted", Pipeline([("extract", EmailFeatureExtractor()),
                                  ("scale", MaxAbsScaler())])),
    ])
    return Pipeline([("features", features), ("clf", CLASSIFIERS[classifier]())])


def load_model(path: str | Path):
    return joblib.load(path)


def predict(model, emails: list[str]) -> list[tuple[str, float]]:
    """Return (label, phishing probability) for each email."""
    probs = model.predict_proba(emails)[:, list(model.classes_).index("Phishing")]
    return [("Phishing" if p >= 0.5 else "Safe", float(p)) for p in probs]
