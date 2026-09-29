"""Train the phishing email classifier and report accuracy + confusion matrix.

Usage:
    python train.py                              # uses data/emails.csv (generated if missing)
    python train.py --data my_emails.csv --classifier random_forest
    python train.py --compare                    # evaluate all classifiers

The CSV must have a text column and a label column. Labels may be
"Phishing"/"Safe", 1/0, or common variants such as "Phishing Email"/"Safe Email".
"""

import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

from phishing_detector.features import FEATURE_NAMES
from phishing_detector.model import CLASSIFIERS, LABELS, build_pipeline

ROOT = Path(__file__).parent
PHISHING_VALUES = {"1", "phishing", "phishing email", "spam", "malicious", "true"}


def load_dataset(path: Path, text_col: str, label_col: str) -> tuple[pd.Series, pd.Series]:
    if not path.exists() and path == ROOT / "data" / "emails.csv":
        subprocess.run([sys.executable, str(ROOT / "data" / "generate_dataset.py"), "--out", str(path)], check=True)
    df = pd.read_csv(path).dropna(subset=[text_col, label_col])
    labels = df[label_col].astype(str).str.strip().str.lower().map(
        lambda v: "Phishing" if v in PHISHING_VALUES else "Safe")
    return df[text_col].astype(str), labels


def evaluate(name, model, X_test, y_test, out_dir: Path) -> float:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, digits=4))
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=[f"actual {l}" for l in LABELS], columns=[f"pred {l}" for l in LABELS]))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} — accuracy {acc:.2%}")
    fig.tight_layout()
    path = out_dir / f"confusion_matrix_{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved confusion matrix plot to {path}")
    return acc


def show_top_features(model, k: int = 15) -> None:
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return
    names = model.named_steps["features"].get_feature_names_out()
    coefs = clf.coef_[0]
    # coef_ points towards classes_[1]; flip so positive always means "more phishing".
    sign = 1 if clf.classes_[1] == "Phishing" else -1
    order = (sign * coefs).argsort()
    print(f"\nTop {k} indicators of PHISHING:")
    for i in order[::-1][:k]:
        print(f"  {names[i]:<40} {sign * coefs[i]:+.3f}")
    print("\nHand-crafted feature weights:")
    for n in FEATURE_NAMES:
        i = list(names).index(f"handcrafted__{n}")
        print(f"  {n:<22} {sign * coefs[i]:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "emails.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--classifier", choices=list(CLASSIFIERS), default="logreg")
    parser.add_argument("--compare", action="store_true", help="train and evaluate every classifier")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-out", type=Path, default=ROOT / "models" / "phishing_model.joblib")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    X, y = load_dataset(args.data, args.text_col, args.label_col)
    print(f"Loaded {len(X)} emails: {y.value_counts().to_dict()}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size,
                                                        stratify=y, random_state=42)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    names = list(CLASSIFIERS) if args.compare else [args.classifier]
    results = {}
    for name in names:
        model = build_pipeline(name).fit(X_train, y_train)
        results[name] = (evaluate(name, model, X_test, y_test, args.out_dir), model)

    best = max(results, key=lambda n: results[n][0])
    if args.compare:
        print("\nSummary:")
        for n, (acc, _) in sorted(results.items(), key=lambda kv: -kv[1][0]):
            print(f"  {n:<15} {acc:.4f}")
    best_model = results[best][1]
    show_top_features(best_model)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.model_out)
    print(f"\nSaved {best} model to {args.model_out}")


if __name__ == "__main__":
    main()
