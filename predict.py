"""Classify emails as Phishing or Safe with a trained model.

Usage:
    python predict.py "Your account is suspended, verify at http://1.2.3.4/login"
    python predict.py --file email.txt
    echo "email body" | python predict.py
"""

import argparse
import sys
from pathlib import Path

from phishing_detector.features import extract_features, extract_urls, requests_credentials
from phishing_detector.model import load_model, predict

ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="*", help="email text(s) to classify")
    parser.add_argument("--file", type=Path, help="read one email from a file")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "phishing_model.joblib")
    parser.add_argument("--explain", action="store_true", help="print extracted URL/keyword features")
    args = parser.parse_args()

    if not args.model.exists():
        sys.exit(f"Model not found at {args.model}. Run `python train.py` first.")
    emails = args.text or ([args.file.read_text(encoding="utf-8")] if args.file else [sys.stdin.read()])

    model = load_model(args.model)
    for email, (label, prob) in zip(emails, predict(model, emails)):
        preview = " ".join(email.split())[:70]
        print(f"[{label:<8}] model phishing probability {prob:6.1%} | {preview}")
        if requests_credentials(email):
            print("    Rule warning: asks for a password or access code.")
        if args.explain:
            print(f"    URLs: {extract_urls(email) or 'none'}")
            for name, value in extract_features(email).items():
                if value:
                    print(f"    {name:<22} {value:g}")


if __name__ == "__main__":
    main()
