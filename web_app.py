"""Run the local web interface for the phishing email detector."""

from pathlib import Path

from flask import Flask, render_template, request

from phishing_detector.features import extract_features, extract_urls, requests_credentials
from phishing_detector.model import load_model, predict

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "phishing_model.joblib"
FEATURE_LABELS = {
    "num_urls": "Number of URLs",
    "has_ip_url": "IP address in URL",
    "has_http_only": "Unencrypted HTTP link",
    "has_suspicious_tld": "Suspicious URL domain ending",
    "has_shortener": "URL shortener",
    "has_at_in_url": "@ symbol in URL",
    "max_url_length": "Longest URL length",
    "max_hyphens_in_host": "Most hyphens in a URL host",
    "max_subdomains": "Most subdomains in a URL host",
    "has_digit_in_host": "Digits mixed into a URL host",
    "urgency_count": "Urgency wording",
    "credential_count": "Credential-related wording",
    "reward_count": "Reward-related wording",
    "generic_greeting": "Generic greeting",
    "exclamation_count": "Exclamation marks",
    "uppercase_ratio": "Uppercase letter ratio",
    "text_length": "Email length",
}


def create_app(model_path: str | Path = MODEL_PATH) -> Flask:
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run `python train.py` first."
        )

    model = load_model(model_path)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1_000_000

    @app.route("/", methods=["GET", "POST"])
    def index():
        email = ""
        result = None
        error = None

        if request.method == "POST":
            email = request.form.get("email", "")
            if not email.strip():
                error = "Paste an email message before checking it."
            else:
                label, probability = predict(model, [email])[0]
                features = extract_features(email)
                result = {
                    "label": label,
                    "probability": probability,
                    "credential_request": requests_credentials(email),
                    "urls": extract_urls(email),
                    "features": [
                        (FEATURE_LABELS[name], value)
                        for name, value in features.items()
                        if value
                    ],
                }

        return render_template("index.html", email=email, result=result, error=error)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
