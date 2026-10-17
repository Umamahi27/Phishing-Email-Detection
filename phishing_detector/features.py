"""Hand-crafted feature extraction for emails: URLs, keywords and style cues."""

import re
from urllib.parse import urlparse

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')]+", re.IGNORECASE)
IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?$")

SUSPICIOUS_TLDS = {"xyz", "top", "info", "ru", "tk", "click", "online", "biz", "cn", "ml", "ga", "gq"}
SHORTENERS = {"bit.ly", "tinyurl.com", "is.gd", "goo.gl", "t.co", "ow.ly", "rebrand.ly"}

URGENCY_WORDS = ["urgent", "immediately", "act now", "expires", "within 24 hours", "within 48 hours",
                 "suspended", "deactivated", "permanent", "final notice", "as soon as possible",
                 "limited time", "do not ignore"]
CREDENTIAL_WORDS = ["verify", "confirm", "password", "login", "log in", "sign in", "credentials",
                    "social security", "bank account", "credit card", "billing information",
                    "update your", "account details","otp", "one-time passcode", "verification code", "security code", "passcode"]
REWARD_WORDS = ["winner", "congratulations", "prize", "gift card", "refund", "claim", "free", "$"]
GENERIC_GREETINGS = ["dear customer", "dear user", "dear valued", "dear account holder", "hello,"]
CREDENTIAL_TERMS = (
    r"(?:passwords?|passcodes?|pass[\s-]?phrases?|credentials?|otps?|codes?|"
    r"one[\s-]?time(?:\s+(?:password|passcode|code|pin))?|"
    r"(?:verification|security|authentication)\s+(?:codes?|pins?)|pins?)"
)
CREDENTIAL_REQUEST_RE = re.compile(
    r"\b(?:send|share|provide|tell|give|email|text|submit|disclose|reveal|enter|type|"
    r"supply|write|return|forward|reply|request|need|required|ask(?:\s+for)?)\b"
    r"(?:\W+\w+){0,10}\W+"
    rf"\b{CREDENTIAL_TERMS}\b",
    re.IGNORECASE,
)
CREDENTIAL_QUESTION_RE = re.compile(
    rf"\b(?:what(?:'s|\s+is)|which\s+is)\s+(?:your|the)\s+{CREDENTIAL_TERMS}\b",
    re.IGNORECASE,
)
CREDENTIAL_HAVE_REQUEST_RE = re.compile(
    rf"\b(?:may|can|could|would)\s+i\s+(?:please\s+)?have\s+(?:your|the)\s+{CREDENTIAL_TERMS}\b",
    re.IGNORECASE,
)
CREDENTIAL_REQUEST_NEGATION_RE = re.compile(
    r"\b(?:never|not|don't|do\s+not|shouldn't|should\s+not|won't|will\s+not|"
    r"cannot|can't)\b(?:\W+\w+){0,3}\W*$",
    re.IGNORECASE,
)


def extract_urls(text: str) -> list[str]:
    return [u.rstrip(".,;:!?") for u in URL_RE.findall(text or "")]


def requests_credentials(text: str) -> bool:
    """Return whether an email asks the recipient to disclose a password or code."""
    text = text or ""
    for pattern in (CREDENTIAL_REQUEST_RE, CREDENTIAL_QUESTION_RE, CREDENTIAL_HAVE_REQUEST_RE):
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 48):match.start()]
            if not CREDENTIAL_REQUEST_NEGATION_RE.search(prefix):
                return True
    return False


def _host(url: str) -> str:
    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url
    return (urlparse(url).hostname or "").lower()


def url_features(urls: list[str]) -> dict[str, float]:
    hosts = [_host(u) for u in urls]
    return {
        "num_urls": len(urls),
        "has_ip_url": float(any(IP_HOST_RE.match(h) for h in hosts)),
        "has_http_only": float(any(u.lower().startswith("http://") for u in urls)),
        "has_suspicious_tld": float(any(h.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS for h in hosts if "." in h)),
        "has_shortener": float(any(h.removeprefix("www.") in SHORTENERS for h in hosts)),
        "has_at_in_url": float(any("@" in u for u in urls)),
        "max_url_length": float(max((len(u) for u in urls), default=0)),
        "max_hyphens_in_host": float(max((h.count("-") for h in hosts), default=0)),
        "max_subdomains": float(max((h.count(".") for h in hosts), default=0)),
        "has_digit_in_host": float(any(re.search(r"[a-z]\d|\d[a-z]", h) for h in hosts)),
    }


def text_features(text: str) -> dict[str, float]:
    lower = (text or "").lower()
    letters = [c for c in text if c.isalpha()] or [" "]
    return {
        "urgency_count": float(sum(lower.count(w) for w in URGENCY_WORDS)),
        "credential_count": float(sum(lower.count(w) for w in CREDENTIAL_WORDS)),
        "reward_count": float(sum(lower.count(w) for w in REWARD_WORDS)),
        "generic_greeting": float(any(g in lower for g in GENERIC_GREETINGS)),
        "exclamation_count": float(text.count("!")),
        "uppercase_ratio": sum(c.isupper() for c in letters) / len(letters),
        "text_length": float(len(text)),
    }


def extract_features(text: str) -> dict[str, float]:
    """Return every hand-crafted feature for a single email as a name -> value dict."""
    text = text or ""
    return {**url_features(extract_urls(text)), **text_features(text)}


FEATURE_NAMES = list(extract_features("").keys())


class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer turning raw email text into the hand-crafted feature matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([[extract_features(t)[n] for n in FEATURE_NAMES] for t in X], dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES, dtype=object)
