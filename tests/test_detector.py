import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from generate_dataset import generate  # noqa: E402
from phishing_detector import (build_pipeline, extract_features, extract_urls, predict,
                               requests_credentials)  # noqa: E402


def test_extract_urls_strips_trailing_punctuation():
    assert extract_urls("See https://example.com/a. And www.test.org!") == [
        "https://example.com/a", "www.test.org"]


def test_url_features_flag_suspicious_links():
    f = extract_features("Verify now at http://10.0.0.1/login or http://paypal-secure-login.xyz/x")
    assert f["num_urls"] == 2
    assert f["has_ip_url"] == 1
    assert f["has_http_only"] == 1
    assert f["has_suspicious_tld"] == 1
    assert f["max_hyphens_in_host"] == 2
    assert f["credential_count"] >= 1


def test_safe_email_has_no_url_flags():
    f = extract_features("Hi Sam, lunch on Friday? Menu: https://www.company.com/menu")
    assert f["has_ip_url"] == f["has_http_only"] == f["has_suspicious_tld"] == 0


@pytest.mark.parametrize("secret", ["password", "OTP", "one-time passcode", "verification code"])
def test_detects_requests_to_disclose_passwords_or_codes(secret):
    assert requests_credentials(f"Please reply with your {secret} to keep your account active.")


@pytest.mark.parametrize("phrasing", [
    "What is your password?",
    "What's the OTP?",
    "Which is your verification code?",
    "May I have your passcode?",
    "Could you please provide the code?",
    "Enter your password to continue.",
])
def test_detects_different_password_and_code_request_phrasings(phrasing):
    assert requests_credentials(phrasing)


@pytest.mark.parametrize("notice", [
    "We will never ask for your password by email.",
    "Do not share your OTP with anyone.",
    "We will never ask what your password is.",
])
def test_security_notices_are_not_credential_requests(notice):
    assert not requests_credentials(notice)


@pytest.fixture(scope="module")
def model():
    rows = generate(600, seed=1)
    texts = [t for t, _ in rows]
    labels = ["Phishing" if l else "Safe" for _, l in rows]
    return build_pipeline().fit(texts, labels)


def test_model_classifies_obvious_examples(model):
    results = predict(model, [
        "URGENT: your account is suspended. Verify your password immediately at http://1.2.3.4/login",
        "Hi Alex, notes from today's meeting are in the shared folder. See you tomorrow!",
    ])
    assert [label for label, _ in results] == ["Phishing", "Safe"]


def test_model_flags_password_or_otp_requests_without_a_url(model):
    results = predict(model, [
        "Please reply with your current password so we can migrate your mailbox.",
        "Please reply with your OTP to keep your account active.",
        "We will never ask for your password by email.",
    ])
    assert [label for label, _ in results] == ["Phishing", "Phishing", "Safe"]


def test_model_accuracy_on_held_out_data(model):
    rows = generate(400, seed=7)
    results = predict(model, [t for t, _ in rows])
    acc = sum((label == "Phishing") == bool(l) for (label, _), (_, l) in zip(results, rows)) / len(rows)
    assert acc > 0.9
