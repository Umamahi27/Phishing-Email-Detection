import numpy as np

import web_app


class FakeModel:
    classes_ = ["Safe", "Phishing"]

    def predict_proba(self, emails):
        return np.array([[0.1, 0.9] for _ in emails])


def test_local_website_shows_prediction_and_indicators(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    monkeypatch.setattr(web_app, "load_model", lambda path: FakeModel())
    client = web_app.create_app(model_path).test_client()

    response = client.post(
        "/",
        data={"email": "URGENT verify your password at http://10.0.0.1/login"},
    )

    assert response.status_code == 200
    assert b"Phishing" in response.data
    assert b"90.0%" in response.data
    assert b"10.0.0.1" in response.data
    assert b"Credential-related wording" in response.data


def test_local_website_warns_on_otp_request_without_url(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    monkeypatch.setattr(web_app, "load_model", lambda path: FakeModel())
    client = web_app.create_app(model_path).test_client()

    response = client.post("/", data={"email": "Please reply with your OTP to keep access."})

    assert response.status_code == 200
    assert b"Credential-disclosure warning" in response.data
    assert b"access code" in response.data


def test_local_website_rejects_blank_email(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    monkeypatch.setattr(web_app, "load_model", lambda path: FakeModel())
    client = web_app.create_app(model_path).test_client()

    response = client.post("/", data={"email": "  "})

    assert response.status_code == 200
    assert b"Paste an email message before checking it." in response.data
