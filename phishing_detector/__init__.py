"""Phishing email detection with scikit-learn."""

from .features import EmailFeatureExtractor, extract_features, extract_urls, requests_credentials
from .model import LABELS, build_pipeline, load_model, predict

__all__ = ["EmailFeatureExtractor", "extract_features", "extract_urls",
           "LABELS", "build_pipeline", "load_model", "predict"]
