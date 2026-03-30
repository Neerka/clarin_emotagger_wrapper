"""Public API for the CLARIN Emotagger wrapper library."""

from .emotagger_models import EmotionOutput
from .emotagger_wrapper import analyze_sentiment_async
from .client import analyze_sentiment

__all__ = ["analyze_sentiment_async", "analyze_sentiment", "EmotionOutput"]
