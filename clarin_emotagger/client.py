"""Convenience client helpers for library consumers."""

import asyncio
from typing import Any, Dict, Optional

from .emotagger_wrapper import analyze_sentiment_async


def analyze_sentiment(text: str) -> Optional[Dict[str, Any]]:
    """Synchronous helper around `analyze_sentiment_async`.

    Use this only in non-async applications.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(analyze_sentiment_async(text))

    raise RuntimeError(
        "analyze_sentiment() called inside a running event loop. "
        "Use await analyze_sentiment_async(text) instead."
    )
