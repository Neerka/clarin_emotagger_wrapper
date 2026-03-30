"""
Pydantic models for Emotagger input/output schemas and response normalization.

Contracts:
- EmotionOutput: Standardized emotion analysis result passed to LLM model
- CLARINResponseRaw: Raw response from CLARIN Emotagger (for mapping/validation)
- Normalization functions convert CLARIN → EmotionOutput
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


EMOTION_LABELS = {
    "joy",
    "sadness",
    "anger",
    "neutral",
    "fear",
    "disgust",
    "surprise",
    "trust",
    "anticipation",
}

SENTIMENT_LABELS = {"positive", "negative", "neutral"}


class EmotionOutput(BaseModel):
    """
    Standardized emotion analysis output passed from pre-processing to LLM model.
    
    This is the STABLE contract; format should not change between success and fallback cases.
    """
    
    label: str = Field(
        ...,
        description="Dominant emotion label (e.g., 'joy', 'sadness', 'anger', 'neutral', 'fear', 'disgust', 'surprise')",
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the emotion label (0.0 = no confidence, 1.0 = certain)",
    )

    sentiment_label: str = Field(
        default="neutral",
        description="Overall sentiment label: positive, negative, or neutral",
    )

    sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Overall sentiment score in range [-1.0, 1.0]",
    )
    
    source: str = Field(
        ...,
        description="Source of emotion data: 'emotagger' (from CLARIN), 'fallback' (timeout/error), or 'neutral' (unanalyzed)",
    )
    
    latency_ms: Optional[int] = Field(
        default=None,
        description="Emotagger response latency in milliseconds (for monitoring)",
    )
    
    status: str = Field(
        default="success",
        description="Status of emotion analysis: 'success', 'timeout', 'error', 'invalid_response', 'skipped'",
    )
    
    raw_emotagger_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw CLARIN response for debugging (omitted in production logs)",
    )
    
    class Config:
        """Pydantic config for JSON serialization."""
        json_schema_extra = {
            "example": {
                "label": "joy",
                "confidence": 0.92,
                "sentiment_label": "positive",
                "sentiment_score": 0.94,
                "source": "emotagger",
                "latency_ms": 145,
                "status": "success",
            }
        }


class CLARINResponseRaw(BaseModel):
    """
    Raw response from CLARIN Emotagger service (placeholder for actual schema).
    
    This is a best-effort model; exact structure depends on CLARIN service version.
    Adapt this based on actual emotagger API response format.
    """
    
    # Placeholder: CLARIN typically returns structured JSON with emotion scores/labels
    # Example (adapt to actual format):
    emotions: Optional[List[str]] = Field(
        default=None,
        description="List of detected emotion labels (e.g., ['joy', 'sadness'])",
    )
    
    scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Emotion confidence scores mapped to labels (e.g., {'joy': 0.9, 'sadness': 0.1})",
    )
    
    dominant_emotion: Optional[str] = Field(
        default=None,
        description="Highest-confidence emotion label",
    )

    sentiment: Optional[str] = Field(
        default=None,
        description="Optional sentiment label if provided directly by service",
    )

    positive: Optional[float] = Field(
        default=None,
        description="Positive sentiment score from flat CLARIN payload",
    )

    negative: Optional[float] = Field(
        default=None,
        description="Negative sentiment score from flat CLARIN payload",
    )

    neutral: Optional[float] = Field(
        default=None,
        description="Neutral sentiment score from flat CLARIN payload",
    )
    
    # Catch-all for additional fields
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields from CLARIN response",
    )
    
    class Config:
        extra = "allow"  # Allow unmapped fields



def normalize_clarin_response(raw_response: Dict[str, Any]) -> EmotionOutput:
    """
    Map CLARIN Emotagger raw response to standardized EmotionOutput.
    
    Expected CLARIN emotagger response structure:
    {
      "emotions": ["joy", "sadness", ...],
      "scores": {"joy": 0.92, "sadness": 0.08, "anger": 0.0, ...},
      "dominant_emotion": "joy"
    }
    
    Handles missing/malformed fields gracefully, falling back to neutral with explanation.
    
    Args:
        raw_response: Raw dict from CLARIN API
        
    Returns:
        EmotionOutput: Standardized result for model input
        
    Raises:
        ValueError: If raw_response is not a dict or is fundamentally unparseable
    """
    if not isinstance(raw_response, dict):
        raise ValueError(f"Expected dict response from CLARIN, got {type(raw_response)}")
    
    try:
        # Try to parse raw response into CLARIN model first (for validation)
        parsed = CLARINResponseRaw(**raw_response)
    except Exception as e:
        # If parsing fails, attempt manual extraction
        parsed = CLARINResponseRaw(extra=raw_response)
    
    # Extract dominant emotion and confidence
    label = "neutral"
    confidence = 0.0

    # Extract overall sentiment
    sentiment_label = "neutral"
    sentiment_score = 0.0

    flat_scores: Dict[str, float] = {}

    # Real CLARIN payload is often flat, e.g. {"joy": 0.99, "positive": 0.98, ...}
    for key, value in raw_response.items():
        if isinstance(value, (int, float)):
            flat_scores[key.lower()] = float(value)
    
    # Strategy 1 (preferred): Use explicit dominant_emotion + scores mapping
    if parsed.dominant_emotion and parsed.scores:
        label = parsed.dominant_emotion
        confidence = parsed.scores.get(label.lower(), 0.0)
    
    # Strategy 2: Find max confidence from scores if dominant_emotion not available
    elif parsed.scores and not parsed.dominant_emotion:
        if parsed.scores:
            label = max(parsed.scores, key=parsed.scores.get)
            confidence = parsed.scores[label]
    
    # Strategy 3: Use first emotion from emotions list if available
    elif parsed.emotions and not parsed.dominant_emotion:
        label = parsed.emotions[0]
        # Try to find score for this emotion
        if parsed.scores:
            confidence = parsed.scores.get(label.lower(), 0.5)
        else:
            confidence = 0.5  # Assume medium confidence if not specified

    # Strategy 4: Flat payload (actual CLARIN output in this project)
    if label == "neutral" and confidence == 0.0 and flat_scores:
        emotion_candidates = {k: v for k, v in flat_scores.items() if k in EMOTION_LABELS and k != "neutral"}
        if emotion_candidates:
            label = max(emotion_candidates, key=emotion_candidates.get)
            confidence = emotion_candidates[label]
        elif "neutral" in flat_scores:
            label = "neutral"
            confidence = flat_scores["neutral"]
    
    # Normalize label to lowercase and validate against allowed emotions
    label = label.lower() if label else "neutral"
    allowed_emotions = EMOTION_LABELS
    
    if label not in allowed_emotions:
        # Unknown emotion label - normalize to neutral
        label = "neutral"
        confidence = 0.0
    
    # Clamp confidence to valid range [0.0, 1.0]
    confidence = max(0.0, min(1.0, float(confidence)))

    # Sentiment from explicit positive/negative/neutral scores (preferred)
    pos = None
    neg = None
    neu = None
    if parsed.scores:
        pos = parsed.scores.get("positive")
        neg = parsed.scores.get("negative")
        neu = parsed.scores.get("neutral")
    if pos is None and "positive" in flat_scores:
        pos = flat_scores["positive"]
    if neg is None and "negative" in flat_scores:
        neg = flat_scores["negative"]
    if neu is None and "neutral" in flat_scores:
        neu = flat_scores["neutral"]

    if pos is not None and neg is not None:
        # Sentiment score in [-1, 1]
        sentiment_score = max(-1.0, min(1.0, float(pos) - float(neg)))
        if pos > neg:
            sentiment_label = "positive"
        elif neg > pos:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
    elif parsed.sentiment and parsed.sentiment.lower() in SENTIMENT_LABELS:
        sentiment_label = parsed.sentiment.lower()
        sentiment_score = 0.0
    else:
        # Polarity fallback based on dominant emotion
        positive_emotions = {"joy", "trust", "anticipation", "surprise"}
        negative_emotions = {"sadness", "anger", "fear", "disgust"}
        if label in positive_emotions:
            sentiment_label = "positive"
            sentiment_score = confidence
        elif label in negative_emotions:
            sentiment_label = "negative"
            sentiment_score = -confidence
        else:
            sentiment_label = "neutral"
            sentiment_score = 0.0
    
    return EmotionOutput(
        label=label,
        confidence=confidence,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        source="emotagger",
        status="success",
        raw_emotagger_response=raw_response,
    )


def create_fallback_emotion(reason: str = "emotagger_unavailable") -> EmotionOutput:
    """
    Create a fallback EmotionOutput when Emotagger is not available.
    
    Args:
        reason: Why Emotagger failed (for status field)
        
    Returns:
        EmotionOutput: Safe fallback with neutral emotion
    """
    from .emotagger_config import emotagger_settings
    
    return EmotionOutput(
        label=emotagger_settings.fallback_emotion_label,
        confidence=emotagger_settings.fallback_confidence,
        sentiment_label="neutral",
        sentiment_score=0.0,
        source="fallback",
        status=reason,
    )


def create_error_emotion(error_type: str, latency_ms: Optional[int] = None) -> EmotionOutput:
    """
    Create EmotionOutput for an error case (timeout, API error, etc.).
    
    Args:
        error_type: Type of error ('timeout', 'api_error', 'invalid_response', etc.)
        latency_ms: Optional latency before error occurred
        
    Returns:
        EmotionOutput: Error case with fallback emotion
    """
    from .emotagger_config import emotagger_settings
    
    return EmotionOutput(
        label=emotagger_settings.fallback_emotion_label,
        confidence=emotagger_settings.fallback_confidence,
        sentiment_label="neutral",
        sentiment_score=0.0,
        source="fallback",
        status=error_type,
        latency_ms=latency_ms,
    )
