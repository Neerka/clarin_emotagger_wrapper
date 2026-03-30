"""
Tests for Emotagger wrapper and sentiment analysis pipeline.

Covers:
- Successful emotion analysis
- Timeout handling
- Authentication errors
- Invalid response handling
- Fallback behavior
"""

import pytest
import asyncio
from unittest.mock import patch

from clarin_emotagger.emotagger_wrapper import (
    analyze_sentiment_async,
    _validate_input,
    _is_transient_error,
    EmotaggerTimeout,
    EmotaggerAuthError,
    EmotaggerAPIError,
)
from clarin_emotagger.emotagger_models import (
    EmotionOutput,
    normalize_clarin_response,
    create_fallback_emotion,
    create_error_emotion,
)
from clarin_emotagger.emotagger_config import emotagger_settings


class TestInputValidation:
    """Test input validation."""
    
    def test_validate_input_normal_text(self):
        """Normal text passes validation."""
        result = _validate_input("Jestem szczęśliwy!")
        assert result == "Jestem szczęśliwy!"
    
    def test_validate_input_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped."""
        result = _validate_input("  hello world  ")
        assert result == "hello world"
    
    def test_validate_input_too_short(self):
        """Text shorter than min_text_length returns None."""
        result = _validate_input("")
        assert result is None
    
    def test_validate_input_too_long(self):
        """Text longer than max_text_length is truncated."""
        long_text = "a" * (emotagger_settings.max_text_length + 100)
        result = _validate_input(long_text)
        assert len(result) == emotagger_settings.max_text_length
    
    def test_validate_input_invalid_type(self):
        """Non-string input raises ValueError."""
        with pytest.raises(ValueError):
            _validate_input(123)


class TestResponseNormalization:
    """Test normalizing CLARIN responses to EmotionOutput."""
    
    def test_normalize_clarin_response_success(self):
        """Successfully normalize valid CLARIN response."""
        clarin_response = {
            "emotions": ["joy"],
            "scores": {"joy": 0.95, "sadness": 0.05},
            "dominant_emotion": "joy",
        }
        
        result = normalize_clarin_response(clarin_response)
        
        assert isinstance(result, EmotionOutput)
        assert result.label == "joy"
        assert result.confidence == 0.95
        assert result.source == "emotagger"
        assert result.status == "success"
        assert result.sentiment_label == "positive"
        assert result.sentiment_score > 0
    
    def test_normalize_clarin_response_missing_confidence(self):
        """Handle CLARIN response without confidence scores."""
        clarin_response = {
            "emotions": ["sadness", "fear"],
            "dominant_emotion": "sadness",
        }
        
        result = normalize_clarin_response(clarin_response)
        
        assert result.label == "sadness"
        assert result.confidence == 0.5  # Default if not specified
        assert result.sentiment_label == "negative"

    def test_normalize_clarin_response_flat_payload(self):
        """Handle real flat CLARIN payload with top-level emotion/sentiment scores."""
        clarin_response = {
            "text": "Jestem bardzo szczęśliwy!",
            "joy": 0.9999,
            "trust": 0.088,
            "anticipation": 0.0006,
            "surprise": 0.0001,
            "fear": 0.0002,
            "sadness": 0.00007,
            "disgust": 0.00016,
            "anger": 0.00007,
            "positive": 0.9997,
            "negative": 0.00002,
            "neutral": 0.0015,
        }

        result = normalize_clarin_response(clarin_response)

        assert result.label == "joy"
        assert result.confidence > 0.9
        assert result.sentiment_label == "positive"
        assert result.sentiment_score > 0.9
    
    def test_normalize_clarin_response_invalid_emotion_label(self):
        """Invalid emotion label normalizes to neutral."""
        clarin_response = {
            "dominant_emotion": "excitement",  # Not in allowed set
        }
        
        result = normalize_clarin_response(clarin_response)
        
        assert result.label == "neutral"
        assert result.confidence == 0.0
    
    def test_normalize_clarin_response_invalid_json(self):
        """Non-dict response raises ValueError."""
        with pytest.raises(ValueError):
            normalize_clarin_response("not a dict")


class TestFallbackEmotion:
    """Test fallback emotion creation."""
    
    def test_create_fallback_emotion(self):
        """Fallback emotion has correct defaults."""
        result = create_fallback_emotion(reason="emotagger_unavailable")
        
        assert result.label == emotagger_settings.fallback_emotion_label
        assert result.confidence == emotagger_settings.fallback_confidence
        assert result.source == "fallback"
        assert result.status == "emotagger_unavailable"
        assert result.sentiment_label == "neutral"
        assert result.sentiment_score == 0.0
    
    def test_create_error_emotion_timeout(self):
        """Error emotion for timeout case."""
        result = create_error_emotion("timeout", latency_ms=30000)
        
        assert result.source == "fallback"
        assert result.status == "timeout"
        assert result.latency_ms == 30000


class TestTransientErrorDetection:
    """Test transient error classification."""
    
    def test_is_transient_error_timeout(self):
        """Timeout is classified as transient."""
        assert _is_transient_error("request timed out")
    
    def test_is_transient_error_5xx(self):
        """5xx errors are classified as transient."""
        assert _is_transient_error("503 Service Unavailable")
        assert _is_transient_error("500 Internal Server Error")
    
    def test_is_transient_error_permanently_unavailable(self):
        """404/403 are not transient."""
        assert not _is_transient_error("404 Not Found")
        assert not _is_transient_error("403 Forbidden")


class TestAnalyzeSentimentAsync:
    """Test main async sentiment analysis function."""
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_success(self):
        """Successful emotion analysis returns normalized result."""
        clarin_response = {
            "dominant_emotion": "joy",
            "scores": {"joy": 0.9},
        }
        
        with patch("clarin_emotagger.emotagger_wrapper._call_lpmn_emotagger") as mock_call:
            mock_call.return_value = clarin_response
            
            result = await analyze_sentiment_async("Jestem szczęśliwy!")
            
            assert result is not None
            assert result["label"] == "joy"
            assert result["source"] == "emotagger"
            assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_timeout(self):
        """Timeout returns error emotion."""
        with patch("clarin_emotagger.emotagger_wrapper._analyze_with_retry") as mock_retry:
            mock_retry.side_effect = asyncio.TimeoutError()
            
            result = await analyze_sentiment_async("Test message")
            
            assert result is not None
            assert result["source"] == "fallback"
            assert result["status"] == "timeout"
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_invalid_input(self):
        """Invalid input returns error emotion."""
        result = await analyze_sentiment_async("")  # Too short
        
        assert result is not None
        assert result["source"] == "fallback"
        assert result["status"] == "invalid_input"
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_auth_error(self):
        """Authentication error returns error emotion."""
        with patch("clarin_emotagger.emotagger_wrapper._analyze_with_retry") as mock_retry:
            mock_retry.side_effect = EmotaggerAuthError("Invalid credentials")
            
            result = await analyze_sentiment_async("Test message")
            
            assert result is not None
            assert result["source"] == "fallback"
            assert "error" in result["status"]
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_malformed_response(self):
        """Malformed CLARIN response handled gracefully."""
        with patch("clarin_emotagger.emotagger_wrapper._call_lpmn_emotagger") as mock_call:
            mock_call.return_value = {"unknown_field": "value"}
            
            result = await analyze_sentiment_async("Test")
            
            # Should still return valid EmotionOutput (with defaults)
            assert result is not None
            assert "label" in result
            assert "source" in result


class TestConfig:
    """Test configuration loading."""
    
    def test_emotagger_settings_defaults(self):
        """Settings have sensible defaults."""
        assert emotagger_settings.enabled is True
        assert emotagger_settings.timeout_seconds == 30
        assert emotagger_settings.max_retries == 1
        assert emotagger_settings.fallback_emotion_label == "neutral"
    
    def test_emotagger_settings_from_env(self, monkeypatch):
        """Settings can be loaded from environment variables."""
        monkeypatch.setenv("CLARIN_EMOTAGGER_TIMEOUT_SECONDS", "60")
        monkeypatch.setenv("CLARIN_EMOTAGGER_MAX_RETRIES", "2")
        
        # Reload settings to pick up env vars
        from clarin_emotagger.emotagger_config import EmotaggerSettings
        settings = EmotaggerSettings.from_env()
        
        assert settings.timeout_seconds == 60
        assert settings.max_retries == 2


class TestEmotionOutput:
    """Test EmotionOutput schema."""
    
    def test_emotion_output_serialization(self):
        """EmotionOutput can be serialized to dict/JSON."""
        emotion = EmotionOutput(
            label="joy",
            confidence=0.95,
            sentiment_label="positive",
            sentiment_score=0.88,
            source="emotagger",
            latency_ms=150,
            status="success",
        )
        
        data = emotion.model_dump()
        
        assert data["label"] == "joy"
        assert data["confidence"] == 0.95
        assert data["sentiment_label"] == "positive"
        assert data["sentiment_score"] == 0.88
        assert data["source"] == "emotagger"
    
    def test_emotion_output_confidence_bounds(self):
        """Confidence is clamped to [0.0, 1.0]."""
        # Confidence > 1.0 should be rejected by Pydantic
        with pytest.raises(Exception):  # Pydantic ValidationError
            EmotionOutput(
                label="joy",
                confidence=1.5,  # Invalid
                source="emotagger",
            )


# Integration tests (require mock CLARIN service or E2E setup)
class TestIntegration:
    """Integration tests for full pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_preprocessing_pipeline(self):
        """Test full pre-processing pipeline from main."""
        from main import preprocess_user_input
        
        # Mock wrapper to avoid real CLARIN call
        with patch("clarin_emotagger.emotagger_wrapper.analyze_sentiment_async") as mock_analyze:
            mock_analyze.return_value = {
                "label": "joy",
                "confidence": 0.9,
                "source": "emotagger",
                "status": "success",
            }
            
            context = await preprocess_user_input("Jestem szczęśliwy!")
            enriched = context.get_enriched_prompt_context()
            
            assert enriched["text"] == "Jestem szczęśliwy!"
            assert enriched["emotions"]["label"] == "joy"
    
    @pytest.mark.asyncio
    async def test_preprocessing_pipeline_fallback(self):
        """Test pre-processing pipeline with Emotagger fallback."""
        from main import preprocess_user_input
        
        # Mock wrapper to return fallback
        with patch("clarin_emotagger.emotagger_wrapper.analyze_sentiment_async") as mock_analyze:
            mock_analyze.return_value = {
                "label": "neutral",
                "confidence": 0.0,
                "source": "fallback",
                "status": "timeout",
            }
            
            context = await preprocess_user_input("Any message")
            enriched = context.get_enriched_prompt_context()
            
            # Flow should not be interrupted
            assert enriched["text"] == "Any message"
            assert enriched["emotions"]["source"] == "fallback"
