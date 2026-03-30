"""
Asynchronous wrapper for CLARIN Emotagger sentiment analysis service.

Provides a single async function `analyze_sentiment_async()` that encapsulates:
- Connection initialization and authentication via lpmn_client_biz
- Request execution with timeout and retry logic
- Response normalization to EmotionOutput schema
- Graceful error handling and fallback behavior

Failures do not raise exceptions; all errors are contained and returned as fallback EmotionOutput.
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .emotagger_config import emotagger_settings
from .emotagger_models import EmotionOutput, normalize_clarin_response, create_error_emotion
from .logging_config import get_emotagger_logger, metrics, log_emotion_analysis


logger = get_emotagger_logger()


class EmotaggerClientError(Exception):
    """Base exception for Emotagger client errors."""
    pass


class EmotaggerTimeout(EmotaggerClientError):
    """Emotagger request timed out."""
    pass


class EmotaggerAuthError(EmotaggerClientError):
    """Failed to authenticate with CLARIN services."""
    pass


class EmotaggerAPIError(EmotaggerClientError):
    """CLARIN API returned an error."""
    pass


async def analyze_sentiment_async(text: str) -> Optional[Dict[str, Any]]:
    """
    Main entry point: analyze text sentiment using CLARIN Emotagger.
    
    This function is:
    - ASYNC-FIRST: Uses asyncio.to_thread() to wrap sync lpmn_client_biz calls
    - FAILURE-TOLERANT: Never raises exceptions, returns EmotionOutput dict or None
    - TIMEOUT-AWARE: Enforces timeout on entire operation
    - RETRY-CAPABLE: Retries on transient errors (timeout, 5xx)
    
    Args:
        text: User message to analyze (will be validated/truncated per config)
        
    Returns:
        dict: EmotionOutput serialized fields (label, confidence, source, etc.)
              or None if analysis failed and fallback should be skipped.
              
    Note:
        This function never raises; all errors are logged and contained.
        Caller should treat None return as "use default neutral emotion".
    """
    
    # Validate and sanitize input
    try:
        text = _validate_input(text)
        if text is None:
            logger.warning("Input validation failed; returning neutral emotion")
            metrics.record_fallback("invalid_input")
            return create_error_emotion("invalid_input").model_dump()
    except Exception as e:
        logger.error(f"Unexpected error during input validation: {e}")
        metrics.record_unexpected_error()
        return create_error_emotion("validation_error").model_dump()
    
    # Run with timeout
    try:
        emotion_output = await asyncio.wait_for(
            _analyze_with_retry(text),
            timeout=float(emotagger_settings.timeout_seconds)
        )
        
        if emotion_output:
            metrics.record_success(emotion_output.latency_ms or 0)
            log_emotion_analysis(text, emotion_output.model_dump(), status="success")
        
        return emotion_output.model_dump() if emotion_output else None
        
    except asyncio.TimeoutError:
        logger.warning(f"Emotagger analysis timed out after {emotagger_settings.timeout_seconds}s")
        metrics.record_timeout()
        return create_error_emotion("timeout", latency_ms=emotagger_settings.timeout_seconds * 1000).model_dump()
        
    except Exception as e:
        logger.error(f"Unexpected error in sentiment analysis: {type(e).__name__}: {e}", exc_info=True)
        metrics.record_unexpected_error()
        return create_error_emotion("unexpected_error").model_dump()


def _validate_input(text: str) -> Optional[str]:
    """
    Validate and sanitize user input before sending to Emotagger.
    
    Args:
        text: Raw user input
        
    Returns:
        Cleaned text, or None if invalid
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected str, got {type(text)}")
    
    text = text.strip()
    
    # Check length bounds
    if len(text) < emotagger_settings.min_text_length:
        logger.debug(f"Input too short ({len(text)} chars); returning neutral")
        return None
    
    if len(text) > emotagger_settings.max_text_length:
        logger.info(f"Input truncated from {len(text)} to {emotagger_settings.max_text_length} chars")
        text = text[:emotagger_settings.max_text_length]
    
    return text


async def _analyze_with_retry(text: str) -> Optional[EmotionOutput]:
    """
    Execute sentiment analysis with retry logic for transient errors.
    
    Args:
        text: Validated user input
        
    Returns:
        EmotionOutput: Emotion analysis result
        
    Raises:
        EmotaggerTimeout: If all retries timed out
        EmotaggerAuthError: If auth failed (no retry)
        EmotaggerAPIError: If API error occurred after all retries
    """
    
    last_error = None
    
    for attempt in range(emotagger_settings.max_retries + 1):
        try:
            start_time = time.time()
            
            # Call CLARIN emotagger in thread pool (blocking I/O)
            result = await asyncio.to_thread(
                _call_lpmn_emotagger,
                text,
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            if emotagger_settings.log_requests:
                logger.info(f"[Attempt {attempt+1}] Emotagger success in {latency_ms}ms")
            
            # Normalize result to EmotionOutput
            emotion_output = normalize_clarin_response(result)
            emotion_output.latency_ms = latency_ms
            return emotion_output
            
        except EmotaggerTimeout as e:
            last_error = e
            if attempt < emotagger_settings.max_retries:
                logger.warning(f"[Attempt {attempt+1}] Timeout; retrying in {emotagger_settings.retry_delay_seconds}s...")
                await asyncio.sleep(emotagger_settings.retry_delay_seconds)
            else:
                logger.error(f"[Attempt {attempt+1}] Timeout after {emotagger_settings.max_retries} retries")
                metrics.record_timeout()
                
        except EmotaggerAuthError as e:
            logger.error(f"Authentication failed (not retrying): {e}")
            metrics.record_auth_error()
            raise  # Don't retry auth errors
            
        except EmotaggerAPIError as e:
            last_error = e
            # Retry on 5xx, but not 4xx (unless specified)
            if attempt < emotagger_settings.max_retries and _is_transient_error(str(e)):
                logger.warning(f"[Attempt {attempt+1}] API error; retrying in {emotagger_settings.retry_delay_seconds}s...")
                await asyncio.sleep(emotagger_settings.retry_delay_seconds)
            else:
                logger.error(f"[Attempt {attempt+1}] API error: {e}")
                metrics.record_api_error()
                
        except Exception as e:
            logger.error(f"[Attempt {attempt+1}] Unexpected error: {type(e).__name__}: {e}", exc_info=True)
            metrics.record_unexpected_error()
            last_error = e
    
    # All retries exhausted
    if isinstance(last_error, EmotaggerTimeout):
        raise last_error
    raise EmotaggerAPIError(f"Failed after {emotagger_settings.max_retries + 1} attempts: {last_error}")


def _call_lpmn_emotagger(text: str) -> Dict[str, Any]:
    """
    Synchronous call to CLARIN LPMN Emotagger service.
    
    Synchronous call to CLARIN LPMN Emotagger service via lpmn_client_biz.
    
    This function wraps lpmn_client_biz and should be called via asyncio.to_thread().
    
    Flow:
    1. Initialize Connection with ~/.clarin/config.yml authentication
    2. Create Task with emotagger pipeline
    3. Run analysis on text (IOType.TEXT = plain string input)
    4. Download result (returns bytes)
    5. Parse JSON response
    
    Args:
        text: Validated user input
        
    Returns:
        dict: CLARIN Emotagger response (raw, before normalization)
              Expected structure: {
                "emotions": ["joy", "sadness"],
                "scores": {"joy": 0.92, "sadness": 0.08, ...},
                "dominant_emotion": "joy"
              }
        
    Raises:
        EmotaggerAuthError: If credentials invalid or missing
        EmotaggerAPIError: If CLARIN API error or response parsing fails
        EmotaggerTimeout: If request timed out
    """
    
    try:
        from lpmn_client_biz import Connection, Task, IOType, download
    except ImportError as e:
        raise EmotaggerAPIError(
            f"lpmn_client_biz not installed. "
            f"Run: pip install --extra-index-url https://pypi.clarin-pl.eu/simple/ lpmn-client-biz"
        )
    
    # Step 1: Initialize connection with CLARIN credentials
    try:
        config_file = emotagger_settings.resolved_config_file
        
        if config_file:
            if emotagger_settings.log_requests:
                logger.debug(f"Connecting to CLARIN using config: {config_file}")
            conn = Connection(config_file=str(config_file))
        else:
            # Try to use default credentials location or environment
            if emotagger_settings.log_requests:
                logger.debug("Connecting to CLARIN using default credentials")
            conn = Connection()
            
    except Exception as e:
        error_msg = str(e).lower()
        if "credential" in error_msg or "auth" in error_msg or "username" in error_msg:
            raise EmotaggerAuthError(
                f"Failed to authenticate with CLARIN services. "
                f"Ensure ~/.clarin/config.yml exists with USERNAME and PASSWORD. Error: {e}"
            )
        raise EmotaggerAPIError(f"Connection initialization failed: {e}")
    
    # Step 2: Build LPMN pipeline for emotagger
    # Static pipeline keeps behavior predictable across environments.
    lpmn_pipeline = ["any2txt", "emotagger"]
    
    try:
        task = Task(lpmn_pipeline, connection=conn)
    except Exception as e:
        raise EmotaggerAPIError(f"Failed to create LPMN Task: {e}")
    
    # Step 3: Run emotion analysis on text
    # IOType.TEXT = plain string input (not file)
    try:
        if emotagger_settings.log_requests:
            logger.debug(f"Running emotagger pipeline on text ({len(text)} chars)")
        
        output_file_id = task.run(text, IOType.TEXT)
        
        if emotagger_settings.log_requests:
            logger.debug(f"Emotagger task completed, result file_id: {output_file_id}")
            
    except TimeoutError as e:
        raise EmotaggerTimeout(f"CLARIN emotagger request timed out: {e}")
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg:
            raise EmotaggerTimeout(f"CLARIN emotagger timed out: {e}")
        raise EmotaggerAPIError(f"CLARIN emotagger task.run() failed: {e}")
    
    # Step 4: Download result from server
    # Result is stored as file on CLARIN server; download to bytes
    try:
        result_bytes = download(conn, output_file_id, IOType.FILE)
        
        if emotagger_settings.log_requests:
            logger.debug(f"Downloaded emotagger result: {len(result_bytes)} bytes")
            
    except Exception as e:
        raise EmotaggerAPIError(f"Failed to download emotagger result: {e}")
    
    # Step 5: Parse response
    # lpmn_client_biz may return bytes, str, dict, or path to a local result file.
    try:
        raw_result = result_bytes

        result = _parse_clarin_response(raw_result)
        
        if emotagger_settings.log_requests:
            logger.debug(f"Emotagger response parsed: {result}")
        
        return result
        
    except json.JSONDecodeError as e:
        preview = str(result_bytes)[:200]
        logger.error(f"Failed to parse emotagger response as JSON. Raw preview: {preview}")
        raise EmotaggerAPIError(f"Invalid JSON response from emotagger: {e}")
    except UnicodeDecodeError as e:
        raise EmotaggerAPIError(f"Failed to decode emotagger response: {e}")
    except Exception as e:
        raise EmotaggerAPIError(f"Unexpected error parsing emotagger response: {e}")


def _is_transient_error(error_str: str) -> bool:
    """
    Determine if error is transient (safe to retry) vs. permanent.
    
    Args:
        error_str: Error message/description
        
    Returns:
        bool: True if error is transient (timeout, 5xx, etc.)
    """
    transient_keywords = ["timeout", "5xx", "503", "502", "504", "temporarily", "unavailable"]
    return any(kw in error_str.lower() for kw in transient_keywords)


def _parse_clarin_response(raw_result: Any) -> Dict[str, Any]:
    """Parse CLARIN output from dict, bytes, text, or path to local result file."""
    if isinstance(raw_result, dict):
        return raw_result

    if isinstance(raw_result, (bytes, bytearray)):
        return _parse_response_text(raw_result.decode("utf-8", errors="replace"))

    if isinstance(raw_result, str):
        # Some lpmn_client_biz flows return local file path from download().
        candidate_path = Path(raw_result)
        if candidate_path.exists() and candidate_path.is_file():
            try:
                file_text = candidate_path.read_text(encoding="utf-8", errors="replace")
                return _parse_response_text(file_text)
            finally:
                try:
                    candidate_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(f"Could not remove temporary CLARIN file {candidate_path}: {exc}")
        return _parse_response_text(raw_result)

    raise EmotaggerAPIError(f"Unsupported emotagger response type: {type(raw_result).__name__}")


def _parse_response_text(text: str) -> Dict[str, Any]:
    """Best-effort parser for JSON, JSONL, and plain-text emotion labels."""
    cleaned = text.strip()
    if not cleaned:
        raise EmotaggerAPIError("Empty response content from emotagger")

    # 1) JSON object/array
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    except json.JSONDecodeError:
        pass

    # 2) JSON Lines: pick first valid object line
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed_line = json.loads(line)
            if isinstance(parsed_line, dict):
                return parsed_line
        except json.JSONDecodeError:
            continue

    # 3) Plain text fallback: detect known emotion label
    lowered = cleaned.lower()
    known = ["joy", "sadness", "anger", "neutral", "fear", "disgust", "surprise"]
    for label in known:
        if label in lowered:
            return {
                "dominant_emotion": label,
                "scores": {label: 1.0},
                "emotions": [label],
            }

    raise EmotaggerAPIError(f"Unsupported emotagger response format. Preview: {cleaned[:200]}")
