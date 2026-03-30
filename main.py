"""
Main entry point for DnD character creation assistant chatbot with CLARIN Emotagger pre-processing.

Flow:
  1. User message received
  2. Pre-processing: Emotagger sentiment analysis (async, graceful fallback)
  3. Enriched input (user message + emotion metadata) → sent to LLM model
  4. Model response returned to user

Pre-processing layer is decoupled and optional; failure in Emotagger does not interrupt chatbot.
"""

import asyncio
from typing import Optional, Any


class PreProcessingContext:
    """Context container for pre-processed user input before sending to LLM model."""
    
    def __init__(self, original_text: str, emotion_metadata: Optional[dict] = None):
        """
        Args:
            original_text: Raw user message
            emotion_metadata: Output from Emotagger wrapper (dict with label, confidence, source)
                             or None if Emotagger failed/disabled and fallback was applied.
        """
        self.original_text = original_text
        self.emotion_metadata = emotion_metadata or {}
        
    def get_enriched_prompt_context(self) -> dict:
        """
        Returns a dictionary suitable for injection into LLM prompt context.
        
        This is the contract between pre-processing layer and model layer.
        Format is stable: always has 'text' and 'emotions', even if Emotagger failed.
        """
        return {
            "text": self.original_text,
            "emotions": self.emotion_metadata,
        }


async def preprocess_user_input(user_message: str) -> PreProcessingContext:
    """
    Main pre-processing entry point: applies sentiment analysis via CLARIN Emotagger.
    
    This function is async-first and gracefully handles Emotagger failures:
    - On timeout/API error: logs warning and continues without emotion metadata
    - On config/auth error: logs error and continues without emotion metadata
    - Returns PreProcessingContext with or without emotion data, never raises
    
    Args:
        user_message: Raw text from user
        
    Returns:
        PreProcessingContext: Contains original text + optional emotion metadata
    """
    # Import wrapper and config here to allow them to be optional module at startup
    try:
        from clarin_emotagger import analyze_sentiment_async
        from clarin_emotagger.emotagger_config import emotagger_settings
        
        if not emotagger_settings.enabled:
            # Emotagger disabled in config: skip analysis, return clean context
            return PreProcessingContext(user_message, emotion_metadata=None)
        
        # Call Emotagger wrapper (async, with internal timeout/retry/fallback)
        emotion_result = await analyze_sentiment_async(user_message)
        return PreProcessingContext(user_message, emotion_metadata=emotion_result)
        
    except ImportError as e:
        # Emotagger wrapper not yet initialized or config missing
        print(f"[WARN] Emotagger pre-processing unavailable: {e}")
        return PreProcessingContext(user_message, emotion_metadata=None)
    except Exception as e:
        # Unlikely but defensive: any other error in wrapper should not break chatbot
        print(f"[ERROR] Pre-processing failed (continuing without Emotagger): {e}")
        return PreProcessingContext(user_message, emotion_metadata=None)


async def main():
    """Application startup and initialization."""
    print("[INFO] Initialized DnD character creation assistant with CLARIN Emotagger pre-processing.")
    
    # Example: test pre-processing on dummy input
    example_input = "Jestem bardzo szczęśliwy!"
    context = await preprocess_user_input(example_input)
    enriched = context.get_enriched_prompt_context()
    print(f"[DEBUG] Pre-processing example: {enriched}")


if __name__ == "__main__":
    asyncio.run(main())
