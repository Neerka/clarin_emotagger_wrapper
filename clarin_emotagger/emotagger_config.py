"""
Configuration for CLARIN Emotagger wrapper.

Reads from:
1. Environment variables (prefix: CLARIN_EMOTAGGER_)
2. ~/.clarin/config.yml (if exists, for fallback auth)
3. Defaults

Supports enable/disable toggle, timeouts, retry policy, and input validation limits.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class EmotaggerSettings(BaseModel):
    """Settings for CLARIN Emotagger integration."""
    
    # Service configuration
    enabled: bool = Field(
        default=True,
        description="Enable/disable Emotagger pre-processing. If False, wrapper returns None metadata.",
    )
    
    base_url: str = Field(
        default="https://services.clarin-pl.eu/api/v1",
        description="CLARIN LPMN services base URL.",
    )
    
    # Authentication (config.yml path)
    config_file: Optional[str] = Field(
        default=None,
        description="Path to CLARIN config.yml. If None, tries ~/.clarin/config.yml",
    )
    
    # Timeout and retry
    timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Emotagger request timeout in seconds.",
    )
    
    max_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Max retries for transient errors (timeout, 5xx).",
    )
    
    retry_delay_seconds: float = Field(
        default=0.5,
        ge=0.1,
        le=10.0,
        description="Delay between retries in seconds.",
    )
    
    # Input validation
    max_text_length: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Max characters accepted for analysis. Longer texts are truncated.",
    )
    
    min_text_length: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Min characters required for analysis. Shorter texts return neutral emotion.",
    )
    
    # Fallback behavior
    fallback_emotion_label: str = Field(
        default="neutral",
        description="Default emotion label when Emotagger fails or is disabled.",
    )
    
    fallback_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Default confidence when Emotagger is unavailable.",
    )
    
    # Logging
    log_requests: bool = Field(
        default=False,
        description="Log Emotagger requests/responses (without sensitive data).",
    )
    
    class Config:
        """Pydantic config."""
        extra = "ignore"
    
    @classmethod
    def from_env(cls) -> "EmotaggerSettings":
        """Load settings from environment variables and create instance."""
        # Read env vars with prefix CLARIN_EMOTAGGER_
        env_dict = {}
        prefix = "CLARIN_EMOTAGGER_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert CLARIN_EMOTAGGER_TIMEOUT_SECONDS → timeout_seconds
                config_key = key[len(prefix):].lower()
                
                # Try to convert to appropriate type
                if config_key in ("enabled", "log_requests"):
                    env_dict[config_key] = value.lower() in ("true", "1", "yes")
                elif config_key in ("timeout_seconds", "max_retries", "max_text_length", "min_text_length"):
                    try:
                        env_dict[config_key] = int(value)
                    except ValueError:
                        pass
                elif config_key in ("retry_delay_seconds", "fallback_confidence"):
                    try:
                        env_dict[config_key] = float(value)
                    except ValueError:
                        pass
                else:
                    env_dict[config_key] = value
        
        return cls(**env_dict)
    
    @property
    def resolved_config_file(self) -> Optional[Path]:
        """Resolve config file path with fallback to ~/.clarin/config.yml."""
        if self.config_file:
            return Path(self.config_file)
        
        clarin_home = Path.home() / ".clarin" / "config.yml"
        if clarin_home.exists():
            return clarin_home
        
        return None


# Global settings instance
emotagger_settings = EmotaggerSettings.from_env()

