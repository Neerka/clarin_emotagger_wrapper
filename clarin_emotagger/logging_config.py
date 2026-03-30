"""
Logging and observability configuration for Emotagger integration.

Provides:
- Structured logging setup (JSON format for easy parsing)
- Metrics/counters for observability (success, fallback, error rates)
- Health check utilities
"""

import logging
import json
import sys
from typing import Dict, Any
from datetime import datetime
from pathlib import Path


# Metrics counters
class MetricsCollector:
    """Simple in-memory metrics for Emotagger operations."""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.fallback_requests = 0
        self.timeout_errors = 0
        self.auth_errors = 0
        self.api_errors = 0
        self.unexpected_errors = 0
        self.total_latency_ms = 0
        
    def record_success(self, latency_ms: int):
        """Record successful emotion analysis."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        
    def record_fallback(self, reason: str):
        """Record fallback (Emotagger unavailable)."""
        self.total_requests += 1
        self.fallback_requests += 1
        
    def record_timeout(self):
        """Record timeout error."""
        self.total_requests += 1
        self.timeout_errors += 1
        
    def record_auth_error(self):
        """Record authentication error."""
        self.total_requests += 1
        self.auth_errors += 1
        
    def record_api_error(self):
        """Record API error."""
        self.total_requests += 1
        self.api_errors += 1
        
    def record_unexpected_error(self):
        """Record unexpected error."""
        self.total_requests += 1
        self.unexpected_errors += 1
        
    def get_stats(self) -> Dict[str, Any]:
        """Return current metrics snapshot."""
        success_rate = (
            (self.successful_requests / self.total_requests * 100)
            if self.total_requests > 0
            else 0.0
        )
        
        avg_latency = (
            (self.total_latency_ms / self.successful_requests)
            if self.successful_requests > 0
            else 0.0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "fallback_requests": self.fallback_requests,
            "success_rate_percent": round(success_rate, 2),
            "error_breakdown": {
                "timeout": self.timeout_errors,
                "authentication": self.auth_errors,
                "api": self.api_errors,
                "unexpected": self.unexpected_errors,
            },
            "avg_latency_ms": round(avg_latency, 2),
            "total_latency_ms": self.total_latency_ms,
        }
    
    def reset(self):
        """Reset all counters."""
        self.__init__()


# Global metrics instance
metrics = MetricsCollector()


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON for easy parsing."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "components"):
            log_data["components"] = record.components
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(level: int = logging.INFO, json_format: bool = True):
    """
    Initialize logging for Emotagger module.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: If True, use JSON formatter; otherwise, use default format
    """
    # Get or create emotagger logger
    logger = logging.getLogger("emotagger")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Also configure lpmn_client_biz logger
    lpmn_logger = logging.getLogger("lpmn_client_biz")
    lpmn_logger.setLevel(logging.WARNING)  # Less verbose for external lib
    
    return logger


def get_emotagger_logger():
    """Get or create emotagger logger."""
    logger = logging.getLogger("emotagger")
    if not logger.handlers:
        setup_logging()
    return logger


def log_emotion_analysis(text: str, result: Dict[str, Any], status: str = "success"):
    """
    Log emotion analysis result (without logging full text for privacy).
    
    Args:
        text: Original text (only length is logged)
        result: Emotion analysis result
        status: Analysis status
    """
    logger = get_emotagger_logger()
    logger.info(
        f"Emotion analysis {status}: text_len={len(text)}, "
        f"emotion={result.get('label', 'unknown')}, "
        f"confidence={result.get('confidence', 0):.2f}, "
        f"latency_ms={result.get('latency_ms', 'N/A')}"
    )


def get_health_status() -> Dict[str, Any]:
    """
    Get health status of Emotagger integration.
    
    Returns:
        dict: Health status with metrics and dependencies
    """
    try:
        from lpmn_client_biz import Connection
        lpmn_available = True
        lpmn_error = None
    except ImportError as e:
        lpmn_available = False
        lpmn_error = str(e)
    
    try:
        from .emotagger_config import emotagger_settings
        config_available = True
        emotagger_enabled = emotagger_settings.enabled
        config_file_present = emotagger_settings.resolved_config_file is not None
    except Exception as e:
        config_available = False
        emotagger_enabled = False
        config_file_present = False
    
    return {
        "status": "healthy" if (lpmn_available and config_available) else "degraded",
        "dependencies": {
            "lpmn_client_biz": {
                "available": lpmn_available,
                "error": lpmn_error,
            },
            "emotagger_config": {
                "available": config_available,
                "enabled": emotagger_enabled,
                "config_file_present": config_file_present,
            },
        },
        "metrics": metrics.get_stats(),
    }
