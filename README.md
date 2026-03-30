# clarin-emotagger-wrapper

Async Python wrapper for CLARIN Emotagger (`lpmn_client_biz`) with:
- robust response parsing (dict/bytes/string/file path),
- dominant emotion extraction,
- overall sentiment extraction,
- retry/timeout/fallback handling,
- stable output contract for chatbot pipelines.

## What You Get

For each input text, wrapper returns normalized metadata:
- `label` - dominant emotion,
- `confidence` - confidence for dominant emotion,
- `sentiment_label` - `positive | negative | neutral`,
- `sentiment_score` - score in range `[-1.0, 1.0]`,
- `status`, `source`, `latency_ms`, `raw_emotagger_response`.

## Requirements

- Python `>=3.13`
- CLARIN account
- credentials file at:
  - `~/.clarin/config.yml`

Expected credentials file format:

```yaml
USERNAME: your_escience_username
PASSWORD: your_escience_password
```

## Install as Git Dependency

### pip

```bash
pip install "git+https://github.com/your-org/clarin-emotagger-wrapper.git"
```

Install specific tag/commit:

```bash
pip install "git+https://github.com/your-org/clarin-emotagger-wrapper.git@v0.1.0"
```

### uv

```bash
uv add "git+https://github.com/your-org/clarin-emotagger-wrapper.git"
```

Specific ref:

```bash
uv add "git+https://github.com/your-org/clarin-emotagger-wrapper.git@v0.1.0"
```

## Quick Start

### Async (recommended)

```python
from clarin_emotagger import analyze_sentiment_async

result = await analyze_sentiment_async("Jestem bardzo szczesliwy!")
print(result)
```

### Sync helper

```python
from clarin_emotagger import analyze_sentiment

result = analyze_sentiment("To mnie frustruje")
print(result)
```

## Output Example

```json
{
  "label": "joy",
  "confidence": 0.9999,
  "sentiment_label": "positive",
  "sentiment_score": 0.998,
  "source": "emotagger",
  "status": "success",
  "latency_ms": 3811,
  "raw_emotagger_response": {
    "text": "Jestem bardzo szczesliwy!",
    "joy": 0.9999,
    "positive": 0.9997,
    "negative": 0.00002,
    "neutral": 0.0015
  }
}
```

## Configuration

Environment variables (`CLARIN_EMOTAGGER_*`):

- `CLARIN_EMOTAGGER_ENABLED` (default: `true`)
- `CLARIN_EMOTAGGER_CONFIG_FILE` (default fallback: `~/.clarin/config.yml`)
- `CLARIN_EMOTAGGER_TIMEOUT_SECONDS` (default: `30`)
- `CLARIN_EMOTAGGER_MAX_RETRIES` (default: `1`)
- `CLARIN_EMOTAGGER_RETRY_DELAY_SECONDS` (default: `0.5`)
- `CLARIN_EMOTAGGER_MAX_TEXT_LENGTH` (default: `5000`)
- `CLARIN_EMOTAGGER_MIN_TEXT_LENGTH` (default: `1`)
- `CLARIN_EMOTAGGER_FALLBACK_EMOTION_LABEL` (default: `neutral`)
- `CLARIN_EMOTAGGER_FALLBACK_CONFIDENCE` (default: `0.0`)
- `CLARIN_EMOTAGGER_LOG_REQUESTS` (default: `false`)

## Chatbot Integration Pattern

Recommended flow:
1. receive user message,
2. call wrapper,
3. attach normalized emotion/sentiment metadata to model input,
4. generate response.

Minimal example:

```python
from clarin_emotagger import analyze_sentiment_async

async def enrich_input(user_text: str) -> dict:
    emotion = await analyze_sentiment_async(user_text)
    return {
        "text": user_text,
        "emotion": {
            "label": emotion.get("label", "neutral"),
            "confidence": emotion.get("confidence", 0.0),
            "sentiment_label": emotion.get("sentiment_label", "neutral"),
            "sentiment_score": emotion.get("sentiment_score", 0.0),
            "status": emotion.get("status", "fallback"),
        },
    }
```

## Local Development

```bash
uv sync
uv run main.py
```

## Build and Publish (optional)

Build wheel/sdist:

```bash
python -m build
```

Then publish to your internal index or keep using git dependency.

## Notes

- Wrapper is async-first.
- Fallback is non-blocking: chatbot flow should continue even when CLARIN is unavailable.
- Keep credentials out of repository.
