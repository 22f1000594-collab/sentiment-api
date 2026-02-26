# Comment Sentiment Analysis API

A FastAPI endpoint that uses OpenAI's structured outputs to analyze comment sentiment.

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Usage

**POST /comment**

```bash
curl -X POST http://localhost:8000/comment \
  -H "Content-Type: application/json" \
  -d '{"comment": "This product is amazing!"}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "rating": 5
}
```

## Deploy to Railway / Render / Fly.io

1. Push this folder to a GitHub repo
2. Connect to Railway/Render
3. Set `OPENAI_API_KEY` as an environment variable
4. Deploy — your public URL will be your endpoint

## Rating Guide

| Rating | Sentiment | Meaning |
|--------|-----------|---------|
| 5 | positive | Highly positive / enthusiastic |
| 4 | positive | Mildly positive |
| 3 | neutral | Neutral |
| 2 | negative | Mildly negative |
| 1 | negative | Highly negative |
