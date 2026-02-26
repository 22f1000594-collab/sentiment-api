from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import json

app = FastAPI(title="Comment Sentiment Analysis API")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class CommentRequest(BaseModel):
    comment: str


class SentimentResponse(BaseModel):
    sentiment: str
    rating: int


# JSON Schema for OpenAI structured outputs
SENTIMENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "Overall sentiment of the comment"
                },
                "rating": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5],
                    "description": "Sentiment intensity: 5=highly positive, 1=highly negative"
                }
            },
            "required": ["sentiment", "rating"],
            "additionalProperties": False
        }
    }
}


@app.post("/comment", response_class=JSONResponse)
async def analyze_comment(request: CommentRequest):
    """Analyze the sentiment of a customer comment."""
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=422, detail="Comment cannot be empty")

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. Analyze the sentiment of customer comments. "
                        "Return 'positive', 'negative', or 'neutral' for sentiment, and a rating from 1-5 "
                        "where 5 is highly positive and 1 is highly negative. "
                        "Use rating 3 for neutral comments. "
                        "Use 4-5 for positive (5 for very enthusiastic), 1-2 for negative (1 for very harsh)."
                    )
                },
                {
                    "role": "user",
                    "content": f"Analyze this comment: {request.comment}"
                }
            ],
            response_format=SENTIMENT_SCHEMA,
            temperature=0
        )

        result = json.loads(response.choices[0].message.content)
        return JSONResponse(
            content=result,
            headers={"Content-Type": "application/json"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
