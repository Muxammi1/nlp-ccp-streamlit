import os
import json
import re
from typing import Dict
from openai import OpenAI
from .utils import chunk_text_chars

# Initialize Groq client once
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
)

def classify_sentiment_with_groq(text: str, model="llama-3.1-8b-instant") -> Dict:
    """
    Ask the model to return strictly parseable JSON:
    {"label": "positive"/"neutral"/"negative", "score": 0.7}
    """
    # If text is very long, truncate or chunk
    if len(text) > 12000:
        chunks = chunk_text_chars(text, max_chars=8000)
        text = " ".join(chunks[:2])

    system = {
        "role": "system",
        "content": "You are a sentiment analysis assistant. Answer strictly in JSON with fields label and score."
    }
    user = {
        "role": "user",
        "content": (
            "Read the following text and return a JSON exactly in this format:\n"
            '{"label":"positive"|"neutral"|"negative","score":<float_between_-1_and_1>} \n\n'
            "Text:\n\n" + text
        )
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[system, user],
        max_tokens=150,
        temperature=0.0,
    )

    raw = resp.choices[0].message.content.strip()

    # Try to extract a JSON object
    json_str = raw
    m = re.search(r'(\{.*\})', raw, flags=re.DOTALL)
    if m:
        json_str = m.group(1)

    try:
        parsed = json.loads(json_str)
        return parsed
    except Exception:
        # Fallback heuristic if model doesn’t return valid JSON
        lower = raw.lower()
        if "positive" in lower:
            return {"label": "positive", "score": 0.7}
        if "negative" in lower:
            return {"label": "negative", "score": -0.7}
        return {"label": "neutral", "score": 0.0}

