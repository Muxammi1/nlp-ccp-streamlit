import os
from openai import OpenAI

# Initialize Groq client (OpenAI-compatible)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def _call_groq_chat(messages, model="llama-3.1-8b-instant", max_tokens=400, temperature=0.0):
    """
    Wrapper calling Groq chat completions with the new OpenAI client.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp

def summarize_chunk(text_chunk, model="llama-3.1-8b-instant"):
    system = {
        "role": "system",
        "content": "You are a helpful assistant that writes concise, factual summaries."
    }
    user = {
        "role": "user",
        "content": f"Summarize this text:\n\n{text_chunk}"
    }
    resp = _call_groq_chat([system, user], model=model, max_tokens=300)
    return resp.choices[0].message.content.strip()

def summarize_text(text, model="llama-3.1-8b-instant", max_chunk_chars=3000):
    """
    Splits text into chunks and summarizes each. 
    If multiple chunks exist, recursively summarizes the summaries.
    """
    chunks = [text[i:i + max_chunk_chars] for i in range(0, len(text), max_chunk_chars)]
    partial_summaries = [summarize_chunk(ch, model=model) for ch in chunks]

    if len(partial_summaries) == 1:
        return partial_summaries[0]
    else:
        # recursively summarize the summaries
        return summarize_text(" ".join(partial_summaries), model=model)

