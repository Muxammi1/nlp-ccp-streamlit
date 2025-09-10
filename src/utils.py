# src/utils.py
import math

def chunk_text_chars(text: str, max_chars: int = 6000):
    """
    Simple char-based chunker: splits text into roughly equal chunks <= max_chars.
    Good enough as a first pass. Returns list of string chunks.
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + max_chars
        # try to split on last newline or space for nicer chunk boundaries
        if end < length:
            split_at = text.rfind("\n", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at <= start:
                split_at = end
        else:
            split_at = length
        chunks.append(text[start:split_at].strip())
        start = split_at
    return chunks
