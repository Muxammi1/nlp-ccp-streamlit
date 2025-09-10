import os
from openai import OpenAI
from src.detection import detect_language
from src.summarization import summarize_text

# Initialize Groq/OpenAI-compatible client once
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
)

# ---------- Core Translation ----------

def translate(text: str, target_lang: str = "en", model: str = "llama-3.1-8b-instant") -> str:
    """
    Force-translate text into the target language.
    Always responds only in target_lang.
    """
    # Very explicit translation instruction
    prompt = (
        f"Translate the following text into {target_lang}. "
        f"Always respond only in {target_lang} with no explanation:\n\n{text}"
    )

    messages = [
        {"role": "system", "content": "You are a professional translator."},
        {"role": "user", "content": prompt}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# ---------- Auto-Detect + Translate ----------

def auto_translate_to_english(text: str) -> str:
    """
    Detects language first. If not English, translates to English.
    If detection fails, still attempts translation to English.
    """
    lang_info = detect_language(text)
    if (not lang_info["lang"]) or (lang_info["lang"] != "en"):
        # Always try to translate to English
        return translate(text, target_lang="en")
    return text

# ---------- Summarize + Translate ----------

def summarize_foreign_text(text: str) -> str:
    """
    Translates to English (if needed), then summarizes.
    """
    english = auto_translate_to_english(text)
    return summarize_text(english)

# ---------- Batch Translation ----------

def batch_translate(texts: list[str], target_lang="en") -> list[str]:
    """
    Translate a list of texts to target_lang.
    """
    return [translate(t, target_lang=target_lang) for t in texts]

# ---------- Example Usage ----------

if __name__ == "__main__":
    sample_foreign = "پاکستان ایک خوبصورت ملک ہے۔"  # Urdu
    print("Auto-translate:", auto_translate_to_english(sample_foreign))
    print("Summarized:", summarize_foreign_text(sample_foreign))

    batch = ["Bonjour, comment ça va?", "La inteligencia artificial cambiará el mundo."]
    print("Batch:", batch_translate(batch, target_lang="en"))

