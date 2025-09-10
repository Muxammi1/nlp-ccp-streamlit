# src/detection.py
from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0  # consistent results

LANG_NAME_MAP = {
    # Top global news languages
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ru": "Russian",
    "de": "German",
    "hi": "Hindi",
    "ur": "Urdu",
    "fa": "Persian (Farsi)",
    "tr": "Turkish",
    "pt": "Portuguese",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "id": "Indonesian",
    "ms": "Malay",
    "bn": "Bengali",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "vi": "Vietnamese",
    "th": "Thai",
    "sw": "Swahili",

    # European news languages
    "pl": "Polish",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "nl": "Dutch",
    "el": "Greek",
    "he": "Hebrew",
    "cs": "Czech",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "hu": "Hungarian",
    "bg": "Bulgarian",
    "sr": "Serbian",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",

    # African news languages
    "am": "Amharic",
    "so": "Somali",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "zu": "Zulu",
    "xh": "Xhosa",
    "af": "Afrikaans",
    "rw": "Kinyarwanda",
    "mg": "Malagasy",

    # Other South Asian news languages
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ne": "Nepali",
    "si": "Sinhala",
    "dv": "Dhivehi (Maldivian)",

    # Central Asian & Middle Eastern
    "kk": "Kazakh",
    "uz": "Uzbek",
    "tk": "Turkmen",
    "ky": "Kyrgyz",
    "az": "Azerbaijani",
    "ps": "Pashto",
    "ku": "Kurdish",

    # Southeast Asian
    "my": "Burmese",
    "km": "Khmer",
    "lo": "Lao",
    "tl": "Tagalog (Filipino)",
}

def detect_language(text: str):
    """
    Returns dict: {'lang': 'en', 'name': 'English', 'score': 0.99}
    """
    if not text or len(text.strip()) < 20:
        return {"lang": None, "name": None, "score": 0.0}
    try:
        langs = detect_langs(text[:10000])  # sample long text to speed up
        if not langs:
            return {"lang": None, "name": None, "score": 0.0}
        top = langs[0]
        code = top.lang
        score = top.prob
        name = LANG_NAME_MAP.get(code, code)
        return {"lang": code, "name": name, "score": float(score)}
    except Exception:
        return {"lang": None, "name": None, "score": 0.0}
