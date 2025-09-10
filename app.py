import streamlit as st
from dotenv import load_dotenv
import os

# --- Load .env locally, but also Streamlit secrets in cloud ---
load_dotenv()

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "GROQ_API_BASE" in st.secrets:
    os.environ["GROQ_API_BASE"] = st.secrets["GROQ_API_BASE"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

import requests
import feedparser
import streamlit as st
from openai import OpenAI
from typing import List

# --- Basic config ---
st.set_page_config(
    page_title="NLP-CCP — News & Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Styling ---
st.markdown(
    """
    <style>
    :root{
        --accent:#1f7a8c;
        --card:#0f1720;
        --muted:#94a3b8;
    }
    /* News ticker */
    .news-ticker {
        white-space: nowrap;
        overflow: hidden;
        box-sizing: border-box;
    }
    .news-ticker span {
        display:inline-block;
        padding-left:100%;
        animation: ticker 60s linear infinite; /* slower ticker */
    }
    @keyframes ticker {
        0% { transform: translateX(0);}
        100% { transform: translateX(-100%);}
    }
    .news-ticker span:hover {
        animation-play-state: paused;
    }
    /* Cards */
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(2,6,23,0.6);
    }
    .muted { color: var(--muted); font-size: 0.9em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Groq/OpenAI client ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

if not GROQ_API_KEY:
    st.error("🚨 GROQ_API_KEY not found. Add it to .env and restart.")
    st.stop()

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_BASE)

# --- Import local modules ---
from src.ingestion import extract_text_from_url, extract_text_from_pdf, save_uploaded_file
from src.detection import detect_language
from src.summarization import summarize_text
from src.sentiment import classify_sentiment_with_groq
from src.translation import auto_translate_to_english, translate

# --- Fetch available models ---
@st.cache_data(ttl=300)
def fetch_available_models() -> List[str]:
    try:
        models_resp = client.models.list()
        models = [m.id for m in models_resp.data if getattr(m, "active", True)]
        return sorted(models, key=lambda s: ("instant" not in s, s))
    except Exception as e:
        st.warning(f"Could not fetch remote model list: {e}")
        return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "groq/compound-mini"]

# --- Fetch RSS headlines ---
RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://feeds.reuters.com/reuters/topNews",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
]

@st.cache_data(ttl=180)
def fetch_rss_headlines(limit=30):
    items = []
    count = 0
    for feed_url in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries[:8]:
                title = entry.get("title", "")
                link = entry.get("link", "")
                source = parsed.feed.get("title", "")
                items.append({"title": title, "link": link, "source": source})
                count += 1
                if count >= limit:
                    break
        except Exception:
            continue
        if count >= limit:
            break
    return items

# --- Sidebar controls ---
st.sidebar.header("Analysis Controls")
available_models = fetch_available_models()
model_choice = st.sidebar.selectbox("Select model", available_models, index=0)
target_lang = st.sidebar.selectbox("Target translation language", ["en", "fr", "es", "ar", "ur", "zh"], index=0)
max_summary_chars = st.sidebar.number_input("Max chunk chars (for summarization)", min_value=800, max_value=10000, value=3000, step=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Top live headlines")
headlines = fetch_rss_headlines(limit=30)
for i, h in enumerate(headlines[:6]):
    st.sidebar.markdown(f"**{h['source']}** — [{h['title']}]({h['link']})")
    if i < 5:
        st.sidebar.markdown("---")

# --- Top ticker ---
ticker_text = "  ∘  ".join([f"{h['source']}: {h['title']}" for h in headlines[:20]])
st.markdown(f'<div class="news-ticker"><span>{ticker_text}</span></div>', unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# --- Layout columns ---
left, right = st.columns([1, 2], gap="large")

with left:
    st.header("Input")
    url_input = st.text_input("Paste URL (optional)")
    uploaded_file = st.file_uploader("Or upload a PDF/TXT", type=["pdf", "txt"])
    run = st.button("🚀 Run Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("Quick Tips:")
    st.markdown("- Use `llama-3.1-8b-instant` for fast runs.")
    st.markdown("- Use `llama-3.3-70b-versatile` for higher-quality outputs.")

with right:
    st.header("Analysis Results")
    status_box = st.empty()
    results_area = st.empty()

# --- Analysis pipeline ---
def run_full_analysis(text: str, model: str, max_chars: int, target_lang: str):
    out = {}
    out["detection"] = detect_language(text)
    try:
        out["summary"] = summarize_text(text, model=model, max_chunk_chars=max_chars)
    except Exception as e:
        out["summary_error"] = str(e)
    try:
        out["sentiment"] = classify_sentiment_with_groq(text, model=model)
    except Exception as e:
        out["sentiment_error"] = str(e)
    try:
        if target_lang != "en":
            out["translation"] = translate(text, target_lang=target_lang, model=model)
        else:
            out["translation"] = auto_translate_to_english(text)
    except Exception as e:
        out["translation_error"] = str(e)
    return out

# --- When run ---
if run:
    status_box.info("📥 Extracting text...")
    text_content = ""

    # Ingestion
    try:
        if uploaded_file is not None:
            if uploaded_file.name.lower().endswith(".pdf"):
                tmp = save_uploaded_file(uploaded_file)
                text_content = extract_text_from_pdf(tmp)
            else:
                text_content = uploaded_file.read().decode("utf-8")
        elif url_input:
            text_content = extract_text_from_url(url_input)
        else:
            st.error("Please upload a file or paste a URL.")
            st.stop()
    except Exception as e:
        st.error(f"Error during ingestion: {e}")
        st.stop()

    if not text_content or len(text_content.strip()) < 20:
        st.error("No substantial text found.")
        st.stop()

    status_box.info("🔎 Running detection, summarization, sentiment, translation...")
    with st.spinner("Running model analysis ..."):
        results = run_full_analysis(text_content, model_choice, max_summary_chars, target_lang)

    # --- Present results ---
    with results_area.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Top row: detection + sentiment in two cards
        c1, c2 = st.columns(2)

        with c1:
            det = results.get("detection", {})
            st.markdown("### 🔎 Language Detection")
            st.markdown(
                f"<p class='muted'>Code: {det.get('lang')} — {det.get('name')} — score: {det.get('score'):.3f}</p>",
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown("### 😊 Sentiment")
            if "sentiment" in results:
                sent = results["sentiment"]
                st.json(sent)
            else:
                st.warning(f"Sentiment error: {results.get('sentiment_error')}")

        # Summary
        if "summary" in results:
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("### 📝 Summary")
            st.write(results["summary"])
        else:
            if "summary_error" in results:
                st.warning(f"Summary error: {results['summary_error']}")

        # Translation
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### 🌐 Translation")
        trans = results.get("translation")
        if trans:
            st.write(trans[:4000])
        else:
            if "translation_error" in results:
                st.warning(f"Translation error: {results['translation_error']}")
            else:
                st.info("No translation produced.")

        st.markdown("</div>", unsafe_allow_html=True)

    status_box.success("✅ Analysis complete.")
    with st.expander("Show full original text"):
        st.text_area("Full text", text_content[:100000], height=300)
