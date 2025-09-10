# src/ingestion.py
import requests
from bs4 import BeautifulSoup
import pdfplumber
import tempfile
import os
from newspaper import Article

def extract_text_from_url(url: str, use_newspaper: bool = True, timeout: int = 10) -> str:
    """
    Try using newspaper3k first (if installed), otherwise fallback to BeautifulSoup.
    """
    if use_newspaper:
        try:
            art = Article(url)
            art.download()
            art.parse()
            text = art.text
            if text and len(text.strip()) > 50:
                return text
        except Exception:
            # fallback
            pass

    # fallback: fetch and parse paragraphs
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "ccp-bot/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # try common article tags
    article = soup.find("article")
    if article:
        txt = " ".join([p.get_text(strip=True) for p in article.find_all("p")])
        if len(txt) > 30:
            return txt

    # fallback: join all <p> tags
    paragraphs = soup.find_all("p")
    text = "\n\n".join([p.get_text(strip=True) for p in paragraphs])
    return text

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file path using pdfplumber.
    """
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            ptext = page.extract_text()
            if ptext:
                text_parts.append(ptext)
    return "\n\n".join(text_parts)

def save_uploaded_file(streamlit_file) -> str:
    """
    Save a Streamlit or Jupyter-uploaded file-like object to a temp file and return the path.
    streamlit_file = st.file_uploader(...) result
    """
    suffix = getattr(streamlit_file, "name", "uploaded")
    fd, path = tempfile.mkstemp(suffix="_" + suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(streamlit_file.getbuffer())
    return path
