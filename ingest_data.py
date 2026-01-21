"""
scripts/ingest_data.py

Ingest EU AI Act content (Articles, Recitals, Annexes), fetch full text,
chunk for embeddings, and save as separate JSON files per section type.
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from pathlib import Path
from time import sleep

# Config
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://artificialintelligenceact.eu"

# Adjust these numbers if the legislation is updated
MAX_ARTICLES = 113
MAX_RECITALS = 180
MAX_ANNEXES = 13

MIN_CHUNK_WORDS = 300
MAX_CHUNK_WORDS = 800

# Helpers
def clean_text(text: str) -> str:
    """Normalize whitespace and remove line breaks."""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def chunk_text(text: str, min_words=MIN_CHUNK_WORDS, max_words=MAX_CHUNK_WORDS):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end
    return chunks


def fetch_page_content(url: str, content_div_class: str) -> str:
    """Fetch a page and extract text from the specified div class."""
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    content_div = soup.find("div", class_=content_div_class)
    if not content_div:
        print(f"Warning: content div not found for {url}")
        return ""

    content_text = clean_text(content_div.get_text(separator=" ", strip=True))
    return content_text


# Parsing functions
def parse_article(article_number: int):
    url = f"{BASE_URL}/article/{article_number}/"
    title = f"Article {article_number}"
    content_text = fetch_page_content(
        url, content_div_class="et_pb_module et_pb_post_content et_pb_post_content_0_tb_body"
    )

    if not content_text:
        return []

    chunks = chunk_text(content_text)
    return [
        {"section_type": "Article", "section_title": title, "content": chunk, "url": url}
        for chunk in chunks
    ]


def parse_recital(recital_number: int):
    url = f"{BASE_URL}/recital/{recital_number}/"
    title = f"Recital {recital_number}"
    content_text = fetch_page_content(
        url, content_div_class="et_pb_module et_pb_post_content et_pb_post_content_0_tb_body"
    )

    if not content_text:
        return []

    chunks = chunk_text(content_text)
    return [
        {"section_type": "Recital", "section_title": title, "content": chunk, "url": url}
        for chunk in chunks
    ]


def parse_annex(annex_number: int):
    url = f"{BASE_URL}/annex/{annex_number}/"
    title = f"Annex {annex_number}"
    content_text = fetch_page_content(
        url, content_div_class="et_pb_module et_pb_post_content et_pb_post_content_0_tb_body"
    )

    if not content_text:
        return []

    chunks = chunk_text(content_text)
    return [
        {"section_type": "Annex", "section_title": title, "content": chunk, "url": url}
        for chunk in chunks
    ]


# Main ingestion
def main():
    sections = {
        "Article": [],
        "Recital": [],
        "Annex": [],
    }

    # Articles 
    print(f"Fetching {MAX_ARTICLES} articles ...")
    for i in range(1, MAX_ARTICLES + 1):
        try:
            chunks = parse_article(i)
            sections["Article"].extend(chunks)
            sleep(0.1)
        except Exception as e:
            print(f"Error fetching article {i}: {e}")

    # Recitals 
    print(f"Fetching {MAX_RECITALS} recitals ...")
    for i in range(1, MAX_RECITALS + 1):
        try:
            chunks = parse_recital(i)
            sections["Recital"].extend(chunks)
            sleep(0.1)
        except Exception as e:
            print(f"Error fetching recital {i}: {e}")

    # Annexes
    print(f"Fetching {MAX_ANNEXES} annexes ...")
    for i in range(1, MAX_ANNEXES + 1):
        try:
            chunks = parse_annex(i)
            sections["Annex"].extend(chunks)
            sleep(0.1)
        except Exception as e:
            print(f"Error fetching annex {i}: {e}")

    # Save JSON files per section type
    for section_type, chunks in sections.items():
        output_file = OUTPUT_DIR / f"{section_type.lower()}s.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(chunks)} {section_type} chunks to {output_file}")


if __name__ == "__main__":
    main()
