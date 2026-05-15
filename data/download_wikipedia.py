"""
Download and extract Simple English Wikipedia as plain text.

Source: https://dumps.wikimedia.org/simplewiki/latest/
Output: data/wikipedia.txt  (~130 MB of clean article text)

Simple English Wikipedia is ideal for training small language models:
  - ~230,000 articles written in clear, simple English
  - ~130 MB of clean text after markup removal
  - No external dependencies — uses Python stdlib only

Usage:
    uv run python data/download_wikipedia.py
    uv run python data/download_wikipedia.py --output data/wiki.txt --max_articles 5000
"""

import urllib.request
import bz2
import xml.etree.ElementTree as ET
import re
import os
import sys
import argparse


DUMP_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-pages-articles.xml.bz2"
)

NAMESPACE = "http://www.mediawiki.org/xml/DTD/MediaWiki"


# ---------------------------------------------------------------------------
# Markup cleaner
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    # Skip redirect pages
    if text.strip().upper().startswith("#REDIRECT"):
        return ""

    # Remove templates  {{...}}  (handles nesting up to 2 levels)
    for _ in range(3):
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)

    # Remove [[File:...]] and [[Image:...]] blocks
    text = re.sub(r'\[\[(?:File|Image|Category|Media)[^\]]*\]\]',
                  '', text, flags=re.IGNORECASE)

    # [[link|display text]] → display text,  [[link]] → link
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)

    # [http://url display text] → display text,  bare URLs → removed
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)

    # Remove HTML tags and comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+/?>', '', text)

    # Remove bold/italic wiki markers
    text = re.sub(r"'{2,}", '', text)

    # Section headers: == Title == → Title
    text = re.sub(r'=+\s*(.*?)\s*=+', r'\1', text)

    # Drop table markup lines
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Table rows, wikitable syntax, list-continuation
        if line[0] in ('|', '!', '{', '}'):
            continue
        lines.append(line)

    return '\n'.join(lines).strip()


# ---------------------------------------------------------------------------
# Stream parser — reads the bz2 dump without loading it all into memory
# ---------------------------------------------------------------------------

def iter_articles(stream):
    """
    Yield (title, clean_text) for each Wikipedia article in the XML stream.
    Uses iterparse so only one element is in memory at a time.
    """
    inside_page = False
    title = text = ns = ""

    for event, elem in ET.iterparse(stream, events=("start", "end")):
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

        if event == "start" and tag == "page":
            inside_page = True
            title = text = ns = ""

        elif event == "end" and inside_page:
            if tag == "title":
                title = (elem.text or "").strip()
            elif tag == "ns":
                ns = (elem.text or "").strip()
            elif tag == "text":
                text = elem.text or ""
            elif tag == "page":
                inside_page = False
                # ns == "0" means main article namespace (skip talk, file, etc.)
                if ns == "0" and title and text:
                    cleaned = clean(text)
                    if len(cleaned) > 200:   # skip stubs
                        yield title, cleaned
                elem.clear()


# ---------------------------------------------------------------------------
# Download with progress
# ---------------------------------------------------------------------------

def download_and_extract(url: str, output_path: str,
                         max_articles: int | None = None) -> int:
    print(f"Downloading {url}")
    print("(Simple English Wikipedia ~250 MB — may take a few minutes)\n")

    written = 0
    article_count = 0
    last_mb = 0

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        decompressor = bz2.BZ2File(response)

        with open(output_path, 'w', encoding='utf-8') as out:
            for title, text in iter_articles(decompressor):
                out.write(text)
                out.write('\n\n')
                written += len(text)
                article_count += 1

                mb = written // (1024 * 1024)
                if mb > last_mb:
                    last_mb = mb
                    sys.stdout.write(
                        f"\r  {article_count:,} articles  |  {mb} MB written"
                    )
                    sys.stdout.flush()

                if max_articles and article_count >= max_articles:
                    break

    print(f"\n\nDone — {article_count:,} articles, "
          f"{written / 1e6:.1f} MB → {output_path}")
    return article_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output',       default='data/wikipedia.txt')
    p.add_argument('--max_articles', type=int, default=None,
                   help='Stop after N articles (default: all ~230k)')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if os.path.exists(args.output):
        print(f"[Warning] {args.output} already exists — overwriting.")

    download_and_extract(DUMP_URL, args.output, args.max_articles)

    print("\nTo train on this dataset:")
    print(f"  uv run python src/train.py --dataset {args.output} \\")
    print("      --model small-cpu --tokenizer tiktoken \\")
    print("      --steps 0 --min_loss 1.5")


if __name__ == '__main__':
    main()
