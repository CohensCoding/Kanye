#!/usr/bin/env python3
"""
Phase 4a — NLP features for professional reviews (joinable on album + source).

Reads:  data/kanye_critical_reception.csv
Writes: data/kanye_review_features.csv

Caches HTML reuse in /tmp/critical_reception_cache (same SHA scheme as phase_4a_build_reception.py).
Extracts full review text from cached/fetched Pitchfork & Rolling Stone pages.

Rows without usable review prose (Metacritic aggregates, no_retrospective, not_found, etc.)
are emitted with NaNs for model-derived columns but lexical counts zeroed.

Lexical + syllable math mirrors scripts/02_extract_features.py conventions.
Sentiment models mirror scripts/04_transformer_analysis.py defaults (CPU/MPS friendly).
"""
from __future__ import annotations

import hashlib
import html as html_lib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECEPTION_CSV = PROJECT_ROOT / "data" / "kanye_critical_reception.csv"
OUT_CSV = PROJECT_ROOT / "data" / "kanye_review_features.csv"
HTML_CACHE = Path("/tmp/critical_reception_cache")

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15 Phase4aNLP/1.0"
)
DOMAIN_LAST: dict[str, float] = defaultdict(float)

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
BATCH_SIZE = 8


def count_syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower().strip())
    if not word:
        return 0
    vowels = "aeiouy"
    count, prev_was_vowel = 0, False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    if len(word) > 2 and word.endswith("le") and word[-3] not in vowels:
        count += 1
    return max(1, count)


def flesch_kincaid_grade(words: list[str], sentences: int) -> float:
    if not words or sentences == 0:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    return round(0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59, 2)


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def cached_get(url: str) -> str:
    HTML_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    path = HTML_CACHE / key
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    dom = _domain(url)
    wait = 2.0 - (time.time() - DOMAIN_LAST[dom])
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    DOMAIN_LAST[dom] = time.time()
    r.raise_for_status()
    text = r.text
    path.write_text(text, encoding="utf-8")
    return text


def parse_pitchfork_body(html: str) -> str:
    best = ""
    for m in re.finditer(r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL):
        try:
            blob = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(blob, dict) and blob.get("@type") == "Review":
            body = blob.get("reviewBody") or ""
            cleaned = html_lib.unescape(re.sub(r"\s+", " ", body)).strip()
            if len(cleaned) > len(best):
                best = cleaned
    return best


def parse_rs_body(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    paras = article.find_all("p") if article else []
    chunks = []
    if paras:
        first = paras[0].get_text(" ", strip=True)
        if first.startswith("By ") or first.startswith("By\t"):
            paras = paras[1:]
        chunks.extend(p.get_text(" ", strip=True) for p in paras)
    return html_lib.unescape(re.sub(r"\s+", " ", "\n".join(chunks))).strip()


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def lexical_metrics(text: str) -> dict:
    words = tokenize_words(text)
    if not words:
        return {
            "word_count": 0,
            "unique_words": 0,
            "ttr": 0.0,
            "mtld": 0.0,
            "avg_word_len": 0.0,
            "avg_syllables": 0.0,
            "flesch_kincaid_grade": 0.0,
        }
    uniq = len(set(words))
    ttr = uniq / len(words)
    try:
        from lexicalrichness import LexicalRichness

        mtld = float(LexicalRichness(" ".join(words)).mtld(threshold=0.72))
        if math.isnan(mtld):
            mtld = 0.0
    except Exception:
        mtld = 0.0
    awl = round(sum(len(w) for w in words) / len(words), 4)
    asyl = round(sum(count_syllables(w) for w in words) / len(words), 4)
    sentences = max(1, len(re.split(r"[.!?]+\s+", text)))
    fk = flesch_kincaid_grade(words, sentences)
    return {
        "word_count": len(words),
        "unique_words": uniq,
        "ttr": round(ttr, 4),
        "mtld": round(mtld, 4),
        "avg_word_len": awl,
        "avg_syllables": asyl,
        "flesch_kincaid_grade": fk,
    }


def pick_device():
    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return -1


def run_batches(pipe, texts: list[str], label: str):
    out = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=label):
        batch = [t if isinstance(t, str) and t.strip() else " " for t in texts[i : i + BATCH_SIZE]]
        out.extend(pipe(batch))
    return out


def main() -> None:
    df = pd.read_csv(RECEPTION_CSV)
    analyzer = SentimentIntensityAnalyzer()
    texts: list[str] = []
    meta: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        album, src = row["album"], row["source"]
        url = str(row.get("review_url") or "")
        prose = ""
        if url and not url.startswith("https://en.wikipedia.org"):
            try:
                html = cached_get(url)
                if src.startswith("pitchfork"):
                    prose = parse_pitchfork_body(html)
                elif src == "rolling_stone":
                    prose = parse_rs_body(html)
            except Exception:
                prose = ""
        texts.append(prose)
        meta.append((album, src))

    # VADER + lexical for all rows
    v_comp, v_pos, v_neu, v_neg = [], [], [], []
    lex_rows = []
    for t in texts:
        if t.strip():
            vs = analyzer.polarity_scores(t)
            v_comp.append(round(vs["compound"], 6))
            v_pos.append(round(vs["pos"], 6))
            v_neu.append(round(vs["neu"], 6))
            v_neg.append(round(vs["neg"], 6))
            lex_rows.append(lexical_metrics(t))
        else:
            v_comp.append(np.nan)
            v_pos.append(np.nan)
            v_neu.append(np.nan)
            v_neg.append(np.nan)
            lex_rows.append(lexical_metrics(""))

    usable_idx = [i for i, t in enumerate(texts) if len(t.strip()) > 40]

    dev = pick_device()
    from transformers import pipeline

    sent_pipe = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        device=dev,
        return_all_scores=True,
        truncation=True,
        max_length=256,
    )
    emo_pipe = pipeline(
        "text-classification",
        model=EMOTION_MODEL,
        device=dev,
        return_all_scores=True,
        truncation=True,
        max_length=256,
    )

    sent_results = [np.nan] * len(texts)
    emo_results = [np.nan] * len(texts)

    if usable_idx:
        sub_texts = [texts[i] for i in usable_idx]
        s_out = run_batches(sent_pipe, sub_texts, "RoBERTa sentiment")
        e_out = run_batches(emo_pipe, sub_texts, "DistilRoBERTa emotion")
        for j, i in enumerate(usable_idx):
            sr = s_out[j]
            scores = {d["label"].lower(): round(float(d["score"]), 4) for d in sr}
            top = max(sr, key=lambda d: d["score"])
            sent_results[i] = json.dumps(
                {
                    "label": top["label"].lower(),
                    "pos": scores.get("positive", 0.0),
                    "neu": scores.get("neutral", 0.0),
                    "neg": scores.get("negative", 0.0),
                }
            )
            er = e_out[j]
            escores = {d["label"].lower(): round(float(d["score"]), 4) for d in er}
            etop = max(er, key=lambda d: d["score"])
            emo_results[i] = json.dumps({"label": etop["label"].lower(), "scores": escores})

    roberta_label = []
    roberta_pos, roberta_neu, roberta_neg = [], [], []
    emotion_label = []
    emotion_joy, emotion_anger, emotion_sadness = [], [], []
    emotion_fear, emotion_disgust, emotion_surprise, emotion_neutral = [], [], [], []

    for i in range(len(texts)):
        if isinstance(sent_results[i], str):
            sd = json.loads(sent_results[i])
            roberta_label.append(sd["label"])
            roberta_pos.append(sd.get("pos", np.nan))
            roberta_neu.append(sd.get("neu", np.nan))
            roberta_neg.append(sd.get("neg", np.nan))
        else:
            roberta_label.append(np.nan)
            roberta_pos.append(np.nan)
            roberta_neu.append(np.nan)
            roberta_neg.append(np.nan)
        if isinstance(emo_results[i], str):
            ed = json.loads(emo_results[i])
            emotion_label.append(ed["label"])
            sc = ed["scores"]
            emotion_joy.append(sc.get("joy", np.nan))
            emotion_anger.append(sc.get("anger", np.nan))
            emotion_sadness.append(sc.get("sadness", np.nan))
            emotion_fear.append(sc.get("fear", np.nan))
            emotion_disgust.append(sc.get("disgust", np.nan))
            emotion_surprise.append(sc.get("surprise", np.nan))
            emotion_neutral.append(sc.get("neutral", np.nan))
        else:
            emotion_label.append(np.nan)
            emotion_joy.append(np.nan)
            emotion_anger.append(np.nan)
            emotion_sadness.append(np.nan)
            emotion_fear.append(np.nan)
            emotion_disgust.append(np.nan)
            emotion_surprise.append(np.nan)
            emotion_neutral.append(np.nan)

    out = pd.DataFrame(
        {
            "album": df["album"],
            "source": df["source"],
            "vader_compound": v_comp,
            "vader_pos": v_pos,
            "vader_neu": v_neu,
            "vader_neg": v_neg,
            "roberta_label": roberta_label,
            "roberta_pos": roberta_pos,
            "roberta_neu": roberta_neu,
            "roberta_neg": roberta_neg,
            "emotion_label": emotion_label,
            "emotion_joy": emotion_joy,
            "emotion_anger": emotion_anger,
            "emotion_sadness": emotion_sadness,
            "emotion_fear": emotion_fear,
            "emotion_disgust": emotion_disgust,
            "emotion_surprise": emotion_surprise,
            "emotion_neutral": emotion_neutral,
            "word_count": [r["word_count"] for r in lex_rows],
            "unique_words": [r["unique_words"] for r in lex_rows],
            "ttr": [r["ttr"] for r in lex_rows],
            "mtld": [r["mtld"] for r in lex_rows],
            "avg_word_len": [r["avg_word_len"] for r in lex_rows],
            "avg_syllables": [r["avg_syllables"] for r in lex_rows],
            "flesch_kincaid_grade": [r["flesch_kincaid_grade"] for r in lex_rows],
        }
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} shape={out.shape}")


if __name__ == "__main__":
    main()
