"""
02_extract_features.py — Compute lexical features (no transformers needed).

Reads:   data/kanye_cleaned.csv
Writes:  data/kanye_track_features.csv  — 287 tracks × 28 columns
         data/kanye_line_features.csv   — ~12,873 lines × 12 columns

What it does:
  - Tokenizes each track's verses into lines (the atomic unit) and verses (paragraphs)
  - Per-line VADER sentiment: compound, pos, neu, neg
  - Aggregates line scores up to track-level (mean, median, std, min, max,
    %positive, %negative, %neutral). Aggregating from lines avoids the
    saturation problem you get when scoring whole songs as one blob.
  - Vocabulary metrics per track:
      * word_count, unique_words, type-token ratio (TTR)
      * MTLD — Measure of Textual Lexical Diversity (length-robust, unlike TTR)
      * avg_word_len, avg_syllables (using a Liang-style heuristic — no NLTK download needed)
      * Flesch-Kincaid grade (using line count as sentence proxy for rap)
      * pct_one_syllable, pct_three_plus_syllable

Sanity-check expectations (validate after running):
  - Heartless:  sent_mean_compound ≈ -0.16, pct_negative ≈ 0.29
  - Runaway:    sent_mean_compound ≈ -0.10, pct_negative ≈ 0.29
  - Ultralight Beam: sent_mean_compound ≈ +0.31, pct_negative = 0.0
  - Real Friends:    sent_mean_compound ≈ +0.08 (KNOWN ISSUE: VADER misses irony.
                     Resolved in 03_transformer_analysis.py via RoBERTa.)

Usage:
  pip install pandas vaderSentiment
  python scripts/02_extract_features.py
"""
from __future__ import annotations
import argparse
import re
import statistics
import sys
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# Syllable counter (Liang-style heuristic, no NLTK dependency)
# Within ~95% of CMU-dict accuracy on English.
# ============================================================
def count_syllables(word: str) -> int:
    word = re.sub(r'[^a-z]', '', word.lower().strip())
    if not word:
        return 0
    vowels = 'aeiouy'
    count, prev_was_vowel = 0, False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith('e') and count > 1:
        count -= 1
    if len(word) > 2 and word.endswith('le') and word[-3] not in vowels:
        count += 1
    return max(1, count)


def flesch_kincaid_grade(words: list[str], sentences: int) -> float:
    if not words or sentences == 0:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    return round(0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59, 2)


# ============================================================
# Cleaning + tokenization
# ============================================================
def strip_annotations(text) -> str:
    """Remove [Verse: Kanye West], [Chorus], (2018) etc."""
    if pd.isna(text):
        return ''
    text = re.sub(r'\[[^\]]*\]', '', str(text))
    text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
    return text.strip()


def split_lines(text: str) -> list[str]:
    return [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 2]


def split_verses(text: str) -> list[str]:
    """Verses = blocks separated by blank lines."""
    blocks = re.split(r'\n\s*\n', text.strip())
    return [b.strip() for b in blocks if b.strip()]


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


# ============================================================
# Lexical diversity — MTLD is robust to text length, TTR is not
# ============================================================
def mtld(words: list[str], threshold: float = 0.72) -> float:
    """McCarthy & Jarvis (2010). Bidirectional average."""
    if len(words) < 50:
        return 0.0

    def factor_count(seq):
        factors, ttr, types, tokens = 0.0, 1.0, set(), 0
        for w in seq:
            tokens += 1
            types.add(w)
            ttr = len(types) / tokens
            if ttr <= threshold:
                factors += 1
                types, tokens = set(), 0
                ttr = 1.0
        if tokens > 0 and threshold < 1:
            factors += (1 - ttr) / (1 - threshold)
        return len(seq) / factors if factors > 0 else 0

    return round((factor_count(words) + factor_count(words[::-1])) / 2, 2)


def vocab_metrics(words: list[str], n_lines: int) -> dict:
    if not words:
        return dict(word_count=0, unique_words=0, ttr=0.0, mtld=0.0,
                    avg_word_len=0.0, avg_syllables=0.0, flesch_kincaid_grade=0.0,
                    pct_one_syl=0.0, pct_three_plus_syl=0.0)
    syllables = [count_syllables(w) for w in words]
    return dict(
        word_count=len(words),
        unique_words=len(set(words)),
        ttr=round(len(set(words)) / len(words), 4),
        mtld=mtld(words),
        avg_word_len=round(sum(len(w) for w in words) / len(words), 2),
        avg_syllables=round(sum(syllables) / len(syllables), 3),
        flesch_kincaid_grade=flesch_kincaid_grade(words, n_lines),
        pct_one_syl=round(sum(1 for s in syllables if s == 1) / len(syllables), 3),
        pct_three_plus_syl=round(sum(1 for s in syllables if s >= 3) / len(syllables), 3),
    )


# ============================================================
# Sentiment aggregation — score lines, aggregate to track
# ============================================================
def sentiment_metrics(lines: list[str], analyzer: SentimentIntensityAnalyzer) -> dict | None:
    if not lines:
        return None
    scores = [analyzer.polarity_scores(l) for l in lines]
    compounds = [s['compound'] for s in scores]
    return dict(
        n_lines=len(lines),
        sent_mean_compound=round(statistics.mean(compounds), 4),
        sent_median_compound=round(statistics.median(compounds), 4),
        sent_std_compound=round(statistics.stdev(compounds), 4) if len(compounds) > 1 else 0.0,
        sent_min_compound=round(min(compounds), 4),
        sent_max_compound=round(max(compounds), 4),
        sent_pct_positive=round(sum(1 for c in compounds if c > 0.05) / len(compounds), 3),
        sent_pct_negative=round(sum(1 for c in compounds if c < -0.05) / len(compounds), 3),
        sent_pct_neutral=round(sum(1 for c in compounds if -0.05 <= c <= 0.05) / len(compounds), 3),
        sent_mean_pos=round(statistics.mean(s['pos'] for s in scores), 4),
        sent_mean_neg=round(statistics.mean(s['neg'] for s in scores), 4),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='data/kanye_cleaned.csv')
    ap.add_argument('--out-dir', default='data')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} tracks from {args.input}")

    analyzer = SentimentIntensityAnalyzer()
    track_rows, line_rows = [], []

    for _, row in df.iterrows():
        cleaned = strip_annotations(row['kanye_verses'])
        lines = split_lines(cleaned)
        verses = split_verses(cleaned)
        words = tokenize_words(cleaned)

        sm = sentiment_metrics(lines, analyzer) or {}
        vm = vocab_metrics(words, len(lines))

        track_rows.append(dict(
            track_id=row['track_id'], track_title=row['track_title'],
            era_clean=row['era_clean'], era_rank=row['era_rank'],
            release_date_clean=row['release_date_clean'], year=row['year'],
            track_type=row['track_type'], n_verses=len(verses),
            **sm, **vm,
        ))

        for li, line in enumerate(lines):
            s = analyzer.polarity_scores(line)
            line_rows.append(dict(
                track_id=row['track_id'], track_title=row['track_title'],
                era_clean=row['era_clean'], era_rank=row['era_rank'],
                year=row['year'], release_date_clean=row['release_date_clean'],
                line_index=li, line_text=line,
                vader_compound=s['compound'], vader_pos=s['pos'],
                vader_neu=s['neu'], vader_neg=s['neg'],
            ))

    track_df = pd.DataFrame(track_rows)
    line_df = pd.DataFrame(line_rows)
    track_df.to_csv(out_dir / 'kanye_track_features.csv', index=False)
    line_df.to_csv(out_dir / 'kanye_line_features.csv', index=False)

    print(f"\nWrote {out_dir/'kanye_track_features.csv'}: {track_df.shape}")
    print(f"Wrote {out_dir/'kanye_line_features.csv'}:  {line_df.shape}")
    print(f"\nQuick sanity check (compare to expected values in script header):")
    for title in ['Heartless', 'Runaway', 'Ultralight Beam', 'Real Friends']:
        m = track_df[track_df['track_title'] == title]
        if not m.empty:
            r = m.iloc[0]
            print(f"  {title:<20} mean={r['sent_mean_compound']:+.3f}  "
                  f"%neg={r['sent_pct_negative']:.2f}  mtld={r['mtld']:.1f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
