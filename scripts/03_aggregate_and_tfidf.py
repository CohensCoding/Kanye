"""
03_aggregate_and_tfidf.py — Era/year rollups + per-era distinctive vocabulary.

Reads:   data/kanye_track_features.csv
         data/kanye_cleaned.csv          (for raw verse text)
Writes:  data/kanye_era_features.csv             — 18 eras × 14 metric columns
         data/kanye_year_features.csv            — 20 years × 12 metric columns
         data/kanye_era_distinctive_words.csv    — top 12 TF-IDF words per era
         data/kanye_era_top_words.csv            — top 20 raw-frequency words per era

What it does:
  - Aggregates track-level features up to era-level and year-level
    (means of sentiment, volatility, vocabulary diversity, etc.)
  - Runs TF-IDF across eras to find DISTINCTIVE vocabulary per era
    (not just common — TF-IDF down-weights words that appear everywhere)
  - Computes raw word frequencies per era for word clouds / bar charts

Why TF-IDF matters here:
  Raw word frequency tells you what's common ("nigga", "yeah" everywhere).
  TF-IDF tells you what's DISTINCTIVE to each era. Example output:
    Graduation:    stronger, faster, harder  (the Daft Punk anthem)
    808s:          lockdown, amazing, heartless  (the album titles)
    Yeezus:        gwaan, piss, blocka  (confrontational/patois)
    Donda:         junya, watanabe, grid  (fashion + Off-the-Grid era)
    Bully era:     quran, allah, worldstar  (religion + media provocation)

Usage:
  pip install pandas scikit-learn
  python scripts/03_aggregate_and_tfidf.py
"""
from __future__ import annotations
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text


# Rap-specific filler/ad-libs to add to sklearn's default English stopwords
EXTRA_STOPWORDS = {
    'yeah', 'uh', 'oh', 'na', 'nah', 'ayy', 'huh', 'hey', 'aw', 'aye',
    'em', 'gon', 'gonna', 'wanna', 'lemme', 'gotta', 'tryna', 'finna',
    'nigga', 'niggas',
    'like', 'just', 'know', 'got', 'get', 'don', 'aint', 'ima',
    'cause', 'cuz', 'bout',
}


def strip_annotations(text) -> str:
    if pd.isna(text):
        return ''
    text = re.sub(r'\[[^\]]*\]', '', str(text))
    return text.strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features',  default='data/kanye_track_features.csv')
    ap.add_argument('--cleaned',   default='data/kanye_cleaned.csv')
    ap.add_argument('--out-dir',   default='data')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    tracks = pd.read_csv(args.features)
    cleaned = pd.read_csv(args.cleaned)
    df = tracks.merge(cleaned[['track_id', 'kanye_verses']], on='track_id')

    # ---------------- Era-level aggregation ----------------
    era_agg = df.groupby(['era_rank', 'era_clean']).agg(
        n_tracks=('track_id', 'count'),
        total_words=('word_count', 'sum'),
        total_lines=('n_lines', 'sum'),
        sent_mean=('sent_mean_compound', 'mean'),
        sent_volatility=('sent_std_compound', 'mean'),
        pct_pos_lines=('sent_pct_positive', 'mean'),
        pct_neg_lines=('sent_pct_negative', 'mean'),
        avg_ttr=('ttr', 'mean'),
        avg_mtld=('mtld', 'mean'),
        avg_word_len=('avg_word_len', 'mean'),
        avg_syllables=('avg_syllables', 'mean'),
        fk_grade=('flesch_kincaid_grade', 'mean'),
        pct_three_plus_syl=('pct_three_plus_syl', 'mean'),
    ).round(3).reset_index().sort_values('era_rank')
    era_agg.to_csv(out_dir / 'kanye_era_features.csv', index=False)
    print(f"Wrote {out_dir/'kanye_era_features.csv'}: {era_agg.shape}")

    # ---------------- Year-level aggregation ----------------
    year_agg = tracks.groupby('year').agg(
        n_tracks=('track_id', 'count'),
        total_words=('word_count', 'sum'),
        total_lines=('n_lines', 'sum'),
        sent_mean=('sent_mean_compound', 'mean'),
        sent_volatility=('sent_std_compound', 'mean'),
        pct_pos_lines=('sent_pct_positive', 'mean'),
        pct_neg_lines=('sent_pct_negative', 'mean'),
        avg_ttr=('ttr', 'mean'),
        avg_mtld=('mtld', 'mean'),
        avg_syllables=('avg_syllables', 'mean'),
        fk_grade=('flesch_kincaid_grade', 'mean'),
    ).round(3).reset_index().sort_values('year')
    year_agg.to_csv(out_dir / 'kanye_year_features.csv', index=False)
    print(f"Wrote {out_dir/'kanye_year_features.csv'}: {year_agg.shape}")

    # ---------------- TF-IDF distinctive words per era ----------------
    eras_sorted = era_agg.sort_values('era_rank')['era_clean'].tolist()
    era_corpus = {}
    for era, group in df.groupby('era_clean'):
        era_corpus[era] = ' '.join(strip_annotations(t) for t in group['kanye_verses'])

    docs = [era_corpus[e] for e in eras_sorted]
    stopwords = list(set(sk_text.ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS)

    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        token_pattern=r"[a-zA-Z']+",
        min_df=1, max_df=0.85,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(docs)
    features = vectorizer.get_feature_names_out()

    distinctive_rows, top_word_rows = [], []
    print(f"\nTop 8 TF-IDF distinctive words per era (preview):")
    for i, era in enumerate(eras_sorted):
        # TF-IDF: distinctive words
        row = tfidf[i].toarray()[0]
        top_idx = row.argsort()[::-1][:12]
        top_words = [features[j] for j in top_idx if row[j] > 0]
        distinctive_rows.append({
            'era_clean': era,
            'distinctive_words': '; '.join(top_words),
        })
        print(f"  {era:<45} {', '.join(top_words[:8])}")

        # Raw frequency
        words = tokenize(era_corpus[era])
        freq = Counter(w for w in words if w not in stopwords and len(w) > 2)
        top_word_rows.append({
            'era_clean': era,
            'top_words': '; '.join(f"{w}:{c}" for w, c in freq.most_common(20)),
        })

    pd.DataFrame(distinctive_rows).to_csv(out_dir / 'kanye_era_distinctive_words.csv', index=False)
    pd.DataFrame(top_word_rows).to_csv(out_dir / 'kanye_era_top_words.csv', index=False)
    print(f"\nWrote {out_dir/'kanye_era_distinctive_words.csv'}")
    print(f"Wrote {out_dir/'kanye_era_top_words.csv'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
