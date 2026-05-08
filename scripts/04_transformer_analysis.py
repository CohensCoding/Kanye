"""
Kanye Transformer Analysis — Step 2 of the NLP pipeline.

Run this on your local machine where huggingface.co is accessible.
It enriches the line-level dataset with:
  1. RoBERTa-Twitter sentiment (3-class, contextual)        — primary sentiment model
  2. DistilRoBERTa emotion classification (7-class)         — joy/anger/sadness/fear/disgust/surprise/neutral
  3. Sentence-transformers embeddings (384-dim, MiniLM)     — for similarity search and clustering
  4. spaCy NER (entities mentioned)                         — tracks who/what/where Kanye mentions over time
  5. BERTopic theme modeling                                — semantic topic clusters across the catalog

INPUTS  (place in same folder as this script):
  - kanye_line_features.csv      (12,873 lines, from sandbox step)
  - kanye_cleaned.csv            (287 tracks, from sandbox step)

OUTPUTS:
  - kanye_line_features_enriched.csv   (lines + all transformer features)
  - kanye_track_topics.csv             (tracks + assigned BERTopic theme)
  - kanye_topics_summary.csv           (topic-id -> representative words)
  - kanye_line_embeddings.npy          (raw 384-dim embeddings, aligned to line CSV row order)

USAGE:
  pip install pandas numpy torch transformers sentence-transformers spacy bertopic tqdm
  python -m spacy download en_core_web_sm
  python kanye_transformer_analysis.py

RUNTIME: ~25-40 min on CPU laptop, ~5 min on GPU.
DISK:    Models cache to ~/.cache/huggingface/  (~2 GB total).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL   = "j-hartmann/emotion-english-distilroberta-base"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL     = "en_core_web_sm"
BATCH_SIZE      = 32

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def log(msg):
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def device():
    import torch
    if torch.cuda.is_available(): return 0           # GPU 0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return -1                                         # CPU

# -------------------------------------------------------------------
# 1. RoBERTa-Twitter sentiment (contextual, handles slang)
# -------------------------------------------------------------------
def run_roberta_sentiment(texts, dev):
    from transformers import pipeline
    log(f"Loading {SENTIMENT_MODEL}")
    pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL,
                    device=dev, return_all_scores=True, truncation=True, max_length=256)
    out = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="RoBERTa sentiment"):
        batch = [t if isinstance(t, str) and t.strip() else " " for t in texts[i:i+BATCH_SIZE]]
        results = pipe(batch)
        for r in results:
            scores = {d['label'].lower(): round(d['score'], 4) for d in r}
            top = max(r, key=lambda d: d['score'])
            out.append({
                'roberta_label':    top['label'].lower(),
                'roberta_pos':      scores.get('positive', 0.0),
                'roberta_neu':      scores.get('neutral',  0.0),
                'roberta_neg':      scores.get('negative', 0.0),
            })
    return pd.DataFrame(out)

# -------------------------------------------------------------------
# 2. Emotion classification (7-class)
# -------------------------------------------------------------------
def run_emotion(texts, dev):
    from transformers import pipeline
    log(f"Loading {EMOTION_MODEL}")
    pipe = pipeline("text-classification", model=EMOTION_MODEL,
                    device=dev, return_all_scores=True, truncation=True, max_length=256)
    out = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Emotion classification"):
        batch = [t if isinstance(t, str) and t.strip() else " " for t in texts[i:i+BATCH_SIZE]]
        results = pipe(batch)
        for r in results:
            scores = {d['label'].lower(): round(d['score'], 4) for d in r}
            top = max(r, key=lambda d: d['score'])
            row = {'emotion_label': top['label'].lower()}
            for emo in ('joy','anger','sadness','fear','disgust','surprise','neutral'):
                row[f'emotion_{emo}'] = scores.get(emo, 0.0)
            out.append(row)
    return pd.DataFrame(out)

# -------------------------------------------------------------------
# 3. Sentence embeddings
# -------------------------------------------------------------------
def run_embeddings(texts):
    from sentence_transformers import SentenceTransformer
    log(f"Loading {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    cleaned = [t if isinstance(t, str) and t.strip() else " " for t in texts]
    emb = model.encode(cleaned, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    return emb

# -------------------------------------------------------------------
# 4. spaCy NER (people, places, organizations)
# -------------------------------------------------------------------
def run_ner(texts):
    import spacy
    log(f"Loading spaCy {SPACY_MODEL}")
    nlp = spacy.load(SPACY_MODEL, disable=["lemmatizer", "tagger"])
    out = []
    for doc in tqdm(nlp.pipe(texts, batch_size=64), total=len(texts), desc="NER"):
        ents = [(e.text, e.label_) for e in doc.ents
                if e.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"}]
        out.append({
            'entities':       json.dumps(ents, ensure_ascii=False) if ents else "[]",
            'entity_count':   len(ents),
            'persons_named':  json.dumps([t for t, l in ents if l == "PERSON"], ensure_ascii=False),
        })
    return pd.DataFrame(out)

# -------------------------------------------------------------------
# 5. BERTopic — thematic clusters across the catalog
# -------------------------------------------------------------------
def run_topics(line_df, embeddings, tracks_csv, out_dir):
    from bertopic import BERTopic
    log("Aggregating verses for BERTopic")
    # Aggregate lines back to track level for cleaner topics
    track_text = line_df.groupby('track_id')['line_text'].apply(lambda xs: ' '.join(xs)).reset_index()
    track_text.columns = ['track_id', 'all_text']
    
    log(f"Embedding {len(track_text)} tracks for topic modeling")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    track_emb = model.encode(track_text['all_text'].tolist(),
                              batch_size=32, show_progress_bar=True)
    
    log("Fitting BERTopic")
    topic_model = BERTopic(min_topic_size=5, calculate_probabilities=False, verbose=False)
    topics, _ = topic_model.fit_transform(track_text['all_text'].tolist(), track_emb)
    track_text['topic_id'] = topics
    
    # Topic summary
    topic_info = topic_model.get_topic_info()
    topic_summary = []
    for tid in sorted(set(topics)):
        words = topic_model.get_topic(tid)
        topic_summary.append({
            'topic_id': tid,
            'topic_size': sum(1 for t in topics if t == tid),
            'top_words': '; '.join(w for w, _ in (words or [])[:10]) if words else 'OUTLIER',
        })
    
    # Merge topic assignment back with track metadata
    tracks = pd.read_csv(tracks_csv)
    track_topics = tracks[['track_id', 'track_title', 'era_clean', 'era_rank',
                            'release_date_clean', 'year']].merge(
        track_text[['track_id', 'topic_id']], on='track_id', how='left')
    
    track_topics.to_csv(out_dir / "kanye_track_topics.csv", index=False)
    pd.DataFrame(topic_summary).to_csv(out_dir / "kanye_topics_summary.csv", index=False)
    log(f"Found {len(set(topics)) - (1 if -1 in topics else 0)} topics + outliers")

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-csv",   default="kanye_line_features.csv")
    parser.add_argument("--tracks-csv", default="kanye_cleaned.csv")
    parser.add_argument("--out-dir",    default=".")
    parser.add_argument("--skip-topics", action="store_true",
                        help="Skip BERTopic (saves ~5 min if you don't need it)")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    
    log(f"Loading {args.line_csv}")
    line_df = pd.read_csv(args.line_csv)
    log(f"  {len(line_df)} lines across {line_df['track_id'].nunique()} tracks")
    
    texts = line_df['line_text'].fillna('').astype(str).tolist()
    dev = device()
    log(f"Using device: {'GPU' if isinstance(dev, int) and dev >= 0 else dev}")
    
    # 1. RoBERTa sentiment
    roberta = run_roberta_sentiment(texts, dev)
    
    # 2. Emotion
    emotion = run_emotion(texts, dev)
    
    # 3. Embeddings
    embeddings = run_embeddings(texts)
    np.save(out_dir / "kanye_line_embeddings.npy", embeddings)
    log(f"Saved embeddings: shape {embeddings.shape}")
    
    # 4. NER
    ner = run_ner(texts)
    
    # Merge everything
    enriched = pd.concat([line_df.reset_index(drop=True),
                          roberta.reset_index(drop=True),
                          emotion.reset_index(drop=True),
                          ner.reset_index(drop=True)], axis=1)
    enriched.to_csv(out_dir / "kanye_line_features_enriched.csv", index=False)
    log(f"Saved enriched line features: {enriched.shape}")
    
    # 5. Topics
    if not args.skip_topics:
        run_topics(line_df, embeddings, args.tracks_csv, out_dir)
    
    log("DONE.")
    log("Send these files back to the chat:")
    log("  - kanye_line_features_enriched.csv  (most important)")
    log("  - kanye_track_topics.csv")
    log("  - kanye_topics_summary.csv")
    log("  - kanye_line_embeddings.npy        (large; optional, only if doing similarity work)")

if __name__ == "__main__":
    main()
