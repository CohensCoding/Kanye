# NLP analysis pipeline

Four scripts that turn `kanye_verses_only.csv` (the scraper's output) into a fully enriched dataset ready for visualization and cross-referencing with life events.

## Pipeline order

```
kanye_verses_only.csv
        ↓
[01] clean_data.py          →  data/kanye_cleaned.csv
        ↓                      data/excluded_tracks.csv
[02] extract_features.py    →  data/kanye_track_features.csv
        ↓                      data/kanye_line_features.csv
[03] aggregate_and_tfidf.py →  data/kanye_era_features.csv
        ↓                      data/kanye_year_features.csv
        ↓                      data/kanye_era_distinctive_words.csv
        ↓                      data/kanye_era_top_words.csv
[04] transformer_analysis.py → data/kanye_line_features_enriched.csv
                               data/kanye_track_topics.csv
                               data/kanye_topics_summary.csv
                               data/kanye_line_embeddings.npy
```

Steps 1–3 are lightweight (no model downloads, runs in seconds).
Step 4 is heavyweight (downloads ~2GB of model weights, takes 25–40 min on CPU).

## Setup

Add to `requirements.txt`:

```
# existing scraper deps
requests
beautifulsoup4
lyricsgenius
python-dotenv

# NLP pipeline (steps 1–3)
pandas
numpy
scikit-learn
vaderSentiment

# Transformer pipeline (step 4 — heavier)
torch
transformers
sentence-transformers
spacy
bertopic
tqdm
```

Then:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # only needed for step 4
```

## Running

```bash
# From repo root
python scripts/01_clean_data.py
python scripts/02_extract_features.py
python scripts/03_aggregate_and_tfidf.py
python scripts/04_transformer_analysis.py
```

Each script writes to `data/` by default. Override with `--out-dir`.

## Reference outputs from prior sandbox run

If you want to validate that your local run matches expected results, here are the key numbers (from `02_extract_features.py`):

| Track             | sent_mean | %neg  | mtld   |
|-------------------|----------:|------:|-------:|
| Heartless         |    -0.159 | 0.290 |  31.77 |
| Runaway           |    -0.101 | 0.288 |  25.74 |
| Power             |    -0.109 | 0.343 |  49.23 |
| Ultralight Beam   |    +0.310 | 0.000 |  10.70 |
| Real Friends      |    +0.082 | 0.194 |  37.10 |  ← VADER misses irony; step 4 fixes
| Stronger          |    +0.067 | 0.106 |  26.11 |

Output shape after each step:
- `kanye_cleaned.csv`:        287 rows × 18 columns
- `kanye_track_features.csv`: 287 rows × 28 columns
- `kanye_line_features.csv`:  ~12,873 rows × 12 columns
- `kanye_era_features.csv`:   18 rows × 14 columns
- `kanye_year_features.csv`:  20 rows × 12 columns

## What each step adds

**Step 1 — Cleaning.** Standardizes era labels (folds stragglers like `2008 features` into `Post-808s feature run`), imputes missing release dates from canonical album dates, builds unique `track_id` per (title, era) — this matters for songs sharing titles like `Hurricane (2018, KSG)` vs `Hurricane (2021, Donda)`. Excludes 29 tracks where Kanye's verses extracted as 0 words (skits, choir-only pieces, feature extraction failures — these are documented in `excluded_tracks.csv`).

**Step 2 — VADER sentiment + vocabulary.** Tokenizes into lines (atomic unit) and verses. Per-line VADER scoring, aggregated to track-level (mean/median/std/min/max/%pos/%neg/%neutral). Vocabulary metrics: TTR, MTLD (length-robust diversity), avg syllables (Liang-style heuristic, no NLTK download), Flesch-Kincaid grade.

**Step 3 — Era/year rollups + TF-IDF.** Aggregates everything to era and year level. TF-IDF identifies distinctive vocabulary per era — words that are *unique* to that period, not just common. This is where you see things like `stronger / faster / harder` on Graduation, or `quran / allah / worldstar` on the Bully era.

**Step 4 — Transformer features.** Layers RoBERTa-Twitter sentiment (3-class, contextual — handles slang and sarcasm that VADER misses), DistilRoBERTa emotion classification (joy / anger / sadness / fear / disgust / surprise / neutral), sentence embeddings (384-dim), spaCy NER (every person/place/org named, tagged), and BERTopic theme clustering. Outputs are line-level so they merge cleanly with step 2's `kanye_line_features.csv`.

## Known limitations

- VADER is lexicon-based and context-blind. It misreads irony — e.g., Real Friends scores positive even though the song is bitter. Step 4's RoBERTa fixes this but the disagreement between the two models is itself useful signal (irony detection by triangulation).
- Date imputation: 278 of 316 tracks had missing or year-only release dates and were imputed to album release dates. Year-level analysis is solid; finer-grained monthly analysis would require more research.
- Hand-validated tracks: Heartless, Runaway, Ultralight Beam, Power, Real Friends, Stronger. Other tracks rely on model performance without spot-check validation.
