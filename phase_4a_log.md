# Phase 4a — Critical reception corpus (May 2026)

## Canonical album scope

**16 albums** per Phase 4a brief, aligned with `Kanye_West_Ye_Complete_Discography_and_Verse_Catalog__2003_to_May_2026.pdf` naming:

12 solo/collab studio LPs through **Donda**, plus **Donda 2**, **Vultures 1 — ¥$ with Ty Dolla $ign**, **Vultures 2 — ¥$**, and **Bully** (18-track streaming edition). Cruel Summer, deluxe-as-separate-title (**Donda Deluxe** folds into **Donda**), live albums, and mixtapes excluded per brief.

## Outputs

| File | Rows | Role |
|------|-----:|------|
| `data/kanye_critical_reception.csv` | **64** (16 × 4 sources) | Scores, metadata, ≤50-word excerpts |
| `data/kanye_review_features.csv` | **64** | VADER + RoBERTa + DistilRoBERTa emotion + lexical metrics |

Scripts:

- `scripts/phase_4a_build_reception.py`
- `scripts/phase_4a_review_nlp.py`

Caches (not committed): `/tmp/critical_reception_cache/` (HTTP replay); `/tmp/review_text_cache/` reserved — NLP reuses the HTTP cache because bodies are extracted client-side.

## Sources & normalization (implementation summary)

Documented in full at top of `scripts/phase_4a_build_reception.py`.

- **Metacritic**: Meta-score + critic counts embedded on Wikipedia album articles were reconciled with **canonical Metacritic album URLs** (`music/<album>/...`). Live Metacritic HTML could not be retrieved reliably here (**Cloudflare**); prose aggregates mirror Wikipedia → cited Metacritic links.

- **Pitchfork**: Original numeric ratings parsed from embedded `"rating": … ,"recircs"` payloads plus JSON-LD `Review`. Multiple JSON-LD `Review` objects occasionally appeared — selector prefers **longest `reviewBody`**.

- **Pitchfork retrospective**: No authenticated Pitchfork **Sunday Review** URLs surfaced via scripted probes for Kanye albums; all retrospective rows use `score_raw = no_retrospective` with searched landing URL `https://pitchfork.com/reviews/albums/`.

- **Rolling Stone**: Mix of legacy `/music/albumreviews/…` and modern `/music/music-album-reviews/…` capsules; `"ratingValue"` JSON extracted where Condé publishes it; Wikipedia star captions (`title="X/5 stars"`) as fallback.

### Duplicate-review / ambiguity surfaced

- **Pitchfork — ¥$ *Vultures 1***: Only one numbered Pitchfork album capsule matched Wikipedia references (`Paul A. Thompson`, headline **“VULTURES 1”**). Its prose deliberately opens with a **Yeezus-era analogy** (confirmed unique longest Review JSON-LD on-page).

### Rolling Stone unscored substantive coverage

- **Donda 2**: Wikipedia routed critique via **`music/features`** (“Post-Kanye world”) piece (`…vibe-shift-1314514/`), captured as substantive **`score_raw = no_score`** per Phase 4a instructions (`score_normalized_100` null, parser_note explains capsule absence).

### Pitchfork pages embedding VH1 Storytellers (*808s* era)

Wikipedia lists `/reviews/albums/13821-vh1-storytellers/` alongside **808s**; slug needles constrained retrieval to **`808s-and-heartbreak`**.

## NLP parity notes (`data/kanye_review_features.csv`)

- Column prefixes **`vader_*`** align with `data/kanye_line_features_enriched.csv`; **`roberta_*`** / **`emotion_*`** align with that enrichment naming where analogous signals exist.
- Lexical block aligns with `data/kanye_track_features.csv` metric meanings (**word_count**, **unique_words**, **ttr**, **mtld**, **avg_word_len**, **avg_syllables**, **flesch_kincaid_grade**).
- Rows lacking prose (**Metacritic aggregates**, **`no_retrospective`**) retain zeros / NaNs appropriately — transformers skipped below minimal token thresholds (`≤40` non-whitespace characters).

### Execution note

This workstation defaulted to **`/usr/bin/python3` (3.9.x)** because Homebrew Python 3.14’s bundled pip/pyexpat is broken; both Phase 4a scripts run cleanly under that interpreter with `--user` site-packages.

## Coverage tally (this integration run)

| `score_raw` token | Count |
|-------------------|------:|
| `not_found` | **0** |
| `no_retrospective` | **16** (every `pitchfork_retrospective` row) |
| `no_score` | **2** Rolling Stone rows with full essays but missing Condé `ratingValue` JSON (**Donda 2** feature review; **Vultures 1** capsule — verify whether RS later assigned a star rating on-page). |

Per-source row counts: **16 each** for `metacritic`, `pitchfork_original`, `pitchfork_retrospective`, `rolling_stone` → **64** total cells.

## Definition-of-done checklist

- [x] `kanye_critical_reception.csv` (64 curated `(album, source)` rows).
- [x] `kanye_review_features.csv` joinable on **`album`, `source`** with NLP parity naming.
- [x] Rate-limit + caching honoured (`two-second per-domain spacing`).
- [x] Git commit recorded (**requires `git add -f`** — `/data/` is gitignored).
