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

Caches (not committed): `/tmp/critical_reception_cache/` (HTTP replay); NLP reuses that cache because bodies are extracted client-side.

## Metacritic sourcing methodology (Phase 5.5 lift)

Metacritic scores for albums where direct fetching was blocked by Cloudflare were sourced from Wikipedia's citations of those Metacritic pages. Wikipedia editors source Metascore values directly from Metacritic; the citation chain (Wikipedia source → Metacritic URL → score) is preserved in the `review_url` column. Manual transcription was deliberately avoided to maintain auditability.

### Wikipedia relay vs direct Metacritic fetch

| Mode | Albums (all 16) |
|------|-----------------|
| **Wikipedia relay** (aggregate prose + cited `review_url` → `metacritic.com/music/...`) | The College Dropout; Late Registration; Graduation; 808s & Heartbreak; My Beautiful Dark Twisted Fantasy; Watch the Throne with Jay-Z; Yeezus; The Life of Pablo; ye; Kids See Ghosts with Kid Cudi; Jesus Is King; Donda; Donda 2; Vultures 1 — ¥$ with Ty Dolla $ign; Vultures 2 — ¥$; Bully |
| **Direct Metacritic HTML scrape** | *None* — live `metacritic.com` responses were Cloudflare-blocked in this automation environment; no parallel unaudited manual score entry was introduced. |

## Sources & normalization (implementation summary)

Full rescaling rules remain documented at the top of `scripts/phase_4a_build_reception.py`.

- **Pitchfork**: Original numeric ratings from `"rating": … ,"recircs"` plus JSON-LD `Review`; when multiple `Review` blobs exist, the **longest `reviewBody`** is chosen.

- **Pitchfork retrospective**: Default placeholder remains `score_raw = no_retrospective` with landing URL `https://pitchfork.com/reviews/albums/`. **808s & Heartbreak** received a **targeted retrospective audit** (see below); outcome unchanged (`no_retrospective`), with an explicit `parser_note` on that row.

- **Rolling Stone**: Legacy `/music/albumreviews/…` vs `/music/music-album-reviews/…`; Condé `"ratingValue"` JSON where published; Wikipedia star captions as fallback.

### Rolling Stone `no_score` (locked handling)

**Donda 2** and **Vultures 1** Rolling Stone rows stay **`no_score`** with full-text NLP features — substantive coverage without numeric grades is intentional schema design; no alternate capsules or manual star transcription.

### Pitchfork — ¥$ *Vultures 1* (single review; Yeezus lede)

One JSON-LD `Review` (headline **“VULTURES 1”**) confirms a **single** capsule — not a merge error; NLP runs on full text.

**Phase 6+ exhibit design:** When *Vultures 1* reception copy is shown visually, consider selecting a **different ≤50-word excerpt** from deeper in the review; the opening paragraphs foreground *Yeezus* as analogy. **Do not change** the Phase 4a CSV excerpt until exhibit design.

### Pitchfork pages embedding VH1 Storytellers (*808s* era)

Wikipedia lists `/reviews/albums/13821-vh1-storytellers/` alongside **808s**; slug needles constrained `pitchfork_original` retrieval to **`808s-and-heartbreak`**.

### *808s & Heartbreak* — targeted Pitchfork retrospective search (locked negative)

High-confidence audit (**May 10, 2026**) — no scored Pitchfork **Sunday Review / second reviews capsule** located:

1. **`pitchfork.com/reviews/albums/`** — indexed hits resolve to the original capsule **`12498-808s-and-heartbreak`** only (plus separate **VH1 Storytellers** live review, out of scope as retrospective rescoring of the studio LP).
2. **`pitchfork.com/features/overtones/9725-the-coldest-story-ever-told-the-influence-of-kanye-wests-808s-heartbreak/`** — major legacy essay (NewsArticle JSON-LD); **no Pitchfork album numeric score** embedded like reviews capsules.
3. **`pitchfork.com/tags/sunday-review/`** — tag hub fetched; **no 808s & Heartbreak** Sunday Review capsule surfaced in the retrieved listing.
4. **Web index / slug probes** — no additional `reviews/albums/` rescored URL discovered.

The `pitchfork_retrospective` row for **808s & Heartbreak** documents this in **`parser_note`** verbatim lead-in plus audit bullets.

## NLP parity notes (`data/kanye_review_features.csv`)

- Column prefixes **`vader_*`** align with `data/kanye_line_features_enriched.csv`; **`roberta_*`** / **`emotion_*`** align where analogous signals exist.
- Lexical block aligns with `data/kanye_track_features.csv` metric meanings (**word_count**, **unique_words**, **ttr**, **mtld**, **avg_word_len**, **avg_syllables**, **flesch_kincaid_grade**).
- Rows lacking prose (**Metacritic aggregates**, **`no_retrospective`**) retain zeros / NaNs — transformers skipped below minimal token thresholds (`≤40` non-whitespace characters).

## Phase 5 analysis planning — preserve `reviewer_name`

**Do not drop `reviewer_name`** when assembling the Phase 5 master dataset. Repeated bylines (e.g. Rob Sheffield across multiple Rolling Stone capsules) support reviewer-effect questions on scores and tone.

### Execution note

This workstation used **`/usr/bin/python3` (3.9.x)** with `--user` site-packages (Homebrew Python 3.14 pip/pyexpat unavailable).

## Coverage tally (locked)

| `score_raw` token | Count |
|-------------------|------:|
| `not_found` | **0** |
| `no_retrospective` | **16** (`pitchfork_retrospective`; **808s** carries extended targeted-search `parser_note`) |
| `no_score` | **2** Rolling Stone (**Donda 2**, **Vultures 1**) — canonical substantive unscored coverage |

Per-source row counts: **16 each** → **64** total `(album, source)` cells.

## Definition-of-done checklist

- [x] `kanye_critical_reception.csv` (64 curated rows).
- [x] `kanye_review_features.csv` joinable on **`album`, `source`**.
- [x] Rate-limit + caching honoured (`two-second per-domain spacing`).
- [x] Methodology + Metacritic relay table + 808s retrospective audit documented here.
- [x] Git commit (`data/` via `git add -f`).
