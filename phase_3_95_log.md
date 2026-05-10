# Phase 3.95 — Tier 2 singles backfill (May 2026)

## Definition of done (this pass)

- **Tier 2 coverage:** `data/audit_summary.txt` → **14/15** (`no AI` deferred).
- **`data/kanye_cleaned.csv`** updated; **`scripts/02_extract_features.py`** + **`scripts/04_transformer_analysis.py`** re-run → **`data/kanye_line_features_enriched.csv`** / **`data/kanye_track_features.csv`** regenerated with **unchanged column schemas** (27 enriched line columns).
- **`data/phase_3_95_unresolved.csv`** records deferred **`no AI`**.
- **Genius credentials:** `scripts/phase_395_step2_fetch.py` loads **`GENIUS_ACCESS_TOKEN` or `GENIUS_API_TOKEN`** from `.env` only (no hardcoded fallback). Errors print `token=***` wording only — never the secret.

## Canonical catalog file

`Kanye_West_Ye_Complete_Discography_and_Verse_Catalog__2003_to_May_2026.md` is absent from the repo; **`discography.md`** was used as substitute (see `data/metadata_disagreements.csv` row 1).

## Resolution A — Theraflu / Cold / Way Too Cold

- **Lyrics:** `https://genius.com/Kanye-west-and-dj-khaled-cold-lyrics` (Chris Brown Theraflu page rejected).
- **`kanye_cleaned.csv` row:** `Theraflu/Way Too Cold` with Resolution A `parser_note`.
- **Album cut `Cold`** (*Cruel Summer*) remains a separate catalog row.

## Ye vs. the People — integrated (2018)

- **Year:** **2018** per `discography.md` §2018 (user-confirmed).
- **`kanye_verses`:** All **T.I.** lines removed using **recording-voice / debate-role attribution**, ignoring Genius section labels.
- **Third-person “Ye”** clauses voiced by **T.I.** are excluded (e.g. closing exchange bars addressed at Kanye).
- **`parser_note`:** Documents structural debate + stripping policy (see `kanye_cleaned.csv`).
- **`extraction_method:** `debate_ti_stripped`.
- **Ambiguity surfaced (needs your OK if you disagree):** Immediately after Kanye’s “found the new me” pivot, the line  
  **`Not worried about some image that I gotta keep up`**  
  could plausibly be read as either voice without stems. **Integrated as Kanye** (continuation of Kanye’s self-narration). Reply if this should be dropped or moved to T.I.

## Billie Jean (2008 Kanye West Mix)

- **`track_type`:** **`intro_outro_only`** (matches vocabulary request; no existing enum matched better).
- **`parser_note`:** `106 words from Kanye-tagged intro/outro markers; remix appearance, not a verse contribution.`
- **Lyrics source:** `https://genius.com/Michael-jackson-billie-jean-2008-kanye-west-mix-lyrics`

## Freaky Girl Edit

- **`parser_note`** records Genius slug fallback:  
  `https://genius.com/Lil-pump-and-kanye-west-i-love-it-freaky-girl-edit-lyrics` (API hit scoring missed; direct slug verified).

## no AI — deferred

- Removed from **`data/phase_3_95_target_list.csv`** for Phase 3.95 execution.
- **`data/phase_3_95_unresolved.csv`** — status **`deferred — sourcing question pending`**.

## Scripts & artifacts

| Path | Role |
|------|------|
| `scripts/phase_395_step2_fetch.py` | Genius fetch → `data/phase_395_fetched_payloads.jsonl` |
| `scripts/phase_395_integrate_cleaned.py` | Merge payloads + Ye vs strip → `data/kanye_cleaned.csv` |
| `data/phase_395_step2_fetch_log.csv` | Fetch statuses |
| `/tmp/genius_cache/*.txt` | Raw lyric cache |

## Final disposition — original target list (10 rows before `no AI` removal)

| Track | Disposition |
|-------|-------------|
| Impossible | Merged from Phase 3.95 payload; NLP regenerated |
| I Love It | Merged from payload; NLP regenerated |
| Only One | Payload merged; Genius blurb prefix stripped at `"As I lay me down to sleep"` |
| All Day | Merged from payload; NLP regenerated |
| Freaky Girl Edit | **New** `kanye_cleaned` row; slug fallback `parser_note`; NLP regenerated |
| FourFiveSeconds | Merged from payload; NLP regenerated |
| Billie Jean (2008 Kanye West Mix) | **New** row; `intro_outro_only`; NLP regenerated |
| no AI | **Deferred** — see `data/phase_3_95_unresolved.csv` |
| Ye vs. the People | **`kanye_verses` rebuilt** with T.I. stripped + debate `parser_note` |
| Theraflu | Row retitled **`Theraflu/Way Too Cold`**; Cold Genius lyrics + Resolution A `parser_note` |

## Security note

If `.env` was ever echoed by tooling, **rotate** the Genius client secret at https://genius.com/api-clients .
