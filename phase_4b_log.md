# Phase 4b log

## Corpus cleanup (2026-05-11)

Ground-truth **line-content review** on ghost candidates from the Phase 4b audit narrowed an initial **13** candidate exclusions (10 from the first exclusion draft plus subsequent review tracks such as *Feel Me*, *Love Yourself*, and collaborators surfaced in the same pass) down to **4** hard exclusions. Eight tracks were **re-flagged as Tier 3 features** (verse presence confirmed or obvious from Genius URL) instead of being removed.

### Final exclusions (4 tracks)

| track_id | track_title | Rationale |
|----------|--------------|-----------|
| `live_from_irving_plaza_pre_dropout_era` | Live from Irving Plaza | Live-performance principle (same as Late Orchestration / VH1 Storytellers): performance date ≠ composition date for era analysis. |
| `eternal_rest_jesus_is_king` | Eternal Rest | Wrong Genius fetch (Mistweaver metal band page); lyrics are not Kanye corpus material. |
| `pro_nails_graduation` | Pro Nails | Kid Sister lead voice in captured lines; no Kanye verse in the extracted text. |
| `skyscrapers_cruel_summer_yeezus_build_up` | Skyscrapers | Swizz Beatz primary; 8 lines could not be attributed with confidence to Kanye vs. other artists; excluded under “ambiguous lyrical material is costlier than missing lines” for verse-level work. |

Full URLs, timestamps, and verbatim `exclusion_reason` strings: **`data/phase_4b_exclusions.csv`**.

### Tier 3 feature re-flags (10 tracks)

`track_type` set to **`Feature`** and **`parser_note`** set in **`data/kanye_track_features.csv`** and **`data/kanye_cleaned.csv`**:

| track_id | Primary (parser_note) | Note class |
|----------|----------------------|------------|
| `feel_me_quiet_year` | Tyga | Line review |
| `love_yourself_quiet_year` | Mary J. Blige | Line review |
| `glow_quiet_year` | Drake | Line review |
| `dat_side_quiet_year` | CyHi | Line review |
| `top_this_late_registration` | Bump J | Line review |
| `buy_you_a_drank_remix_graduation` | T-Pain | Line review |
| `go_late_registration` | Common | URL / original `track_type` correction (not line-review-driven) |
| `because_of_you_remix_graduation` | Ne-Yo | URL / original classification correction |
| `go2damoon_donda` | Playboi Carti | URL / original classification correction |
| `rock_n_roll_donda_2` | Pusha T | URL / original classification correction |

### Row counts (4 deletions)

| File | Rows removed | Notes |
|------|-------------:|-------|
| `data/kanye_line_features_enriched.csv` | 95 | Irving 47 + Eternal Rest 32 + Pro Nails 8 + Skyscrapers 8 |
| `data/kanye_line_features.csv` | 95 | same |
| `data/kanye_track_features.csv` | 4 | **397** tracks remain |
| `data/kanye_track_topics.csv` | 4 | |
| `data/kanye_cleaned.csv` | 4 | |
| `data/kanye_track_performance.csv` | 4 | |
| `data/kanye_track_sonic_derived.csv` | 4 | |
| `data/phase_3_5_pronoun_features.csv` | 95 | line-aligned |
| `kanye_track_event_proximity.csv` | 4 | repo root |
| `kanye_verses_only.csv` | 4 | matched `genius_url` / `(track_title, album_or_era)` |

Regenerated: **`data/phase_4b_ghost_tracks.csv`**, **`data/phase_4b_ghost_tracks_enriched.csv`** (`scripts/phase_4b_ghost_tracks.py`, `scripts/phase_4b_ghost_tracks_enriched.py`).

Re-aggregated (no NLP recompute): **`data/kanye_era_features.csv`**, **`data/kanye_year_features.csv`**, **`data/kanye_era_top_words.csv`**, **`data/kanye_era_distinctive_words.csv`** via **`scripts/03_aggregate_and_tfidf.py`**.

Audit summary: **`data/audit_summary.txt`** from **`scripts/00c_deep_audit.py --md discography.md`** (dataset row count reflects **`kanye_verses_only.csv`**).

### Reversibility

Pre-cleanup state is available in **git history** (`git checkout` / revert this commit) if any exclusion or re-flag should be reversed.

### Next step

Re-run **`scripts/phase_4b_build_performance.py`** on the **397-track** `kanye_track_features.csv` corpus for Spotify / sonic metrics on the cleaned set.
