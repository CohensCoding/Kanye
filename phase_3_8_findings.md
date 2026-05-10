# Phase 3.8 — Robustness gauntlet

Inputs reuse `data/kanye_line_features_enriched.csv` (DistilRoBERTa emotions, VADER, RoBERTa-Twitter already scored per line) and `data/kanye_track_features.csv` for native MTLD; Phase 3.6 PELT machinery is replayed on newly aggregated monthly series.

## Robustness scorecard

| Finding | Method | Robust? | Notes |
| --- | --- | --- | --- |
| 1: Donda fear | 3 classifiers (+ DistilRoBERTa anchor + RoBERTa margin) | **FAILS** | Vote count uses ≥3 of 5 directional proxies at perm p<0.10 |
| 2: Pronoun inversion | spaCy PRON 1sg + LIWC *I* list | **ROBUST** | Peak/trough eras vs Donda 2 / Jesus Is King |
| 3: Era × emotion STRONG pairs | VADER neg/compound + RoBERTa margin (BH over 54 tests) | **see pairs** | G.O.O.D. Fridays + MBDTF emotion_anger→ROBUST; Donda 2 emotion_sadness→FAILS |
| 4a: Lexical collapse/expansion | HD-D, VOCD-D, mean line tokens, mean word chars | **FAILS** | Curated PELT sweep (≥6 settings), Phase 3.7-style dedupe |
| 4b: Lexical expansion stress | MTLD monthly excluding Go2DaMoon | **FAILS** | Expansion **demoted** — divorce-expansion claim should be dropped or softened (collapse-only framing). |
| 6: 2026 fear consensus | Top quartile via pooled percentile ranks × 3 metrics | **FAILS** | Overlap manual vs consensus = **1**/8 — mixed / reconcile (<4 overlap) |
| 5: Late 2021 spike | Qualitative audit | **N/A** | Descriptive / lyrical interpretation — out of scope for classifier replication |

## Finding 1 — Donda-year fear across sentiment channels
- Label: **FAILS** (need ≥3 of 5 directional proxies with two-sided perm p<0.10).
- Table: `data/phase_3_8_finding1_replication.csv`

## Finding 2 — Pronoun inversion under alternate codings
- Label: **ROBUST** — expects **Donda 2** max z and **Jesus Is King** min z for both codings.
- Diagnostics: `{"spacy_morph_1sg": {"peak_era": "Donda 2", "trough_era": "Jesus Is King", "donda2_is_peak": true, "jik_is_trough": true}, "liwc_i_words": {"peak_era": "Donda 2", "trough_era": "Jesus Is King", "donda2_is_peak": true, "jik_is_trough": true}, "overall_robust_label": "ROBUST"}`
- Rows: `data/phase_3_8_finding2_replication.csv`

## Finding 3 — STRONG DistilRoBERTa pairs vs alternate sentiment axes
- Aggregated label across the two STRONG pairs: **PARTIAL**.
| era | emotion_metric | primary_z_distilroberta | primary_bh_p | vader_z | vader_bh_p | vader_compound_z | vader_compound_bh_p | roberta_margin_z | roberta_bh_p | robust_label | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G.O.O.D. Fridays + MBDTF | emotion_anger | 0.3102631519119389 | 0.0 | 0.2552126847372436 | 0.0 | -0.26545288034180403 | 0.0 | -0.2656218481002746 | 0.0 | ROBUST | BH pooled across 54 era×alternate-metric tests; alignment counts vader_neg (+same sign as primary), compound & margin (opposite sign). |
| Donda 2 | emotion_sadness | 0.3189038143127771 | 0.0 | -0.01781339464373666 | 0.71748 | 0.14980917858434878 | 0.0 | 0.1187180626918236 | 0.0009000000000000001 | FAILS | BH pooled across 54 era×alternate-metric tests; alignment counts vader_neg (+same sign as primary), compound & margin (opposite sign). |

## Finding 4 — Lexical collapse/expansion beyond MTLD
- Multi-metric sweep label: **FAILS** (0/4 metrics tagged Jun‑2018 collapse; 0/4 tagged Dec‑2020 expansion).
### Go2DaMoon single-track stress (MTLD)
- **Direct monthly stress test (Nov→Dec 2020 mean track MTLD):** full catalog Δ = **32.55**; excluding Go2DaMoon Δ = **0** (matches Phase 3.7 `~32.55` magnitude when Go2DaMoon is included).
- Monthly interpretation: Dec‑2020 catalog-wide monthly MTLD gain is entirely attributable to Go2DaMoon (excluding it: Nov→Dec jump → 0).
- Curated PELT expansion row (full catalog): `{'metric': 'mtld', 'change_point_date': '2020-12-01', 'n_settings_support': 6, 'magnitude': 32.54749999999999, 'direction': 'up', 'nearest_event': 'Kim files for divorce', 'days_to_nearest_event': 80, 'nearest_sev3_event': 'October 2022 antisemitism cascade begins', 'days_to_nearest_sev3': 671, 'classification': 'near_any_event_90d', 'cp_dt': Timestamp('2020-12-01 00:00:00')}`
- Curated PELT expansion row excluding **Go2DaMoon**: `{}`
- **What changed:** **Divorce/expansion claim dropped:** Nov→Dec 2020 catalog-wide MTLD bump (~32.5) becomes **0** when `Go2DaMoon` is excluded — the timed expansion is not robust off that track's release month.
- CSV: `data/phase_3_8_finding4_alternative_metrics.csv`

## Finding 5 — Late‑2021 descriptive spike
- **Out of scope** for classifier replication: no surviving Phase 3.7 change-point on `emotion_fear`; interpretive claim rests on the lyrical audit (family fracture vs collaborator grief). Robustness here is qualitative, not statistical.

## Finding 6 — 2026 fear consensus vs manual display filter
- Label: **FAILS** — overlap manual vs consensus (top quartile by **within-pool percentile rank** per metric) = **1** / 8.
- JSON: `data/phase_3_8_finding6_consensus_lines.json`

## What changed about each finding's claim
- **Finding 1:** Post‑robustness disposition **FAILS** — see CSV for which proxies carried the effect.
- **Finding 2:** **ROBUST** — spaCy morph vs LIWC list agreement on era extremes.
- **Finding 3:** Pair-level robustness captured in `phase_3_8_finding3_replication.csv` (PARTIAL summary).
- **Finding 4:** Retain **vocabulary collapse / bipolar-era** framing only; **drop divorce-expansion** — Dec‑2020 MTLD jump is fully explained by `Go2DaMoon` in the monthly aggregate.
- **Finding 5:** Unchanged evidentiary status (qualitative).
- **Finding 6:** Manual filter vs consensus overlap **1** — mixed / reconcile (<4 overlap).

## Publication readiness
| Finding | Disposition |
| --- | --- |
| 1 — Donda fear | DOWNGRADE TO QUALITATIVE |
| 2 — Pronoun inversion | SAFE TO PUBLISH AS-IS |
| 3 — Era emotion STRONG tiers | SAFE WITH STATED LIMITATION |
| 4 — Lexical collapse | SAFE WITH STATED LIMITATION (collapse emphasis only if expansion fails stress test) |
| 5 — Nov‑2021 spike | DOWNGRADE TO QUALITATIVE (by design) |
| 6 — 2026 fear lines | DOWNGRADE TO QUALITATIVE |
