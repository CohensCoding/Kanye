# Kanye West life events timeline

Companion data file for the Kanye lyrical analysis project. 39 dated events from October 2002 through May 2026, curated for cross-referencing against the lyrics dataset.

## File

- `kanye_life_events.csv` — 39 rows, 6 columns

## Schema

| column | type | notes |
|---|---|---|
| `date` | YYYY-MM-DD | every row populated; some month-only events use first of month |
| `event_label` | string | short scannable label, ~3–7 words |
| `event_type` | string | one of: `personal`, `controversy`, `business`, `political`, `biographical` |
| `severity` | int | 1 = notable, 2 = major life shift, 3 = catastrophic/foundational |
| `description` | string | one sentence on why it matters and the proposed mechanism for lyrical influence |
| `source` | string | citation(s) |

## Joining to lyrics data

Joins to `kanye_track_features.csv` and `kanye_line_features_enriched.csv` on date. The lyrics tables use `release_date_clean` (also YYYY-MM-DD), so a temporal join is direct — no key normalization needed.

Useful derived fields for analysis:
- `days_since_last_event` per track (any event, or filtered by type/severity)
- `events_in_Nd_window` for rolling counts (90/180/365/730 day windows are reasonable)
- `last_event_label` for grouping tracks by their most recent inflection

When events cluster (October 2022 had three severity-3 events in 22 days), a single `nearest_event` field will lose context. Use rolling-window counts for cascade analysis.

## Severity distribution

| severity | count |
|---|---|
| 3 (catastrophic) | 8 |
| 2 (major) | 26 |
| 1 (notable) | 5 |

The eight severity-3 events are the foundational ruptures: 2002 car crash, 2007 Donda's death, 2016 hospitalization, 2018 bipolar disclosure, October 2022 antisemitism cascade, 2022 Adidas collapse, 2025 Super Bowl swastika ad, 2025 'Heil Hitler' release.

## Known data caveats

- Two events use approximate dates because no public day-level source exists: `2002-10-01` (Roc-A-Fella signing as recording artist) and `2012-04-01` (start of Kim Kardashian relationship). Both are flagged in the description field.
- The Bianca Censori marriage (`2022-12-01`) is approximate; no public marriage license was filed and sources differ between December 2022 and January 2023.
- The October 2022 antisemitism period is intentionally consolidated into a single event (`2022-10-03`) covering the full cascade through the Mar-a-Lago dinner, rather than split into separate Drink Champs / Tucker Carlson / Alex Jones rows. Same logic applies to the Pete Davidson feud (one row, `2022-03-02`).
- There is a 16-month gap with no studio releases between *Donda 2* (Feb 2022) and *Vultures 1* (Feb 2024), which means the Oct 2022 cascade and Adidas collapse have no tracks in their immediate 365-day window. This is itself a finding, not a data gap.

## Phase 3 questions this file enables

- Sentiment / emotion trajectories before vs. after specific events
- Vocabulary shifts at severity-3 inflection points
- Entity-mention frequency over time (Donda, Kim, Bianca, Trump appear in the enriched line features)
- Per-era characterization tied to the events that bracket each era
