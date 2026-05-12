# Phase 4b log

Generated: 2026-05-12T16:42:07Z

## Methodology pivot (Spotify removed)

Phase 4b **no longer calls the Spotify Web API**. Earlier attempts hit recurring operational failures: **HTTP 400** on quoted field-style queries; **HTTP 400** on unquoted free-text queries containing symbols Lucene treats as operators; **HTTP 401** from cached vs. rotated client credentials; **HTTP 429** throttling that persisted across backoff attempts; and **HTTP 401** under heavy-throttle conditions. The pipeline is now **Spotify-free**: Kworb cumulative streams (title fuzzy match), Wikipedia singles discography (Billboard + RIAA), and Wikipedia song-page **Length** for durations.

## Final human corrections (Phase 4b lock)

- **King (Bully vs Vultures):** Wikipedia singles table lists a **Bully** «King» row under release year **2026** with Hot 100 **peak 40**, separate from the ¥$ / Ty Dolla $ign **Vultures** «King» row (Hot 100 **94**). The Bully corpus row **keeps** Billboard; **Kworb** cumulative streams on that row were cleared because Kworb anchors the 35M figure to the Vultures-era opener only (one table entry per title).

- **Number One (College vs Late Registration):** Hot 100 lineage for Pharrell feat. Kanye (2006) is documented on `number_one_late_registration`; the College Dropout album cut `number_one_the_college_dropout` carries an explicit `parser_note` (no Hot 100).

- **Sanctified (Yeezus vs TLOP):** Chart data attaches to the 2014-era corpus row (`sanctified_the_life_of_pablo`); the Yeezus session row has Billboard cleared and a matching `parser_note`.

- **Manual Kworb (+2):** `buy_you_a_drank_remix_graduation` → Kworb «Buy U A Drank…» (T-Pain spelling); `i_don_t_like_remix_cruel_summer_yeezus_build_up` → Kworb «Don't Like.1» (duplicate suffix). Both use `kworb_match_score=100.00` after lyric review.

## Rates

- Kworb / Wikipedia: minimum **2 seconds** between network requests per registrable domain.
- HTTP cache directory: `/tmp/performance_cache`

## Kworb (title-only)

- Merged song rows (both artist pages): **527**
- RapidFuzz `token_set_ratio` threshold: **≥ 88** vs Kworb anchor title.
- Corpus tracks with non-empty `kworb_total_streams`: **334** / **397** (84.1%).
- Tracks below threshold this run (listed in `parser_note` as `no_kworb_match`): **64**

### Kworb gate override

Run used `--continue-after-kworb-review` with **64** tracks below 88 (more than the usual stop threshold of 20). Many are expected: mixtape / leak / alternate titles not present on Kworb Spotify tables.

## Billboard Hot 100

- Singles table rows carry a **release year** when Wikipedia lists a leading year cell (including rowspan blocks). Chart matching requires RapidFuzz ≥ 85 on the normalized title and, when both corpus `year` and table year exist, they must agree within **±2 years** (or **exact year** when the same `track_title` appears under multiple `era_clean` values in the features corpus). Rows are keyed by `wiki_slug` (`norm_slug`). Non–Kanye-West primaries must align with that slug (`partial_ratio` gate) so feature rows do not latch onto the wrong song.

- Rows with `charted_hot100=True`: **120**

## RIAA

- Rows with non-empty `riaa_certification`: **109**

## Wikipedia duration (infobox Length)

- Tracks with non-null `duration_seconds`: **103** / **397** (25.9%).
- **Words per minute** and **lines per minute** in `kanye_track_sonic_derived.csv` are only computed where `duration_seconds` is present (Wikipedia subset); other rows leave WPM/LPM empty.

## Sonic derived

- `kanye_verse_share` computable (collab + full lyrics): **252**
- Rows with non-empty `words_per_minute` (duration-backed): **103**
- Collab rows with null `kanye_verse_share` (backfill candidates): **0**

## Audit outputs

- `data/phase_4b_title_collisions.csv` — titles appearing under multiple `era_clean` values (6 titles).
- `data/phase_4b_kworb_misses.csv` — tracks below Kworb fuzzy 88 (62 rows), sorted by best score descending.

## Wikipedia ambiguity / duplicate peaks

- Multiple distinct Hot 100 peaks for chart row 'Father Stretch My Hands': [37, 54]
- Multiple distinct Hot 100 peaks for chart row 'Tell the Vision': [49, 90]
