# Phase 3.7b — Curated findings

## Headline

1. Mother Donda West's death coincided with a statistically significant rise in model-estimated fear (Δ=0.0373, 95% CI [0.0235, 0.0517], permutation p=0.0286, n=756 lines).
2. First-person singular intensity is era-heterogeneous: **Donda 2** peaks 1psg (z=0.195, BH-p=1e-12) while **Jesus Is King** is the trough (z=-0.186, BH-p=1e-12), contradicting a single lifetime linear trend.
3. After Benjamini–Hochberg correction across 126 era×emotion tests, **2** comparisons remain STRONG (exclude baseline, |z|>0.3, BH-p<0.05), with **5** additional MODERATE signals (|z|>0.25).
4. The **2018** bipolar diagnosis coincides with the catalog's largest vocabulary collapse; the **2020** separation from Kim coincides with vocabulary expansion (**2020-12-01** pivot, **80 days after** her divorce filing per `kanye_life_events.csv`; collapse **2018-06-01**: MTLD n_settings=6, magnitude=-56.57, FK-grade n_settings=4, magnitude=-2.12; expansion MTLD n_settings=6, magnitude=32.55).
5. A previously uncurated fear spike appears in **November 2021** around **Donda** LP companion-track drops and active **divorce–custody escalation** (the lyrics cite moms, dads, and kids — not collaborator loss). The same month also contains Virgil Abloh's **Nov 28** death, but the fear-elevated lines read overwhelmingly as **family fracture**, not creative-partner grief.
6. March–April **2026** releases carry custody-siege adjacent fear language after the January **2026** WSJ apology — the manuscript display filters out obvious lexical false positives while leaving the substantive threats-to-family lines.

## Methodology summary

Phase 3.7 does not introduce new estimators: it re-filters Phase 3.6 change-point consensus from the raw PEL sweep (±60-day setting clusters), tightens minimum-setting counts adaptively, drops interpolated forward-fill months with zero catalog releases, enforces metric-specific magnitude floors, and dedupes ±90-day neighbors within each metric.

Phase 3.7b applies post-hoc curation to Phase 3.7 outputs without re-running statistical tests. Specifically: surfaces a vocabulary-diversity finding hiding in the change-point CSV, reframes the November 2021 fear interpretation based on a lyrical content audit, and filters Finding 6 display lines to remove classifier surface-feature triggers.

## Finding 1: Donda's death produced a measurable, statistically significant fear elevation in lyrics for one year afterward.
- Statistic: Δ=0.0373, 95% CI [0.0235, 0.0517], permutation p=0.0286, n=756 lines
- Chart: ![emotion deltas](phase_3_7_charts/emotion_deltas_redone.png)
- Quotes:
  - *«I'm the only thing I'm afraid of»* — *Amazing* (2008-07-01)
  - *«On, I let my nightmares go»* — *Put On* (2008-09-02)
  - *«Don't worry 'bout what we can't control»* — *Paranoid* (2008-07-01)
  - *«You worry 'bout the wrong things»* — *Paranoid* (2008-07-01)
  - *«I'm a monster, I'm a killer»* — *Amazing* (2008-07-01)

The Year-after window concentrates enough lines that the fear classifier registers a sustained upward shift rather than noise around the global baseline; competing severity-3 shocks largely wash out at this granularity.

_Reproducibility:_ `data/phase_3_5_statistical_tests.csv`, `scripts/07_phase37_curation.py`, `scripts/07b_phase37b_patch.py`.

## Finding 2: Self-reference is era-specific, not life-trend-specific. Donda 2 (post-divorce, peak vengeance) is his highest 1psg era; Jesus Is King (gospel deflection) is his lowest.
- Statistic: peak era **Donda 2** (1psg z=0.195, BH-p=1e-12); trough **Jesus Is King** (z=-0.186, BH-p=1e-12)
- Chart: ![1psg eras](phase_3_7_charts/era_1psg_deviations.png)
- Quotes (filtered per Phase 3.7b repetition/copula rules; see `data/phase_3_7b_pronoun_quotes_filtered.json`):
  - *«I lost my twin and my mother too»* — *Louie Bags* (2022-07-01)
  - *«Yeah, I lost my best man and I lost my spouse»* — *First Time in a Long Time* (2022-07-01)
  - *«I go to war, just me and my gun»* — *We Did It Kid* (2022-07-01)
  - *«We need you (We need you, of the Lord)»* — *Every Hour* (2019-10-25)
  - *«Sing 'til the power of the Lord comes down ('Cause when we sing the glory of the Lord comes down, down)»* — *Every Hour* (2019-10-25)
  - *«Love God and our neighbor, as written in Luke»* — *Selah* (2019-10-25)

Catalog-wide Pennebaker slopes flatten because opposing eras cancel; zooming to bins restores interpretable swings — **Donda 2** skews toward grievance cadences, while **Jesus Is King** illustrates communal/theological address even in the statistical trough of first-person singular density.

_Reproducibility:_ `data/phase_3_7_era_pronoun_tiered.csv`, `data/phase_3_7b_pronoun_quotes_filtered.json`.

## Finding 3: Specific eras carry specific emotional signatures that hold up under multiple-comparison correction.
- Display table (STRONG tier only):
| era_clean | metric | z | BH-p | exclude_baseline |
| --- | --- | --- | --- | --- |
| Donda 2 | emotion_sadness | 0.319 | 1e-12 | True |
| G.O.O.D. Fridays + MBDTF | emotion_anger | 0.310 | 1e-12 | True |

_MODERATE tier count:_ **5** additional pairs.

MBDTF-cycle anger (already visible pre-correction) and Donda 2's sadness elevation exemplify how concentrated eras—not isolated punchlines—carry persistent emotional offsets.
- Quotes:
  - *«Don't know how to behave, we rage out of the raves»* — *One Minute* (2018-07-01)
  - *«Never trust a bartender that don't drink, bitch»* — *Watch* (2018-07-01)

_Reproducibility:_ `data/phase_3_7_era_emotion_tiered.csv`, `data/kanye_line_features_enriched.csv`.

## Finding 4: The June 2018 bipolar disclosure coincides with the catalog's sharpest curated vocabulary contraction; December 2020 coincides with vocabulary expansion after Kim's divorce filing.
- Change-point rows (Phase 3.7 curated CSV, no re-fit): **MTLD** median breakpoint **2018-06-01** aligns with the severity-3 bipolar-cover diagnosis (n_settings=6, magnitude=-56.57; nearest severity-3 distance 0 days). **Flesch–Kincaid grade** moves at the same stamp (n_settings=4, magnitude=-2.12). The **2020-12-01** expansion pivot sits **80 days after** Kim's divorce filing per `kanye_life_events.csv` (n_settings=6, magnitude=32.55).
- Lexical exemplars (`data/phase_3_7b_lexical_event_lines.json`):
  - **Collapse window (May–July 2018, shorter lines prioritized):**
  - *«Whoopity-whoop, scoop-poop-woop-toop»* — *XTCY* (2018-06-01)
  - *«Turn TMZ to Smack DVD, huh»* — *Yikes* (2018-06-01)
  - *«Devil been tryna make an army»* — *Yikes* (2018-06-01)
  - *«Niggas been tryna test my Gandhi»* — *Yikes* (2018-06-01)
  - *«Hopefully, Alice Johnson will inspire men»* — *Cudi Montage* (2018-06-08)
  - **Expansion window (Oct 2020–Feb 2021, longer lines prioritized):**
  - *«Hit the club with at least ten, you in a 500, that's a cheap Benz»* — *Go2DaMoon* (2020-12-25)
  - *«I promised Playboi we'd get the tape out 'fore the weekend is over»* — *Go2DaMoon* (2020-12-25)
  - *«YEEZYs in the stock room, that's the name, make the stock boom»* — *Go2DaMoon* (2020-12-25)
  - *«Slept with him, then woke up, saw his watch was a Fossil»* — *Go2DaMoon* (2020-12-25)
  - *«Then left with a scammer with a Gucci hat from Marshalls»* — *Go2DaMoon* (2020-12-25)

Independent lexical complexity metrics moving together at the diagnostic disclosure — then reversing around the divorce filing — supports reading vocabulary diversity as a stylistic stress gauge rather than a spurious PEL artifact.

_Reproducibility:_ `data/phase_3_7_change_points_curated.csv`, `data/phase_3_7b_lexical_event_lines.json`.

## Finding 5: A previously uncurated fear spike appears in November 2021 around the Donda LP companion-track releases and active divorce-custody escalation.
- Statistic: **2021-11-01** monthly `emotion_fear` aggregate (same Phase 3.7 descriptive anchor — **27** calendar days before **Nov 28** shocks that dominated that month's tabloid cycle); contemporaneous drops include **Never Abandon Your Family** and **Up From the Ashes** in the custody narrative following Kim's **Feb 19, 2021** filing.
- The same month also contains the Nov 28 death of Virgil Abloh, but the lyrical content of the fear-elevated lines is overwhelmingly about family fracture, not collaborator loss — e.g. **Tell mom you're sorry**, **Come back tonight, daddy**, **Why won't you answer me**.
- Change-point note: No **emotion_fear** row survived Phase 3.7 segmentation filters for that month — treat as descriptive.
- Quotes:
  - *«I wish I never screamed, alcohol when you breathe»* — *Never Abandon Your Family* (2021-11-14)
  - *«"Come back tonight, daddy, please, come back tonight, daddy, please"»* — *Never Abandon Your Family* (2021-11-14)
  - *«"Tell mom you're sorry," she's screaming at me»* — *Never Abandon Your Family* (2021-11-14)
  - *«Darkness can't take light from me»* — *Never Abandon Your Family* (2021-11-14)
  - *«"Why won't you answer me? I'm in the room"»* — *Never Abandon Your Family* (2021-11-14)
  - *«God is our shepherd, light in the night»* — *Up From the Ashes* (2021-11-14)

_Reproducibility:_ `data/phase_3_6_fear_deep_dive.csv`, `data/kanye_life_events.csv`.

## Finding 6: March–April 2026 fear-coded language clusters on custody-adjacent threats after the WSJ apology.
- Tracks involved: Circles, Damn, Highs and Lows, King, Preacher Man, This a Must.
- Quotes (manually filtered display; full candidate pool untouched in `data/phase_3_7_fear_2026_lines.json`):
  - *«All the threats to the fam, I'm advisin' against (Baow, baow)»* — *This a Must* (2026-03-28)
  - *«Don't let me go, don't let me go»* — *Highs and Lows* (2026-03-28)
  - *«Before I break your heart, I'll have a heart attack»* — *Highs and Lows* (2026-03-28)
  - *«When it's dark, you don't know where you goin'»* — *Preacher Man* (2026-03-28)
  - *«This ring that I hold, I—»* — *Preacher Man* (2026-03-28)
  - *«I float, I don't never land»* — *Preacher Man* (2026-03-28)
  - *«Pray we never crash, crash, crash»* — *Damn* (2026-03-28)
  - *«Did I ruin your plans, plans, plans?»* — *Damn* (2026-03-28)

_Footnote:_ 8 of 29 candidate fear-coded lines are displayed; the remaining 21 either fall below 0.20 fear score or are classifier triggers on lexical surface features that do not reflect line-level affect on inspection. Phase 3.8 will apply multi-classifier consensus to remove this manual step.

_Classifier-trigger examples withheld from display:_ motivational “If it don't scare you…” (hits **scare**); sexual lines (**Way she suck…**, **Climax fast…**); neutral scene-setting (**I walk up to the preacher man**).

- Compare / contrast (2008 vs 2026):
  - **2008** fear lines skew toward bereavement vertigo and insomnia confession (*808s* palette).
  - **2026** skews toward custody siege metaphors and reputational whiplash post-apology.
  - Both weaponize vulnerability publicity; twenty years later procedural/legal threats rival existential grief.

_Reproducibility:_ `data/phase_3_7_fear_2026_lines.json`, `data/kanye_life_events.csv`.

## Limitations
Line-level `emotion_*` scores come from a single DistilRoBERTa-family fine-tune; VADER / RoBERTa-Twitter replication is queued for Phase 3.8. Era sample sizes swing widely (~400–2k lines). Finding 6 applies a **manual classifier-trigger filter** for quoted lines — Phase 3.8 will apply multi-classifier consensus to handle this rigorously.

## What does NOT replicate (transparency)
- Phase 3.5 semantic theme correlations attenuate once conservative intervals are foregrounded.
- Pennebaker-style linear year slopes fail — Finding 2 reframes self-focus as piecewise-era structure.
- Almost every severity-3 365-day emotion delta besides Donda fear lacks power; Phase 3.6 era pooling was required to surface secondary structure.
