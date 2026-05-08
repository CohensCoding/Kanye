"""
01_clean_data.py — Build the canonical analyzable dataset.

Reads:   kanye_verses_only.csv (output of fetch_kanye_lyrics.py)
Writes:  data/kanye_cleaned.csv          — 287 analyzable tracks
         data/excluded_tracks.csv        — tracks dropped (with reasons)

What it does:
  1. Standardizes era labels (folds stragglers like "2008 features" into the right canonical era)
  2. Imputes missing release_dates from canonical album release dates
  3. Builds a unique track_id per (title, era) — handles real title collisions
     like Hurricane 2018 (KSG) vs Hurricane 2021 (Donda)
  4. Adds chronological era_rank (0 = Pre-Dropout, 17 = Cuck-era)
  5. Excludes 29 tracks where Kanye's verses extracted as 0 words
     (skits, choir-only pieces, feature extraction failures)

Usage:
  python scripts/01_clean_data.py [--input kanye_verses_only.csv] [--out-dir data/]
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# Canonical era release dates (used for imputing missing release_date values)
ERA_DATES = {
    'Pre-Dropout era':                          '2003-01-01',
    'The College Dropout':                      '2004-02-10',
    'Late Registration':                        '2005-08-30',
    'Graduation':                               '2007-09-11',
    '808s & Heartbreak':                        '2008-11-24',
    'Post-808s feature run':                    '2009-06-15',
    'G.O.O.D. Fridays + MBDTF':                 '2010-11-22',
    'Watch the Throne':                         '2011-08-08',
    'Cruel Summer + Yeezus build-up':           '2012-09-18',
    'Yeezus':                                   '2013-06-18',
    'The Life of Pablo':                        '2016-02-14',
    'Quiet year':                               '2017-06-01',
    'Wyoming Sessions (ye / KSG)':              '2018-06-01',
    'Jesus Is King':                            '2019-10-25',
    'Donda':                                    '2021-08-29',
    'Donda 2':                                  '2022-02-22',
    'Vultures 1 + Vultures 2':                  '2024-02-10',
    'Bully build-up + Cuck/IAPW controversy':   '2025-06-01',
}

# Stragglers that came out of the scraper with non-canonical era labels
ERA_REMAP = {
    '2008 features':                '2008-06-01',  # bug in scraper; fold into Post-808s
    '2009 — Post-808s feature run': 'Post-808s feature run',
    '2020':                         'Donda',
    '2024':                         'Vultures 1 + Vultures 2',
}
ERA_REMAP_RENAMES = {
    '2008 features':                'Post-808s feature run',
    '2009 — Post-808s feature run': 'Post-808s feature run',
    '2020':                         'Donda',
    '2024':                         'Vultures 1 + Vultures 2',
}

# Tracks that came back as `auto_recovered` need explicit overrides
TRACK_OVERRIDES = {
    'Grammy Family': {'era_clean': 'Late Registration',         'date': '2006-06-01'},
    'Vultures':      {'era_clean': 'Vultures 1 + Vultures 2',   'date': '2024-02-10'},
}

# Chronological order for the era_rank column
ERA_ORDER = [
    'Pre-Dropout era', 'The College Dropout', 'Late Registration', 'Graduation',
    '808s & Heartbreak', 'Post-808s feature run', 'G.O.O.D. Fridays + MBDTF',
    'Watch the Throne', 'Cruel Summer + Yeezus build-up', 'Yeezus',
    'The Life of Pablo', 'Quiet year', 'Wyoming Sessions (ye / KSG)',
    'Jesus Is King', 'Donda', 'Donda 2', 'Vultures 1 + Vultures 2',
    'Bully build-up + Cuck/IAPW controversy',
]
ERA_RANK = {era: i for i, era in enumerate(ERA_ORDER)}


def impute_date(row: pd.Series) -> str | None:
    raw = str(row['release_date']) if not pd.isna(row['release_date']) else ''
    if re.match(r'^\d{4}-\d{2}-\d{2}', raw):
        return raw[:10]
    if re.match(r'^\d{4}$', raw):
        return f"{raw}-07-01"  # mid-year approximation for year-only dates
    if row['track_title'] in TRACK_OVERRIDES:
        return TRACK_OVERRIDES[row['track_title']]['date']
    return ERA_DATES.get(row['era_clean'])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='kanye_verses_only.csv')
    ap.add_argument('--out-dir', default='data')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} tracks from {args.input}")

    # 1. Standardize era labels
    df['era_clean'] = df['album_or_era'].replace(ERA_REMAP_RENAMES)
    for title, override in TRACK_OVERRIDES.items():
        df.loc[df['track_title'] == title, 'era_clean'] = override['era_clean']

    # 2. Impute missing dates
    df['release_date_clean'] = df.apply(impute_date, axis=1)
    df['date_imputed'] = df.apply(
        lambda r: pd.isna(r['release_date'])
                  or not re.match(r'^\d{4}-\d{2}-\d{2}', str(r['release_date'])),
        axis=1,
    )
    missing = df[df['release_date_clean'].isna()]
    if len(missing):
        print(f"WARN: {len(missing)} tracks could not be dated:")
        print(missing[['track_title', 'album_or_era']].to_string(index=False))
        return 1

    # 3. Build unique track_id
    df['track_id'] = df.apply(
        lambda r: re.sub(r'[^a-z0-9]+', '_', f"{r['track_title']}_{r['era_clean']}".lower()).strip('_'),
        axis=1,
    )
    if df['track_id'].nunique() != len(df):
        dupes = df[df['track_id'].duplicated(keep=False)]
        print(f"WARN: {len(dupes)} non-unique track_ids:")
        print(dupes[['track_title', 'era_clean', 'track_id']].to_string(index=False))
        return 1

    # 4. Status flag
    df['analysis_status'] = df['kanye_verse_word_count'].apply(
        lambda w: 'excluded_no_content' if w == 0 else 'analyze'
    )

    # 5. Chronology
    df['era_rank'] = df['era_clean'].map(ERA_RANK).fillna(99).astype(int)
    df['date_obj'] = pd.to_datetime(df['release_date_clean'], errors='coerce')
    df['year'] = df['date_obj'].dt.year.astype('Int64')
    df = df.sort_values(['era_rank', 'date_obj', 'track_title']).reset_index(drop=True)

    # Report
    n_analyze = (df['analysis_status'] == 'analyze').sum()
    n_excl = (df['analysis_status'] == 'excluded_no_content').sum()
    print(f"\nCleaning report:")
    print(f"  Total:      {len(df)}")
    print(f"  Analyzable: {n_analyze}")
    print(f"  Excluded:   {n_excl} (no Kanye verse content)")
    print(f"  Imputed:    {df['date_imputed'].sum()} dates filled from era defaults")
    print(f"  Year range: {df['year'].min()}–{df['year'].max()}")

    # Save
    analyzable = df[df['analysis_status'] == 'analyze'].copy()
    analyzable.to_csv(out_dir / 'kanye_cleaned.csv', index=False)
    df[df['analysis_status'] == 'excluded_no_content'][
        ['track_title', 'era_clean', 'release_date_clean', 'extraction_method']
    ].to_csv(out_dir / 'excluded_tracks.csv', index=False)
    print(f"\nWrote {out_dir/'kanye_cleaned.csv'} ({n_analyze} rows)")
    print(f"Wrote {out_dir/'excluded_tracks.csv'} ({n_excl} rows)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
