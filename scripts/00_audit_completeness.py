"""
00_audit_completeness.py — Audit kanye_verses_only.csv against canonical album tracklists.

Reads:   kanye_verses_only.csv (main scrape output)
Writes:  data/missing_tracks.csv  — every canonical album track NOT in the dataset,
                                    with the album/era it belongs to and the artist
                                    attribution to use for re-fetching.

Run this first, then 00b_fetch_missing.py to actually pull the missing tracks.

The canonical tracklists below are the source of truth. They cover every Kanye solo
studio album, plus collaborative albums where Kanye is a primary artist
(Watch the Throne, Cruel Summer, Kids See Ghosts, Vultures 1/2). Standalone singles,
features on other artists' albums, and unreleased leaks are deliberately excluded —
the user's requirement is "100% on every album track."
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# Canonical album tracklists. (album_label, era_clean, primary_artist) → [tracks]
# ============================================================
CANONICAL = {
    'The College Dropout (2004)': {
        'era_clean': 'The College Dropout',
        'release_date': '2004-02-10',
        'primary_artist': 'Kanye West',
        'tracks': [
            'Intro', "We Don't Care", 'Graduation Day', 'All Falls Down', "I'll Fly Away",
            'Spaceship', 'Jesus Walks', 'Never Let Me Down', 'Get Em High',
            'Workout Plan', 'The New Workout Plan', 'Slow Jamz', 'Breathe In Breathe Out',
            'School Spirit Skit 1', 'School Spirit', 'School Spirit Skit 2', 'Lil Jimmy Skit',
            'Two Words', 'Through the Wire', 'Family Business', 'Last Call',
        ],
    },
    'Late Registration (2005)': {
        'era_clean': 'Late Registration',
        'release_date': '2005-08-30',
        'primary_artist': 'Kanye West',
        'tracks': [
            'Wake Up Mr. West', "Heard 'Em Say", 'Touch the Sky', 'Gold Digger',
            'Skit #1', 'Drive Slow', 'My Way Home', 'Crack Music', 'Roses', 'Bring Me Down',
            'Addiction', 'Skit #2', 'Diamonds from Sierra Leone (Remix)', 'We Major',
            'Skit #3', 'Hey Mama', 'Celebration', 'Skit #4', 'Gone',
            'Diamonds from Sierra Leone', 'Late',
        ],
    },
    'Graduation (2007)': {
        'era_clean': 'Graduation',
        'release_date': '2007-09-11',
        'primary_artist': 'Kanye West',
        'tracks': [
            'Good Morning', 'Champion', 'Stronger', 'I Wonder', 'Good Life',
            "Can't Tell Me Nothing", 'Barry Bonds', 'Drunk and Hot Girls',
            'Flashing Lights', 'Everything I Am', 'The Glory', 'Homecoming', 'Big Brother',
        ],
    },
    '808s & Heartbreak (2008)': {
        'era_clean': '808s & Heartbreak',
        'release_date': '2008-11-24',
        'primary_artist': 'Kanye West',
        'tracks': [
            'Say You Will', 'Welcome to Heartbreak', 'Heartless', 'Amazing',
            'Love Lockdown', 'Paranoid', 'RoboCop', 'Street Lights', 'Bad News',
            'See You in My Nightmares', 'Coldest Winter', 'Pinocchio Story',
        ],
    },
    'My Beautiful Dark Twisted Fantasy (2010)': {
        'era_clean': 'G.O.O.D. Fridays + MBDTF',
        'release_date': '2010-11-22',
        'primary_artist': 'Kanye West',
        'tracks': [
            'Dark Fantasy', 'Gorgeous', 'Power', 'All of the Lights (Interlude)',
            'All of the Lights', 'Monster', 'So Appalled', 'Devil in a New Dress',
            'Runaway', 'Hell of a Life', 'Blame Game', 'Lost in the World',
            'Who Will Survive in America',
        ],
    },
    'Watch the Throne (2011)': {
        'era_clean': 'Watch the Throne',
        'release_date': '2011-08-08',
        'primary_artist': 'JAY-Z & Kanye West',
        'tracks': [
            'No Church in the Wild', 'Lift Off', 'Niggas in Paris', 'Otis',
            'Gotta Have It', 'New Day', "That's My Bitch", 'Welcome to the Jungle',
            'Who Gon Stop Me', 'Murder to Excellence', 'Made in America',
            'Why I Love You', 'Illest Motherfucker Alive', 'H•A•M', 'Primetime', 'The Joy',
        ],
    },
    'Cruel Summer (2012)': {
        'era_clean': 'Cruel Summer + Yeezus build-up',
        'release_date': '2012-09-18',
        'primary_artist': 'Kanye West',  # GOOD Music — most tracks credited to specific artists, but Kanye is primary
        'tracks': [
            'To the World', 'Clique', 'Mercy', 'New God Flow', 'The Morning',
            'Cold', 'Higher', 'Sin City', 'The One', 'Creepers',
            'Bliss', "Don't Like (Remix)",
        ],
    },
    'Yeezus (2013)': {
        'era_clean': 'Yeezus',
        'release_date': '2013-06-18',
        'primary_artist': 'Kanye West',
        'tracks': [
            'On Sight', 'Black Skinhead', 'I Am a God', 'New Slaves',
            'Hold My Liquor', "I'm In It", 'Blood on the Leaves',
            'Guilt Trip', 'Send It Up', 'Bound 2',
        ],
    },
    'The Life of Pablo (2016)': {
        'era_clean': 'The Life of Pablo',
        'release_date': '2016-02-14',
        'primary_artist': 'Kanye West',
        'tracks': [
            'Ultralight Beam', 'Father Stretch My Hands Pt. 1', 'Pt. 2', 'Famous',
            'Feedback', 'Low Lights', 'Highlights', 'Freestyle 4', 'I Love Kanye',
            'Waves', 'FML', 'Real Friends', 'Wolves', "Frank's Track",
            'Siiiiiiiiilver Surffffeeeeer Intermission', '30 Hours',
            'No More Parties in LA', 'Facts (Charlie Heat Version)', 'Fade', 'Saint Pablo',
        ],
    },
    'ye (2018)': {
        'era_clean': 'Wyoming Sessions (ye / KSG)',
        'release_date': '2018-06-01',
        'primary_artist': 'Kanye West',
        'tracks': [
            'I Thought About Killing You', 'Yikes', 'All Mine', "Wouldn't Leave",
            'No Mistakes', 'Ghost Town', 'Violent Crimes',
        ],
    },
    'Kids See Ghosts (2018)': {
        'era_clean': 'Wyoming Sessions (ye / KSG)',
        'release_date': '2018-06-08',
        'primary_artist': 'KIDS SEE GHOSTS',
        'tracks': [
            'Feel the Love', 'Fire', '4th Dimension', 'Freeee (Ghost Town Pt. 2)',
            'Reborn', 'Kids See Ghosts', 'Cudi Montage',
        ],
    },
    'Jesus Is King (2019)': {
        'era_clean': 'Jesus Is King',
        'release_date': '2019-10-25',
        'primary_artist': 'Kanye West',
        'tracks': [
            'Every Hour', 'Selah', 'Follow God', 'Closed on Sunday', 'On God',
            'Everything We Need', 'Water', 'God Is', 'Hands On',
            'Use This Gospel', 'Jesus Is Lord',
        ],
    },
    'Donda (2021)': {
        'era_clean': 'Donda',
        'release_date': '2021-08-29',
        'primary_artist': 'Kanye West',
        'tracks': [
            'Donda Chant', 'Jail', 'God Breathed', 'Off the Grid', 'Hurricane',
            'Praise God', 'Jonah', 'Ok Ok', 'Junya', 'Believe What I Say',
            '24', 'Remote Control', 'Moon', 'Heaven and Hell', 'Donda',
            'Keep My Spirit Alive', 'Jesus Lord', 'New Again', 'Tell the Vision',
            'Lord I Need You', 'Pure Souls', 'Come to Life', 'No Child Left Behind',
            'Jail Pt. 2', 'Ok Ok Pt. 2', 'Junya Pt. 2', 'Jesus Lord Pt. 2',
        ],
    },
    'Donda Deluxe (2021)': {
        'era_clean': 'Donda',
        'release_date': '2021-11-14',
        'primary_artist': 'Kanye West',
        'tracks': [
            'Life of the Party', 'Up From the Ashes', 'Never Abandon Your Family',
            'Keep My Spirit Alive Pt. 2', 'Remote Control Pt. 2',
        ],
    },
    'Donda 2 (2022)': {
        'era_clean': 'Donda 2',
        'release_date': '2022-02-22',
        'primary_artist': 'Kanye West',
        'tracks': [
            'True Love', 'Broken Road', 'Get Lost', 'Too Easy', 'Flowers',
            'Security', 'We Did It Kid', 'Pablo', 'Louie Bags', 'Happy',
            'Sci Fi', 'City of Gods', 'Lord Lift Me Up', 'First Time in a Long Time',
            'Selfish', 'Link Up', 'Daylight', 'Possessions', 'Life',
            'My Life Was Never Eazy', 'We Made It', 'Bridge to Lay Down',
        ],
    },
    'Vultures 1 (2024)': {
        'era_clean': 'Vultures 1 + Vultures 2',
        'release_date': '2024-02-10',
        'primary_artist': 'Kanye West & Ty Dolla $ign',
        'tracks': [
            'Stars', 'Keys to My Life', 'Paid', 'Talking', 'Back to Me',
            'Hoodrat', 'Do It', 'Paperwork', 'Burn', 'Fuk Sumn',
            'Vultures', 'Carnival', 'Beg Forgiveness', "Good (Don't Die)",
            'Problematic', 'King',
        ],
    },
    'Vultures 2 (2024)': {
        'era_clean': 'Vultures 1 + Vultures 2',
        'release_date': '2024-08-03',
        'primary_artist': 'Kanye West & Ty Dolla $ign',
        'tracks': [
            'Slide', 'Time Moving Slow', 'Field Trip', 'Lifestyle', 'River',
            'Promotion', 'Forever Rolling', 'Husband', 'Dead', 'Bomb',
            '530', 'Forever', 'Maybe', 'Fried', 'Sky City', 'My Soul',
        ],
    },
    'Bully (2026)': {
        'era_clean': 'Bully build-up + Cuck/IAPW controversy',
        'release_date': '2026-03-28',
        'primary_artist': 'Kanye West',
        'tracks': [
            'King', 'This a Must', 'Father', 'All the Love', 'Sisters and Brothers',
            "Mama's Favorite", 'Punch Drunk', 'Whatever Works', "I Can't Wait",
            'Bully', 'Highs and Lows', 'Preacher Man', 'Beauty and the Beast',
            'Last Breath', 'White Lines', 'Circles', 'Damn', 'This One Here',
        ],
    },
}


def normalize(s: str) -> str:
    """Lowercase + strip non-alphanumeric for fuzzy matching."""
    return re.sub(r'[^a-z0-9]+', ' ', str(s).lower().strip()).strip()


def find_match(canonical: str, existing_norms: set) -> bool:
    c_norm = normalize(canonical)
    if c_norm in existing_norms:
        return True
    # Substring match for parenthetical variants ("Diamonds (Remix)" vs "Diamonds")
    for e in existing_norms:
        if len(c_norm) > 5 and len(e) > 5 and (c_norm in e or e in c_norm):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', default='kanye_verses_only.csv')
    ap.add_argument('--out-dir', default='data')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    df = pd.read_csv(args.input)
    existing_norms = {normalize(t) for t in df['track_title']}
    print(f"Loaded {len(df)} tracks from {args.input}")

    missing_rows, total_canonical, total_found = [], 0, 0
    print("\n" + "=" * 80)
    print("COMPLETENESS BY ALBUM")
    print("=" * 80)

    for album, meta in CANONICAL.items():
        tracks = meta['tracks']
        found = sum(1 for t in tracks if find_match(t, existing_norms))
        total_canonical += len(tracks)
        total_found += found
        pct = found / len(tracks)
        flag = '+' if pct == 1.0 else ('!' if pct >= 0.8 else 'x')
        print(f"  [{flag}] {album:<46} {found}/{len(tracks)}  ({pct:.0%})")
        for t in tracks:
            if not find_match(t, existing_norms):
                missing_rows.append({
                    'album_label':    album,
                    'track_title':    t,
                    'era_clean':      meta['era_clean'],
                    'release_date':   meta['release_date'],
                    'primary_artist': meta['primary_artist'],
                })

    print("=" * 80)
    print(f"TOTAL: {total_found}/{total_canonical} ({total_found/total_canonical:.0%}) — MISSING {len(missing_rows)} tracks")

    out_path = out_dir / 'missing_tracks.csv'
    pd.DataFrame(missing_rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(missing_rows)} rows)")
    print("Next step: python scripts/00b_fetch_missing.py")
    return 0


if __name__ == '__main__':
    sys.exit(main())
