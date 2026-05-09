"""
00c_deep_audit.py — Multi-tier completeness audit against the discography MD.

Tier 1 — Canonical album tracks (267 hardcoded across 18 studio/collab albums).
         These MUST be near-100%. This is what 00_audit_completeness.py already checks,
         re-checked here for a unified report.
Tier 2 — Kanye-led non-album officially-released work (singles, EP cuts, soundtrack
         contributions where Kanye is the primary artist).
Tier 3 — Major officially-released features (Kanye on others' tracks).
Tier 4 — Leaks / mixtapes / unreleased / demos / Sunday Service Choir tracks where
         Kanye doesn't have lead vocals. Out of scope; counted but not chased.

The script parses the discography MD line-by-line, tracking which year section and
which subsection type (album / singles / features / leaks / mixtape) each quoted
track appears in. Each candidate gets tagged with a tier. Matches against the
dataset are computed with a normalized fuzzy comparison.

Reads:   kanye_verses_only.csv
         Kanye_West_Ye_Complete_Discography_and_Verse_Catalog__2003_to_May_2026.md
Writes:  data/audit_tier1_album.csv      — per-album coverage
         data/audit_tier2_singles.csv    — Kanye-led singles coverage
         data/audit_tier3_features.csv   — feature coverage
         data/audit_summary.txt          — human-readable summary

Usage:
  /usr/bin/python3 scripts/00c_deep_audit.py
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Re-use the canonical tracklists from 00_audit_completeness.py.
# Importing across script files isn't clean for a flat scripts/ folder, so duplicate.
CANONICAL_ALBUMS = {
    'The College Dropout (2004)': ['Intro', "We Don't Care", 'Graduation Day', 'All Falls Down', "I'll Fly Away", 'Spaceship', 'Jesus Walks', 'Never Let Me Down', 'Get Em High', 'Workout Plan', 'The New Workout Plan', 'Slow Jamz', 'Breathe In Breathe Out', 'School Spirit Skit 1', 'School Spirit', 'School Spirit Skit 2', 'Lil Jimmy Skit', 'Two Words', 'Through the Wire', 'Family Business', 'Last Call'],
    'Late Registration (2005)': ['Wake Up Mr. West', "Heard 'Em Say", 'Touch the Sky', 'Gold Digger', 'Skit #1', 'Drive Slow', 'My Way Home', 'Crack Music', 'Roses', 'Bring Me Down', 'Addiction', 'Skit #2', 'Diamonds from Sierra Leone (Remix)', 'We Major', 'Skit #3', 'Hey Mama', 'Celebration', 'Skit #4', 'Gone', 'Diamonds from Sierra Leone', 'Late'],
    'Graduation (2007)': ['Good Morning', 'Champion', 'Stronger', 'I Wonder', 'Good Life', "Can't Tell Me Nothing", 'Barry Bonds', 'Drunk and Hot Girls', 'Flashing Lights', 'Everything I Am', 'The Glory', 'Homecoming', 'Big Brother'],
    '808s & Heartbreak (2008)': ['Say You Will', 'Welcome to Heartbreak', 'Heartless', 'Amazing', 'Love Lockdown', 'Paranoid', 'RoboCop', 'Street Lights', 'Bad News', 'See You in My Nightmares', 'Coldest Winter', 'Pinocchio Story'],
    'My Beautiful Dark Twisted Fantasy (2010)': ['Dark Fantasy', 'Gorgeous', 'Power', 'All of the Lights (Interlude)', 'All of the Lights', 'Monster', 'So Appalled', 'Devil in a New Dress', 'Runaway', 'Hell of a Life', 'Blame Game', 'Lost in the World', 'Who Will Survive in America'],
    'Watch the Throne (2011)': ['No Church in the Wild', 'Lift Off', 'Niggas in Paris', 'Otis', 'Gotta Have It', 'New Day', "That's My Bitch", 'Welcome to the Jungle', 'Who Gon Stop Me', 'Murder to Excellence', 'Made in America', 'Why I Love You', 'Illest Motherfucker Alive', 'H•A•M', 'Primetime', 'The Joy'],
    'Cruel Summer (2012)': ['To the World', 'Clique', 'Mercy', 'New God Flow', 'The Morning', 'Cold', 'Higher', 'Sin City', 'The One', 'Creepers', 'Bliss', "Don't Like (Remix)"],
    'Yeezus (2013)': ['On Sight', 'Black Skinhead', 'I Am a God', 'New Slaves', 'Hold My Liquor', "I'm In It", 'Blood on the Leaves', 'Guilt Trip', 'Send It Up', 'Bound 2'],
    'The Life of Pablo (2016)': ['Ultralight Beam', 'Father Stretch My Hands Pt. 1', 'Pt. 2', 'Famous', 'Feedback', 'Low Lights', 'Highlights', 'Freestyle 4', 'I Love Kanye', 'Waves', 'FML', 'Real Friends', 'Wolves', "Frank's Track", 'Siiiiiiiiilver Surffffeeeeer Intermission', '30 Hours', 'No More Parties in LA', 'Facts (Charlie Heat Version)', 'Fade', 'Saint Pablo'],
    'ye (2018)': ['I Thought About Killing You', 'Yikes', 'All Mine', "Wouldn't Leave", 'No Mistakes', 'Ghost Town', 'Violent Crimes'],
    'Kids See Ghosts (2018)': ['Feel the Love', 'Fire', '4th Dimension', 'Freeee (Ghost Town Pt. 2)', 'Reborn', 'Kids See Ghosts', 'Cudi Montage'],
    'Jesus Is King (2019)': ['Every Hour', 'Selah', 'Follow God', 'Closed on Sunday', 'On God', 'Everything We Need', 'Water', 'God Is', 'Hands On', 'Use This Gospel', 'Jesus Is Lord'],
    'Donda (2021)': ['Donda Chant', 'Jail', 'God Breathed', 'Off the Grid', 'Hurricane', 'Praise God', 'Jonah', 'Ok Ok', 'Junya', 'Believe What I Say', '24', 'Remote Control', 'Moon', 'Heaven and Hell', 'Donda', 'Keep My Spirit Alive', 'Jesus Lord', 'New Again', 'Tell the Vision', 'Lord I Need You', 'Pure Souls', 'Come to Life', 'No Child Left Behind', 'Jail Pt. 2', 'Ok Ok Pt. 2', 'Junya Pt. 2', 'Jesus Lord Pt. 2'],
    'Donda Deluxe (2021)': ['Life of the Party', 'Up From the Ashes', 'Never Abandon Your Family', 'Keep My Spirit Alive Pt. 2', 'Remote Control Pt. 2'],
    'Donda 2 (2022)': ['True Love', 'Broken Road', 'Get Lost', 'Too Easy', 'Flowers', 'Security', 'We Did It Kid', 'Pablo', 'Louie Bags', 'Happy', 'Sci Fi', 'City of Gods', 'Lord Lift Me Up', 'First Time in a Long Time', 'Selfish', 'Link Up', 'Daylight', 'Possessions', 'Life', 'My Life Was Never Eazy', 'We Made It', 'Bridge to Lay Down'],
    'Vultures 1 (2024)': ['Stars', 'Keys to My Life', 'Paid', 'Talking', 'Back to Me', 'Hoodrat', 'Do It', 'Paperwork', 'Burn', 'Fuk Sumn', 'Vultures', 'Carnival', 'Beg Forgiveness', "Good (Don't Die)", 'Problematic', 'King'],
    'Vultures 2 (2024)': ['Slide', 'Time Moving Slow', 'Field Trip', 'Lifestyle', 'River', 'Promotion', 'Forever Rolling', 'Husband', 'Dead', 'Bomb', '530', 'Forever', 'Maybe', 'Fried', 'Sky City', 'My Soul'],
    'Bully (2026)': ['King', 'This a Must', 'Father', 'All the Love', 'Sisters and Brothers', "Mama's Favorite", 'Punch Drunk', 'Whatever Works', "I Can't Wait", 'Bully', 'Highs and Lows', 'Preacher Man', 'Beauty and the Beast', 'Last Breath', 'White Lines', 'Circles', 'Damn', 'This One Here'],
}


def normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', str(s).lower().strip()).strip()


def is_match(a: str, b: str) -> bool:
    """Fuzzy match between a candidate and a dataset title."""
    na, nb = normalize(a), normalize(b)
    if na == nb: return True
    if len(na) > 5 and len(nb) > 5 and (na in nb or nb in na): return True
    return False


def find_in_dataset(candidate: str, existing_norms: list[str]) -> bool:
    return any(is_match(candidate, e) for e in existing_norms)


# =====================================================================
# MD parser
# =====================================================================
LEAK_KEYWORDS = (
    'leak', 'unreleased', 'mixtape', 'demo', 'freshmen adjustment',
    'sunday service', 'production with', 'production-only', 'production credit',
    'misattributed', 'frequently misattributed', 'b-side',
    'live album', 'listening party', 'tour',
)
SINGLES_KEYWORDS = ('single', 'non-album release')
FEATURES_KEYWORDS = ('features', 'featur', 'guest spot')
ALBUM_KEYWORDS = ('feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', '(20')

# Strings that look like quoted titles but aren't tracks
NOT_A_TRACK = {
    'rap genius senior project', '#1epicrant', 'top this', 'genius picks: songs 2018',
    'kanye west', 'ye', 'mr.', 'donald trump', 'nigga heil hitler',  # too short or noise
    'production', 'production only', 'a written testimony', 'mr. wrong', 'wants and needs',
    'wants & needs', 'replace me', 'brunch on sundays', 'ztfo', 'lithuania',
    'bigger than me', 'sacrifices',  # explicitly noted as misattributed in MD
}


def classify_section(line: str, prev_section: str) -> str:
    line_lower = line.lower()
    # Year header resets context
    if re.match(r'^## \d{4}', line):
        return 'unknown'
    # Subsection header (bold text)
    bold_match = re.findall(r'\*\*([^*]+?)\*\*', line)
    if not bold_match:
        return prev_section
    headers = ' '.join(bold_match).lower()
    if any(k in headers for k in LEAK_KEYWORDS): return 'leaks'
    if any(k in headers for k in FEATURES_KEYWORDS): return 'features'
    if any(k in headers for k in SINGLES_KEYWORDS): return 'singles'
    if any(k in headers for k in ALBUM_KEYWORDS): return 'album'
    return prev_section


def extract_titles(line: str) -> list[str]:
    titles = re.findall(r'"([^"]+)"', line)
    out = []
    for t in titles:
        t = t.strip().rstrip(',.;:?!').strip()
        # Skip if too short, too long, or obvious non-track
        if not (3 <= len(t) <= 80): continue
        if normalize(t) in NOT_A_TRACK: continue
        out.append(t)
    return out


def parse_md(md_path: Path) -> pd.DataFrame:
    text = md_path.read_text()
    rows, current_year, current_section = [], None, 'unknown'

    for line in text.splitlines():
        # Year section
        ym = re.match(r'^## (\d{4})', line)
        if ym:
            current_year, current_section = int(ym.group(1)), 'unknown'
            continue
        if current_year is None:
            continue

        # Section context update
        current_section = classify_section(line, current_section)

        # Skip leak/mixtape/SS/production sections entirely
        if current_section == 'leaks':
            continue

        # Per-line override: the line itself contains [Leak/Unreleased] etc
        if any(tag in line for tag in (
            '[Leak/Unreleased]', '[Mixtape]', '[Production]', 'production-only',
            'production with vocal', 'all **[Leak', 'all **[Mixtape',
        )):
            continue

        # Extract titles
        for t in extract_titles(line):
            rows.append({
                'year':    current_year,
                'section': current_section,
                'title':   t,
            })

    df = pd.DataFrame(rows)
    # Dedupe by (year, normalized_title); keep best section (album > singles > features > unknown)
    section_priority = {'album': 0, 'singles': 1, 'features': 2, 'unknown': 3}
    df['priority'] = df['section'].map(section_priority).fillna(4).astype(int)
    df['title_norm'] = df['title'].apply(normalize)
    df = df.sort_values('priority').drop_duplicates(['year', 'title_norm'], keep='first')
    return df.drop(columns=['priority']).reset_index(drop=True)


# =====================================================================
# Main audit
# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--master', default='kanye_verses_only.csv')
    ap.add_argument('--md',     default='Kanye_West_Ye_Complete_Discography_and_Verse_Catalog__2003_to_May_2026.md')
    ap.add_argument('--out-dir', default='data')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    md_path = Path(args.md)
    if not md_path.exists():
        print(f"ERROR: {md_path} not found.")
        return 1

    df = pd.read_csv(args.master)
    existing_norms = list({normalize(t) for t in df['track_title']})
    print(f"Dataset: {len(df)} tracks loaded\n")

    # ---------------- Tier 1: canonical albums ----------------
    print("=" * 78)
    print("TIER 1 — Canonical album tracks (must-have)")
    print("=" * 78)
    tier1_rows, t1_total, t1_found = [], 0, 0
    for album, tracks in CANONICAL_ALBUMS.items():
        n_found = sum(1 for t in tracks if find_in_dataset(t, existing_norms))
        t1_total += len(tracks); t1_found += n_found
        pct = n_found / len(tracks)
        flag = '+' if pct == 1.0 else ('!' if pct >= 0.9 else 'x')
        print(f"  [{flag}] {album:<46} {n_found}/{len(tracks)}  ({pct:.0%})")
        for t in tracks:
            in_data = find_in_dataset(t, existing_norms)
            tier1_rows.append({'album': album, 'track_title': t, 'in_dataset': in_data})
    pd.DataFrame(tier1_rows).to_csv(out_dir / 'audit_tier1_album.csv', index=False)
    print(f"  -- TIER 1 SUBTOTAL: {t1_found}/{t1_total} ({t1_found/t1_total:.0%})")

    # ---------------- Parse MD for tiers 2 & 3 ----------------
    md = parse_md(md_path)
    print(f"\nParsed {len(md)} unique candidate tracks from {md_path.name}")
    print(f"  By section: {dict(md['section'].value_counts())}")

    # Drop any candidates that match a Tier 1 canonical album track
    canonical_norms = {normalize(t) for tracks in CANONICAL_ALBUMS.values() for t in tracks}
    md['is_canonical'] = md['title_norm'].apply(
        lambda n: any(n == c or (len(n) > 5 and len(c) > 5 and (n in c or c in n))
                       for c in canonical_norms))

    tier2 = md[(md['section'] == 'singles') & (~md['is_canonical'])].copy()
    tier3 = md[(md['section'] == 'features') & (~md['is_canonical'])].copy()
    other = md[(md['section'].isin(['unknown'])) & (~md['is_canonical'])].copy()

    # ---------------- Tier 2: Kanye-led non-album ----------------
    print("\n" + "=" * 78)
    print("TIER 2 — Kanye-led non-album officially released tracks")
    print("=" * 78)
    tier2['in_dataset'] = tier2['title'].apply(lambda t: find_in_dataset(t, existing_norms))
    t2_total, t2_found = len(tier2), tier2['in_dataset'].sum()
    print(f"  Total: {t2_total}, in dataset: {t2_found} ({t2_found/max(t2_total,1):.0%})")
    print(f"  Missing:")
    for _, r in tier2[~tier2['in_dataset']].sort_values('year').iterrows():
        print(f"    [{r['year']}] {r['title']}")
    tier2.to_csv(out_dir / 'audit_tier2_singles.csv', index=False)

    # ---------------- Tier 3: features ----------------
    print("\n" + "=" * 78)
    print("TIER 3 — Features (Kanye guest spots on others' tracks)")
    print("=" * 78)
    tier3['in_dataset'] = tier3['title'].apply(lambda t: find_in_dataset(t, existing_norms))
    t3_total, t3_found = len(tier3), tier3['in_dataset'].sum()
    print(f"  Total: {t3_total}, in dataset: {t3_found} ({t3_found/max(t3_total,1):.0%})")
    print(f"  (Per user note: feature gaps are forgivable. Showing top 15 missing.)")
    for _, r in tier3[~tier3['in_dataset']].sort_values('year').head(15).iterrows():
        print(f"    [{r['year']}] {r['title']}")
    tier3.to_csv(out_dir / 'audit_tier3_features.csv', index=False)

    # ---------------- Summary ----------------
    summary_lines = [
        "DEEP COMPLETENESS AUDIT",
        "=" * 78,
        f"Dataset rows:                                 {len(df)}",
        "",
        f"Tier 1 — canonical album tracks:              {t1_found}/{t1_total}  ({t1_found/t1_total:.0%})",
        f"Tier 2 — Kanye-led non-album (singles/EP):    {t2_found}/{t2_total}  ({t2_found/max(t2_total,1):.0%})",
        f"Tier 3 — features on others' tracks:          {t3_found}/{t3_total}  ({t3_found/max(t3_total,1):.0%})",
        "",
        "Tier 1 needs near-100%. Tier 2 should be high. Tier 3 gaps are acceptable.",
        "",
        "Detailed missing-tracks lists are in:",
        f"  {out_dir/'audit_tier1_album.csv'}",
        f"  {out_dir/'audit_tier2_singles.csv'}",
        f"  {out_dir/'audit_tier3_features.csv'}",
    ]
    print("\n" + "\n".join(summary_lines))
    (out_dir / 'audit_summary.txt').write_text('\n'.join(summary_lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
