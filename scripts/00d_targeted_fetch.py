"""
00d_targeted_fetch.py — Fetch the specific Tier 2 and Tier 3 tracks the user
flagged as worth chasing after the deep audit.

This is a focused supplement to 00b_fetch_missing.py with hardcoded targets and
role-aware extraction (features require explicit Kanye markers; Kanye-led tracks
fall back to taking the whole song).

18 targets:
  Tier 2 — Kanye-led non-album (7):
    Impossible, Theraflu, Only One, All Day, FourFiveSeconds, I Love It, Ye vs. the People
  Tier 3 — Major features (11):
    American Boy, Put On, Lollipop (Remix), Swagga Like Us, Knock You Down,
    Run This Town, Make Her Say, Forever, Birthday Song, Sanctified, All We Got

Reads:   kanye_verses_only.csv (master)
Writes:  data/missing_tracks_supplement_2.csv
         data/supplement_fetch_log_2.csv
         kanye_verses_only.csv (merged — old version backed up)

Usage:
  /usr/bin/python3 scripts/00d_targeted_fetch.py
  /usr/bin/python3 scripts/00d_targeted_fetch.py --dry-run   # preview without merging
"""
from __future__ import annotations
import argparse
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
    '(KHTML, like Gecko) Version/17.5 Safari/605.1.15'
)

KANYE_MARKERS = re.compile(
    r'\[(?:Verse|Hook|Chorus|Bridge|Refrain|Pre-Chorus|Intro|Outro|Interlude)[^\]]*?'
    r'(?:Kanye West|Ye(?!es?zus)|Yeezy|Mr\.?\s*West)[^\]]*?\]',
    re.IGNORECASE,
)
SECTION_HEADER = re.compile(r'\[[^\]]+\]')


# =====================================================================
# TARGETS — every track we want to add, with metadata
# Fields: title, primary_artist (for Genius search), era_clean, date,
#         kanye_role ('lead' or 'feature').
# 'feature' = REQUIRE explicit Kanye section markers (other artist's song)
# 'lead'    = treat as Kanye-primary; if no markers, take whole song
# =====================================================================
TARGETS = [
    # ========== Tier 2: Kanye-led non-album ==========
    {'title': 'Impossible',         'artist': 'Kanye West',  'era_clean': 'Late Registration',
     'date': '2006-05-09', 'kanye_role': 'lead'},
    {'title': 'Theraflu',           'artist': 'Kanye West',  'era_clean': 'Cruel Summer + Yeezus build-up',
     'date': '2012-04-04', 'kanye_role': 'lead'},
    {'title': 'Only One',           'artist': 'Kanye West',  'era_clean': 'The Life of Pablo',
     'date': '2014-12-31', 'kanye_role': 'lead'},
    {'title': 'All Day',            'artist': 'Kanye West',  'era_clean': 'The Life of Pablo',
     'date': '2015-03-01', 'kanye_role': 'lead'},
    {'title': 'FourFiveSeconds',    'artist': 'Rihanna',     'era_clean': 'The Life of Pablo',
     'date': '2015-01-24', 'kanye_role': 'lead'},  # Rihanna+Kanye+McCartney all sing — treat as lead
    {'title': 'I Love It',          'artist': 'Kanye West',  'era_clean': 'Wyoming Sessions (ye / KSG)',
     'date': '2018-09-06', 'kanye_role': 'lead'},
    {'title': 'Ye vs. the People',  'artist': 'Kanye West',  'era_clean': 'The Life of Pablo',
     'date': '2018-04-27', 'kanye_role': 'lead'},

    # ========== Tier 3: Major features ==========
    {'title': 'American Boy',       'artist': 'Estelle',           'era_clean': '808s & Heartbreak',
     'date': '2008-03-21', 'kanye_role': 'feature'},
    {'title': 'Put On',             'artist': 'Young Jeezy',       'era_clean': '808s & Heartbreak',
     'date': '2008-09-02', 'kanye_role': 'feature'},
    {'title': 'Lollipop (Remix)',   'artist': 'Lil Wayne',         'era_clean': '808s & Heartbreak',
     'date': '2008-08-26', 'kanye_role': 'feature'},
    {'title': 'Swagga Like Us',     'artist': 'JAY-Z',             'era_clean': '808s & Heartbreak',
     'date': '2008-09-09', 'kanye_role': 'feature'},  # T.I. & Jay-Z feat. Kanye, Wayne
    {'title': 'Knock You Down',     'artist': 'Keri Hilson',       'era_clean': 'Post-808s feature run',
     'date': '2009-03-10', 'kanye_role': 'feature'},
    {'title': 'Run This Town',      'artist': 'JAY-Z',             'era_clean': 'Post-808s feature run',
     'date': '2009-07-24', 'kanye_role': 'feature'},
    {'title': 'Make Her Say',       'artist': 'Kid Cudi',          'era_clean': 'Post-808s feature run',
     'date': '2009-09-15', 'kanye_role': 'feature'},
    {'title': 'Forever',            'artist': 'Drake',             'era_clean': 'Post-808s feature run',
     'date': '2009-09-15', 'kanye_role': 'feature'},
    {'title': 'Birthday Song',      'artist': '2 Chainz',          'era_clean': 'Cruel Summer + Yeezus build-up',
     'date': '2012-08-14', 'kanye_role': 'feature'},
    {'title': 'Sanctified',         'artist': 'Rick Ross',         'era_clean': 'The Life of Pablo',
     'date': '2014-03-04', 'kanye_role': 'feature'},
    {'title': 'All We Got',         'artist': 'Chance the Rapper', 'era_clean': 'The Life of Pablo',
     'date': '2016-05-13', 'kanye_role': 'feature'},
]


# =====================================================================
# Genius API + scraping
# =====================================================================
def genius_search(token: str, query: str, max_results: int = 5) -> list[dict]:
    r = requests.get(
        'https://api.genius.com/search',
        params={'q': query},
        headers={'Authorization': f'Bearer {token}', 'User-Agent': USER_AGENT},
        timeout=20,
    )
    r.raise_for_status()
    hits = r.json().get('response', {}).get('hits', [])
    return [
        {'title': h['result']['title'], 'url': h['result']['url'],
         'artist': h['result']['primary_artist']['name']}
        for h in hits[:max_results]
    ]


def genius_fetch_lyrics(url: str) -> str:
    r = requests.get(url, headers={'User-Agent': USER_AGENT}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    containers = soup.find_all('div', attrs={'data-lyrics-container': 'true'})
    if containers:
        for c in containers:
            for br in c.find_all('br'):
                br.replace_with('\n')
        return '\n'.join(c.get_text() for c in containers).strip()
    legacy = soup.find('div', class_='lyrics')
    if legacy:
        for br in legacy.find_all('br'):
            br.replace_with('\n')
        return legacy.get_text().strip()
    return ''


def normalize_title(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())


def find_best_hit(hits: list[dict], wanted_title: str, wanted_artist: str) -> dict | None:
    if not hits: return None
    wt = normalize_title(wanted_title)
    wa = normalize_title(wanted_artist)
    scored = []
    for h in hits:
        ht = normalize_title(h['title'])
        ha = normalize_title(h['artist'])
        score = 0
        if wt == ht: score += 100
        elif wt in ht or ht in wt:
            score += 80 if abs(len(wt) - len(ht)) <= 8 else 40
        if wa and (wa in ha or ha in wa): score += 30
        if 'kanye' in ha: score += 15
        scored.append((score, h))
    scored.sort(key=lambda x: -x[0])
    best_score, best = scored[0]
    return best if best_score >= 60 else None


# =====================================================================
# Cleaning + extraction
# =====================================================================
def clean_lyrics_text(raw: str) -> str:
    if not raw: return ''
    raw = re.sub(r'^\d+\s*Contributors?.*?Lyrics', '', raw, flags=re.DOTALL)
    raw = re.sub(r'You might also like.*$', '', raw, flags=re.DOTALL)
    raw = re.sub(r'\d*Embed\s*$', '', raw)
    raw = re.sub(r'See .*? LiveGet tickets.*?(?=\n|$)', '', raw, flags=re.DOTALL)
    return raw.strip()


def extract_kanye_verses(full_lyrics: str, kanye_role: str) -> tuple[str, str]:
    """
    kanye_role='feature' => REQUIRE explicit Kanye markers (don't take other artist's verses)
    kanye_role='lead'    => take everything if no markers, else take Kanye-marked sections
    """
    lyrics = clean_lyrics_text(full_lyrics)
    parts = re.split(r'(\[[^\]]+\])', lyrics)
    sections, current_header = [], None
    for p in parts:
        if SECTION_HEADER.match(p.strip()):
            current_header = p.strip()
        elif p.strip():
            sections.append((current_header, p.strip()))

    kanye_chunks = [body for header, body in sections
                    if header and KANYE_MARKERS.search(header)]
    if kanye_chunks:
        return '\n\n'.join(kanye_chunks), 'marker_match'

    if kanye_role == 'feature':
        # No Kanye markers found → don't take other artists' verses
        return '', 'feature_no_kanye_markers'

    # Kanye-led: take everything
    return '\n\n'.join(body for _, body in sections if body), 'lead_no_markers'


# =====================================================================
# Main
# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--master',  default='kanye_verses_only.csv')
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--min-delay', type=float, default=1.0)
    ap.add_argument('--max-delay', type=float, default=2.5)
    args = ap.parse_args()

    load_dotenv()
    token = os.getenv('GENIUS_API_TOKEN')
    if not token:
        print("ERROR: GENIUS_API_TOKEN not in .env"); return 1

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    print(f"Targeted fetch of {len(TARGETS)} tracks "
          f"({sum(1 for t in TARGETS if t['kanye_role']=='lead')} Kanye-led, "
          f"{sum(1 for t in TARGETS if t['kanye_role']=='feature')} features)\n")

    new_rows, fetch_log = [], []
    for i, t in enumerate(TARGETS, 1):
        title, artist, era, date, role = t['title'], t['artist'], t['era_clean'], t['date'], t['kanye_role']
        print(f"[{i:>2}/{len(TARGETS)}] {title}  ({artist}, {role})", flush=True)

        # Search query strategies
        queries = [f"{artist} {title}", f"Kanye West {title}", title]

        hit, last_err = None, None
        for q in queries:
            try:
                hits = genius_search(token, q)
                hit = find_best_hit(hits, title, artist)
                if hit: break
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:100]}"
                time.sleep(2)
            time.sleep(0.4)

        if not hit:
            print(f"      NOT FOUND ({last_err or 'no good match'})")
            fetch_log.append({'track_title': title, 'status': 'not_found',
                              'detail': last_err or 'no good match'})
            time.sleep(random.uniform(args.min_delay, args.max_delay))
            continue

        try:
            full_lyrics = genius_fetch_lyrics(hit['url'])
        except Exception as e:
            print(f"      LYRICS FETCH FAILED: {e}")
            fetch_log.append({'track_title': title, 'status': 'lyrics_fetch_failed',
                              'detail': str(e)[:120]})
            time.sleep(random.uniform(args.min_delay, args.max_delay))
            continue

        if not full_lyrics or len(full_lyrics) < 50:
            print(f"      EMPTY LYRICS PAGE")
            fetch_log.append({'track_title': title, 'status': 'empty_lyrics', 'detail': hit['url']})
            time.sleep(random.uniform(args.min_delay, args.max_delay))
            continue

        kanye_verses, method = extract_kanye_verses(full_lyrics, role)
        cleaned_full = clean_lyrics_text(full_lyrics)
        full_wc = len(re.findall(r"[a-zA-Z']+", cleaned_full))
        kanye_wc = len(re.findall(r"[a-zA-Z']+", kanye_verses))

        if kanye_wc == 0 and role == 'feature':
            print(f"      NO KANYE CONTENT EXTRACTED ({method}) — feature with no markers found")
            fetch_log.append({'track_title': title, 'status': 'no_kanye_extracted', 'detail': hit['url']})
            time.sleep(random.uniform(args.min_delay, args.max_delay))
            continue

        new_rows.append({
            'track_title':              title,
            'album_or_era':             era,
            'release_date':             date,
            'track_type':               'Solo' if role == 'lead' else 'Feature',
            'primary_artist':           artist if role == 'feature' else 'Kanye West',
            'featured_artists':         '',
            'parser_note':              '',
            'genius_url':               hit['url'],
            'lyrics':                   cleaned_full,
            'lyrics_word_count':        full_wc,
            'original_word_count':      full_wc,
            'full_track_lyrics':        cleaned_full,
            'kanye_verses':             kanye_verses,
            'kanye_verse_word_count':   kanye_wc,
            'extraction_method':        f'targeted_{method}',
        })
        print(f"      OK  ({method}, {kanye_wc} Kanye / {full_wc} total words)")
        fetch_log.append({'track_title': title, 'status': f'ok_{method}',
                          'detail': f'kanye={kanye_wc} total={full_wc}'})

        time.sleep(random.uniform(args.min_delay, args.max_delay))

    # Save
    pd.DataFrame(new_rows).to_csv(out_dir / 'missing_tracks_supplement_2.csv', index=False)
    pd.DataFrame(fetch_log).to_csv(out_dir / 'supplement_fetch_log_2.csv', index=False)
    print(f"\nWrote data/missing_tracks_supplement_2.csv ({len(new_rows)} new tracks)")
    print(f"Wrote data/supplement_fetch_log_2.csv")

    statuses = pd.Series([r['status'].split('_')[0] for r in fetch_log]).value_counts()
    print("\nFetch summary:")
    for status, count in statuses.items():
        print(f"  {status:<15} {count}")

    if args.dry_run:
        print("\n--dry-run: skipping merge")
        return 0

    if new_rows:
        master_path = Path(args.master)
        if master_path.exists():
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup = master_path.with_name(f"{master_path.stem}.backup_{ts}.csv")
            shutil.copy2(master_path, backup)
            print(f"\nBacked up {master_path} -> {backup}")

        master_df = pd.read_csv(master_path)
        supplement = pd.DataFrame(new_rows)
        for c in master_df.columns:
            if c not in supplement.columns: supplement[c] = ''
        supplement = supplement[master_df.columns]
        merged = pd.concat([master_df, supplement], ignore_index=True)
        merged.to_csv(master_path, index=False)
        print(f"Merged. {master_path} now has {len(merged)} rows ({len(supplement)} new).")
        print("\nNext: re-run scripts 01-04 in order.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
