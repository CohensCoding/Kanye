"""
00b_fetch_missing.py — Fetch missing tracks via the OFFICIAL Genius API.

This version bypasses lyricsgenius's default search (which hits a Cloudflare-blocked
unofficial endpoint and returns 403). Instead it:
  1. Searches via api.genius.com/search using the official Bearer-token API
  2. Fetches the lyrics page HTML directly via requests + a realistic browser User-Agent
  3. Parses lyrics out with BeautifulSoup (handles both modern and legacy Genius layouts)

Reads:   data/missing_tracks.csv   (output of 00_audit_completeness.py)
         kanye_verses_only.csv     (master file)
Writes:  data/missing_tracks_supplement.csv     (just the new tracks)
         data/supplement_fetch_log.csv          (per-track success/failure log)
         kanye_verses_only.csv                  (merged — old version backed up)

Usage:
  /usr/bin/python3 scripts/00b_fetch_missing.py
  /usr/bin/python3 scripts/00b_fetch_missing.py --dry-run        # don't merge
  /usr/bin/python3 scripts/00b_fetch_missing.py --limit 5        # test on 5 tracks first
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

try:
    from dotenv import load_dotenv
except ImportError:
    print("Run: pip install python-dotenv beautifulsoup4 requests")
    sys.exit(1)


# Realistic Safari UA — Cloudflare blocks default Python UA on genius.com
USER_AGENT = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
    '(KHTML, like Gecko) Version/17.5 Safari/605.1.15'
)

COLLAB_ALBUMS = {
    'JAY-Z & Kanye West', 'KIDS SEE GHOSTS', 'Kanye West & Ty Dolla $ign',
}

KANYE_MARKERS = re.compile(
    r'\[(?:Verse|Hook|Chorus|Bridge|Refrain|Pre-Chorus|Intro|Outro|Interlude)[^\]]*?'
    r'(?:Kanye West|Ye(?!es?zus)|Yeezy|Mr\.?\s*West)[^\]]*?\]',
    re.IGNORECASE,
)
SECTION_HEADER = re.compile(r'\[[^\]]+\]')


# =====================================================================
# Genius API + page scraping (replaces lyricsgenius)
# =====================================================================
def genius_search(token: str, query: str, max_results: int = 5) -> list[dict]:
    """Hit the OFFICIAL api.genius.com/search endpoint."""
    r = requests.get(
        'https://api.genius.com/search',
        params={'q': query},
        headers={'Authorization': f'Bearer {token}', 'User-Agent': USER_AGENT},
        timeout=20,
    )
    r.raise_for_status()
    hits = r.json().get('response', {}).get('hits', [])
    return [
        {
            'id':         h['result']['id'],
            'title':      h['result']['title'],
            'full_title': h['result']['full_title'],
            'url':        h['result']['url'],
            'artist':     h['result']['primary_artist']['name'],
        }
        for h in hits[:max_results]
    ]


def genius_fetch_lyrics(url: str) -> str:
    """GET the Genius lyrics page and parse lyrics out of the HTML."""
    r = requests.get(url, headers={'User-Agent': USER_AGENT}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')

    # Modern layout: <div data-lyrics-container="true">...</div> (one or more)
    containers = soup.find_all('div', attrs={'data-lyrics-container': 'true'})
    if containers:
        for c in containers:
            for br in c.find_all('br'):
                br.replace_with('\n')
        return '\n'.join(c.get_text() for c in containers).strip()

    # Legacy layout: <div class="lyrics"><p>...</p></div>
    legacy = soup.find('div', class_='lyrics')
    if legacy:
        for br in legacy.find_all('br'):
            br.replace_with('\n')
        return legacy.get_text().strip()

    return ''


def normalize_title(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())


def find_best_hit(hits: list[dict], wanted_title: str, wanted_artist: str) -> dict | None:
    """Pick the hit whose title best matches, with bonus for Kanye-related artist."""
    if not hits: return None
    wanted_n = normalize_title(wanted_title)
    wanted_artist_n = normalize_title(wanted_artist)

    scored = []
    for h in hits:
        title_n = normalize_title(h['title'])
        artist_n = normalize_title(h['artist'])
        score = 0
        if wanted_n == title_n:
            score += 100
        elif wanted_n in title_n or title_n in wanted_n:
            score += 80 if abs(len(wanted_n) - len(title_n)) <= 8 else 40
        if 'kanye' in artist_n or 'kidsseeghosts' in artist_n or 'tydolla' in artist_n or 'jayz' in artist_n:
            score += 30
        if wanted_artist_n and (wanted_artist_n in artist_n or artist_n in wanted_artist_n):
            score += 20
        scored.append((score, h))

    scored.sort(key=lambda x: -x[0])
    best_score, best = scored[0]
    return best if best_score >= 60 else None


# =====================================================================
# Lyrics cleaning + Kanye-verse extraction
# =====================================================================
def clean_lyrics_text(raw: str) -> str:
    if not raw: return ''
    raw = re.sub(r'^\d+\s+Contributors?\b.*?Lyrics', '', raw, flags=re.DOTALL)
    raw = re.sub(r'You might also like.*$', '', raw, flags=re.DOTALL)
    raw = re.sub(r'\d*Embed\s*$', '', raw)
    raw = re.sub(r'See .*? LiveGet tickets.*?(?=\n|$)', '', raw, flags=re.DOTALL)
    return raw.strip()


def extract_kanye_verses(full_lyrics: str, primary_artist: str) -> tuple[str, str]:
    lyrics = clean_lyrics_text(full_lyrics)
    is_collab = primary_artist in COLLAB_ALBUMS

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

    if not is_collab:
        other_marked = [body for header, body in sections
            if header and re.search(r'\[(?:Verse|Hook|Chorus|Bridge)[^\]]*?:[^\]]+\]', header)
            and not KANYE_MARKERS.search(header)]
        if not other_marked:
            return '\n\n'.join(body for _, body in sections), 'solo_no_markers'
        kanye_implicit = [body for header, body in sections
            if not (header and re.search(r'\[(?:Verse|Hook|Chorus|Bridge)[^\]]*?:[^\]]+\]', header)
                    and not KANYE_MARKERS.search(header or ''))]
        if kanye_implicit:
            return '\n\n'.join(kanye_implicit), 'solo_partial'

    return '', 'no_kanye_content'


# =====================================================================
# Main
# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',  default='data/missing_tracks.csv')
    ap.add_argument('--master', default='kanye_verses_only.csv')
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--min-delay', type=float, default=1.0)
    ap.add_argument('--max-delay', type=float, default=2.5)
    args = ap.parse_args()

    load_dotenv()
    token = os.getenv('GENIUS_API_TOKEN')
    if not token:
        print("ERROR: GENIUS_API_TOKEN not in .env"); return 1

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    missing = pd.read_csv(args.input)
    if args.limit: missing = missing.head(args.limit)
    print(f"Fetching {len(missing)} missing tracks via OFFICIAL Genius API\n")

    new_rows, fetch_log = [], []
    for i, row in missing.iterrows():
        title, artist = row['track_title'], row['primary_artist']
        era, date = row['era_clean'], row['release_date']
        print(f"[{i+1:>3}/{len(missing)}] {title}  ({artist})", flush=True)

        # Try multiple search query variants
        queries = [f"Kanye West {title}", title]
        if artist not in {'Kanye West'} | COLLAB_ALBUMS:
            queries = [f"{artist} {title}"] + queries

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
            print(f"      LYRICS FETCH FAILED: {type(e).__name__}: {str(e)[:80]}")
            fetch_log.append({'track_title': title, 'status': 'lyrics_fetch_failed',
                              'detail': str(e)[:120]})
            time.sleep(random.uniform(args.min_delay, args.max_delay))
            continue

        if not full_lyrics or len(full_lyrics) < 50:
            print(f"      EMPTY LYRICS PAGE")
            fetch_log.append({'track_title': title, 'status': 'empty_lyrics',
                              'detail': hit['url']})
            time.sleep(random.uniform(args.min_delay, args.max_delay))
            continue

        kanye_verses, method = extract_kanye_verses(full_lyrics, artist)
        cleaned_full = clean_lyrics_text(full_lyrics)
        full_wc = len(re.findall(r"[a-zA-Z']+", cleaned_full))
        kanye_wc = len(re.findall(r"[a-zA-Z']+", kanye_verses))

        new_rows.append({
            'track_title':              title,
            'album_or_era':             era,
            'release_date':             date,
            'track_type':               'Solo' if artist == 'Kanye West' else 'Collaboration',
            'primary_artist':           artist,
            'featured_artists':         '',
            'parser_note':              '',
            'genius_url':               hit['url'],
            'lyrics':                   cleaned_full,
            'lyrics_word_count':        full_wc,
            'original_word_count':      full_wc,
            'full_track_lyrics':        cleaned_full,
            'kanye_verses':             kanye_verses,
            'kanye_verse_word_count':   kanye_wc,
            'extraction_method':        f'supplement_{method}',
        })
        print(f"      OK  ({method}, {kanye_wc} Kanye / {full_wc} total words)")
        fetch_log.append({'track_title': title, 'status': f'ok_{method}',
                          'detail': f'kanye={kanye_wc} total={full_wc}'})

        time.sleep(random.uniform(args.min_delay, args.max_delay))

    # Save outputs
    pd.DataFrame(new_rows).to_csv(out_dir / 'missing_tracks_supplement.csv', index=False)
    pd.DataFrame(fetch_log).to_csv(out_dir / 'supplement_fetch_log.csv', index=False)
    print(f"\nWrote data/missing_tracks_supplement.csv ({len(new_rows)} new tracks)")
    print(f"Wrote data/supplement_fetch_log.csv")

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
