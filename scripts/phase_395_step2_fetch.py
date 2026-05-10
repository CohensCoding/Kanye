"""
Phase 3.95 Step 2 — fetch Genius lyrics for Phase 3.95 clean targets only.

Does not fetch no AI (metadata disagreement — handled outside this script via unresolved CSV).

Theraflu lineage (Resolution A):
  - genius_url forced to kanye-west-and-dj-khaled-cold-lyrics (not Chris Brown page)
  - track_title uses discography form: Theraflu/Way Too Cold
  - parser_note set per Resolution A

Reads:  data/phase_3_95_target_list.csv
Writes: data/phase_395_step2_fetch_log.csv
        data/phase_395_fetched_payloads.jsonl
Cache:  /tmp/genius_cache/*.txt (raw lyric text)

Requires GENIUS_ACCESS_TOKEN or GENIUS_API_TOKEN in .env (no hardcoded fallback;
~1 req/sec between Genius hits). Never log raw tokens or Authorization headers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from importlib.util import module_from_spec, spec_from_file_location

_spec = spec_from_file_location("targeted_fetch", PROJECT_ROOT / "scripts" / "00d_targeted_fetch.py")
_mod = module_from_spec(_spec)
assert _spec.loader
_spec.loader.exec_module(_mod)

genius_search = _mod.genius_search
genius_fetch_lyrics = _mod.genius_fetch_lyrics
clean_lyrics_text = _mod.clean_lyrics_text
extract_kanye_verses = _mod.extract_kanye_verses
find_best_hit = _mod.find_best_hit
USER_AGENT = _mod.USER_AGENT

COLD_LYRICS_URL = "https://genius.com/Kanye-west-and-dj-khaled-cold-lyrics"
FREAKY_GIRL_FALLBACK_URL = (
    "https://genius.com/Lil-pump-and-kanye-west-i-love-it-freaky-girl-edit-lyrics"
)
THERAFLU_PARSER_NOTE = (
    "Lyrics fetched from kanye-west-and-dj-khaled-cold-lyrics; track also released "
    "under aliases Way Too Cold and Cold."
)
CACHE_DIR = Path("/tmp/genius_cache")


def _load_genius_token() -> str:
    """Bearer token from environment only (.env via load_dotenv)."""
    raw = (os.getenv("GENIUS_ACCESS_TOKEN") or os.getenv("GENIUS_API_TOKEN") or "").strip()
    return raw.strip('"').strip("'")


def _parse_genius_url(detail: str) -> str | None:
    m = re.search(r"genius_url=(https://genius\.com/[^\s|]+)", detail or "")
    return m.group(1).rstrip(",") if m else None


def _slug_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def _save_cache(url: str, text: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_slug_from_url(url)}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _search_pick(token: str, title: str, artist: str, queries: list[str]) -> dict | None:
    last_err = None
    for q in queries:
        try:
            hits = genius_search(token, q)
            hit = find_best_hit(hits, title, artist)
            if hit:
                return hit
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.05)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-csv", default=str(PROJECT_ROOT / "data" / "phase_3_95_target_list.csv"))
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    token = _load_genius_token()
    if not token:
        print(
            "ERROR: Set GENIUS_ACCESS_TOKEN or GENIUS_API_TOKEN in .env (token=*** if debugging)",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(args.target_csv)
    rows_out = []
    payloads = []

    for _, row in df.iterrows():
        title = str(row["title"]).strip()
        year = int(row["year"]) if pd.notna(row.get("year")) else ""
        detail = str(row.get("prior_attempt_detail") or "")

        genius_url = None
        parser_note = ""
        fetch_title = title
        primary_for_search = "Kanye West"
        kanye_role = "lead"

        if title == "Theraflu":
            genius_url = COLD_LYRICS_URL
            fetch_title = "Theraflu/Way Too Cold"
            parser_note = THERAFLU_PARSER_NOTE
            primary_for_search = "Kanye West"
        elif title == "FourFiveSeconds":
            genius_url = _parse_genius_url(detail)
            primary_for_search = "Rihanna"
        elif title == "I Love It":
            genius_url = _parse_genius_url(detail)
            primary_for_search = "Lil Pump"
        else:
            genius_url = _parse_genius_url(detail)

        if title == "Freaky Girl Edit" and not genius_url:
            hit = _search_pick(
                token,
                "Freaky Girl",
                "Kanye West",
                ["Kanye West Freaky Girl Edit", "Kanye West freaky girl", "I Love It Freaky Girl Edit Kanye"],
            )
            genius_url = (hit or {}).get("url") or ""
            if not genius_url:
                # API match scoring can miss; direct slug verified May 2026.
                genius_url = FREAKY_GIRL_FALLBACK_URL

        if title == "Billie Jean (2008 Kanye West Mix)" and not genius_url:
            hit = _search_pick(
                token,
                "Billie Jean",
                "Michael Jackson",
                [
                    "Billie Jean 2008 Kanye West Mix",
                    "Michael Jackson Billie Jean Kanye West remix",
                    "Billie Jean Kanye West remix Thriller 25",
                ],
            )
            if not hit:
                rows_out.append(
                    {
                        "track_title": title,
                        "year": year,
                        "status": "search_failed",
                        "genius_url": "",
                        "detail": "Genius search returned no hit",
                    }
                )
                time.sleep(1.05)
                continue
            genius_url = hit["url"]

        if not genius_url:
            rows_out.append(
                {
                    "track_title": title,
                    "year": year,
                    "status": "skipped_no_url",
                    "genius_url": "",
                    "detail": "No genius_url in supplement detail and no search override",
                }
            )
            continue

        try:
            raw = genius_fetch_lyrics(genius_url)
        except Exception as e:
            rows_out.append(
                {
                    "track_title": fetch_title,
                    "year": year,
                    "status": "lyrics_fetch_failed",
                    "genius_url": genius_url,
                    "detail": str(e)[:200],
                }
            )
            time.sleep(1.05)
            continue

        _save_cache(genius_url, raw)
        cleaned = clean_lyrics_text(raw)
        extract_role = kanye_role
        if title == "Billie Jean (2008 Kanye West Mix)":
            # Michael-led remix: require explicit Kanye section markers (same as features).
            # If Genius transcribes only MJ vocals with no Kanye tags, word count stays 0 → production-only.
            extract_role = "feature"
        kanye_verses, method = extract_kanye_verses(raw, extract_role)
        kanye_wc = len(re.findall(r"[a-zA-Z']+", kanye_verses))
        full_wc = len(re.findall(r"[a-zA-Z']+", cleaned))

        billie_status = ""
        if title == "Billie Jean (2008 Kanye West Mix)":
            if method == "feature_no_kanye_markers" or kanye_wc == 0:
                billie_status = "EXCLUDED — production-only contribution (no Kanye-tagged verse on Genius)"
            else:
                billie_status = "has_kanye_markers_or_extracted_content"

        status = "ok"
        if billie_status.startswith("EXCLUDED"):
            status = "excluded_production_only"
        elif billie_status.startswith("ambiguous"):
            status = "ambiguous_billie_jean"

        rows_out.append(
            {
                "track_title": fetch_title,
                "year": year,
                "status": status,
                "genius_url": genius_url,
                "detail": f"{method} kanye_wc={kanye_wc} full_wc={full_wc} {billie_status}".strip(),
                "parser_note": parser_note,
            }
        )

        payloads.append(
            {
                "catalog_track_title": title,
                "track_title": fetch_title,
                "year": year,
                "genius_url": genius_url,
                "parser_note": parser_note,
                "extraction_method": method,
                "kanye_verse_word_count": kanye_wc,
                "full_lyrics_word_count": full_wc,
                "kanye_verses": kanye_verses,
                "full_lyrics": cleaned,
                "billie_jean_note": billie_status,
            }
        )

        time.sleep(1.05)

    log_path = PROJECT_ROOT / "data" / "phase_395_step2_fetch_log.csv"
    pd.DataFrame(rows_out).to_csv(log_path, index=False)

    jsonl_path = PROJECT_ROOT / "data" / "phase_395_fetched_payloads.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for obj in payloads:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {log_path}")
    print(f"Wrote {jsonl_path} ({len(payloads)} payloads)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
