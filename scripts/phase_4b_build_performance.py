#!/usr/bin/env python3
"""
Phase 4b — Build data/kanye_track_performance.csv and data/kanye_track_sonic_derived.csv.

Spotify: client-credentials only; GET /v1/search + GET /v1/tracks (NO /audio-features).

Kworb: cumulative streams from Kanye West + ¥$ song tables (Spotify IDs in anchors).

Billboard Hot 100 peak: Wikipedia «Kanye West singles discography» wikitables (US column).

Optional enrichment (weeks / debut date): cached Wikipedia song articles + regex.

RIAA strings: parsed from Certifications column cells when present (singles-oriented tables).

Caching: /tmp/performance_cache/<sha256(url)>.

Rate limits: Spotify ~5 req/s; kworb.net / wikipedia.org ≥2s between requests.

Requires SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET in .env (never logged).

Low Spotify fuzzy match (<85 RapidFuzz token_set_ratio): exit code 2 unless
--accept-low-confidence (writes rows as not_on_spotify with parser_note).

Spotify /v1/search errors: HTTP 400 → skip row with parser_note (no crash); 429 / 5xx →
retry (Retry-After / exponential backoff); HTTP 401 → hard fail (SpotifyAuthError).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_lib
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from dotenv import load_dotenv
from rapidfuzz import fuzz
from requests import Response

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = PROJECT_ROOT / "data" / "kanye_track_features.csv"
CLEANED_CSV = PROJECT_ROOT / "data" / "kanye_cleaned.csv"
OUT_PERF = PROJECT_ROOT / "data" / "kanye_track_performance.csv"
OUT_SONIC = PROJECT_ROOT / "data" / "kanye_track_sonic_derived.csv"
LOG_MD = PROJECT_ROOT / "data" / "phase_4b_log.md"

CACHE_ROOT = Path("/tmp/performance_cache")
TOKEN_CACHE = CACHE_ROOT / "spotify_token.json"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15 Phase4bBot/1.0"
)

KWORB_PAGES = [
    "https://kworb.net/spotify/artist/5K4W6rqBFWDnAN6FQUkS6x_songs.html",
    "https://kworb.net/spotify/artist/4xPQFgDA5M2xa0ZGo5iIsv_songs.html",
]

WIKI_KANYE_SINGLES = "https://en.wikipedia.org/wiki/Kanye_West_singles_discography"

DOMAIN_LAST: dict[str, float] = defaultdict(float)

CRITICAL_ALBUM_SUBSTRINGS: list[tuple[str, tuple[str, ...]]] = [
    ("Watch the Throne with Jay-Z", ("watch the throne",)),
    ("Kids See Ghosts with Kid Cudi", ("kids see ghosts",)),
    ("Vultures 1 — ¥$ with Ty Dolla $ign", ("vultures 1",)),
    ("Vultures 2 — ¥$", ("vultures 2",)),
    ("The College Dropout", ("college dropout",)),
    ("Late Registration", ("late registration",)),
    ("Graduation", ("graduation",)),
    ("808s & Heartbreak", ("808s", "heartbreak")),
    ("My Beautiful Dark Twisted Fantasy", ("beautiful dark twisted fantasy", "mbdtf")),
    ("Yeezus", ("yeezus",)),
    ("The Life of Pablo", ("life of pablo", "tlop")),
    ("Jesus Is King", ("jesus is king",)),
    ("Donda", ("donda",)),
    ("Donda 2", ("donda 2",)),
    ("Bully", ("bully",)),
]

FUZZ_THRESHOLD = 85.0
SPOTIFY_GAP = 0.21

_spotify_last = 0.0

MSG_SPOTIFY_400 = (
    "Spotify search returned 400 — query may have failed URL encoding or "
    "contained unsupported characters"
)


class SpotifyAuthError(RuntimeError):
    """Invalid or expired Spotify client credentials (HTTP 401)."""


def _retry_after_seconds(resp: Response, *, cap: float = 30.0) -> float:
    """Spotify may send large Retry-After during outages; cap avoids multi-hour bulk stalls."""
    raw = (resp.headers.get("Retry-After") or "").strip()
    if not raw:
        return min(3.0, cap)
    try:
        return min(max(0.5, float(raw)), cap)
    except ValueError:
        return min(5.0, cap)


def spotify_search(
    token: str,
    query: str,
    *,
    track_title: str,
    limit: int = 15,
) -> tuple[dict[str, Any], str | None]:
    """
    GET /v1/search (track). Returns (payload, parser_note).

    parser_note is set when the search must be skipped for this row (HTTP 400,
    exhausted 429/5xx retries, or other client errors). Caller writes not_on_spotify.

    Raises SpotifyAuthError on HTTP 401 (credentials).
    """
    empty: dict[str, Any] = {"tracks": {"items": []}}
    params = {"q": query, "type": "track", "limit": limit}
    max_retries = 3  # after first failure → up to 4 attempts total

    for attempt in range(max_retries + 1):
        spotify_throttle()
        try:
            r = requests.get(
                "https://api.spotify.com/v1/search",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=60,
            )
        except requests.exceptions.RequestException as exc:
            if attempt >= max_retries:
                print(
                    f"[phase4b] Spotify search network error track_title={track_title!r}: {exc!r}",
                    file=sys.stderr,
                )
                return empty, f"Spotify search network error after retries: {exc!r}"
            time.sleep(2**attempt)
            continue
        sc = r.status_code
        if sc == 401:
            raise SpotifyAuthError(
                "Spotify API returned 401 Unauthorized — check SPOTIFY_CLIENT_ID and "
                "SPOTIFY_CLIENT_SECRET in .env (client credentials must be valid)."
            )
        if sc == 400:
            print(
                f"[phase4b] Spotify 400 Bad Request search track_title={track_title!r} query={query!r}",
                file=sys.stderr,
            )
            return empty, MSG_SPOTIFY_400
        if sc == 429:
            if attempt >= max_retries:
                print(
                    f"[phase4b] Spotify 429 rate limit after {max_retries} retries "
                    f"track_title={track_title!r} query={query!r}",
                    file=sys.stderr,
                )
                return empty, "Spotify search rate-limited (429) after retries"
            wait = _retry_after_seconds(r)
            time.sleep(wait)
            continue
        if 500 <= sc < 600:
            if attempt >= max_retries:
                print(
                    f"[phase4b] Spotify {sc} after {max_retries} retries "
                    f"track_title={track_title!r}",
                    file=sys.stderr,
                )
                return empty, f"Spotify search server error {sc} after retries"
            time.sleep(2**attempt)
            continue
        if sc == 200:
            try:
                return r.json(), None
            except ValueError:
                print(
                    f"[phase4b] Spotify 200 but invalid JSON track_title={track_title!r}",
                    file=sys.stderr,
                )
                return empty, "Spotify search returned invalid JSON"

        print(
            f"[phase4b] Spotify HTTP {sc} search track_title={track_title!r} query={query!r}",
            file=sys.stderr,
        )
        return empty, f"Spotify search failed (HTTP {sc})"

    return empty, "Spotify search exhausted retries"


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def cached_get(url: str, min_interval: float = 2.0) -> str:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    path = CACHE_ROOT / key
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    dom = _domain(url)
    wait = min_interval - (time.time() - DOMAIN_LAST[dom])
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=90)
    DOMAIN_LAST[dom] = time.time()
    r.raise_for_status()
    txt = r.text
    path.write_text(txt, encoding="utf-8")
    return txt


def spotify_throttle() -> None:
    global _spotify_last
    gap = SPOTIFY_GAP - (time.time() - _spotify_last)
    if gap > 0:
        time.sleep(gap)
    _spotify_last = time.time()


def load_spotify_token(client_id: str, client_secret: str) -> str:
    now = time.time()
    if TOKEN_CACHE.exists():
        try:
            meta = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
            exp = float(meta.get("expires_at", 0))
            tok = (meta.get("access_token") or "").strip()
            if tok and now < exp - 60:
                return tok
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    spotify_throttle()
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(
            "Spotify token request failed "
            f"HTTP {r.status_code}: {r.text[:500]!r} "
            "(check SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET in .env)"
        )
    payload = r.json()
    tok = payload["access_token"]
    ttl = int(payload.get("expires_in", 3600))
    TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE.write_text(
        json.dumps({"access_token": tok, "expires_at": now + ttl}),
        encoding="utf-8",
    )
    return tok


def normalize_title(s: str) -> str:
    s = html_lib.unescape(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().strip('"“”«»').lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\(.*?\)", "", s).strip()
    s = re.sub(r"\[.*?\]", "", s).strip()
    return s


def album_critical_name(album_or_era: str) -> str:
    raw = (album_or_era or "").strip()
    if not raw:
        return ""
    low = raw.lower().strip()
    if low in {"ye", "ye (album)", "kanye west ye"}:
        return "ye"
    if low == "ye":
        return "ye"
    for canon, needles in CRITICAL_ALBUM_SUBSTRINGS:
        if any(n in low for n in needles):
            return canon
    return raw


def build_spotify_search_query(track_title: str, album_or_era: str) -> str:
    """
    Spotify search `q` grammar requires quoted strings for multi-token field values.
    Unquoted `track:All Falls Down` → HTTP 400; use track:\"…\" artist:\"…\" instead.
    """
    tq = track_title.replace('"', '\\"')
    hint = collab_album_search_hint(album_or_era)
    if hint:
        return f'track:"{tq}" {hint}'
    return f'track:"{tq}" artist:"Kanye West"'


def collab_album_search_hint(album_or_era: str) -> str | None:
    low = (album_or_era or "").lower()
    if "watch the throne" in low:
        return "Kanye West JAY-Z"
    if "kids see ghosts" in low:
        return "Kanye West Kid Cudi"
    if "vultures" in low or "\u00a5" in (album_or_era or ""):
        return "\u00a5$"
    return None


def artists_blob(track: dict[str, Any]) -> str:
    parts: list[str] = []
    for a in track.get("artists") or []:
        nm = (a.get("name") or "").lower()
        if nm:
            parts.append(nm)
    return " ".join(parts)


def artist_gate_ok(track: dict[str, Any], album_or_era: str) -> bool:
    blob = artists_blob(track)
    low = (album_or_era or "").lower()
    if "watch the throne" in low:
        return "kanye" in blob and ("jay-z" in blob or "jay z" in blob)
    if "kids see ghosts" in low:
        return "cudi" in blob and "kanye" in blob
    if "vultures" in low or "\u00a5" in (album_or_era or ""):
        return ("\u00a5" in blob) or ("ty dolla" in blob) or ("kanye" in blob)
    return "kanye" in blob


def spotify_tracks_batch(token: str, ids: list[str]) -> list[dict[str, Any] | None]:
    """GET /v1/tracks in chunks of 50; 429/5xx retries; 401 raises SpotifyAuthError."""
    out: list[dict[str, Any] | None] = []
    max_retries = 3
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        for attempt in range(max_retries + 1):
            spotify_throttle()
            try:
                r = requests.get(
                    "https://api.spotify.com/v1/tracks",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"ids": ",".join(chunk)},
                    timeout=60,
                )
            except requests.exceptions.RequestException as exc:
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"Spotify /v1/tracks network error after retries: {exc!r}"
                    ) from exc
                time.sleep(2**attempt)
                continue
            sc = r.status_code
            if sc == 401:
                raise SpotifyAuthError(
                    "Spotify API returned 401 Unauthorized while fetching /v1/tracks — "
                    "check SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."
                )
            if sc == 429:
                if attempt >= max_retries:
                    raise RuntimeError(
                        "Spotify /v1/tracks rate-limited (429) after retries; "
                        f"chunk starting {chunk[0]!r}"
                    )
                time.sleep(_retry_after_seconds(r, cap=60.0))
                continue
            if 500 <= sc < 600:
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"Spotify /v1/tracks server error {sc} after retries; "
                        f"chunk starting {chunk[0]!r}"
                    )
                time.sleep(2**attempt)
                continue
            if sc != 200:
                raise RuntimeError(f"Spotify /v1/tracks unexpected HTTP {sc}: {r.text[:300]!r}")
            tracks = (r.json() or {}).get("tracks") or []
            out.extend(tracks)
            break
    return out


def pick_best_spotify_match(
    local_title: str,
    album_or_era: str,
    search_payload: dict[str, Any],
) -> tuple[str | None, float, str | None]:
    items = ((search_payload.get("tracks") or {}).get("items")) or []
    best_id: str | None = None
    best_score = -1.0
    best_name: str | None = None
    nt = normalize_title(local_title)
    for it in items:
        if not it or not it.get("id"):
            continue
        name = it.get("name") or ""
        if not artist_gate_ok(it, album_or_era):
            continue
        score = float(fuzz.token_set_ratio(nt, normalize_title(name)))
        if score > best_score:
            best_score = score
            best_id = it["id"]
            best_name = name
    return best_id, best_score, best_name


def parse_kworb_songs_html(html: str) -> tuple[dict[str, int], dict[str, str], str | None]:
    """sid -> streams, sid -> anchor title, last-updated ISO date."""
    soup = BeautifulSoup(html, "lxml")
    m = re.search(r"Last updated:\s*(\d{4}/\d{2}/\d{2})", html)
    lu = m.group(1).replace("/", "-") if m else None
    streams: dict[str, int] = {}
    titles: dict[str, str] = {}
    for a in soup.select('td.text a[href*="open.spotify.com/track/"]'):
        href = a.get("href") or ""
        m2 = re.search(r"/track/([A-Za-z0-9]+)", href)
        if not m2:
            continue
        sid = m2.group(1)
        row = a.find_parent("tr")
        if not row:
            continue
        tds = row.find_all("td")
        if len(tds) < 2:
            continue
        raw = tds[1].get_text(strip=True).replace(",", "")
        if raw.isdigit():
            streams[sid] = int(raw)
            titles[sid] = re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()
    return streams, titles, lu


def merge_kworb() -> tuple[dict[str, int], dict[str, str], str | None]:
    merged_s: dict[str, int] = {}
    merged_t: dict[str, str] = {}
    dates: list[str] = []
    for url in KWORB_PAGES:
        html = cached_get(url, min_interval=2.0)
        part_s, part_t, lu = parse_kworb_songs_html(html)
        merged_s.update(part_s)
        merged_t.update(part_t)
        if lu:
            dates.append(lu)
    return merged_s, merged_t, max(dates) if dates else None


def strip_title_from_row_th(th: Tag) -> tuple[str, str | None]:
    wiki = None
    song_link = None
    for a in th.find_all("a", href=True):
        href = a.get("href") or ""
        if "/wiki/" not in href or "/wiki/File:" in href:
            continue
        slug = unquote(href.split("/wiki/", 1)[-1])
        if "#" in slug:
            slug = slug.split("#", 1)[0]
        if slug.startswith(("Billboard_", "Recording_", "List_of_", "Help:", "Template:", "Category:")):
            continue
        song_link = a
        wiki = slug.replace("_", " ")
        break

    if song_link is not None:
        txt = song_link.get_text(" ", strip=True)
    else:
        txt = th.get_text(" ", strip=True)
    txt = re.sub(r'^["“]+|["”]+$', "", txt).strip()
    return txt, wiki


def parse_peak_cell(text: str) -> int | None:
    text = re.sub(r"\s+", "", (text or "").strip())
    if not text or text in {"—", "–", "-"}:
        return None
    m = re.match(r"^(\d{1,3})", text)
    if not m:
        return None
    return int(m.group(1))


def row_has_year_column(tds: list) -> bool:
    if not tds:
        return False
    return bool(re.fullmatch(r"\d{4}", tds[0].get_text(strip=True)))


def find_riaa_cell(tds: list[Any]) -> Any | None:
    for td in tds:
        txt = td.get_text(" ", strip=True)
        if "RIAA" in txt:
            return td
    return None


def parse_wikipedia_singles_discography(html: str) -> tuple[list[dict[str, Any]], list[str]]:
    soup = BeautifulSoup(html, "lxml")
    rows_out: list[dict[str, Any]] = []
    peaks_by_norm: dict[str, list[int]] = defaultdict(list)
    ambiguities: list[str] = []

    for table in soup.select("table.wikitable"):
        header_row = None
        for tr in table.find_all("tr")[:6]:
            if tr.find("a", href=lambda h: h and "Billboard_Hot_100" in h):
                header_row = tr
                break
        if not header_row:
            continue

        hot_idx = None
        cells = header_row.find_all(["th", "td"])
        for i, cell in enumerate(cells):
            if cell.find("a", href=lambda h: h and "Billboard_Hot_100" in h):
                hot_idx = i
                break
        if hot_idx is None:
            continue

        for tr in table.find_all("tr"):
            th = tr.find("th", attrs={"scope": "row"})
            if not th:
                continue
            title, wiki_slug = strip_title_from_row_th(th)
            if not title:
                continue
            tds = tr.find_all("td")
            if not tds:
                continue
            offset = 1 if row_has_year_column(tds) else 0
            pi = offset + hot_idx
            if pi >= len(tds):
                continue
            peak = parse_peak_cell(tds[pi].get_text(" ", strip=True))
            cert_td = find_riaa_cell(tds)
            riaa_blob = cert_td.get_text(" ", strip=True) if cert_td else ""

            norm = normalize_title(title)
            rows_out.append(
                {
                    "title": title,
                    "norm": norm,
                    "peak": peak,
                    "wiki_slug": wiki_slug,
                    "riaa_blob": riaa_blob,
                }
            )
            if peak is not None:
                peaks_by_norm[norm].append(peak)

    for norm, plist in peaks_by_norm.items():
        u = sorted(set(plist))
        if len(u) > 1:
            ambiguities.append(f'Multiple distinct Hot 100 peaks for title variant {norm!r}: {u}')

    return rows_out, ambiguities


def parse_riaa_cert(blob: str) -> tuple[str | None, str | None]:
    if not blob or "RIAA" not in blob:
        return None, None
    m = re.search(r"RIAA[^:]*:\s*((?:\d+[×x]\s*)?(?:Diamond|Platinum|Gold)[^<\n]*)", blob, re.I)
    if not m:
        return None, None
    frag = re.sub(r"\s+", " ", m.group(1)).strip()
    frag = frag.replace("×", "x")
    cert = frag.split("[")[0].strip()
    dm = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        blob,
    )
    cert_date = None
    if dm:
        try:
            cert_date = datetime.strptime(dm.group(0), "%B %d, %Y").date().isoformat()
        except ValueError:
            cert_date = None
    return cert, cert_date


def wiki_article_plain_summary(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    core = soup.select_one("#mw-content-text .mw-parser-output") or soup
    for tag in core(["sup", "style", "script"]):
        tag.decompose()
    return core.get_text(" ", strip=True)


def extract_hot100_extras_from_wiki_html(html: str) -> tuple[int | None, str | None]:
    text = wiki_article_plain_summary(html)
    text = re.sub(r"\s+", " ", text)
    weeks_m = re.search(r"spent (\d+) weeks on the Hot 100", text, re.I)
    weeks = int(weeks_m.group(1)) if weeks_m else None
    debut = None
    dm = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        text,
    )
    if dm:
        try:
            debut = datetime.strptime(dm.group(0), "%B %d, %Y").date().isoformat()
        except ValueError:
            debut = None
    return weeks, debut


def build_chart_lookup(
    html: str,
    *,
    enrich_wiki: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    chart_rows, ambiguities = parse_wikipedia_singles_discography(html)

    best_by_norm: dict[str, dict[str, Any]] = {}
    for row in chart_rows:
        if row["peak"] is None:
            continue
        prev = best_by_norm.get(row["norm"])
        if prev is None or row["peak"] < prev["peak"]:
            best_by_norm[row["norm"]] = row

    extras_by_norm: dict[str, dict[str, Any]] = {}
    if enrich_wiki:
        fetched_slugs = 0
        for norm, row in best_by_norm.items():
            slug = row.get("wiki_slug")
            if not slug or fetched_slugs > 220:
                continue
            wiki_url = "https://en.wikipedia.org/wiki/" + slug.replace(" ", "_")
            whtml = cached_get(wiki_url, min_interval=2.0)
            fetched_slugs += 1
            wk, db = extract_hot100_extras_from_wiki_html(whtml)
            extras_by_norm[norm] = {"weeks": wk, "debut_date": db}

    lookup: dict[str, dict[str, Any]] = {}
    for norm, row in best_by_norm.items():
        ex = extras_by_norm.get(norm, {})
        blob = row.get("riaa_blob") or ""
        cert, cdate = parse_riaa_cert(blob)
        lookup[norm] = {
            "peak": row["peak"],
            "weeks": ex.get("weeks"),
            "debut_date": ex.get("debut_date"),
            "riaa_certification": cert,
            "riaa_cert_date": cdate,
            "riaa_blob": blob,
        }

    return chart_rows, lookup, ambiguities


def match_chart_row(
    track_title: str,
    chart_rows: list[dict[str, Any]],
    *,
    require_peak: bool,
    require_riaa: bool = False,
) -> dict[str, Any] | None:
    nt = normalize_title(track_title)
    rows = chart_rows
    if require_peak:
        rows = [r for r in rows if r["peak"] is not None]
    if require_riaa:
        rows = [r for r in rows if "RIAA" in (r.get("riaa_blob") or "")]
    best = None
    best_score = -1.0
    for row in rows:
        score = float(fuzz.token_set_ratio(nt, row["norm"]))
        if score > best_score:
            best_score = score
            best = row
    if best is None or best_score < FUZZ_THRESHOLD:
        return None
    return best


def is_collab_row(track_type: str, featured_artists: str) -> bool:
    tt = (track_type or "").strip().lower()
    if tt and tt != "solo":
        return True
    return bool((featured_artists or "").strip())


def compute_verse_share(
    track_type: str,
    featured_artists: str,
    full_track_lyrics: str,
    lyrics_word_count: Any,
    kanye_verse_word_count: Any,
) -> float | None:
    if not is_collab_row(track_type, featured_artists):
        return None
    if not (full_track_lyrics or "").strip():
        return None
    tw = int(float(lyrics_word_count or 0))
    kv = int(float(kanye_verse_word_count or 0))
    if tw <= 0:
        return None
    return round(kv / tw, 6)


def write_sonic_csv(
    perf_rows: list[dict[str, Any]],
    feats: dict[str, dict[str, Any]],
    cleaned_by_id: dict[str, dict[str, Any]],
    fetch_ts: str,
) -> list[str]:
    collab_null: list[str] = []
    sonic_rows: list[dict[str, Any]] = []

    for prow in perf_rows:
        tid = prow["track_id"]
        ft = feats.get(tid, {})
        cl = cleaned_by_id.get(tid, {})

        wc = int(float(ft.get("word_count") or 0))
        n_lines = int(float(ft.get("n_lines") or 0))
        n_verses = int(float(ft.get("n_verses") or 0))

        dm = prow.get("spotify_duration_ms")
        dur_s: float | None = None
        if dm not in ("", None):
            try:
                dur_s = round(float(dm) / 1000.0, 4)
            except (TypeError, ValueError):
                dur_s = None

        if dur_s and dur_s > 0:
            wps = round(wc / dur_s, 6)
            wpm = round(wps * 60.0, 6)
            lpm = round(n_lines / (dur_s / 60.0), 6)
        else:
            wps = wpm = lpm = None

        avg_ll = round(wc / n_lines, 6) if n_lines > 0 else None

        share = compute_verse_share(
            str(cl.get("track_type") or ""),
            str(cl.get("featured_artists") or ""),
            str(cl.get("full_track_lyrics") or ""),
            cl.get("lyrics_word_count"),
            cl.get("kanye_verse_word_count"),
        )
        if is_collab_row(str(cl.get("track_type") or ""), str(cl.get("featured_artists") or "")) and share is None:
            collab_null.append(tid)

        sonic_rows.append(
            {
                "track_id": tid,
                "duration_seconds": f"{dur_s:.4f}" if dur_s is not None else "",
                "line_count": n_lines,
                "verse_count": n_verses,
                "word_count": wc,
                "words_per_second": f"{wps:.6f}" if wps is not None else "",
                "words_per_minute": f"{wpm:.6f}" if wpm is not None else "",
                "lines_per_minute": f"{lpm:.6f}" if lpm is not None else "",
                "avg_line_length_words": f"{avg_ll:.6f}" if avg_ll is not None else "",
                "kanye_verse_share": f"{share:.6f}" if share is not None else "",
                "fetch_timestamp": fetch_ts,
            }
        )

    OUT_SONIC.parent.mkdir(parents=True, exist_ok=True)
    fnames = list(sonic_rows[0].keys())
    with OUT_SONIC.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader()
        w.writerows(sonic_rows)
    return collab_null


def fmt_perf_preview(rows: list[dict[str, Any]], limit: int = 20) -> str:
    cols = [
        "track_id",
        "track_title",
        "album",
        "spotify_track_id",
        "spotify_popularity",
        "spotify_duration_ms",
        "kworb_total_streams",
        "billboard_hot100_peak",
        "billboard_hot100_weeks_on_chart",
        "charted_hot100",
        "parser_note",
    ]
    lines = ["=== First {} rows (selected columns) ===".format(min(limit, len(rows)))]
    for r in rows[:limit]:
        cells = []
        for c in cols:
            val = str(r.get(c, ""))
            if c == "parser_note" and len(val) > 70:
                val = val[:67] + "..."
            cells.append(val.replace("\n", " "))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def null_stats(rows: list[dict[str, Any]], cols: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    n = len(rows)
    for c in cols:
        empties = sum(1 for r in rows if str(r.get(c, "")).strip() == "")
        out[c] = empties
    out["_row_count"] = n
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accept-low-confidence",
        action="store_true",
        help="Continue when Spotify fuzzy score < 85 (rows marked not_on_spotify).",
    )
    parser.add_argument(
        "--skip-wiki-enrich",
        action="store_true",
        help="Do not fetch song articles for weeks/debut (peak-only).",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    client_id = (os.getenv("SPOTIFY_CLIENT_ID") or "").strip().strip('"').strip("'")
    client_secret = (os.getenv("SPOTIFY_CLIENT_SECRET") or "").strip().strip('"').strip("'")
    if not client_id or not client_secret:
        print("ERROR: Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env", file=sys.stderr)
        return 2

    fetch_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    feats: dict[str, dict[str, Any]] = {}
    with FEATURES_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            feats[row["track_id"]] = row

    cleaned_by_id: dict[str, dict[str, Any]] = {}
    with CLEANED_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cleaned_by_id[row["track_id"]] = row

    wiki_html = cached_get(WIKI_KANYE_SINGLES, min_interval=2.0)
    chart_rows, chart_lookup, ambiguities = build_chart_lookup(
        wiki_html,
        enrich_wiki=not args.skip_wiki_enrich,
    )

    kworb_streams, kworb_titles, kworb_date = merge_kworb()

    try:
        token = load_spotify_token(client_id, client_secret)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    low_conf: list[dict[str, Any]] = []
    match_scores: dict[str, float] = {}
    id_for_track: dict[str, str | None] = {}
    search_fail_notes: dict[str, str] = {}

    for tid, frow in feats.items():
        title = frow["track_title"]
        cl = cleaned_by_id.get(tid, {})
        aoe = str(cl.get("album_or_era") or "")
        q = build_spotify_search_query(title, aoe)

        try:
            payload, search_note = spotify_search(token, q, track_title=title)
        except SpotifyAuthError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

        if search_note:
            search_fail_notes[tid] = search_note
            id_for_track[tid] = None
            match_scores[tid] = 0.0
            continue

        sid, score, matched_name = pick_best_spotify_match(title, aoe, payload)
        match_scores[tid] = score
        if sid is None or score < FUZZ_THRESHOLD:
            low_conf.append(
                {
                    "track_id": tid,
                    "track_title": title,
                    "best_score": score,
                    "matched_spotify_title": matched_name,
                    "query": q,
                }
            )
            id_for_track[tid] = None
            continue
        id_for_track[tid] = sid

    if low_conf and not args.accept_low_confidence:
        print("STOP: Spotify fuzzy match below 0.85 for one or more tracks.", file=sys.stderr)
        print(json.dumps(low_conf, indent=2), file=sys.stderr)
        return 2

    uniq_ids = sorted({i for i in id_for_track.values() if i})
    meta_by_id: dict[str, dict[str, Any]] = {}
    if uniq_ids:
        try:
            batch = spotify_tracks_batch(token, uniq_ids)
        except SpotifyAuthError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        for tid_meta, meta in zip(uniq_ids, batch):
            if meta:
                meta_by_id[tid_meta] = meta

    our_ids = {i for i in id_for_track.values() if i}
    kworb_orphans: list[tuple[str, str]] = []
    for sid in kworb_streams:
        if sid not in our_ids:
            kworb_orphans.append((sid, kworb_titles.get(sid, "?")))

    perf_rows: list[dict[str, Any]] = []
    for tid, frow in feats.items():
        title = frow["track_title"]
        cl = cleaned_by_id.get(tid, {})
        aoe = str(cl.get("album_or_era") or "")
        album_disp = album_critical_name(aoe)

        sid = id_for_track.get(tid)
        score = match_scores[tid]

        if sid is None or score < FUZZ_THRESHOLD:
            pnote = search_fail_notes.get(tid) or f"spotify_fuzzy_below_{int(FUZZ_THRESHOLD)};score={score:.1f}"
            perf_rows.append(
                {
                    "track_id": tid,
                    "track_title": title,
                    "album": album_disp,
                    "spotify_track_id": "not_on_spotify",
                    "spotify_popularity": "",
                    "spotify_duration_ms": "",
                    "spotify_explicit": "",
                    "spotify_isrc": "",
                    "kworb_total_streams": "",
                    "kworb_fetch_date": "",
                    "billboard_hot100_peak": "",
                    "billboard_hot100_weeks_on_chart": "",
                    "billboard_hot100_debut_date": "",
                    "charted_hot100": "False",
                    "riaa_certification": "",
                    "riaa_cert_date": "",
                    "fetch_timestamp": fetch_iso,
                    "parser_note": pnote,
                }
            )
            continue

        meta = meta_by_id.get(sid, {})
        pops = meta.get("popularity")
        dm = meta.get("duration_ms")
        explicit = meta.get("explicit")
        isrc = (meta.get("external_ids") or {}).get("isrc")

        streams = kworb_streams.get(sid)

        matched_bb = match_chart_row(title, chart_rows, require_peak=True)
        matched_riaa = (
            None
            if matched_bb and (matched_bb.get("riaa_blob") or "").find("RIAA") >= 0
            else match_chart_row(title, chart_rows, require_peak=False, require_riaa=True)
        )

        peak = weeks = debut = None
        riaa_cert = riaa_dt = ""
        charted = False
        if matched_bb:
            info = chart_lookup.get(matched_bb["norm"])
            if info:
                peak = info.get("peak")
                weeks = info.get("weeks")
                debut = info.get("debut_date")
                riaa_cert = info.get("riaa_certification") or ""
                riaa_dt = info.get("riaa_cert_date") or ""
                charted = peak is not None

        if (not riaa_cert) and matched_riaa:
            rc, rd = parse_riaa_cert(matched_riaa.get("riaa_blob") or "")
            riaa_cert = riaa_cert or (rc or "")
            riaa_dt = riaa_dt or (rd or "")

        notes: list[str] = []
        if streams is None:
            notes.append("kworb:not_on_kworb")

        perf_rows.append(
            {
                "track_id": tid,
                "track_title": title,
                "album": album_disp,
                "spotify_track_id": sid,
                "spotify_popularity": pops if pops is not None else "",
                "spotify_duration_ms": dm if dm is not None else "",
                "spotify_explicit": ("True" if explicit else "False") if explicit is not None else "",
                "spotify_isrc": isrc or "",
                "kworb_total_streams": streams if streams is not None else "",
                "kworb_fetch_date": kworb_date if streams is not None else "",
                "billboard_hot100_peak": peak if peak is not None else "",
                "billboard_hot100_weeks_on_chart": weeks if weeks is not None else "",
                "billboard_hot100_debut_date": debut if debut is not None else "",
                "charted_hot100": "True" if charted else "False",
                "riaa_certification": riaa_cert,
                "riaa_cert_date": riaa_dt,
                "fetch_timestamp": fetch_iso,
                "parser_note": "; ".join(notes),
            }
        )

    collab_null_share_ids = write_sonic_csv(perf_rows, feats, cleaned_by_id, fetch_iso)

    OUT_PERF.parent.mkdir(parents=True, exist_ok=True)
    pfnames = list(perf_rows[0].keys())
    with OUT_PERF.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pfnames)
        w.writeheader()
        w.writerows(perf_rows)

    n_tracks = len(perf_rows)
    n_spotify = sum(1 for r in perf_rows if r["spotify_track_id"] != "not_on_spotify")
    n_spotify_search_400 = sum(1 for n in search_fail_notes.values() if n == MSG_SPOTIFY_400)
    n_spotify_search_other = len(search_fail_notes) - n_spotify_search_400
    n_kworb = sum(1 for r in perf_rows if str(r.get("kworb_total_streams", "")).strip() != "")
    n_bb = sum(1 for r in perf_rows if r.get("charted_hot100") == "True")
    n_riaa = sum(1 for r in perf_rows if str(r.get("riaa_certification", "")).strip() != "")

    stat_cols = [
        "spotify_popularity",
        "spotify_duration_ms",
        "spotify_explicit",
        "spotify_isrc",
        "kworb_total_streams",
        "kworb_fetch_date",
        "billboard_hot100_peak",
        "billboard_hot100_weeks_on_chart",
        "billboard_hot100_debut_date",
        "riaa_certification",
        "riaa_cert_date",
        "parser_note",
    ]
    stats = null_stats(perf_rows, stat_cols)

    sonic_feats = list(csv.DictReader(OUT_SONIC.open(encoding="utf-8")))
    n_share_comp = sum(1 for r in sonic_feats if str(r.get("kanye_verse_share", "")).strip() != "")

    log_lines = [
        "# Phase 4b log",
        "",
        f"Generated: {fetch_iso}",
        "",
        "## Spotify setup",
        "",
        "Add to `.env` (never commit secrets):",
        "",
        "- `SPOTIFY_CLIENT_ID`",
        "- `SPOTIFY_CLIENT_SECRET`",
        "",
        f"HTTP/token cache directory: `{CACHE_ROOT}`",
        "",
        "## Rates",
        "",
        f"- Spotify: ~5 requests/sec (`sleep {SPOTIFY_GAP:.2f}s` between calls).",
        "- Kworb / Wikipedia: minimum 2s between hits per registrable domain.",
        "",
        "## Aggregate metrics",
        "",
        f"- Tracks: **{n_tracks}**",
        f"- Spotify matched (not `not_on_spotify`): **{n_spotify}** ({n_spotify / max(n_tracks,1):.1%})",
        f"- Rows skipped after Spotify search HTTP 400: **{n_spotify_search_400}**",
        f"- Rows skipped after other Spotify search failures: **{n_spotify_search_other}**",
        "- Run `scripts/phase_4b_ghost_tracks.py` before integration to audit non-catalog rows "
        "(`data/phase_4b_ghost_tracks.csv`).",
        f"- Kworb streams present: **{n_kworb}** ({n_kworb / max(n_tracks,1):.1%})",
        f"- Hot 100 charted (`charted_hot100=True`): **{n_bb}**",
        f"- Non-empty RIAA certification column: **{n_riaa}**",
        f"- `kanye_verse_share` computable (collab + full lyrics): **{n_share_comp}**",
        f"- Collab rows with null verse_share (backfill candidates): **{len(collab_null_share_ids)}**",
        "",
        "## Spotify fuzzy gate",
        "",
        f"RapidFuzz `token_set_ratio` ≥ **{int(FUZZ_THRESHOLD)}** plus contextual artist-token gate.",
        "",
        "## Kworb rows without catalog Spotify ID",
        "",
        f"{len(kworb_orphans)} Spotify IDs on merged Kworb pages were not used after matching.",
        "",
    ]
    for sid, ttl in sorted(kworb_orphans, key=lambda x: -kworb_streams.get(x[0], 0))[:40]:
        log_lines.append(f"- `{sid}` — {ttl}")
    if len(kworb_orphans) > 40:
        log_lines.append(f"- … ({len(kworb_orphans) - 40} more)")
    log_lines.extend(["", "## Wikipedia ambiguity / duplicate peaks", ""])
    log_lines.extend(f"- {x}" for x in ambiguities[:80])

    LOG_MD.parent.mkdir(parents=True, exist_ok=True)
    LOG_MD.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(fmt_perf_preview(perf_rows))
    print("\n=== Match rates ===")
    print(f"spotify_match_rate={n_spotify}/{n_tracks}")
    print(f"kworb_match_rate={n_kworb}/{n_tracks}")
    print(f"billboard_hot100_charted={n_bb}")
    print(f"riaa_non_empty_rows={n_riaa}")
    print("\n=== Null / empty counts (performance CSV) ===")
    for c in stat_cols:
        print(f"  empty({c})={stats[c]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
