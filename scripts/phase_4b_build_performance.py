#!/usr/bin/env python3
"""
Phase 4b — Build data/kanye_track_performance.csv and data/kanye_track_sonic_derived.csv.

Kworb: cumulative streams from Kanye West + ¥$ song tables; title-only fuzzy match
(RapidFuzz token_set_ratio ≥ 88) against anchor titles.

Billboard Hot 100: Wikipedia «Kanye West singles discography» wikitables (US column).

Optional enrichment: cached Wikipedia song articles for weeks, debut date, and infobox Length.

RIAA: parsed from Certifications column cells when present.

Caching: /tmp/performance_cache/<sha256(url)>.

Rate limits: kworb.net / wikipedia.org ≥ 2s between requests per registrable domain.

No authenticated external APIs (Spotify removed).

Kworb gate: if more than 20 tracks have best fuzzy score < 88 vs the merged Kworb list,
the script exits 2 and prints JSON for review. Re-run with `--continue-after-kworb-review`
after human review when many misses are expected (off-Kworb titles).

Wikipedia duration gate: when song-page enrichment is enabled, if fewer than 30 tracks
receive a parsed infobox Length, exit 2 (parser likely broken).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_lib
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = PROJECT_ROOT / "data" / "kanye_track_features.csv"
CLEANED_CSV = PROJECT_ROOT / "data" / "kanye_cleaned.csv"
OUT_PERF = PROJECT_ROOT / "data" / "kanye_track_performance.csv"
OUT_SONIC = PROJECT_ROOT / "data" / "kanye_track_sonic_derived.csv"
LOG_MD = PROJECT_ROOT / "data" / "phase_4b_log.md"
OUT_TITLE_COLLISIONS = PROJECT_ROOT / "data" / "phase_4b_title_collisions.csv"
OUT_KWORB_MISSES = PROJECT_ROOT / "data" / "phase_4b_kworb_misses.csv"

CACHE_ROOT = Path("/tmp/performance_cache")

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

# Billboard / RIAA table fuzzy match (unchanged behavior).
CHART_FUZZ_THRESHOLD = 85.0
# Kworb title-only (no Spotify artist cross-check).
KWORB_FUZZ_THRESHOLD = 88.0
MIN_WIKI_DURATION_COVERAGE = 30
MAX_KWORB_REVIEW_FAILS = 20


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


def normalize_title(s: str) -> str:
    s = html_lib.unescape(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u2032", "'")
    s = s.strip().strip('"“”«»').lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\(.*?\)", "", s).strip()
    s = re.sub(r"\[.*?\]", "", s).strip()
    return s


def norm_from_wiki_slug(slug: Optional[str]) -> str:
    """Lowercase slug text with parentheses kept (disambiguates «Number One» rows)."""
    if not slug:
        return ""
    s = unquote(slug.split("#")[0])
    s = s.replace("_", " ")
    s = html_lib.unescape(s)
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def chart_row_uid(row: dict[str, Any]) -> str:
    """Stable id for a Wikipedia singles row (slug when present)."""
    ws = (row.get("wiki_slug") or "").strip()
    if ws:
        return ws
    norm = row.get("norm") or ""
    ry = row.get("release_year")
    return f"{norm}##{ry if ry is not None else 'none'}"


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


def parse_kworb_songs_html(html: str) -> tuple[list[tuple[str, int]], str | None]:
    """(title, streams) per table row; last-updated ISO date from page header."""
    soup = BeautifulSoup(html, "lxml")
    m = re.search(r"Last updated:\s*(\d{4}/\d{2}/\d{2})", html)
    lu = m.group(1).replace("/", "-") if m else None
    rows_out: list[tuple[str, int]] = []
    for a in soup.select('td.text a[href*="open.spotify.com/track/"]'):
        row = a.find_parent("tr")
        if not row:
            continue
        tds = row.find_all("td")
        if len(tds) < 2:
            continue
        raw = tds[1].get_text(strip=True).replace(",", "")
        if not raw.isdigit():
            continue
        title = re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()
        if title:
            rows_out.append((title, int(raw)))
    return rows_out, lu


def merge_kworb() -> tuple[list[tuple[str, int]], str | None]:
    merged: list[tuple[str, int]] = []
    dates: list[str] = []
    for url in KWORB_PAGES:
        html = cached_get(url, min_interval=2.0)
        part, lu = parse_kworb_songs_html(html)
        merged.extend(part)
        if lu:
            dates.append(lu)
    return merged, max(dates) if dates else None


def repair_mojibake(s: str) -> str:
    """Kworb anchor text sometimes carries UTF-8 decoded as Latin-1 (e.g. MAMAâ\x80\x99S)."""
    if not s or ("â" not in s and "Ã" not in s):
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def kworb_fuzz_score(nt: str, kw_title: str) -> float:
    """
    token_set_ratio on normalized titles, plus a path that strips Spotify-style
    censorship asterisks so local «Niggas in Paris» can align with «Ni**as In Paris».
    """
    kwn = normalize_title(repair_mojibake(kw_title.strip()))
    s1 = float(fuzz.token_set_ratio(nt, kwn))
    s2 = float(
        fuzz.token_set_ratio(
            nt.replace("*", ""),
            kwn.replace("*", ""),
        )
    )
    return max(s1, s2)


def best_kworb_match(
    track_title: str, kworb_rows: list[tuple[str, int]]
) -> tuple[int | None, float, str | None]:
    """
    Returns (streams, best_score, best_kworb_title).
    Caller applies KWORB_FUZZ_THRESHOLD to decide whether streams is usable.
    On score ties, prefers higher stream counts.
    """
    nt = normalize_title(track_title)
    best_score = -1.0
    best_streams: int | None = None
    best_title: str | None = None
    for kw_title, streams in kworb_rows:
        sc = kworb_fuzz_score(nt, kw_title)
        if sc > best_score:
            best_score = sc
            best_streams = streams
            best_title = kw_title
        elif sc == best_score and best_streams is not None and streams > best_streams:
            best_streams = streams
            best_title = kw_title
    if best_score < 0:
        return None, 0.0, None
    return best_streams, best_score, best_title


def strip_title_from_row_th(th: Tag) -> tuple[str, str | None, str | None]:
    wiki = None
    song_link = None
    wiki_display_title: str | None = None
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
        wiki_display_title = (a.get("title") or "").strip() or None
        break

    if song_link is not None:
        txt = song_link.get_text(" ", strip=True)
    else:
        txt = th.get_text(" ", strip=True)
    txt = re.sub(r'^["“]+|["”]+$', "", txt).strip()
    return txt, wiki, wiki_display_title


def parse_peak_cell(text: str) -> int | None:
    text = re.sub(r"\s+", "", (text or "").strip())
    if not text or text in {"—", "–", "-"}:
        return None
    m = re.match(r"^(\d{1,3})", text)
    if not m:
        return None
    return int(m.group(1))


def find_riaa_cell(tds: list[Any]) -> Any | None:
    for td in tds:
        txt = td.get_text(" ", strip=True)
        if "RIAA" in txt:
            return td
    return None


def parse_wikipedia_singles_discography(html: str) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Parse singles wikitables. Carries `release_year` from a leading 4-digit year cell
    (including rowspan) so continuation rows (e.g. 2016 «Champions») inherit the year
    from «Famous»'s rowspan cell.
    """
    soup = BeautifulSoup(html, "lxml")
    rows_out: list[dict[str, Any]] = []
    peaks_by_uid: dict[str, list[int]] = defaultdict(list)
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

        active_year: int | None = None
        for tr in table.find_all("tr"):
            th = tr.find("th", attrs={"scope": "row"})
            if not th:
                continue
            title, wiki_slug, wiki_display_title = strip_title_from_row_th(th)
            if not title:
                continue
            tds = tr.find_all("td", recursive=False)
            if not tds:
                continue
            offset = 0
            release_year: int | None = active_year
            first_txt = tds[0].get_text(strip=True)
            if re.fullmatch(r"\d{4}", first_txt):
                active_year = int(first_txt)
                release_year = active_year
                offset = 1
            pi = offset + hot_idx
            if pi >= len(tds):
                continue
            peak = parse_peak_cell(tds[pi].get_text(" ", strip=True))
            all_tds = tr.find_all("td")
            cert_td = find_riaa_cell(all_tds)
            riaa_blob = cert_td.get_text(" ", strip=True) if cert_td else ""

            norm = normalize_title(title)
            norm_slug = norm_from_wiki_slug(wiki_slug) if wiki_slug else norm
            crow = {
                "title": title,
                "norm": norm,
                "norm_slug": norm_slug,
                "peak": peak,
                "wiki_slug": wiki_slug,
                "wiki_display_title": wiki_display_title,
                "release_year": release_year,
                "riaa_blob": riaa_blob,
            }
            rows_out.append(crow)
            if peak is not None:
                peaks_by_uid[chart_row_uid(crow)].append(peak)

    for uid, plist in peaks_by_uid.items():
        u = sorted(set(plist))
        if len(u) > 1:
            ambiguities.append(
                f"Multiple distinct Hot 100 peaks for chart row {uid!r}: {u}"
            )

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


def parse_length_to_seconds(text: str) -> int | None:
    """Infobox Length cell: '3:42', '3 : 41 (album version)', '3 minutes 42 seconds', etc."""
    if not text:
        return None
    t = re.sub(r"\[[^\]]*\]", "", text)
    # First timed segment only when multiple versions (e.g. album vs single).
    t = t.split("(")[0].strip()

    m = re.search(
        r"(\d+)\s*minutes?\s*(?:and\s*)?(\d+)\s*seconds?",
        t,
        re.I,
    )
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))

    # Allow spaces around ':' (MediaWiki duration spans: "3 : 41").
    m = re.search(r"\b(\d{1,3})\s*:\s*(\d{2})\b", t)
    if m:
        mins, secs = int(m.group(1)), int(m.group(2))
        if mins < 60 and secs < 60:
            return mins * 60 + secs
    return None


def extract_length_seconds_from_wiki_html(html: str) -> int | None:
    soup = BeautifulSoup(html, "lxml")
    infobox = soup.select_one("table.infobox") or soup.find(
        "table", class_=lambda c: c and "infobox" in " ".join(c).lower()
    )
    if not infobox:
        return None
    for tr in infobox.find_all("tr"):
        th = tr.find("th")
        td = tr.find("td")
        if not th or not td:
            continue
        lab = th.get_text(" ", strip=True).lower()
        if "length" not in lab and lab not in ("duration", "running time"):
            continue
        return parse_length_to_seconds(td.get_text(" ", strip=True))
    return None


ChartRowId = str


def build_chart_lookup(
    html: str,
    *,
    enrich_wiki: bool,
) -> tuple[list[dict[str, Any]], dict[ChartRowId, dict[str, Any]], list[str]]:
    chart_rows, ambiguities = parse_wikipedia_singles_discography(html)

    best_by_uid: dict[ChartRowId, dict[str, Any]] = {}
    for row in chart_rows:
        if row["peak"] is None:
            continue
        uid = chart_row_uid(row)
        prev = best_by_uid.get(uid)
        if prev is None or row["peak"] < prev["peak"]:
            best_by_uid[uid] = row

    extras_by_uid: dict[ChartRowId, dict[str, Any]] = {}
    if enrich_wiki:
        fetched_slugs = 0
        for uid, row in best_by_uid.items():
            slug = row.get("wiki_slug")
            if not slug or fetched_slugs > 220:
                continue
            wiki_url = "https://en.wikipedia.org/wiki/" + slug.replace(" ", "_")
            whtml = cached_get(wiki_url, min_interval=2.0)
            fetched_slugs += 1
            wk, db = extract_hot100_extras_from_wiki_html(whtml)
            dur = extract_length_seconds_from_wiki_html(whtml)
            extras_by_uid[uid] = {
                "weeks": wk,
                "debut_date": db,
                "duration_seconds": dur,
            }

    lookup: dict[ChartRowId, dict[str, Any]] = {}
    for uid, row in best_by_uid.items():
        ex = extras_by_uid.get(uid, {})
        blob = row.get("riaa_blob") or ""
        cert, cdate = parse_riaa_cert(blob)
        lookup[uid] = {
            "peak": row["peak"],
            "weeks": ex.get("weeks"),
            "debut_date": ex.get("debut_date"),
            "duration_seconds": ex.get("duration_seconds"),
            "riaa_certification": cert,
            "riaa_cert_date": cdate,
            "riaa_blob": blob,
        }

    return chart_rows, lookup, ambiguities


def find_attribution_era(
    title: str, chart_year: int, exclude_tid: str, feats: dict[str, dict[str, Any]]
) -> str | None:
    """Other corpus row with same display title whose year is closest to Wikipedia chart year."""
    best_era: str | None = None
    best_delta = 10**9
    for tid2, f2 in feats.items():
        if tid2 == exclude_tid or (f2.get("track_title") or "").strip() != title.strip():
            continue
        try:
            y2 = int(float(f2.get("year") or 0))
        except (TypeError, ValueError):
            continue
        d = abs(y2 - chart_year)
        if d < best_delta:
            best_delta = d
            best_era = (f2.get("era_clean") or "").strip() or None
    return best_era


def is_corpus_kanye_primary(pa_n: str) -> bool:
    """Primary artist column is Kanye on solo cuts (skip slug-artist gate for those)."""
    s = pa_n.replace(" ", "").lower()
    return "kanyewest" in s or pa_n.strip().lower() in ("kanye", "ye", "")


def match_chart_row(
    track_title: str,
    chart_rows: list[dict[str, Any]],
    *,
    require_peak: bool,
    require_riaa: bool = False,
    track_release_year: int | None = None,
    ignore_release_year: bool = False,
    corpus_primary: str = "",
    strict_year_match: bool = False,
) -> dict[str, Any] | None:
    nt = normalize_title(track_title)
    pa_n = normalize_title((corpus_primary or "").strip())
    rows = chart_rows
    if require_peak:
        rows = [r for r in rows if r["peak"] is not None]
    if require_riaa:
        rows = [r for r in rows if "RIAA" in (r.get("riaa_blob") or "")]
    best = None
    best_score = -1.0
    for row in rows:
        cry = row.get("release_year")
        weak_slug = "(song)" in ((row.get("wiki_slug") or "").lower())
        if not ignore_release_year and track_release_year is not None and cry is not None:
            if strict_year_match:
                if weak_slug:
                    if int(cry) != int(track_release_year):
                        continue
                elif abs(int(cry) - int(track_release_year)) > 1:
                    continue
            elif abs(int(cry) - int(track_release_year)) > 2:
                continue

        slug_n = row.get("norm_slug") or row["norm"]
        s_title = float(fuzz.token_set_ratio(nt, slug_n))
        if s_title < CHART_FUZZ_THRESHOLD:
            continue
        if pa_n:
            s_art = max(
                float(fuzz.token_set_ratio(pa_n, slug_n)),
                float(fuzz.partial_ratio(pa_n, slug_n)),
            )
        else:
            s_art = 0.0
        if pa_n and not is_corpus_kanye_primary(pa_n) and s_art < 55 and not weak_slug:
            continue
        score = s_title + 0.0001 * s_art
        if score > best_score:
            best_score = score
            best = row
    if best is None or best_score < CHART_FUZZ_THRESHOLD:
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

        ds_raw = prow.get("duration_seconds")
        dur_s: float | None = None
        if ds_raw not in ("", None):
            try:
                dur_s = float(int(str(ds_raw).strip()))
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
        "kworb_total_streams",
        "kworb_match_score",
        "billboard_hot100_peak",
        "billboard_hot100_weeks_on_chart",
        "charted_hot100",
        "duration_seconds",
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


PERF_FIELDNAMES = [
    "track_id",
    "track_title",
    "album",
    "kworb_total_streams",
    "kworb_match_score",
    "kworb_fetch_date",
    "billboard_hot100_peak",
    "billboard_hot100_weeks_on_chart",
    "billboard_hot100_debut_date",
    "charted_hot100",
    "riaa_certification",
    "riaa_cert_date",
    "duration_seconds",
    "fetch_timestamp",
    "parser_note",
]


def _merge_parser_note(existing: str, new: str) -> str:
    e = (existing or "").strip()
    if not new:
        return e
    if not e:
        return new
    if new in e:
        return e
    return f"{e}; {new}"


MANUAL_KWORB_NOTE = (
    "Manual confirmation of kworb match below fuzzy threshold; verified by lyric content review."
)


def apply_final_phase4b_corrections(
    perf_rows: list[dict[str, Any]],
    *,
    kworb_date: str,
) -> None:
    """Human-verified fixes after automated fuzzy merge (Phase 4b final)."""
    by_id = {r["track_id"]: r for r in perf_rows}

    tid_bully = "king_bully_build_up_cuck_iapw_controversy"
    if tid_bully in by_id:
        r = by_id[tid_bully]
        r["kworb_total_streams"] = ""
        r["kworb_match_score"] = ""
        r["kworb_fetch_date"] = ""
        note = (
            "Kworb match was attributed to Vultures 1 'King' (track from 2024); "
            "Bully-era 'King' is a different track and has no distinct kworb entry as of fetch date."
        )
        r["parser_note"] = _merge_parser_note(r.get("parser_note", ""), note)

    tid_college_no = "number_one_the_college_dropout"
    if tid_college_no in by_id:
        r = by_id[tid_college_no]
        r["billboard_hot100_peak"] = ""
        r["billboard_hot100_weeks_on_chart"] = ""
        r["billboard_hot100_debut_date"] = ""
        r["charted_hot100"] = "False"
        note = (
            "Chart data for 'Number One' attaches to Pharrell feat. Kanye 2006 release; "
            "captured on Late Registration corpus row, not this College Dropout album track."
        )
        r["parser_note"] = _merge_parser_note(r.get("parser_note", ""), note)

    tid_yee_san = "sanctified_yeezus"
    if tid_yee_san in by_id:
        r = by_id[tid_yee_san]
        r["billboard_hot100_peak"] = ""
        r["billboard_hot100_weeks_on_chart"] = ""
        r["billboard_hot100_debut_date"] = ""
        r["charted_hot100"] = "False"
        r["duration_seconds"] = ""
        r["parser_note"] = (
            "Chart data for 'Sanctified' attaches to Rick Ross feat. Kanye 2014 release; "
            "captured on Life of Pablo corpus row, not this Yeezus session track."
        )

    manual_kworb: dict[str, tuple[str, str]] = {
        "buy_you_a_drank_remix_graduation": (
            "9594701",
            "Buy U A Drank (Shawty Snappin') (feat. Kanye West) - Remix",
        ),
        "i_don_t_like_remix_cruel_summer_yeezus_build_up": ("289690852", "Don't Like.1"),
    }
    for tid, (streams, _kw_title) in manual_kworb.items():
        if tid not in by_id:
            continue
        r = by_id[tid]
        r["kworb_total_streams"] = streams
        r["kworb_match_score"] = "100.00"
        r["kworb_fetch_date"] = kworb_date
        r["parser_note"] = MANUAL_KWORB_NOTE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-wiki-enrich",
        action="store_true",
        help="Do not fetch song articles (no weeks/debut/duration from Wikipedia song pages).",
    )
    parser.add_argument(
        "--continue-after-kworb-review",
        action="store_true",
        help=(
            "Complete the run even when more than 20 tracks have best Kworb score < "
            f"{KWORB_FUZZ_THRESHOLD:g} (default: exit 2 with JSON). Use after reviewing the list."
        ),
    )
    args = parser.parse_args()

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

    kworb_rows, kworb_date = merge_kworb()

    kworb_review: list[dict[str, Any]] = []
    for tid, frow in feats.items():
        title = frow["track_title"]
        _streams, score, kw_title = best_kworb_match(title, kworb_rows)
        if score < KWORB_FUZZ_THRESHOLD:
            kworb_review.append(
                {
                    "track_id": tid,
                    "track_title": title,
                    "best_score": round(score, 2),
                    "best_kworb_title": kw_title,
                }
            )

    if len(kworb_review) > MAX_KWORB_REVIEW_FAILS and not args.continue_after_kworb_review:
        print(
            f"STOP: {len(kworb_review)} tracks have best Kworb fuzzy score < {KWORB_FUZZ_THRESHOLD:g} "
            f"(limit {MAX_KWORB_REVIEW_FAILS} before stop-and-ask). "
            "Re-run with --continue-after-kworb-review after human review.",
            file=sys.stderr,
        )
        print(json.dumps(kworb_review, indent=2), file=sys.stderr)
        return 2

    by_title_eras: dict[str, set[str]] = defaultdict(set)
    for f in feats.values():
        by_title_eras[f["track_title"]].add((f.get("era_clean") or "").strip())
    multi_era_titles = {t for t, es in by_title_eras.items() if len(es) >= 2}

    perf_rows: list[dict[str, Any]] = []
    kworb_miss_rows: list[dict[str, str]] = []

    for tid, frow in feats.items():
        title = frow["track_title"]
        cl = cleaned_by_id.get(tid, {})
        aoe = str(cl.get("album_or_era") or "")
        album_disp = album_critical_name(aoe)

        try:
            track_year = int(float(frow.get("year") or ""))
        except (TypeError, ValueError):
            track_year = None

        streams, k_score, kw_title = best_kworb_match(title, kworb_rows)
        if k_score >= KWORB_FUZZ_THRESHOLD and streams is not None:
            kw_streams_s = str(streams)
            kw_score_s = f"{k_score:.2f}"
            kw_date_s = kworb_date or ""
            kw_note = ""
        else:
            kw_streams_s = ""
            kw_score_s = ""
            kw_date_s = ""
            kw_note = f"no_kworb_match;best_kworb_title={kw_title!r};best_score={k_score:.2f}"
            kworb_miss_rows.append(
                {
                    "track_id": tid,
                    "track_title": title,
                    "era_clean": (frow.get("era_clean") or "").strip(),
                    "year": (frow.get("year") or "").strip(),
                    "best_kworb_match": kw_title or "",
                    "best_kworb_score": f"{k_score:.2f}",
                }
            )

        strict_year = title in multi_era_titles

        matched_bb = match_chart_row(
            title,
            chart_rows,
            require_peak=True,
            track_release_year=track_year,
            corpus_primary=str(cl.get("primary_artist") or ""),
            strict_year_match=strict_year,
        )
        matched_loose = match_chart_row(
            title,
            chart_rows,
            require_peak=True,
            ignore_release_year=True,
            corpus_primary=str(cl.get("primary_artist") or ""),
            strict_year_match=False,
        )
        matched_riaa = (
            None
            if matched_bb and (matched_bb.get("riaa_blob") or "").find("RIAA") >= 0
            else match_chart_row(
                title,
                chart_rows,
                require_peak=False,
                require_riaa=True,
                track_release_year=track_year,
                corpus_primary=str(cl.get("primary_artist") or ""),
                strict_year_match=strict_year,
            )
        )

        peak = weeks = debut = None
        riaa_cert = riaa_dt = ""
        charted = False
        duration_sec: int | None = None
        if matched_bb:
            ck: ChartRowId = chart_row_uid(matched_bb)
            info = chart_lookup.get(ck)
            if info:
                peak = info.get("peak")
                weeks = info.get("weeks")
                debut = info.get("debut_date")
                duration_sec = info.get("duration_seconds")
                riaa_cert = info.get("riaa_certification") or ""
                riaa_dt = info.get("riaa_cert_date") or ""
                charted = peak is not None

        if (not riaa_cert) and matched_riaa:
            rc, rd = parse_riaa_cert(matched_riaa.get("riaa_blob") or "")
            riaa_cert = riaa_cert or (rc or "")
            riaa_dt = riaa_dt or (rd or "")

        notes: list[str] = []
        if kw_note:
            notes.append(kw_note)
        if matched_bb is None and matched_loose and track_year is not None:
            cry = matched_loose.get("release_year")
            if cry is not None:
                wslug = "(song)" in ((matched_loose.get("wiki_slug") or "").lower())
                if strict_year:
                    mismatch = (wslug and int(cry) != int(track_year)) or (
                        not wslug and abs(int(cry) - int(track_year)) > 1
                    )
                else:
                    mismatch = abs(int(cry) - int(track_year)) > 2
                if mismatch:
                    other_era = find_attribution_era(title, int(cry), tid, feats)
                    if other_era:
                        notes.append(
                            f'Billboard chart data removed; chart attribution belongs to {other_era}\'s '
                            f'"{title}" track, not this era\'s track.'
                        )
                    else:
                        notes.append(
                            f'Billboard chart data removed; chart attribution belongs to the "{title}" '
                            f"Hot 100 listing for Wikipedia release year {cry}, not this era's track."
                        )

        perf_rows.append(
            {
                "track_id": tid,
                "track_title": title,
                "album": album_disp,
                "kworb_total_streams": kw_streams_s,
                "kworb_match_score": kw_score_s,
                "kworb_fetch_date": kw_date_s,
                "billboard_hot100_peak": peak if peak is not None else "",
                "billboard_hot100_weeks_on_chart": weeks if weeks is not None else "",
                "billboard_hot100_debut_date": debut if debut is not None else "",
                "charted_hot100": "True" if charted else "False",
                "riaa_certification": riaa_cert,
                "riaa_cert_date": riaa_dt,
                "duration_seconds": duration_sec if duration_sec is not None else "",
                "fetch_timestamp": fetch_iso,
                "parser_note": "; ".join(notes),
            }
        )

    apply_final_phase4b_corrections(perf_rows, kworb_date=str(kworb_date or ""))

    _perf_by_id_post = {r["track_id"]: r for r in perf_rows}
    kworb_miss_rows = [
        r
        for r in kworb_miss_rows
        if not str(_perf_by_id_post.get(r["track_id"], {}).get("kworb_total_streams", "")).strip()
    ]

    n_dur = sum(1 for r in perf_rows if str(r.get("duration_seconds", "")).strip() != "")
    if not args.skip_wiki_enrich and n_dur < MIN_WIKI_DURATION_COVERAGE:
        print(
            f"STOP: Wikipedia infobox Length parsed for only {n_dur} tracks "
            f"(minimum {MIN_WIKI_DURATION_COVERAGE}). Check extract_length_seconds_from_wiki_html.",
            file=sys.stderr,
        )
        return 2

    collab_null_share_ids = write_sonic_csv(perf_rows, feats, cleaned_by_id, fetch_iso)

    OUT_PERF.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PERF.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PERF_FIELDNAMES)
        w.writeheader()
        w.writerows(perf_rows)

    kworb_miss_rows.sort(key=lambda r: -float(r["best_kworb_score"]))

    perf_by_id = {r["track_id"]: r for r in perf_rows}
    by_title_eras: dict[str, set[str]] = defaultdict(set)
    for f in feats.values():
        by_title_eras[f["track_title"]].add((f.get("era_clean") or "").strip())

    collision_rows: list[dict[str, str]] = []
    for title, eras in sorted(by_title_eras.items()):
        if len(eras) < 2:
            continue
        era_sorted = sorted(eras)
        has_bb = False
        for tid, f in feats.items():
            if f["track_title"] != title:
                continue
            if perf_by_id.get(tid, {}).get("charted_hot100") == "True":
                has_bb = True
                break
        collision_rows.append(
            {
                "track_title": title,
                "era_list": ";".join(era_sorted),
                "n_eras": str(len(era_sorted)),
                "has_billboard_data": "True" if has_bb else "False",
            }
        )

    OUT_TITLE_COLLISIONS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TITLE_COLLISIONS.open("w", encoding="utf-8", newline="") as f:
        wc = csv.DictWriter(
            f,
            fieldnames=["track_title", "era_list", "n_eras", "has_billboard_data"],
        )
        wc.writeheader()
        wc.writerows(collision_rows)

    with OUT_KWORB_MISSES.open("w", encoding="utf-8", newline="") as f:
        wkm = csv.DictWriter(
            f,
            fieldnames=[
                "track_id",
                "track_title",
                "era_clean",
                "year",
                "best_kworb_match",
                "best_kworb_score",
            ],
        )
        wkm.writeheader()
        wkm.writerows(kworb_miss_rows)

    n_tracks = len(perf_rows)
    n_kworb = sum(1 for r in perf_rows if str(r.get("kworb_total_streams", "")).strip() != "")
    n_bb = sum(1 for r in perf_rows if r.get("charted_hot100") == "True")
    n_riaa = sum(1 for r in perf_rows if str(r.get("riaa_certification", "")).strip() != "")

    stat_cols = [
        "kworb_total_streams",
        "kworb_match_score",
        "kworb_fetch_date",
        "billboard_hot100_peak",
        "billboard_hot100_weeks_on_chart",
        "billboard_hot100_debut_date",
        "riaa_certification",
        "riaa_cert_date",
        "duration_seconds",
        "parser_note",
    ]
    stats = null_stats(perf_rows, stat_cols)

    sonic_feats = list(csv.DictReader(OUT_SONIC.open(encoding="utf-8")))
    n_share_comp = sum(1 for r in sonic_feats if str(r.get("kanye_verse_share", "")).strip() != "")
    n_wpm = sum(1 for r in sonic_feats if str(r.get("words_per_minute", "")).strip() != "")

    log_lines = [
        "# Phase 4b log",
        "",
        f"Generated: {fetch_iso}",
        "",
        "## Methodology pivot (Spotify removed)",
        "",
        "Phase 4b **no longer calls the Spotify Web API**. Earlier attempts hit recurring "
        "operational failures: **HTTP 400** on quoted field-style queries; **HTTP 400** on "
        "unquoted free-text queries containing symbols Lucene treats as operators; **HTTP 401** "
        "from cached vs. rotated client credentials; **HTTP 429** throttling that persisted "
        "across backoff attempts; and **HTTP 401** under heavy-throttle conditions. The pipeline "
        "is now **Spotify-free**: Kworb cumulative streams (title fuzzy match), Wikipedia "
        "singles discography (Billboard + RIAA), and Wikipedia song-page **Length** for durations.",
        "",
        "## Final human corrections (Phase 4b lock)",
        "",
        "- **King (Bully vs Vultures):** Wikipedia singles table lists a **Bully** «King» row under "
        "release year **2026** with Hot 100 **peak 40**, separate from the ¥$ / Ty Dolla $ign "
        "**Vultures** «King» row (Hot 100 **94**). The Bully corpus row **keeps** Billboard; "
        "**Kworb** cumulative streams on that row were cleared because Kworb anchors the 35M figure "
        "to the Vultures-era opener only (one table entry per title).",
        "",
        "- **Number One (College vs Late Registration):** Hot 100 lineage for Pharrell feat. Kanye "
        "(2006) is documented on `number_one_late_registration`; the College Dropout album cut "
        "`number_one_the_college_dropout` carries an explicit `parser_note` (no Hot 100).",
        "",
        "- **Sanctified (Yeezus vs TLOP):** Chart data attaches to the 2014-era corpus row "
        "(`sanctified_the_life_of_pablo`); the Yeezus session row has Billboard cleared and a "
        "matching `parser_note`.",
        "",
        "- **Manual Kworb (+2):** `buy_you_a_drank_remix_graduation` → Kworb «Buy U A Drank…» "
        "(T-Pain spelling); `i_don_t_like_remix_cruel_summer_yeezus_build_up` → Kworb «Don't Like.1» "
        "(duplicate suffix). Both use `kworb_match_score=100.00` after lyric review.",
        "",
        "## Rates",
        "",
        "- Kworb / Wikipedia: minimum **2 seconds** between network requests per registrable domain.",
        f"- HTTP cache directory: `{CACHE_ROOT}`",
        "",
        "## Kworb (title-only)",
        "",
        f"- Merged song rows (both artist pages): **{len(kworb_rows)}**",
        f"- RapidFuzz `token_set_ratio` threshold: **≥ {KWORB_FUZZ_THRESHOLD:g}** vs Kworb anchor title.",
        f"- Corpus tracks with non-empty `kworb_total_streams`: **{n_kworb}** / **{n_tracks}** "
        f"({n_kworb / max(n_tracks, 1):.1%}).",
        f"- Tracks below threshold this run (listed in `parser_note` as `no_kworb_match`): **{len(kworb_review)}**",
        "",
    ]
    if args.continue_after_kworb_review and len(kworb_review) > MAX_KWORB_REVIEW_FAILS:
        log_lines.extend(
            [
                "### Kworb gate override",
                "",
                f"Run used `--continue-after-kworb-review` with **{len(kworb_review)}** tracks below "
                f"{KWORB_FUZZ_THRESHOLD:g} (more than the usual stop threshold of {MAX_KWORB_REVIEW_FAILS}). "
                "Many are expected: mixtape / leak / alternate titles not present on Kworb Spotify tables.",
                "",
            ]
        )
    log_lines.extend(
        [
            "## Billboard Hot 100",
            "",
            "- Singles table rows carry a **release year** when Wikipedia lists a leading year cell "
            "(including rowspan blocks). Chart matching requires RapidFuzz ≥ 85 on the normalized title "
        "and, when both corpus `year` and table year exist, they must agree within **±2 years** "
        "(or **exact year** when the same `track_title` appears under multiple `era_clean` values in "
        "the features corpus). Rows are keyed by `wiki_slug` (`norm_slug`). Non–Kanye-West primaries "
        "must align with that slug (`partial_ratio` gate) so feature rows do not latch onto the wrong song.",
            "",
            f"- Rows with `charted_hot100=True`: **{n_bb}**",
            "",
            "## RIAA",
            "",
            f"- Rows with non-empty `riaa_certification`: **{n_riaa}**",
            "",
            "## Wikipedia duration (infobox Length)",
            "",
            f"- Tracks with non-null `duration_seconds`: **{n_dur}** / **{n_tracks}** "
            f"({n_dur / max(n_tracks, 1):.1%}).",
            "- **Words per minute** and **lines per minute** in `kanye_track_sonic_derived.csv` are only "
            "computed where `duration_seconds` is present (Wikipedia subset); other rows leave WPM/LPM empty.",
            "",
            "## Sonic derived",
            "",
            f"- `kanye_verse_share` computable (collab + full lyrics): **{n_share_comp}**",
            f"- Rows with non-empty `words_per_minute` (duration-backed): **{n_wpm}**",
            f"- Collab rows with null `kanye_verse_share` (backfill candidates): **{len(collab_null_share_ids)}**",
            "",
            "## Audit outputs",
            "",
            f"- `data/phase_4b_title_collisions.csv` — titles appearing under multiple `era_clean` values "
            f"({len(collision_rows)} titles).",
            f"- `data/phase_4b_kworb_misses.csv` — tracks below Kworb fuzzy {KWORB_FUZZ_THRESHOLD:g} "
            f"({len(kworb_miss_rows)} rows), sorted by best score descending.",
            "",
            "## Wikipedia ambiguity / duplicate peaks",
            "",
        ]
    )
    log_lines.extend([f"- {x}" for x in ambiguities[:80]])

    LOG_MD.parent.mkdir(parents=True, exist_ok=True)
    LOG_MD.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(fmt_perf_preview(perf_rows))
    print("\n=== Summary statistics ===")
    print(f"kworb_match_rate={n_kworb}/{n_tracks}")
    print(f"billboard_hot100_charted={n_bb}")
    print(f"riaa_non_empty_rows={n_riaa}")
    print(f"wikipedia_duration_coverage={n_dur}/{n_tracks}")
    print(f"sonic_wpm_populated={n_wpm}/{n_tracks}")
    print("\n=== Null / empty counts (performance CSV) ===")
    for c in stat_cols:
        print(f"  empty({c})={stats[c]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
