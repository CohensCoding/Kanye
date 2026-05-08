from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional

import lyricsgenius
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

# -----------------------------
# Constants
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCOGRAPHY_FILENAME = "compass_artifact_wf-c161937f-b826-4936-bdd1-26dcb933d60a_text_markdown.md"

OUT_CSV = PROJECT_ROOT / "kanye_lyrics.csv"
OUT_JSON = PROJECT_ROOT / "kanye_lyrics.json"
FAILED_TXT = PROJECT_ROOT / "failed_tracks.txt"

CACHE_DIR = PROJECT_ROOT / ".cache"
CHECKPOINT_JSONL = CACHE_DIR / "kanye_lyrics_records.jsonl"

GENIUS_TIMEOUT_S = 15
GENIUS_RETRIES = 3
DELAY_BETWEEN_CALLS_S = 1.5

RATE_LIMIT_BACKOFF_S = [30, 60, 120]  # max 3 retries

SKIP_TAGS = {"Leak/Unreleased", "Listening Party"}
SKIP_IF_TYPE_CONTAINS = {"Leak", "Unreleased", "Listening Party"}

DATE_PATTERN = re.compile(
    r"^(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|"
    r"Aug(ust)?|Sep(t(ember)?)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)"
    r"\s+\d{1,2}(,?\s+\d{4})?$",
    re.IGNORECASE,
)

METADATA_INDICATORS = [
    "with ",
    "uncredited",
    "originally",
    "produced",
    "sample",
    "demo",
    "snippet",
    "clip",
    "leaked",
    "unreleased",
    "studio",
    "early ",
    "late ",
    "version",
    "audio",
    "interpolation",
    "feat.",
    "featuring",
    "became",
]

END_OF_DATA_MARKERS = [
    "misattributed",
    "not kanye",
    "frequently misattributed",
    "notes on items",
]

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # Keep the user's requested value; if the runtime can't decode br, we retry without it.
    "Accept-Encoding": "gzip, deflate, br",
}


# -----------------------------
# Models
# -----------------------------


@dataclass(frozen=True)
class ParsedTrack:
    """
    Parsed track metadata from the discography markdown.

    - `release_date` is stored as the most specific string available:
      ISO `YYYY-MM-DD` when we can infer it, otherwise a year (`YYYY`) or a
      best-effort fragment (e.g., "Sept 30") if the year can't be resolved.
    """

    track_title: str
    album_or_era: str
    release_date: str
    track_type: str
    primary_artist: str
    featured_artists: list[str]
    parser_note: str = ""


@dataclass
class LyricRecord:
    track_title: str
    album_or_era: str
    release_date: str
    track_type: str
    primary_artist: str
    featured_artists: list[str]
    parser_note: str
    genius_url: str
    lyrics: str
    lyrics_word_count: int


# -----------------------------
# Parsing utilities
# -----------------------------


def parse_discography(md_path: Path) -> list[ParsedTrack]:
    """
    Parse the discography markdown into track records.

    Handles patterns seen in the provided artifact:
    - Era headings like `## 2004 — *The College Dropout*` (sets default year + context)
    - Album paragraphs like `**The College Dropout (Feb 10, 2004)** ... **[Solo]**: "Intro," ...`
    - Bullet entries like `- "Through the Wire" — Kanye West [Single, Sept 30]`
    - Inline feature lists like `"Pop Style" (Drake, *Views*); "All We Got" (Chance, *Coloring Book*);`
    - Feature-run paragraphs that list many `"Title" (Artist)` entries separated by semicolons/commas.

    Returns:
        List of ParsedTrack (not yet deduped/filtered for skits/leaks).
    """
    text = md_path.read_text(encoding="utf-8", errors="replace")
    tracks: list[ParsedTrack] = []

    current_year = ""
    current_context = ""

    era_heading_re = re.compile(r"^\s*##\s*(?P<year>\d{4})\s*—\s*(?P<era>.+?)\s*$")
    album_tracklist_re = re.compile(
        r"^\s*\*\*(?P<album>.+?)\s*\((?P<date>[^)]+)\)\*\*.*?\*\*\[(?P<tag>[^\]]+)\]\*\*.*?:\s*(?P<body>.+?)\s*$"
    )
    bullet_re = re.compile(
        r'^\s*[-*]\s*"(?P<title>[^"]+)"\s*—\s*(?P<credits>.+?)\s*\[(?P<meta>[^\]]+)\]\s*$'
    )

    # Any `"Title" ( ... )` occurrences within a line; we’ll interpret the parenthetical.
    inline_title_paren_re = re.compile(r'"(?P<title>[^"]+)"\s*\((?P<inside>[^)]+)\)')

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# "):
            continue
        if line.startswith("##"):
            # End-of-data marker: the discography includes a "misattributed/not Kanye" notes section.
            lowered = _strip_markdown(line).lower()
            if any(marker in lowered for marker in END_OF_DATA_MARKERS):
                break

        m = era_heading_re.match(line)
        if m:
            current_year = m.group("year").strip()
            current_context = _strip_markdown(m.group("era").strip())
            continue

        # Album tracklist paragraphs (bulk quoted titles)
        m = album_tracklist_re.match(line)
        if m:
            album = _strip_markdown(m.group("album").strip())
            release_date = _normalize_date_string(m.group("date").strip())
            tag = _normalize_type_tag(m.group("tag").strip())
            quoted_titles = [t.strip().rstrip(",") for t in re.findall(r'"([^"]+)"', m.group("body"))]
            for title in quoted_titles:
                if not title:
                    continue
                tracks.append(
                    ParsedTrack(
                        track_title=title,
                        album_or_era=album,
                        release_date=release_date,
                        track_type=tag,
                        primary_artist="Kanye West",
                        featured_artists=_extract_features_from_title(title),
                    )
                )
            continue

        # Bullets with explicit credits + [type, date]
        m = bullet_re.match(line)
        if m:
            title = m.group("title").strip()
            credits = _strip_markdown(m.group("credits").strip())
            meta = _strip_markdown(m.group("meta").strip())
            track_type, date_fragment = _split_type_date_fragment(meta)
            track_type = _normalize_type_tag(track_type)
            rd = _resolve_date(date_fragment, current_year) if date_fragment else ""
            if not rd:
                rd = current_year

            primary_artist, featured = _parse_artist_credits(credits)
            primary_artist, note = _fix_primary_artist(primary_artist)
            tracks.append(
                ParsedTrack(
                    track_title=title,
                    album_or_era=current_context or album_from_credits(credits) or current_year,
                    release_date=rd,
                    track_type=track_type,
                    primary_artist=primary_artist,
                    featured_artists=featured,
                    parser_note=note,
                )
            )
            continue

        # Inline `"Title" (Artist, *Album*)` feature lists in prose.
        # Example: **2016 features**: "Pop Style" (Drake, *Views*); "All We Got" (Chance, *Coloring Book*);
        inlines = list(inline_title_paren_re.finditer(line))
        if inlines:
            for mm in inlines:
                title = mm.group("title").strip()
                inside = _strip_markdown(mm.group("inside").strip())
                primary_artist, featured = _parse_inline_paren_inside(inside)
                primary_artist, note = _fix_primary_artist(primary_artist)
                tracks.append(
                    ParsedTrack(
                        track_title=title,
                        album_or_era=current_context or current_year,
                        release_date=current_year,
                        track_type="Feature" if primary_artist.lower() != "kanye west" else "Solo",
                        primary_artist=primary_artist,
                        featured_artists=featured,
                        parser_note=note,
                    )
                )
            continue

    return tracks


def album_from_credits(credits: str) -> str:
    """
    If the credits include `, *Album*` we treat it as context, otherwise empty.
    This is intentionally conservative and only used for bullet lines when present.
    """
    m = re.search(r",\s*\*([^*]+)\*", credits)
    return m.group(1).strip() if m else ""


def _parse_inline_paren_inside(inside: str) -> tuple[str, list[str]]:
    """
    Parse the content inside parentheses after a quoted title.
    Common forms:
    - "Drake, *Views*" -> primary_artist=Drake
    - "Twista" -> primary_artist=Twista
    - "Jay-Z feat. Kanye West" -> primary_artist=Jay-Z, featured includes Kanye West
    """
    # Take the first comma-separated segment as the artist/credits piece.
    first = inside.split(";")[0].strip()
    first = first.split(",")[0].strip()
    primary, featured = _parse_artist_credits(first)
    primary_fixed, _note = _fix_primary_artist(primary)
    if primary_fixed != primary:
        return primary_fixed, featured
    return primary, featured


def looks_like_date(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if DATE_PATTERN.match(s):
        return True
    if re.match(r"^\d{4}(-\d{4})?$", s):
        return True
    months = {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }
    return s.lower() in months


def is_metadata_not_artist(s: str) -> bool:
    s_lower = s.lower().strip()
    if not s_lower:
        return True
    if any(ind in s_lower for ind in METADATA_INDICATORS):
        return True
    if looks_like_date(s):
        return True
    if re.match(r"^\d{4}$", s_lower):
        return True
    return False


def _fix_primary_artist(primary_artist: str) -> tuple[str, str]:
    """
    Guardrail for discography parsing:
    if the extracted "artist" looks like a date or metadata, treat it as a parser note
    and fall back to `Kanye West`.

    Returns:
        (fixed_primary_artist, parser_note_or_empty)
    """
    s = (primary_artist or "").strip()
    if not s:
        return "Kanye West", ""
    if looks_like_date(s) or is_metadata_not_artist(s):
        return "Kanye West", s
    return s, ""


def _strip_markdown(s: str) -> str:
    return s.replace("**", "").replace("*", "").strip()


def _normalize_type_tag(tag: str) -> str:
    tag = tag.strip()
    if not tag:
        return ""
    # Keep the broad label; prefer canonical capitalization.
    # Handles "Feature/Single", "Feature/Remix", etc.
    parts = re.split(r"\s*/\s*", tag)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return tag
    normalized = []
    for p in parts:
        key = p.lower()
        if key == "solo":
            normalized.append("Solo")
        elif key == "single":
            normalized.append("Single")
        elif key == "feature":
            normalized.append("Feature")
        elif key == "remix":
            normalized.append("Remix")
        elif key == "mixtape":
            normalized.append("Mixtape")
        elif key in ("leak", "leak/unreleased", "unreleased"):
            normalized.append("Leak/Unreleased")
        elif key == "listening party":
            normalized.append("Listening Party")
        elif key == "production":
            normalized.append("Production")
        else:
            normalized.append(p)
    # If there are multiple, keep slash form (it’s helpful downstream).
    return "/".join(normalized)


def _split_type_date_fragment(meta: str) -> tuple[str, str]:
    """
    Split bracket meta like:
    - "Single, Sept 30"
    - "Feature/Single, Nov 10"
    - "Feature/Remix"
    Returns (track_type, date_fragment_or_empty).
    """
    parts = [p.strip() for p in meta.split(",") if p.strip()]
    if len(parts) >= 2:
        return parts[0], ", ".join(parts[1:])
    return meta.strip(), ""


def _normalize_date_string(s: str) -> str:
    """
    Normalize common date forms to ISO `YYYY-MM-DD` when possible.
    Otherwise returns the original trimmed string.
    """
    s = s.strip()
    if not s:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s

    month_map = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    # "Feb 10, 2004" or "Feb 10 2004"
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})(?:,)?\s+(\d{4})", s)
    if m:
        mon = month_map.get(m.group(1).lower())
        if mon:
            try:
                return date(int(m.group(3)), mon, int(m.group(2))).isoformat()
            except ValueError:
                return s
    return s


def _resolve_date(fragment: str, current_year: str) -> str:
    """
    Resolve date fragments like "Sept 30" against the current era year.
    Returns ISO date when possible, else returns the best available string.
    """
    fragment = fragment.strip()
    if not fragment:
        return ""

    # Already ISO
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", fragment):
        return fragment

    # Try "Month Day" optionally with year already attached.
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})(?:,)?(?:\s+(\d{4}))?", fragment)
    if m:
        month = m.group(1)
        day = m.group(2)
        year = m.group(3) or current_year
        if year and len(year) == 4:
            return _normalize_date_string(f"{month} {day}, {year}")
        return f"{month} {int(day)}"

    # If it’s a plain year, keep it.
    if re.fullmatch(r"\d{4}", fragment):
        return fragment

    return fragment


def _extract_features_from_title(title: str) -> list[str]:
    """
    Extract featured artists from title parentheticals like:
    - "All Falls Down" (feat. Syleena Johnson)
    - "Spaceship" (feat. GLC, Consequence)
    Returns a best-effort list, excluding Kanye himself.
    """
    m = re.search(r"\((?:feat\.|featuring)\s+([^)]+)\)", title, flags=re.I)
    if not m:
        return []
    raw = m.group(1)
    artists = _split_artists(raw)
    return [a for a in artists if a.lower() not in {"kanye west", "ye"}]


def _split_artists(s: str) -> list[str]:
    s = s.replace("&", ",")
    parts = [p.strip() for p in re.split(r",|/| and ", s) if p.strip()]
    cleaned: list[str] = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            cleaned.append(p)
    # De-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for a in cleaned:
        key = a.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def _parse_artist_credits(credits: str) -> tuple[str, list[str]]:
    """
    Parse artist credits like:
    - "Kanye West"
    - "Twista feat. Kanye West & Jamie Foxx"
    - "Talib Kweli feat. Jay-Z, Mos Def, Kanye West & Busta Rhymes"
    - "Jay-Z, *The Black Album* [Production with vocal ad-libs]" (we’ll take primary=Jay-Z)
    """
    c = credits.strip()
    c = re.sub(r"\s+", " ", c)

    # Remove trailing album italic if present: ", *Views*"
    c = re.sub(r",\s*\*[^*]+\*\s*$", "", c).strip()

    feat_split = re.split(r"\bfeat\.|\bfeaturing\b", c, flags=re.I, maxsplit=1)
    primary = feat_split[0].strip().rstrip(",")
    featured: list[str] = []
    if len(feat_split) == 2:
        featured = _split_artists(feat_split[1])

    # Also handle "Artist1 & Artist2" without feat. (rare; keep Artist1 as primary)
    if not featured and "&" in primary:
        parts = [p.strip() for p in primary.split("&") if p.strip()]
        primary = parts[0]
        featured = parts[1:]

    return primary or "Kanye West", featured


def _is_skippable(track: ParsedTrack) -> tuple[bool, str]:
    """
    Apply skip rules:
    - Skip [Leak/Unreleased] and [Listening Party]
    - Skip production-only
    - Skip pure skits/interludes/intro (heuristic)
    """
    ttype = track.track_type or ""
    for bad in SKIP_IF_TYPE_CONTAINS:
        if bad.lower() in ttype.lower():
            return True, f"tagged {bad}"
    if "production" in ttype.lower():
        return True, "production-only"

    title_l = track.track_title.strip().lower()
    if "skit" in title_l:
        return True, "skit"
    if re.fullmatch(r"interlude", title_l):
        return True, "interlude"
    if re.fullmatch(r"intro", title_l):
        return True, "intro"
    if re.search(r"\(interlude\)", title_l):
        return True, "interlude"
    if re.search(r"\(skit", title_l):
        return True, "skit"

    return False, ""


def _dedup_key(track_title: str, primary_artist: str) -> str:
    t = re.sub(r"\s+", " ", track_title.strip().lower())
    a = re.sub(r"\s+", " ", primary_artist.strip().lower())
    return f"{t}||{a}"


# -----------------------------
# Genius fetch + cleaning
# -----------------------------


def create_genius_client(api_token: str) -> lyricsgenius.Genius:
    """
    Create a lyricsgenius client with the required settings.
    """
    genius = lyricsgenius.Genius(
        api_token,
        timeout=GENIUS_TIMEOUT_S,
        retries=GENIUS_RETRIES,
        verbose=False,
        remove_section_headers=True,
        skip_non_songs=True,
        # excluded_terms is applied conditionally per-track via search strategy below
        excluded_terms=[],
    )
    genius.sleep_time = 0
    _inject_browser_headers_into_lyricsgenius(genius)
    return genius


def _inject_browser_headers_into_lyricsgenius(genius: lyricsgenius.Genius) -> None:
    """
    Inject browser-like headers into lyricsgenius' internal HTTP session.

    The 403s you observed happen on `https://genius.com/api/search/multi`, which is
    called inside `lyricsgenius.search_song()`. That call uses an internal requests
    session on the Genius client. In lyricsgenius==3.0.1 this is typically exposed
    as `genius._session` (requests.Session).

    If the attribute name differs, we try a small set of likely session attributes.
    """
    session = None
    for attr in ("_session", "session", "_client", "client"):
        if hasattr(genius, attr):
            candidate = getattr(genius, attr)
            # requests.Session has `.headers` and `.request`
            if hasattr(candidate, "headers") and hasattr(candidate, "request"):
                session = candidate
                break
    if session is None:
        return
    try:
        session.headers.update(BROWSER_HEADERS)
    except Exception:
        # If something unexpected happens, don't fail script startup.
        return


def fetch_lyrics(
    genius: lyricsgenius.Genius,
    track: ParsedTrack,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Fetch lyrics for a track.

    Strategy:
    - search_song(title, primary_artist)
    - if no match: retry with fuzzed title (strip parentheticals, remove feat. fragments)
    - verify match by checking title keyword overlap
    - on success: return a payload dict with genius_url + lyrics

    Returns:
        (payload, failure_reason)
    """
    title = track.track_title.strip()
    artist = track.primary_artist.strip()
    is_remix = "remix" in (track.track_type or "").lower() or "(remix" in title.lower()
    is_live = "(live" in title.lower()

    excluded_terms = []
    if not is_remix:
        excluded_terms.append("(Remix)")
    if not is_live:
        excluded_terms.append("(Live)")

    def do_search(search_title: str) -> Optional[Any]:
        # lyricsgenius reads excluded_terms off the object
        genius.excluded_terms = excluded_terms
        return genius.search_song(search_title, artist=artist)

    attempts = [title]
    fuzz = _fuzz_title(title)
    if fuzz and fuzz != title:
        attempts.append(fuzz)

    last_err: Optional[str] = None
    for search_title in attempts:
        try:
            song = _search_with_backoff(do_search, search_title)
        except Exception as e:
            # If Genius' web search is blocked (403), fall back to official API.
            status = _extract_http_status(e)
            if status == 403:
                fb_lyrics, fb_url = fetch_lyrics_fallback(search_title, artist, os.getenv("GENIUS_API_TOKEN", "").strip())
                if fb_lyrics and fb_url:
                    return {"genius_url": fb_url, "lyrics": fb_lyrics}, None
                last_err = "403 from genius.com search; official API fallback failed"
                continue
            last_err = f"error: {type(e).__name__}: {e}"
            continue
        if not song:
            # No match via web search; try the official API fallback before giving up.
            fb_lyrics, fb_url = fetch_lyrics_fallback(search_title, artist, os.getenv("GENIUS_API_TOKEN", "").strip())
            if fb_lyrics and fb_url:
                return {"genius_url": fb_url, "lyrics": fb_lyrics}, None
            last_err = "no match (web) and fallback miss"
            continue

        genius_title = getattr(song, "title", "") or ""
        if not _sanity_check_title(search_title, genius_title):
            last_err = f"sanity check failed (got '{genius_title}')"
            continue

        url = getattr(song, "url", "") or ""
        if not url:
            last_err = "matched song missing url"
            continue

        # Primary fix: ALWAYS scrape only data-lyrics-container blocks from the URL.
        lyrics = scrape_lyrics_from_genius_url(url, track_title=title)
        if not lyrics:
            # If lyricsgenius returned an object but lyrics are empty, try official API fallback.
            fb_lyrics, fb_url = fetch_lyrics_fallback(search_title, artist, os.getenv("GENIUS_API_TOKEN", "").strip())
            if fb_lyrics and fb_url:
                return {"genius_url": fb_url, "lyrics": fb_lyrics}, None
            last_err = "empty lyrics (web) and fallback miss"
            continue

        return {"genius_url": url, "lyrics": lyrics}, None

    return None, (last_err or "unknown failure")


def _search_with_backoff(search_fn, title: str) -> Optional[Any]:
    """
    Wrap Genius search in rate-limit backoff handling.
    """
    for i, wait_s in enumerate([0] + RATE_LIMIT_BACKOFF_S):
        if wait_s:
            time.sleep(wait_s)
        try:
            return search_fn(title)
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None)
            if status == 429 and i < len(RATE_LIMIT_BACKOFF_S):
                continue
            raise
        except requests.RequestException as e:
            # Some rate limiting surfaces as RequestException with 429 in message
            msg = str(e).lower()
            if "429" in msg and i < len(RATE_LIMIT_BACKOFF_S):
                continue
            raise
    return None


def _fuzz_title(title: str) -> str:
    """
    Fuzzy title variant for retry:
    - strips parentheticals like "(Remix)" or "(Live from ...)"
    - strips "feat." tails if present in raw title string
    """
    t = title
    t = re.sub(r"\s*\([^)]*\)\s*", " ", t).strip()
    t = re.sub(r"\bfeat\.?\b.*$", "", t, flags=re.I).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _sanity_check_title(searched: str, returned: str) -> bool:
    """
    Basic sanity check to avoid false matches:
    Require overlap of meaningful keywords from the searched title.
    """
    def tokens(s: str) -> set[str]:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        parts = [p for p in s.split() if len(p) >= 3]
        stop = {"the", "and", "for", "with", "from", "remix", "live"}
        return {p for p in parts if p not in stop}

    a = tokens(searched)
    b = tokens(returned)
    if not a or not b:
        return True
    return len(a & b) >= max(1, min(2, len(a) // 2))


def _is_genius_untranscribed_placeholder(raw: str) -> bool:
    s = (raw or "").lower()
    return "yet to be transcribed" in s or "lyrics for this song have yet" in s


def scrape_lyrics_from_genius_url(url: str, *, track_title: str = "", timeout: int = 20) -> str:
    """
    Scrape lyrics from a Genius song URL using ONLY the lyrics containers.

    This avoids Genius page chrome (contributors, translations, title headers,
    descriptions) which are outside `div[data-lyrics-container="true"]`.
    """
    try:
        resp = _http_get(url, timeout=timeout)
    except Exception:
        return ""

    soup = BeautifulSoup(resp.text, "lxml")

    containers = soup.select('div[data-lyrics-container="true"]')
    if containers:
        raw = "\n".join(c.get_text("\n", strip=False) for c in containers)
    else:
        modern = soup.select('div[class*="Lyrics__Container"]')
        if modern:
            raw = "\n".join(c.get_text("\n", strip=False) for c in modern)
        else:
            legacy = soup.select_one("div.lyrics")
            raw = legacy.get_text("\n", strip=False) if legacy else ""

    if _is_genius_untranscribed_placeholder(raw):
        return ""

    return clean_lyrics(raw, track_title)


def fetch_lyrics_direct_from_url(url: str, track_title: str) -> tuple[Optional[str], Optional[str]]:
    """
    Last-resort recovery: fetch lyrics from a known Genius song URL using only
    lyrics-container scraping (legacy data-lyrics-container, modern Lyrics__Container,
    or legacy div.lyrics).

    Returns:
        (lyrics, url) if lyrics were parsed; otherwise (None, None).
    """
    url = (url or "").strip()
    if not url:
        return None, None
    lyrics = scrape_lyrics_from_genius_url(url, track_title=track_title)
    if not lyrics.strip():
        return None, None
    return lyrics, url


def clean_lyrics(raw: str, _track_title: str = "") -> str:
    """
    Aggressively clean Genius lyrics text to remove common metadata blocks.

    This is intentionally title-agnostic because the garbage often includes
    contributor/translation blocks and annotation blurbs that precede the lyrics.
    """
    if not raw:
        return ""

    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\u200b", "")

    # 1. Strip everything from start up through "Read More" if present
    text = re.sub(r"^.*?Read More\s*", "", text, count=1, flags=re.DOTALL)

    # 2. If no "Read More" was found, strip the contributor/translation/title header pattern
    text = re.sub(r"^\d+\s*Contributors?.*?Lyrics\s*", "", text, count=1, flags=re.DOTALL)

    # 3. Strip "You might also like" blocks (inline and footer)
    text = re.sub(r"You might also like", "\n", text, flags=re.I)

    # 4. Strip "See <something> Live\nGet tickets..." promo blocks
    text = re.sub(r"See .+? LiveGet tickets as low as \$\d+", "", text, flags=re.I | re.DOTALL)

    # 5. Strip trailing "Embed" with optional digit prefix
    text = re.sub(r"\d*Embed\s*$", "", text, flags=re.I)

    # 6. Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 7. Strip leading/trailing whitespace
    text = text.strip()

    # 8. Safety net: strip leading description prose (third-person commentary about the song)
    description_indicators = [
        "is a ",
        "was released",
        "features",
        "produced by",
        "samples",
        "from the album",
        "from kanye",
        "kanye west's",
        "track from",
        "song from",
        "this song",
        "on the album",
        "debut album",
    ]

    lines = text.split("\n")
    stripped_count = 0
    while lines and stripped_count < 5:
        first = lines[0].strip()
        if not first:
            lines.pop(0)
            continue
        is_description = (
            len(first) > 80
            and first.endswith(".")
            and any(ind in first.lower() for ind in description_indicators)
        )
        if is_description:
            lines.pop(0)
            stripped_count += 1
        else:
            break
    text = "\n".join(lines).strip()

    return text


def fetch_lyrics_fallback(title: str, artist: str, token: str) -> tuple[Optional[str], Optional[str]]:
    """
    Fallback that uses the official Genius API (api.genius.com) instead of the
    Cloudflare-protected web search at genius.com.

    Returns:
        (lyrics, url) on success, otherwise (None, None).
    """
    if not token:
        return None, None

    q = f"{title} {artist}".strip()
    search_url = "https://api.genius.com/search"
    headers = {**BROWSER_HEADERS, "Authorization": f"Bearer {token}"}

    try:
        resp = _http_get(search_url, timeout=GENIUS_TIMEOUT_S, headers=headers, params={"q": q})
        data = resp.json()
    except Exception:
        return None, None

    hits = (((data or {}).get("response") or {}).get("hits") or [])
    best = _pick_best_hit(hits, title=title, artist=artist)
    if not best:
        return None, None

    url = best.get("url") or ""
    if not url:
        return None, None
    cleaned = scrape_lyrics_from_genius_url(url, track_title=title)
    return (cleaned, url) if cleaned else (None, None)


def _pick_best_hit(hits: list[dict[str, Any]], *, title: str, artist: str) -> Optional[dict[str, Any]]:
    """
    Choose the best Genius API search hit given a title + artist.
    Prefers artist name match, then title token overlap.
    """
    want_artist = artist.strip().lower()
    want_title = title.strip().lower()

    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    want_artist_n = norm(want_artist)
    want_title_n = norm(_fuzz_title(want_title))

    def score(hit: dict[str, Any]) -> int:
        r = hit.get("result") or {}
        hit_title = norm(r.get("title") or "")
        hit_artist = norm((r.get("primary_artist") or {}).get("name") or "")
        s = 0
        if want_artist_n and hit_artist and (want_artist_n in hit_artist or hit_artist in want_artist_n):
            s += 50
        # Token overlap in title
        a = set([t for t in want_title_n.split() if len(t) >= 3])
        b = set([t for t in hit_title.split() if len(t) >= 3])
        s += 5 * len(a & b)
        return s

    best_hit: Optional[dict[str, Any]] = None
    best_score = -1
    for h in hits:
        try:
            sc = score(h)
        except Exception:
            continue
        if sc > best_score:
            best_score = sc
            best_hit = h.get("result") or None

    if not best_hit:
        return None
    # Basic guard against totally wrong results
    if not _sanity_check_title(title, best_hit.get("title") or ""):
        return None
    return best_hit


def _extract_http_status(e: Exception) -> Optional[int]:
    """
    Best-effort extraction of HTTP status code from common requests exceptions.
    """
    resp = getattr(e, "response", None)
    status = getattr(resp, "status_code", None)
    if isinstance(status, int):
        return status
    msg = str(e)
    m = re.search(r"\b(403|429)\b", msg)
    return int(m.group(1)) if m else None


def _http_get(
    url: str,
    *,
    timeout: int,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, str]] = None,
) -> requests.Response:
    """
    HTTP GET helper using browser headers, with a safety retry if the runtime can't
    decode brotli (`br`) responses.
    """
    hdrs = {**BROWSER_HEADERS, **(headers or {})}
    try:
        resp = requests.get(url, timeout=timeout, headers=hdrs, params=params)
        resp.raise_for_status()
        return resp
    except requests.exceptions.ContentDecodingError:
        # Retry without br if brotli isn't available in this Python build.
        hdrs2 = dict(hdrs)
        hdrs2["Accept-Encoding"] = "gzip, deflate"
        resp = requests.get(url, timeout=timeout, headers=hdrs2, params=params)
        resp.raise_for_status()
        return resp


# -----------------------------
# IO / checkpointing
# -----------------------------


def load_checkpoint(path: Path) -> dict[str, LyricRecord]:
    """
    Load `.cache/kanye_lyrics_records.jsonl` and return key->LyricRecord.
    Unique key is `track_title + primary_artist` (normalized).
    """
    if not path.exists():
        return {}
    out: dict[str, LyricRecord] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("_key")
            if not key:
                continue
            out[key] = LyricRecord(
                track_title=obj["track_title"],
                album_or_era=obj["album_or_era"],
                release_date=obj["release_date"],
                track_type=obj["track_type"],
                primary_artist=obj["primary_artist"],
                featured_artists=list(obj.get("featured_artists", [])),
                parser_note=obj.get("parser_note", "") or "",
                genius_url=obj.get("genius_url", ""),
                lyrics=obj.get("lyrics", ""),
                lyrics_word_count=int(obj.get("lyrics_word_count", 0)),
            )
    return out


def append_checkpoint(path: Path, key: str, rec: LyricRecord) -> None:
    """
    Append one successful record to JSONL (crash-safe resumability).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_key": key, **asdict(rec)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def reclean_checkpoint_and_outputs(path: Path) -> tuple[int, int]:
    """
    Re-clean lyrics for all records in an existing checkpoint file, recompute
    `lyrics_word_count`, rewrite the checkpoint, and regenerate CSV/JSON outputs.

    Returns:
        (records_seen, records_updated)
    """
    if not path.exists():
        return 0, 0

    lines = path.read_text(encoding="utf-8").splitlines()
    rewritten: list[str] = []
    updated = 0
    seen = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        seen += 1

        lyrics_raw = obj.get("lyrics", "") or ""
        lyrics_clean = clean_lyrics(lyrics_raw, obj.get("track_title", "") or "")
        if lyrics_clean != lyrics_raw:
            updated += 1
        obj["lyrics"] = lyrics_clean
        obj["lyrics_word_count"] = len(re.findall(r"\b\w+\b", lyrics_clean))

        rewritten.append(json.dumps(obj, ensure_ascii=False))

    # Rewrite checkpoint deterministically (same objects, one per line).
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")

    # Regenerate outputs from checkpoint content.
    records = load_checkpoint(path)
    save_outputs(records.values())
    return seen, updated


def save_outputs(records: Iterable[LyricRecord]) -> None:
    """
    Save `kanye_lyrics.csv` and `kanye_lyrics.json` to the project root.
    """
    rows = []
    for r in records:
        rows.append(
            {
                "track_title": r.track_title,
                "album_or_era": r.album_or_era,
                "release_date": r.release_date,
                "track_type": r.track_type,
                "primary_artist": r.primary_artist,
                "featured_artists": r.featured_artists,
                "featured_artists_pipe": "|".join(r.featured_artists),
                "parser_note": r.parser_note,
                "genius_url": r.genius_url,
                "lyrics": r.lyrics,
                "lyrics_word_count": r.lyrics_word_count,
            }
        )

    # CSV (pipe-separated featured artists)
    df = pd.DataFrame(rows)
    csv_cols = [
        "track_title",
        "album_or_era",
        "release_date",
        "track_type",
        "primary_artist",
        "featured_artists_pipe",
        "parser_note",
        "genius_url",
        "lyrics",
        "lyrics_word_count",
    ]
    for c in csv_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[csv_cols].rename(columns={"featured_artists_pipe": "featured_artists"})
    OUT_CSV.write_text("", encoding="utf-8") if not OUT_CSV.exists() else None
    df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    # JSON (featured_artists is an array)
    json_rows = [
        {
            "track_title": r["track_title"],
            "album_or_era": r["album_or_era"],
            "release_date": r["release_date"],
            "track_type": r["track_type"],
            "primary_artist": r["primary_artist"],
            "featured_artists": r["featured_artists"],
            "parser_note": r.get("parser_note", ""),
            "genius_url": r["genius_url"],
            "lyrics": r["lyrics"],
            "lyrics_word_count": r["lyrics_word_count"],
        }
        for r in rows
    ]
    OUT_JSON.write_text(json.dumps(json_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def log_failed(reason: str, track: ParsedTrack, path: Path = FAILED_TXT) -> None:
    """
    Append a single failure line:
    [reason] track_title | primary_artist | album_or_era
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{reason}] {track.track_title} | {track.primary_artist} | {track.album_or_era}\n")


def retry_failed_tracks(token: str) -> tuple[int, int]:
    """
    Retry failures listed in `failed_tracks.txt`.

    Input line format:
        [reason] track_title | original_artist | era

    Behavior:
    - Re-attempt fetch using corrected primary_artist guardrails:
      if original_artist looks like a date/metadata => use "Kanye West" and store it as parser_note.
    - Append successes to checkpoint and regenerate outputs.
    - Rewrite failed_tracks.txt with only entries still failing.

    Returns:
        (recovered_count, still_failed_count)
    """
    if not FAILED_TXT.exists():
        return 0, 0

    lines = FAILED_TXT.read_text(encoding="utf-8", errors="replace").splitlines()
    existing = load_checkpoint(CHECKPOINT_JSONL)
    genius = create_genius_client(token)

    recovered = 0
    still_failed_lines: list[str] = []

    for line in tqdm(lines, desc="Retrying failed", unit="track"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\[(?P<reason>.*?)\]\s*(?P<title>.*?)\s*\|\s*(?P<artist>.*?)\s*\|\s*(?P<era>.*)\s*$", line)
        if not m:
            still_failed_lines.append(line)
            continue

        title = m.group("title").strip()
        orig_artist = m.group("artist").strip()
        era = m.group("era").strip()

        fixed_artist, note = _fix_primary_artist(orig_artist)
        key = _dedup_key(title, fixed_artist)
        if key in existing:
            # Already recovered on a previous run
            recovered += 1
            continue

        t = ParsedTrack(
            track_title=title,
            album_or_era=era,
            release_date="",
            track_type="",
            primary_artist=fixed_artist,
            featured_artists=[],
            parser_note=note,
        )

        payload, reason = fetch_lyrics(genius, t)
        if not payload:
            still_failed_lines.append(f"[{reason or 'failed'}] {title} | {orig_artist} | {era}")
            continue

        lyrics = payload["lyrics"]
        rec = LyricRecord(
            track_title=t.track_title,
            album_or_era=t.album_or_era,
            release_date=t.release_date,
            track_type=t.track_type,
            primary_artist=t.primary_artist,
            featured_artists=t.featured_artists,
            parser_note=t.parser_note,
            genius_url=payload.get("genius_url", ""),
            lyrics=lyrics,
            lyrics_word_count=len(re.findall(r"\b\w+\b", lyrics)),
        )
        existing[key] = rec
        append_checkpoint(CHECKPOINT_JSONL, key, rec)
        recovered += 1

        save_outputs(existing.values())
        time.sleep(DELAY_BETWEEN_CALLS_S)

    # Rewrite failed list with remaining failures only
    FAILED_TXT.write_text("\n".join(still_failed_lines) + ("\n" if still_failed_lines else ""), encoding="utf-8")
    save_outputs(existing.values())
    return recovered, len(still_failed_lines)


def _normalize_track_title_for_dedup(title: str) -> tuple[str, bool]:
    """
    Normalize a title for duplicate detection.

    Returns:
        (normalized_title, has_parenthetical)
    """
    t = (title or "").strip().lower()
    has_paren = bool(re.search(r"\([^)]*\)", t))
    # Keep parenthetical content in the normalized form so we don't merge remix vs non-remix
    # unless BOTH records share the same "paren-ness".
    t = re.sub(r"[^\w\s()]", " ", t)  # keep parens
    t = re.sub(r"\s+", " ", t).strip()
    return t, has_paren


def _should_consider_duplicate(a: LyricRecord, b: LyricRecord) -> bool:
    """
    Two records are considered duplicates if:
    - normalized titles match, AND
    - either primary artists match OR one is Kanye West OR one looks like metadata/date garbage.
    """
    ta, pa = _normalize_track_title_for_dedup(a.track_title)
    tb, pb = _normalize_track_title_for_dedup(b.track_title)
    if ta != tb:
        return False
    if pa != pb:
        return False

    aa = (a.primary_artist or "").strip()
    bb = (b.primary_artist or "").strip()
    if aa.lower() == bb.lower():
        return True
    if aa.lower() == "kanye west" or bb.lower() == "kanye west":
        return True
    if is_metadata_not_artist(aa) or is_metadata_not_artist(bb):
        return True
    return False


def _record_preference_score(rec: LyricRecord, *, order_index: int) -> tuple[int, int, int, int]:
    """
    Preference ordering when deduping.
    Higher tuple wins.

    Priority:
    - more complete lyrics (word count)
    - more authoritative classification for this project (Kanye primary > non-Kanye > garbage)
    - newer in checkpoint (order_index)
    """
    wc = int(rec.lyrics_word_count or 0)
    artist = (rec.primary_artist or "").strip()
    if artist.lower() == "kanye west":
        artist_score = 3
    elif is_metadata_not_artist(artist):
        artist_score = 0
    else:
        artist_score = 2
    # additional tiny bias: having a genius_url
    url_score = 1 if (rec.genius_url or "").strip() else 0
    return (wc, artist_score, url_score, order_index)


def dedupe_checkpoint_jsonl(path: Path) -> tuple[int, int]:
    """
    Deduplicate records in the checkpoint JSONL and rewrite it in-place.

    Returns:
        (records_before, records_after)
    """
    if not path.exists():
        return 0, 0

    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    parsed: list[tuple[int, str, dict[str, Any]]] = []
    for idx, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        key = obj.get("_key") or ""
        parsed.append((idx, key, obj))

    # Build LyricRecord objects for scoring/duplicate checks
    recs: list[tuple[int, str, LyricRecord, dict[str, Any]]] = []
    for idx, key, obj in parsed:
        rec = LyricRecord(
            track_title=obj.get("track_title", ""),
            album_or_era=obj.get("album_or_era", ""),
            release_date=obj.get("release_date", ""),
            track_type=obj.get("track_type", ""),
            primary_artist=obj.get("primary_artist", ""),
            featured_artists=list(obj.get("featured_artists", []) or []),
            parser_note=obj.get("parser_note", "") or "",
            genius_url=obj.get("genius_url", "") or "",
            lyrics=obj.get("lyrics", "") or "",
            lyrics_word_count=int(obj.get("lyrics_word_count", 0) or 0),
        )
        recs.append((idx, key, rec, obj))

    kept_indices: set[int] = set()
    groups: list[list[tuple[int, str, LyricRecord, dict[str, Any]]]] = []

    for i, item in enumerate(recs):
        if i in kept_indices:
            continue
        # group duplicates to item i
        group = [item]
        for j in range(i + 1, len(recs)):
            if j in kept_indices:
                continue
            if _should_consider_duplicate(item[2], recs[j][2]):
                group.append(recs[j])
        for g in group:
            kept_indices.add(recs.index(g))
        groups.append(group)

    # Choose winner per group
    winners: list[tuple[int, str, dict[str, Any]]] = []
    for group in groups:
        best = None
        best_score = None
        for (idx, key, rec, obj) in group:
            score = _record_preference_score(rec, order_index=idx)
            if best is None or score > best_score:
                best = (idx, key, obj)
                best_score = score
        assert best is not None
        winners.append(best)

    winners.sort(key=lambda t: t[0])  # preserve chronological checkpoint order
    rewritten = [json.dumps(obj, ensure_ascii=False) for (_idx, _key, obj) in winners]
    path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")
    return len(recs), len(winners)


# Known-missing recovery list: 6-tuple (no URL) or 7-tuple with optional Genius URL
# for direct fetch when search indexing doesn't match primary_artist.
KNOWN_MISSING_WITH_URLS: list[tuple[Any, ...]] = [
    ("American Boy", "Estelle", "2008 features", "2008-03-21", "Feature/Single", ["Kanye West"]),
    ("Knock You Down", "Keri Hilson", "2009 — Post-808s feature run", "2009", "Feature", ["Kanye West", "Ne-Yo"]),
    (
        "Niggas in Paris",
        "JAY-Z",
        "Watch the Throne",
        "2011-08-08",
        "Solo",
        ["Kanye West"],
        "https://genius.com/Jay-z-and-kanye-west-niggas-in-paris-lyrics",
    ),
    (
        "Go2DaMoon",
        "Playboi Carti",
        "2020",
        "2020-12-25",
        "Feature",
        ["Kanye West"],
        "https://genius.com/Playboi-carti-go2damoon-lyrics",
    ),
    (
        "Like That (Remix)",
        "Future",
        "2024",
        "2024-04-21",
        "Remix",
        ["Metro Boomin", "Kanye West", "Ty Dolla Sign"],
        "https://genius.com/Future-and-metro-boomin-like-that-remix-lyrics",
    ),
]


def _should_prefer_direct_genius_url(track_title: str, search_url: str, direct_url: str) -> bool:
    """
    Prefer scraping the explicit recovery URL when Genius search landed on a different song page
    (e.g. album cut vs remix, or different artist slug).
    """
    su = (search_url or "").lower().rstrip("/")
    du = (direct_url or "").lower().rstrip("/")
    if not du:
        return False
    if su == du:
        return False
    tl = track_title.lower()
    if "(remix)" in tl or tl.endswith(" remix"):
        su_slug = su.split("/")[-1] if su else ""
        du_slug = du.split("/")[-1] if du else ""
        if "remix" not in su_slug and "remix" in du_slug:
            return True
    su_slug = su.split("/")[-1] if su else ""
    du_slug = du.split("/")[-1] if du else ""
    return bool(su_slug and du_slug and su_slug != du_slug)


def _pick_lyrics_search_vs_direct(
    track_title: str,
    payload: Optional[dict[str, Any]],
    direct_url: Optional[str],
) -> tuple[str, str, str]:
    """
    Try search payload first; prefer direct Genius URL when appropriate.

    Returns:
        (lyrics, genius_url, note) note empty when search path used exclusively.
    """
    search_lyrics = (payload.get("lyrics") or "").strip() if payload else ""
    search_url = (payload.get("genius_url") or "").strip() if payload else ""

    direct_lyrics = ""
    direct_u = ""
    if direct_url:
        direct_lyrics, direct_u = fetch_lyrics_direct_from_url(direct_url.strip(), track_title)

    if direct_url and direct_lyrics:
        if not search_lyrics:
            return direct_lyrics, direct_u or direct_url.strip(), "used direct URL (search miss)"
        if _should_prefer_direct_genius_url(track_title, search_url, direct_url.strip()):
            return direct_lyrics, direct_u or direct_url.strip(), "used direct URL (search mismatch)"
        return search_lyrics, search_url, ""

    if search_lyrics:
        return search_lyrics, search_url, ""

    return "", "", "no lyrics"


def _scan_markdown_for_bold_quoted_titles(md_path: Path) -> list[tuple[str, str, str, str, str, list[str]]]:
    """
    Scan discography markdown for patterns like **"American Boy"** (Estelle, ...)
    and return tuples for titles that look recoverable.
    """
    text = md_path.read_text(encoding="utf-8", errors="replace")
    out: list[tuple[str, str, str, str, str, list[str]]] = []
    pat = re.compile(r'\*\*"\s*(?P<title>[^"]+?)\s*"\*\*(?P<rest>[^\n]*)')
    for line in text.splitlines():
        m = pat.search(line)
        if not m:
            continue
        title = m.group("title").strip()
        rest = m.group("rest") or ""
        # Attempt to parse "(Artist ...)" immediately following.
        mm = re.search(r"\((?P<inside>[^)]+)\)", rest)
        primary_artist = "Kanye West"
        featured: list[str] = []
        if mm:
            inside = _strip_markdown(mm.group("inside"))
            first = inside.split(",")[0].strip()
            primary_artist, featured = _parse_artist_credits(first)
            primary_artist, note = _fix_primary_artist(primary_artist)
        else:
            note = ""

        track_type = "Feature" if primary_artist.lower() != "kanye west" else "Solo"
        # Date best-effort: look for "Mon dd" patterns in the line, otherwise blank.
        date_guess = ""
        md = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:,\s*(\d{4}))?\b", line)
        if md:
            # don't guess year if absent
            frag = md.group(0)
            date_guess = _normalize_date_string(frag) if md.group(3) else frag
        out.append((title, primary_artist, "auto_recovered", date_guess, track_type, featured))
    return out


def _normalize_recovery_row(row: tuple[Any, ...]) -> tuple[str, str, str, str, str, list[str], Optional[str]]:
    if len(row) == 7:
        title, artist, era, rd, ttype, featured, url = row
        return (
            str(title),
            str(artist),
            str(era),
            str(rd),
            str(ttype),
            list(featured or []),
            str(url).strip() if url else None,
        )
    title, artist, era, rd, ttype, featured = row[:6]
    return (
        str(title),
        str(artist),
        str(era),
        str(rd),
        str(ttype),
        list(featured or []),
        None,
    )


def fetch_known_missing_tracks(token: str, md_path: Path) -> tuple[int, list[str]]:
    """
    Fetch known-missing tracks plus any additional bold-quoted titles missing
    from the dataset, and append successes to checkpoint.
    """
    genius = create_genius_client(token)
    existing = load_checkpoint(CHECKPOINT_JSONL)

    # Build normalized title index of existing records
    existing_titles = set()
    for rec in existing.values():
        nt, _hp = _normalize_track_title_for_dedup(rec.track_title)
        existing_titles.add(nt)

    # Normalized titles we explicitly recover — skip markdown duplicates with wrong artists.
    known_recovery_norm_titles = set()
    for row in KNOWN_MISSING_WITH_URLS:
        t = _normalize_recovery_row(tuple(row))[0]
        nt, _hp = _normalize_track_title_for_dedup(t)
        known_recovery_norm_titles.add(nt)

    recovery: list[tuple[str, str, str, str, str, list[str], Optional[str]]] = [
        _normalize_recovery_row(tuple(r)) for r in KNOWN_MISSING_WITH_URLS
    ]
    # Add auto-scanned bold-quoted titles if missing by normalized title
    for title, artist, era, rd, ttype, featured in _scan_markdown_for_bold_quoted_titles(md_path):
        nt, _hp = _normalize_track_title_for_dedup(title)
        if nt in existing_titles:
            continue
        if nt in known_recovery_norm_titles:
            continue
        recovery.append((title, artist, era, rd, ttype, featured, None))

    recovered = 0
    failed: list[str] = []

    for title, artist, era, rd, ttype, featured, direct_url in tqdm(
        recovery, desc="Recovering missing", unit="track"
    ):
        key = _dedup_key(title, artist)

        # Upgrade existing row if we have a canonical URL but checkpoint points elsewhere.
        if key in existing and direct_url:
            cur = existing[key]
            su = (cur.genius_url or "").strip()
            du = direct_url.strip()
            if su.lower().rstrip("/") != du.lower().rstrip("/"):
                dl, u = fetch_lyrics_direct_from_url(du, title)
                if dl:
                    existing[key] = LyricRecord(
                        track_title=title,
                        album_or_era=era,
                        release_date=rd,
                        track_type=ttype,
                        primary_artist=artist,
                        featured_artists=featured,
                        parser_note=cur.parser_note,
                        genius_url=u or du,
                        lyrics=dl,
                        lyrics_word_count=len(re.findall(r"\b\w+\b", dl)),
                    )
                    append_checkpoint(CHECKPOINT_JSONL, key, existing[key])
                    recovered += 1
                    save_outputs(existing.values())
                    time.sleep(DELAY_BETWEEN_CALLS_S)
            continue

        if key in existing:
            continue

        t = ParsedTrack(
            track_title=title,
            album_or_era=era,
            release_date=rd,
            track_type=ttype,
            primary_artist=artist,
            featured_artists=featured,
            parser_note="",
        )
        payload, reason = fetch_lyrics(genius, t)
        lyrics, genius_url, _note = _pick_lyrics_search_vs_direct(title, payload, direct_url)

        if not lyrics.strip():
            failed.append(f"{title} — {artist} ({reason or 'no lyrics'})")
            continue

        rec = LyricRecord(
            track_title=t.track_title,
            album_or_era=t.album_or_era,
            release_date=t.release_date,
            track_type=t.track_type,
            primary_artist=t.primary_artist,
            featured_artists=t.featured_artists,
            parser_note=t.parser_note,
            genius_url=genius_url,
            lyrics=lyrics,
            lyrics_word_count=len(re.findall(r"\b\w+\b", lyrics)),
        )
        existing[key] = rec
        append_checkpoint(CHECKPOINT_JSONL, key, rec)
        recovered += 1
        save_outputs(existing.values())
        time.sleep(DELAY_BETWEEN_CALLS_S)

    save_outputs(existing.values())
    return recovered, failed


def cleanup_and_recover(token: str) -> None:
    """
    Consolidated pass:
    - dedupe checkpoint
    - recover known-missing tracks (+ auto-scanned bold-quoted titles missing)
    - regenerate outputs
    """
    before, after = dedupe_checkpoint_jsonl(CHECKPOINT_JSONL)
    md_path = autodetect_discography_md()
    recovered, failed = fetch_known_missing_tracks(token, md_path)
    # final output regen
    records = load_checkpoint(CHECKPOINT_JSONL)
    save_outputs(records.values())

    print("\nCleanup summary")
    print("--------------")
    print(f"Records before dedup: {before}")
    print(f"Duplicates removed: {before - after}")
    print(f"Records after dedup: {after}")
    print(f"Newly recovered tracks: {recovered}")
    print(f"Final track count: {len(records)}")
    if failed:
        print("Known-missing still failing:")
        for s in failed[:20]:
            print(f"  - {s}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


# -----------------------------
# Main
# -----------------------------


def autodetect_discography_md() -> Path:
    """
    Find the discography markdown in the project root:
    - prefer the exact known filename when present
    - otherwise pick the most recently modified `.md` file in root
    """
    exact = PROJECT_ROOT / DEFAULT_DISCOGRAPHY_FILENAME
    if exact.exists():
        return exact
    md_files = sorted([p for p in PROJECT_ROOT.glob("*.md") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not md_files:
        raise SystemExit(f"No .md files found in project root: {PROJECT_ROOT}")
    return md_files[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Kanye-related track lyrics from Genius using a discography markdown."
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse only; no Genius API calls.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N tracks (after filtering).")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete outputs and checkpoint before running (clean smoke tests).",
    )
    parser.add_argument(
        "--reclean",
        action="store_true",
        help="Re-clean lyrics in the checkpoint and regenerate CSV/JSON (no refetch).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed_tracks.txt using improved artist/date parsing; rewrites failed list.",
    )
    parser.add_argument(
        "--cleanup-and-recover",
        action="store_true",
        help="Dedup checkpoint, recover known-missing tracks, and regenerate outputs.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("GENIUS_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("Missing GENIUS_API_TOKEN. Add it to .env (see .env.example).")

    if args.reclean:
        seen, updated = reclean_checkpoint_and_outputs(CHECKPOINT_JSONL)
        print(f"Recleaned checkpoint: {seen} records, {updated} changed.")
        print(f"Outputs: {OUT_CSV} | {OUT_JSON}")
        return 0

    if args.retry_failed:
        recovered, still_failed = retry_failed_tracks(token)
        print(f"Recovered: {recovered}")
        print(f"Still failed: {still_failed}")
        print(f"Outputs: {OUT_CSV} | {OUT_JSON} | {FAILED_TXT}")
        print(f"Checkpoint: {CHECKPOINT_JSONL}")
        return 0

    if args.cleanup_and_recover:
        cleanup_and_recover(token)
        return 0

    md_path = autodetect_discography_md()
    parsed = parse_discography(md_path)

    # Filter + dedup
    filtered: list[ParsedTrack] = []
    seen: set[str] = set()
    for t in parsed:
        skip, _reason = _is_skippable(t)
        if skip:
            continue
        key = _dedup_key(t.track_title, t.primary_artist)
        if key in seen:
            continue
        seen.add(key)
        # Ensure we always have at least an era year
        if not t.release_date:
            t = ParsedTrack(
                track_title=t.track_title,
                album_or_era=t.album_or_era,
                release_date="",
                track_type=t.track_type,
                primary_artist=t.primary_artist,
                featured_artists=t.featured_artists,
            )
        filtered.append(t)

    if args.dry_run:
        print(f"Discography file: {md_path}")
        print(f"Total tracks parsed from markdown: {len(parsed)}")
        print(f"Tracks after skip+dedup filters: {len(filtered)}")
        return 0

    if args.reset:
        for p in (FAILED_TXT, OUT_CSV, OUT_JSON, CHECKPOINT_JSONL):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    if args.limit and args.limit > 0:
        filtered = filtered[: args.limit]

    start = time.time()

    genius = create_genius_client(token)
    existing = load_checkpoint(CHECKPOINT_JSONL)

    attempted = 0
    ok = 0
    failed = 0
    skipped_cached = 0

    for t in tqdm(filtered, desc="Fetching lyrics", unit="track"):
        key = _dedup_key(t.track_title, t.primary_artist)
        if key in existing:
            skipped_cached += 1
            tqdm.write(f"↺ skipped (cached) {t.track_title} — {t.primary_artist}")
            continue

        attempted += 1

        payload, reason = fetch_lyrics(genius, t)
        if not payload:
            failed += 1
            tqdm.write(f"✗ failed ({reason}) {t.track_title} — {t.primary_artist}")
            log_failed(reason or "failed", t)
        else:
            lyrics = payload["lyrics"]
            rec = LyricRecord(
                track_title=t.track_title,
                album_or_era=t.album_or_era,
                release_date=t.release_date,
                track_type=t.track_type,
                primary_artist=t.primary_artist,
                featured_artists=t.featured_artists,
                parser_note=t.parser_note,
                genius_url=payload.get("genius_url", ""),
                lyrics=lyrics,
                lyrics_word_count=len(re.findall(r"\b\w+\b", lyrics)),
            )
            existing[key] = rec
            append_checkpoint(CHECKPOINT_JSONL, key, rec)
            ok += 1
            tqdm.write(f"✓ fetched {t.track_title} — {t.primary_artist}")

        save_outputs(existing.values())
        time.sleep(DELAY_BETWEEN_CALLS_S)

    save_outputs(existing.values())

    elapsed = time.time() - start
    print("\nSummary")
    print("-------")
    print(f"Discography file: {md_path}")
    print(f"Total tracks parsed from markdown: {len(parsed)}")
    print(f"Tracks attempted (after dedup and skip filter): {attempted}")
    print(f"Successful fetches: {ok}")
    print(f"Failed fetches: {failed}")
    print(f"Already-cached skips: {skipped_cached}")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"Outputs: {OUT_CSV} | {OUT_JSON} | {FAILED_TXT}")
    print(f"Checkpoint: {CHECKPOINT_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
