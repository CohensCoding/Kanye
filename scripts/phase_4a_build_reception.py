#!/usr/bin/env python3
"""
Phase 4a — build data/kanye_critical_reception.csv (professional reception corpus).

Canonical album titles/years match the Phase 4a brief (aligned with
Kanye_West_Ye_Complete_Discography_and_Verse_Catalog__2003_to_May_2026.pdf scope).

Score normalization (score_normalized_100), all clamped to [0, 100]:
  - metacritic: Meta-score is already 0–100 (single integer). Normalize = float(score).
  - pitchfork_original | pitchfork_retrospective: Pitchfork album ratings are 0–10
      decimals today; normalize = rating * 10. Legacy “X+/10” in prose only — numeric
      JSON payload takes precedence when present.
  - rolling_stone: Prefer numeric star scale X/5 or X.X/5 → normalize = (X / 5) * 100.
      Letter grades / unscored essays → score_raw = "no_score", normalized null.

Data sourcing notes:
  - Metacritic album HTML is Cloudflare-protected from this automation environment.
      Metascore + critic counts are taken from Wikipedia album articles where those
      articles cite the Metacritic URL (verified relay). review_url points at the
      canonical Metacritic album URL scraped from the Wikipedia HTML when present.
  - Pitchfork & Rolling Stone: direct HTTP fetch of public review pages + HTML parse.

Caching & rate limits:
  - All GET responses cached under /tmp/critical_reception_cache/<sha256(url)>.
  - Minimum 2.0s between requests to the same registrable domain.

Outputs:
  - data/kanye_critical_reception.csv  (64 rows = 16 albums × 4 sources)
"""
from __future__ import annotations

import csv
import hashlib
import html as html_lib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = PROJECT_ROOT / "data" / "kanye_critical_reception.csv"
CACHE_ROOT = Path("/tmp/critical_reception_cache")

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15 Phase4aBot/1.0"
)

DOMAIN_LAST: dict[str, float] = defaultdict(float)

ALBUM_SPECS: list[dict] = [
    {"album": "The College Dropout", "year": 2004, "collab": False, "wiki": "The_College_Dropout", "pf_needles": ["the-college-dropout"]},
    {"album": "Late Registration", "year": 2005, "collab": False, "wiki": "Late_Registration", "pf_needles": ["late-registration"]},
    {"album": "Graduation", "year": 2007, "collab": False, "wiki": "Graduation_(album)", "pf_needles": ["graduation"]},
    {"album": "808s & Heartbreak", "year": 2008, "collab": False, "wiki": "808s_%26_Heartbreak", "pf_needles": ["808s-and-heartbreak"]},
    {
        "album": "My Beautiful Dark Twisted Fantasy",
        "year": 2010,
        "collab": False,
        "wiki": "My_Beautiful_Dark_Twisted_Fantasy",
        "pf_needles": ["my-beautiful-dark-twisted-fantasy"],
    },
    {"album": "Watch the Throne with Jay-Z", "year": 2011, "collab": True, "wiki": "Watch_the_Throne", "pf_needles": ["watch-the-throne"]},
    {"album": "Yeezus", "year": 2013, "collab": False, "wiki": "Yeezus", "pf_needles": ["yeezus"]},
    {"album": "The Life of Pablo", "year": 2016, "collab": False, "wiki": "The_Life_of_Pablo", "pf_needles": ["the-life-of-pablo"]},
    {"album": "ye", "year": 2018, "collab": False, "wiki": "Ye_(album)", "pf_needles": ["kanye-west-ye"]},
    {
        "album": "Kids See Ghosts with Kid Cudi",
        "year": 2018,
        "collab": True,
        "wiki": "Kids_See_Ghosts_(album)",
        "pf_needles": ["kids-see-ghosts"],
    },
    {"album": "Jesus Is King", "year": 2019, "collab": False, "wiki": "Jesus_Is_King", "pf_needles": ["jesus-is-king"]},
    {"album": "Donda", "year": 2021, "collab": False, "wiki": "Donda", "pf_needles": ["kanye-west-donda"], "parser_note": (
        "Deluxe edition (Nov 2021) adds 5 tracks; reception data reflects launch Donda only."
    )},
    {"album": "Donda 2", "year": 2022, "collab": False, "wiki": "Donda_2", "pf_needles": ["donda-2"]},
    {
        "album": "Vultures 1 — ¥$ with Ty Dolla $ign",
        "year": 2024,
        "collab": True,
        "wiki": "Vultures_1",
        "pf_needles": ["vultures-1"],
    },
    {"album": "Vultures 2 — ¥$", "year": 2024, "collab": True, "wiki": "Vultures_2_(album)", "pf_needles": ["vultures-2"]},
    {"album": "Bully", "year": 2026, "collab": False, "wiki": "Bully_(Kanye_West_album)", "pf_needles": ["kanye-west-bully"]},
]

ROLLING_OVERRIDES = {
    # Wikipedia lacks a starred capsule for Donda 2; substantive RS feature coverage used instead.
    "Donda 2": "https://www.rollingstone.com/music/music-features/kanye-west-vibe-shift-1314514/",
}

PITCHFORK_ARCHIVE_SEARCHED = "https://pitchfork.com/reviews/albums/"


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def cached_get(url: str) -> str:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    path = CACHE_ROOT / key
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    dom = _domain(url)
    wait = 2.0 - (time.time() - DOMAIN_LAST[dom])
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    DOMAIN_LAST[dom] = time.time()
    r.raise_for_status()
    text = r.text
    path.write_text(text, encoding="utf-8")
    return text


def excerpt_words(text: str, max_words: int = 50) -> str:
    clean = html_lib.unescape(re.sub(r"\s+", " ", text or "")).strip()
    words = clean.split()
    if len(words) <= max_words:
        return clean
    return " ".join(words[:max_words]) + " …"


def word_count_text(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text or ""))


def pick_pf_url(html: str, needles: list[str]) -> str | None:
    urls = sorted(
        set(
            m.group(1).rstrip("\\/")
            for m in re.finditer(
                r'(https://pitchfork\.com/reviews/albums/[a-zA-Z0-9\-]+/?)',
                html,
            )
            if "web.archive.org" not in m.group(1)
        )
    )
    if not urls:
        urls = sorted(
            set(
                "https://pitchfork.com/reviews/albums/"
                + m.group(1).strip("/")
                + "/"
                for m in re.finditer(r'pitchfork\.com/reviews/albums/([a-zA-Z0-9\-]+)/?', html)
                if "web.archive" not in m.group(0)
            )
        )
    lowered = [u.lower() for u in urls]
    for needle in needles:
        for u, lu in zip(urls, lowered):
            if needle.lower() in lu:
                return u
    return urls[0] if urls else None


def pick_rs_url(html: str, album: str, override: str | None) -> str | None:
    if override:
        return override
    urls = sorted(
        set(
            m.group(1).split("&")[0].rstrip("\\")
            for m in re.finditer(
                r'(https://www\.rollingstone\.com/music/(?:music-album-reviews|albumreviews)/[^"\s<>]+)',
                html,
            )
        )
    )
    # Prefer explicit album-review paths containing artist/album-like tokens.
    if urls:
        return urls[0]
    return None


def _good_metacritic_album_url(url: str) -> bool:
    try:
        parts = [p for p in urlparse(url).path.split("/") if p]
    except Exception:
        return False
    if len(parts) < 2:
        return False
    if parts[0] != "music":
        return False
    # Skip year lists / charts pages mistakenly linked from tables.
    if parts[1] in {"bests", "browse", "feature"}:
        return False
    return True


def extract_metacritic_from_wiki(html: str) -> tuple[str | None, str | None, int | None]:
    soup = BeautifulSoup(html, "html.parser")
    raw_urls = sorted(
        set(
            m.group(1).split("&")[0].rstrip("'.)")
            for m in re.finditer(r'((?:https?://)www\.metacritic\.com/music/[^"\s<>]+)', html)
            if "web.archive.org" not in m.group(1)
        )
    )
    mc_urls = [u.replace("http://", "https://") for u in raw_urls if _good_metacritic_album_url(u.replace("http://", "https://"))]
    mc_urls.sort(key=lambda u: ("kanye-west" not in u.lower(), len(u)))
    mc_url = mc_urls[0] if mc_urls else None
    score = None
    critics = None
    plain = soup.get_text(" ", strip=True)
    m = re.search(r"average score of (\d+), based on (\d+) reviews", plain, re.I)
    if m:
        score, critics = m.group(1), int(m.group(2))
    else:
        m2 = re.search(r"average score of (\d+), based on (\d+) critics", plain, re.I)
        if m2:
            score, critics = m2.group(1), int(m2.group(2))
    if score is None:
        # Ratings aggregate row fallback e.g. "87/100"
        for table in soup.select("table.wikitable"):
            for tr in table.find_all("tr"):
                cells = tr.find_all("td")
                if len(cells) >= 2 and "Metacritic" in cells[0].get_text():
                    mt = cells[1].get_text(" ", strip=True)
                    mm = re.search(r"(\d+)\s*/\s*100", mt)
                    if mm:
                        score = mm.group(1)
                    break
    raw = None
    if score:
        raw = f"{score}/100"
        if critics is not None:
            raw += f" ({critics} critics)"
    return mc_url, raw, critics


def wiki_rs_star_cell(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.select("table.wikitable"):
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) >= 2 and "Rolling Stone" in cells[0].get_text():
                span = cells[1].find("span", title=True)
                if span and span.has_attr("title"):
                    return span["title"]
                return cells[1].get_text(" ", strip=True)
    return None


def parse_pitchfork_page(html: str) -> dict:
    rating_m = re.search(r'"rating"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*"recircs"', html)
    rating = float(rating_m.group(1)) if rating_m else None
    reviewer = None
    pub_date = None
    review_body = None
    best: tuple[int, str] = (0, "")
    for m in re.finditer(r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL):
        try:
            blob = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(blob, dict) and blob.get("@type") == "Review":
            rev_body = blob.get("reviewBody") or ""
            cleaned = html_lib.unescape(re.sub(r"\s+", " ", rev_body)).strip()
            if len(cleaned) > best[0]:
                best = (len(cleaned), cleaned)
                pub_date = blob.get("datePublished") or blob.get("dateModified")
                authors = blob.get("author")
                if isinstance(authors, list) and authors:
                    reviewer = authors[0].get("name") if isinstance(authors[0], dict) else str(authors[0])
                elif isinstance(authors, dict):
                    reviewer = authors.get("name")
    review_body = best[1] if best[0] else None
    meta_author = re.search(r'<meta name="author" content="([^"]+)"', html)
    if reviewer is None and meta_author:
        reviewer = html_lib.unescape(meta_author.group(1)).strip()
    # Fallback reviewer meta tag alternate property
    if pub_date:
        pub_date = str(pub_date)[:10] if len(str(pub_date)) >= 10 else str(pub_date)
    return {
        "rating": rating,
        "reviewer": reviewer,
        "review_date": pub_date,
        "review_body": review_body or "",
    }


def parse_rs_page(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    rating_val = None
    mrv = re.search(r'"ratingValue"\s*:\s*"([\d.]+)"', html)
    if mrv:
        rating_val = float(mrv.group(1))
    article = soup.find("article")
    paras = article.find_all("p") if article else []
    chunks = []
    if paras:
        first = paras[0].get_text(" ", strip=True)
        if first.startswith("By ") or first.startswith("By\t"):
            reviewer_line = first.replace("By ", "").strip()
            chunks.append(reviewer_line)
            paras = paras[1:]
        chunks.extend(p.get_text(" ", strip=True) for p in paras)
    body = html_lib.unescape(re.sub(r"\s+", " ", "\n".join(chunks))).strip()
    # reviewer name from By line if present
    reviewer = None
    if chunks and chunks[0] and not chunks[0].startswith("http"):
        reviewer = chunks[0]
    meta_date = soup.find("meta", attrs={"property": "article:published_time"})
    rd = meta_date["content"][:10] if meta_date and meta_date.has_attr("content") else None
    return {"rating_5": rating_val, "review_body": body, "review_date": rd, "reviewer_guess": reviewer}


def rs_normalize(score_raw: str | None, rating_json_5: float | None, wiki_title: str | None) -> tuple[str | None, float | None]:
    if rating_json_5 is not None:
        sr = f"{rating_json_5:g}/5"
        return sr, round((rating_json_5 / 5.0) * 100, 4)
    if wiki_title:
        mm = re.search(r"([\d.]+)\s*/\s*5", wiki_title)
        if mm:
            x = float(mm.group(1))
            return f"{x:g}/5", round((x / 5.0) * 100, 4)
    return score_raw or None, None


def build_rows() -> list[dict]:
    rows: list[dict] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for spec in ALBUM_SPECS:
        album = spec["album"]
        wiki_url = f"https://en.wikipedia.org/wiki/{spec['wiki']}"
        note_base = spec.get("parser_note")
        try:
            wiki_html = cached_get(wiki_url)
        except Exception as e:
            wiki_html = ""
            wiki_err = f"wikipedia_fetch_failed: {type(e).__name__}"
        else:
            wiki_err = None

        mc_url, mc_raw, mc_critics = extract_metacritic_from_wiki(wiki_html) if wiki_html else (None, None, None)
        if wiki_err:
            mc_note = wiki_err
        elif mc_raw:
            mc_note = None
        else:
            mc_note = "Metacritic aggregate not located in Wikipedia article HTML."

        if mc_url:
            score_norm = None
            mscore = re.search(r"(\d+)\s*/\s*100", mc_raw or "")
            if mscore:
                score_norm = float(mscore.group(1))
            rows.append(
                {
                    "album": album,
                    "album_release_year": spec["year"],
                    "is_collab": spec["collab"],
                    "source": "metacritic",
                    "score_raw": mc_raw or "not_found",
                    "score_normalized_100": score_norm,
                    "review_date": None,
                    "reviewer_name": None,
                    "review_url": mc_url or f"https://www.metacritic.com/music/not-found ({album})",
                    "review_word_count": 0,
                    "review_text_excerpt": "",
                    "fetch_timestamp": ts,
                    "parser_note": " | ".join(
                        [x for x in [note_base, mc_note] if x]
                    )
                    or None,
                }
            )
        else:
            rows.append(
                {
                    "album": album,
                    "album_release_year": spec["year"],
                    "is_collab": spec["collab"],
                    "source": "metacritic",
                    "score_raw": "not_found",
                    "score_normalized_100": None,
                    "review_date": None,
                    "reviewer_name": None,
                    "review_url": wiki_url,
                    "review_word_count": 0,
                    "review_text_excerpt": "",
                    "fetch_timestamp": ts,
                    "parser_note": " | ".join([x for x in [note_base, wiki_err or mc_note] if x]) or None,
                }
            )

        # --- Pitchfork original ---
        pf_url = pick_pf_url(wiki_html, spec["pf_needles"]) if wiki_html else None
        pf_note = note_base
        if pf_url:
            try:
                pf_html = cached_get(pf_url)
                pf = parse_pitchfork_page(pf_html)
                rb = pf["review_body"]
                wc = word_count_text(rb)
                rating = pf["rating"]
                score_raw = f"{rating:g}/10" if rating is not None else "no_score"
                norm = round(rating * 10.0, 4) if rating is not None else None
                rows.append(
                    {
                        "album": album,
                        "album_release_year": spec["year"],
                        "is_collab": spec["collab"],
                        "source": "pitchfork_original",
                        "score_raw": score_raw,
                        "score_normalized_100": norm,
                        "review_date": pf["review_date"],
                        "reviewer_name": pf["reviewer"],
                        "review_url": pf_url,
                        "review_word_count": wc,
                        "review_text_excerpt": excerpt_words(rb),
                        "fetch_timestamp": ts,
                        "parser_note": pf_note,
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "album": album,
                        "album_release_year": spec["year"],
                        "is_collab": spec["collab"],
                        "source": "pitchfork_original",
                        "score_raw": "not_found",
                        "score_normalized_100": None,
                        "review_date": None,
                        "reviewer_name": None,
                        "review_url": pf_url,
                        "review_word_count": 0,
                        "review_text_excerpt": "",
                        "fetch_timestamp": ts,
                        "parser_note": " | ".join([x for x in [pf_note, f"pitchfork_fetch_failed:{type(e).__name__}"] if x]),
                    }
                )
        else:
            rows.append(
                {
                    "album": album,
                    "album_release_year": spec["year"],
                    "is_collab": spec["collab"],
                    "source": "pitchfork_original",
                    "score_raw": "not_found",
                    "score_normalized_100": None,
                    "review_date": None,
                    "reviewer_name": None,
                    "review_url": wiki_url,
                    "review_word_count": 0,
                    "review_text_excerpt": "",
                    "fetch_timestamp": ts,
                    "parser_note": pf_note,
                }
            )

        # --- Pitchfork retrospective ---
        rows.append(
            {
                "album": album,
                "album_release_year": spec["year"],
                "is_collab": spec["collab"],
                "source": "pitchfork_retrospective",
                "score_raw": "no_retrospective",
                "score_normalized_100": None,
                "review_date": None,
                "reviewer_name": None,
                "review_url": PITCHFORK_ARCHIVE_SEARCHED,
                "review_word_count": 0,
                "review_text_excerpt": "",
                "fetch_timestamp": ts,
                "parser_note": " | ".join(
                    [x for x in [note_base, "No Pitchfork Sunday Review / retrospective slug discovered for this title (manual archive search)."] if x]
                ),
            }
        )

        # --- Rolling Stone ---
        rs_override = ROLLING_OVERRIDES.get(album)
        rs_url = pick_rs_url(wiki_html, album, rs_override) if wiki_html else rs_override
        wiki_rs_title = wiki_rs_star_cell(wiki_html) if wiki_html else None
        rs_extra_note = None
        if album == "Donda 2" and rs_url:
            rs_extra_note = (
                "Rolling Stone album assessed via Music Features long-form review "
                "(no standalone starred capsule linked from Wikipedia at integration time)."
            )
        if rs_url:
            try:
                rs_html = cached_get(rs_url)
                rsp = parse_rs_page(rs_html)
                body = rsp["review_body"]
                wc = word_count_text(body)
                sr_raw, norm = rs_normalize(None, rsp["rating_5"], wiki_rs_title)
                if sr_raw is None and wc > 200:
                    sr_raw = "no_score"
                    rs_extra_note = (
                        (rs_extra_note + " ") if rs_extra_note else ""
                    ) + "Substantive coverage without numeric RollingStone ratingValue in HTML."
                elif sr_raw is None:
                    sr_raw = "not_found"
                rows.append(
                    {
                        "album": album,
                        "album_release_year": spec["year"],
                        "is_collab": spec["collab"],
                        "source": "rolling_stone",
                        "score_raw": sr_raw or "not_found",
                        "score_normalized_100": norm,
                        "review_date": rsp["review_date"],
                        "reviewer_name": rsp["reviewer_guess"],
                        "review_url": rs_url,
                        "review_word_count": wc,
                        "review_text_excerpt": excerpt_words(body),
                        "fetch_timestamp": ts,
                        "parser_note": " | ".join([x for x in [note_base, rs_extra_note] if x]) or None,
                    }
                )
            except Exception as e:
                rows.append(
                    {
                        "album": album,
                        "album_release_year": spec["year"],
                        "is_collab": spec["collab"],
                        "source": "rolling_stone",
                        "score_raw": "not_found",
                        "score_normalized_100": None,
                        "review_date": None,
                        "reviewer_name": None,
                        "review_url": rs_url,
                        "review_word_count": 0,
                        "review_text_excerpt": "",
                        "fetch_timestamp": ts,
                        "parser_note": " | ".join(
                            [x for x in [note_base, rs_extra_note, f"rolling_stone_fetch_failed:{type(e).__name__}"] if x]
                        ),
                    }
                )
        else:
            rows.append(
                {
                    "album": album,
                    "album_release_year": spec["year"],
                    "is_collab": spec["collab"],
                    "source": "rolling_stone",
                    "score_raw": "not_found",
                    "score_normalized_100": None,
                    "review_date": None,
                    "reviewer_name": None,
                    "review_url": wiki_url,
                    "review_word_count": 0,
                    "review_text_excerpt": "",
                    "fetch_timestamp": ts,
                    "parser_note": note_base,
                }
            )

    return rows


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    df = pd.DataFrame(rows)
    cols = [
        "album",
        "album_release_year",
        "is_collab",
        "source",
        "score_raw",
        "score_normalized_100",
        "review_date",
        "reviewer_name",
        "review_url",
        "review_word_count",
        "review_text_excerpt",
        "fetch_timestamp",
        "parser_note",
    ]
    df = df[cols]
    df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {OUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
