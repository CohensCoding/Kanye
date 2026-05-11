#!/usr/bin/env python3
"""
Phase 4b — Enrich ghost-track rows (in_canonical_catalog=False) with PDF + Genius URL audit.

Reads:  data/phase_4b_ghost_tracks.csv
         Kanye_West_Ye_Complete_Discography_and_Verse_Catalog__2003_to_May_2026.pdf
         data/kanye_cleaned.csv, missing_tracks_supplement*.csv, phase_395_fetched_payloads.jsonl

Writes: data/phase_4b_ghost_tracks_enriched.csv

Does not modify other datasets.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GHOST_CSV = PROJECT_ROOT / "data" / "phase_4b_ghost_tracks.csv"
OUT_CSV = PROJECT_ROOT / "data" / "phase_4b_ghost_tracks_enriched.csv"
PDF_PATH = PROJECT_ROOT / "Kanye_West_Ye_Complete_Discography_and_Verse_Catalog__2003_to_May_2026.pdf"
KANYE_CLEANED = PROJECT_ROOT / "data" / "kanye_cleaned.csv"
SUPP1 = PROJECT_ROOT / "data" / "missing_tracks_supplement.csv"
SUPP2 = PROJECT_ROOT / "data" / "missing_tracks_supplement_2.csv"
JSONL = PROJECT_ROOT / "data" / "phase_395_fetched_payloads.jsonl"

# Strip trailing parenthetical / bracketed release tags (iterative).
_SUFFIX_CHUNK = re.compile(
    r"""
    \s*[\[(]
    (?:
        remix | remaster | skit | live | explicit | clean \s* version
        | album \s* version | radio \s* edit | extended \s* version | extended
        | single \s* version | instrumental | acoustic | demo | edit | main
        | pt \.? \s* \d+ | part \s* \d+ | vol \.? \s* \d+
        | feat [^)\]]* | featuring [^)\]]*
    )
    [^)\]]*
    [\])]
    \s*$
    """,
    re.I | re.VERBOSE,
)


def strip_suffixes(title: str) -> str:
    t = title.strip()
    for _ in range(12):
        nt = _SUFFIX_CHUNK.sub("", t).strip()
        if nt == t:
            break
        t = nt
    return t


def soft_normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compact_alnum(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def title_variants(title: str) -> list[str]:
    raw = title.strip()
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    add(raw)
    add(strip_suffixes(raw))
    add(raw.rstrip("."))
    add(strip_suffixes(raw.rstrip(".")))
    return out


def title_keys(title: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for v in title_variants(title):
        k = re.sub(r"\s+", " ", v.lower().strip())
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
        if k.endswith("."):
            k2 = k.rstrip(".")
            if k2 and k2 not in seen:
                seen.add(k2)
                keys.append(k2)
    return keys


def extract_pdf_quotes(pdf_path: Path) -> list[str]:
    text_parts: list[str] = []
    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    full = "\n".join(text_parts)

    quotes: list[str] = []
    seen: set[str] = set()
    for pat in (
        r'"([^"]{2,200})"',
        r"\u201c([^\u201d]{2,200})\u201d",
        r"\u2018([^\u2019]{2,200})\u2019",
    ):
        for m in re.finditer(pat, full):
            q = m.group(1).strip()
            if len(q) >= 2 and q not in seen:
                seen.add(q)
                quotes.append(q)

    return quotes


def best_pdf_match(track_title: str, pdf_quotes: list[str]) -> tuple[bool, str]:
    """Match only against quoted spans in the PDF (not unquoted prose)."""
    variants = title_variants(track_title)
    best_score = -1.0
    best_evidence = ""

    for var in variants:
        sn = soft_normalize(var)
        comp = compact_alnum(var)

        for q in pdf_quotes:
            qn = soft_normalize(q)
            if not qn:
                continue
            ts = fuzz.token_set_ratio(sn, qn)
            if ts > best_score:
                best_score = float(ts)
                best_evidence = q

        if len(comp) >= 8 and comp:
            for q in pdf_quotes:
                qc = compact_alnum(q)
                if not qc:
                    continue
                if comp in qc or (len(qc) >= 8 and qc in comp):
                    best_score = max(best_score, 95.0)
                    best_evidence = q
                    break

        if len(comp) <= 7 and comp:
            for q in pdf_quotes:
                qc = compact_alnum(q)
                if comp == qc or (len(comp) >= 4 and comp == qc[: len(comp)]):
                    best_score = max(best_score, 98.0)
                    best_evidence = q
                    break

    threshold = 82.0
    if best_score >= threshold and best_evidence:
        return True, best_evidence
    return False, "none"


def genius_slug(url: str) -> str:
    u = url.strip().lower()
    if "genius.com" not in u:
        return ""
    tail = u.split("genius.com", 1)[1].split("?", 1)[0].strip("/")
    slug = tail.split("/")[-1]
    if slug.endswith("-lyrics"):
        slug = slug[: -len("-lyrics")]
    if slug.endswith("-annotated"):
        slug = slug[: -len("-annotated")]
    return slug


def genius_url_plausible(url: str, track_title: str) -> bool:
    """Reject obvious track_id ↔ Genius mismatches in upstream CSVs."""
    slug = genius_slug(url)
    if not slug:
        return False
    sc = compact_alnum(slug)
    tc = compact_alnum(track_title)
    if not tc:
        return False
    if len(tc) < 6:
        return tc in sc
    if tc in sc:
        return True
    words = [w for w in soft_normalize(track_title).split() if len(w) >= 4]
    if len(words) >= 2:
        hit = sum(1 for w in words if w in slug)
        return hit >= max(2, (len(words) + 1) // 2)
    if len(words) == 1:
        return words[0] in slug
    return False


def url_credits_kanye(url: str) -> str:
    if not url or url.strip().lower() == "unknown":
        return "unknown"
    slug = genius_slug(url)
    if not slug:
        return "unknown"

    if "kanye-west" in slug:
        return "True"
    if "-kanye-west" in slug or slug.endswith("-kanye-west"):
        return "True"
    if slug.startswith("kanye-west-"):
        return "True"
    if "feat-kanye-west" in slug or "featuring-kanye-west" in slug:
        return "True"
    if "with-kanye-west" in slug or "and-kanye-west" in slug:
        return "True"

    if slug.startswith("ye-"):
        bad = (
            "year-",
            "years-",
            "yeast-",
            "yellow-",
            "yeet-",
            "yeah-",
            "yes-",
            "yesterday-",
            "yet-",
            "yelp-",
            "yen-",
        )
        if any(slug.startswith(b) for b in bad):
            return "False"
        return "True"

    if "feat-ye-" in slug or "-feat-ye-" in slug or slug.endswith("-feat-ye"):
        return "True"

    return "False"


def recommended_action(pdf_match: bool, url_cred: str) -> str:
    if pdf_match or url_cred == "True":
        return "keep — matcher missed it"
    if not pdf_match and url_cred == "False":
        return "exclude — real ghost"
    return "review — ambiguous"


def load_genius_by_track_id(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = (row.get("track_id") or "").strip()
            url = (row.get("genius_url") or "").strip()
            if tid and url:
                out[tid] = url
    return out


def load_genius_by_title(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                title = (row.get("track_title") or "").strip()
                url = (row.get("genius_url") or "").strip()
                if not title or not url:
                    continue
                for k in title_keys(title):
                    out[k] = url
    return out


def load_genius_from_jsonl(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = (obj.get("track_title") or obj.get("catalog_track_title") or "").strip()
            url = (obj.get("genius_url") or "").strip()
            if not title or not url:
                continue
            for k in title_keys(title):
                out[k] = url
    return out


def resolve_genius_url(
    track_id: str,
    track_title: str,
    by_id: dict[str, str],
    by_title_supp: dict[str, str],
    by_title_jsonl: dict[str, str],
) -> str:
    u = by_id.get(track_id, "").strip()
    if u and genius_url_plausible(u, track_title):
        return u
    for k in title_keys(track_title):
        u = by_title_supp.get(k, "").strip()
        if u and genius_url_plausible(u, track_title):
            return u
    for k in title_keys(track_title):
        u = by_title_jsonl.get(k, "").strip()
        if u and genius_url_plausible(u, track_title):
            return u
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pdf",
        type=Path,
        default=PDF_PATH,
        help="Path to the Kanye discography PDF",
    )
    args = ap.parse_args()

    pdf_quotes = extract_pdf_quotes(args.pdf)

    by_id = load_genius_by_track_id(KANYE_CLEANED)
    by_title_supp = load_genius_by_title([SUPP1, SUPP2])
    by_title_jsonl = load_genius_from_jsonl(JSONL)

    fieldnames = [
        "track_id",
        "track_title",
        "era_clean",
        "year",
        "in_canonical_catalog",
        "pdf_catalog_match",
        "pdf_catalog_match_evidence",
        "genius_url",
        "url_credits_kanye",
        "recommended_action",
    ]

    rows_out: list[dict[str, str]] = []
    with GHOST_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            flag = (row.get("in_canonical_catalog") or "").strip().lower()
            if flag != "false":
                continue
            tid = (row.get("track_id") or "").strip()
            title = (row.get("track_title") or "").strip()
            era = (row.get("era_clean") or "").strip()
            year = (row.get("year") or "").strip()

            pdf_m, evidence = best_pdf_match(title, pdf_quotes)
            gurl = resolve_genius_url(tid, title, by_id, by_title_supp, by_title_jsonl)
            url_cred = url_credits_kanye(gurl)
            action = recommended_action(pdf_m, url_cred)

            rows_out.append(
                {
                    "track_id": tid,
                    "track_title": title,
                    "era_clean": era,
                    "year": year,
                    "in_canonical_catalog": "False",
                    "pdf_catalog_match": str(pdf_m),
                    "pdf_catalog_match_evidence": evidence if pdf_m else "none",
                    "genius_url": gurl,
                    "url_credits_kanye": url_cred,
                    "recommended_action": action,
                }
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
