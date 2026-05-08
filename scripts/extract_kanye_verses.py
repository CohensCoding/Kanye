from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_CSV = PROJECT_ROOT / "kanye_lyrics.csv"
IN_JSON = PROJECT_ROOT / "kanye_lyrics.json"

OUT_CSV = PROJECT_ROOT / "kanye_verses_only.csv"
OUT_JSON = PROJECT_ROOT / "kanye_verses_only.json"
WARNINGS_TXT = PROJECT_ROOT / "extraction_warnings.txt"


SECTION_RE = re.compile(r"\[(?P<marker>[^\]]+)\]\s*(?P<body>.*?)(?=\n\s*\[[^\]]+\]|\Z)", re.DOTALL)

# Marker header kinds that are not actual lyric sections.
NON_ARTIST_MARKER_HINTS = [
    "sample",
    "samples",
    "translation",
    "translations",
    "producer",
    "produced",
    "interpolation",
]


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def parse_sections(lyrics: str) -> list[tuple[str, str, Optional[str]]]:
    """
    Split lyrics into (marker, body, credit_str_or_none) sections.

    - marker: full marker contents without brackets, e.g. "Verse 1: Kanye West"
    - body: text after the marker up to the next marker (not including the next marker)
    - credit_str: whatever appears after ':' in the marker, stripped, or None if absent
    """
    sections: list[tuple[str, str, Optional[str]]] = []
    for m in SECTION_RE.finditer(lyrics or ""):
        marker = (m.group("marker") or "").strip()
        body = (m.group("body") or "").strip("\n")
        credit = None
        if ":" in marker:
            _kind, rhs = marker.split(":", 1)
            credit = rhs.strip()
        sections.append((marker, body.strip(), credit))
    return sections


def marker_is_non_artist(marker: str) -> bool:
    ml = (marker or "").lower()
    return any(h in ml for h in NON_ARTIST_MARKER_HINTS) or ml.startswith("intro: sample")


def split_artist_credits(credit: str) -> list[str]:
    """
    Parse marker credits like:
    - "Kanye West & Rihanna"
    - "Kanye West, Rihanna"
    - "Kanye West with Charlie Wilson"
    """
    c = (credit or "").strip()
    if not c:
        return []
    c = re.sub(r"\s+", " ", c)
    # normalize separators
    c = re.sub(r"\s+with\s+", ", ", c, flags=re.I)
    c = c.replace("&", ",")
    parts = [p.strip() for p in c.split(",") if p.strip()]
    return parts


def credits_include_kanye(credit: Optional[str], *, primary_artist: str) -> bool:
    if not credit:
        return False
    names = [n.strip().lower() for n in split_artist_credits(credit)]
    if any(n in {"kanye west", "kanye"} for n in names):
        return True
    # "ye" is only treated as Kanye if primary_artist is Kanye West (per prompt)
    if primary_artist.strip().lower() == "kanye west" and "ye" in names:
        return True
    return False


def credits_are_explicit_non_kanye_only(credit: Optional[str], *, primary_artist: str) -> bool:
    """
    For Kanye-primary tracks: detect markers that explicitly credit ONLY non-Kanye artists,
    so we can exclude those sections.
    """
    if not credit:
        return False
    names = [n.strip().lower() for n in split_artist_credits(credit)]
    if not names:
        return False
    # if any kanye variant included, not "non-kanye only"
    if any(n in {"kanye west", "kanye"} for n in names):
        return False
    if primary_artist.strip().lower() == "kanye west" and "ye" in names:
        return False
    return True


@dataclass
class ExtractionResult:
    kanye_verses: str
    method: str
    excluded_sections: int
    markers_found: bool


def extract_kanye_verses_for_track(
    lyrics: str,
    *,
    primary_artist: str,
) -> ExtractionResult:
    sections = parse_sections(lyrics)
    markers_found = len(sections) > 0

    primary_is_kanye = primary_artist.strip().lower() == "kanye west"

    if not markers_found:
        if primary_is_kanye:
            return ExtractionResult(lyrics.strip(), "solo_track_no_markers", 0, False)
        return ExtractionResult(lyrics.strip(), "feature_no_markers_ambiguous", 0, False)

    # Markers exist
    kept: list[str] = []
    excluded = 0

    if primary_is_kanye:
        # Take everything by default, but drop sections explicitly credited to non-Kanye artists.
        for marker, body, credit in sections:
            if marker_is_non_artist(marker):
                excluded += 1
                continue
            if credits_are_explicit_non_kanye_only(credit, primary_artist=primary_artist):
                excluded += 1
                continue
            # keep sections that include kanye OR have no credit (default to lead on his own track)
            kept.append(body)
        method = "solo_album_filtered" if excluded > 0 else "solo_album_full"
        return ExtractionResult(_join_blocks(kept), method, excluded, True)

    # Feature track: only sections explicitly crediting Kanye
    for marker, body, credit in sections:
        if marker_is_non_artist(marker):
            continue
        if credits_include_kanye(credit, primary_artist=primary_artist):
            kept.append(body)
    return ExtractionResult(_join_blocks(kept), "feature_kanye_verses_only", excluded, True)


def _join_blocks(blocks: Iterable[str]) -> str:
    cleaned = [b.strip() for b in blocks if (b or "").strip()]
    return "\n\n".join(cleaned).strip()


def load_input_dataset() -> pd.DataFrame:
    """
    Prefer CSV for speed and schema stability; fall back to JSON if CSV missing.
    """
    if IN_CSV.exists():
        return pd.read_csv(IN_CSV)
    if IN_JSON.exists():
        data = json.loads(IN_JSON.read_text(encoding="utf-8"))
        return pd.DataFrame(data)
    raise SystemExit(f"Missing input dataset: expected {IN_CSV} or {IN_JSON}")


def write_outputs(df_out: pd.DataFrame) -> None:
    df_out.to_csv(OUT_CSV, index=False)
    # JSON array
    OUT_JSON.write_text(
        json.dumps(df_out.to_dict(orient="records"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Kanye-only verses from the Kanye lyrics dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print summary only; do not write files.")
    args = parser.parse_args()

    df = load_input_dataset()

    required_cols = {"track_title", "primary_artist", "lyrics", "lyrics_word_count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Input dataset missing required columns: {sorted(missing)}")

    warnings: list[str] = []
    method_counts: Counter[str] = Counter()
    big_cuts: list[str] = []

    total_orig_words = 0
    total_kanye_words = 0

    out_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        title = str(row.get("track_title", "") or "")
        primary_artist = str(row.get("primary_artist", "") or "")
        lyrics = str(row.get("lyrics", "") or "")

        orig_wc = int(row.get("lyrics_word_count", 0) or 0)
        total_orig_words += orig_wc

        res = extract_kanye_verses_for_track(lyrics, primary_artist=primary_artist)
        kanye_text = res.kanye_verses
        kanye_wc = word_count(kanye_text)
        total_kanye_words += kanye_wc

        method_counts[res.method] += 1

        if res.method == "feature_no_markers_ambiguous":
            warnings.append(f"[no markers] {title} — feature track with no section markers")

        if primary_artist.strip().lower() != "kanye west" and res.method == "feature_kanye_verses_only" and kanye_wc < 20:
            warnings.append(f"[short extraction] {title} — {kanye_wc} words")

        if orig_wc > 0 and kanye_wc / orig_wc < 0.2:
            big_cuts.append(f"{title} — {primary_artist} ({kanye_wc}/{orig_wc} words)")

        out = dict(row)
        out["original_word_count"] = orig_wc
        out["full_track_lyrics"] = lyrics
        out["kanye_verses"] = kanye_text
        out["kanye_verse_word_count"] = kanye_wc
        out["extraction_method"] = res.method

        # Replace lyrics column with kanye-only verses for downstream NLP focus
        out["lyrics"] = kanye_text
        out["lyrics_word_count"] = kanye_wc

        out_rows.append(out)

    df_out = pd.DataFrame(out_rows)

    # Write warnings
    if warnings and not args.dry_run:
        WARNINGS_TXT.write_text("\n".join(warnings) + "\n", encoding="utf-8")
    elif not args.dry_run:
        WARNINGS_TXT.write_text("", encoding="utf-8")

    # Summary
    print("\nSummary")
    print("-------")
    print(f"Total tracks processed: {len(df_out)}")
    print("Breakdown by extraction_method:")
    for k in sorted(method_counts.keys()):
        print(f"  {k}: {method_counts[k]}")
    print(f"Total Kanye-only word count: {total_kanye_words}")
    print(f"Total original word count: {total_orig_words}")
    print(f"Tracks flagged in extraction_warnings.txt: {len(warnings)}")
    print(f"Tracks with >80% reduction: {len(big_cuts)}")
    if big_cuts:
        # Print a small sample (highest-signal)
        print("Sample aggressive cuts (first 10):")
        for s in big_cuts[:10]:
            print(f"  {s}")

    if args.dry_run:
        return 0

    write_outputs(df_out)
    print(f"\nWrote: {OUT_CSV}")
    print(f"Wrote: {OUT_JSON}")
    print(f"Wrote: {WARNINGS_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

