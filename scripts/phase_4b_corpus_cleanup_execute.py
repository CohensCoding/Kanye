#!/usr/bin/env python3
"""
One-shot Phase 4b corpus cleanup: 4 hard-deletes + 10 feature re-flags.

Run from repo root after reviewing row-count preview.
Regenerates ghost CSVs via phase_4b_ghost_tracks*.py (invoked by subprocess).
"""
from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"

EXCLUDE_IDS = frozenset(
    {
        "live_from_irving_plaza_pre_dropout_era",
        "eternal_rest_jesus_is_king",
        "pro_nails_graduation",
        "skyscrapers_cruel_summer_yeezus_build_up",
    }
)

EXCLUSION_ROWS: list[dict[str, str]] = [
    {
        "track_id": "live_from_irving_plaza_pre_dropout_era",
        "track_title": "Live from Irving Plaza",
        "era_clean": "Pre-Dropout era",
        "year": "2003",
        "exclusion_reason": (
            "Live performance of pre-existing material; excluded per same principle as Late Orchestration "
            "and VH1 Storytellers — live performances distort era-level analysis by counting performance dates "
            "rather than composition dates"
        ),
        "genius_url_at_exclusion": "https://genius.com/Kanye-west-live-from-irving-plaza-ny-lyrics",
        "fetch_source": "kanye_cleaned.csv",
    },
    {
        "track_id": "eternal_rest_jesus_is_king",
        "track_title": "Eternal Rest",
        "era_clean": "Jesus Is King",
        "year": "2019",
        "exclusion_reason": (
            "Genius URL resolves to Mistweaver (metal band) page, not Kanye; URL appears to be a wrong-fetch "
            "from original extraction; no evidence of Kanye contribution to a track of this title"
        ),
        "genius_url_at_exclusion": "https://genius.com/Mistweaver-eternal-rest-lyrics",
        "fetch_source": "kanye_cleaned.csv",
    },
    {
        "track_id": "pro_nails_graduation",
        "track_title": "Pro Nails",
        "era_clean": "Graduation",
        "year": "2007",
        "exclusion_reason": (
            "Non-Kanye Genius page; primary artist is Kid Sister; no Kanye verse contribution in captured lines"
        ),
        "genius_url_at_exclusion": "https://genius.com/Kid-sister-pro-nails-lyrics",
        "fetch_source": "kanye_cleaned.csv",
    },
    {
        "track_id": "skyscrapers_cruel_summer_yeezus_build_up",
        "track_title": "Skyscrapers",
        "era_clean": "Cruel Summer + Yeezus build-up",
        "year": "2012",
        "exclusion_reason": (
            "Non-Kanye Genius page; primary artist is Swizz Beatz; no confident Kanye verse attribution in corpus. "
            "Line content review found 8 lines of substantive lyrical material that could not be attributed with "
            "confidence to Kanye vs. other artists on the collaboration; excluded on the principle that ambiguous "
            "content is more costly than missing content for verse-level analysis."
        ),
        "genius_url_at_exclusion": "https://genius.com/Swizz-beatz-skyscrapers-lyrics",
        "fetch_source": "kanye_cleaned.csv",
    },
]

NOTE_LINE = (
    "{primary} primary; Kanye featured. Originally categorized as ghost/non-feature in Phase 4b audit; "
    "line content review confirmed Kanye verse contribution. Re-flagged as Tier 3 feature."
)

NOTE_URL = (
    "{primary} primary; Kanye featured. Genius URL credits Kanye as featured artist; original pipeline "
    "track_type did not reflect Tier 3 feature status. Phase 4b corpus cleanup corrected misclassification "
    "(not based on dedicated line content review). Re-flagged as Tier 3 feature."
)

REFLAG: dict[str, tuple[str, str]] = {
    "feel_me_quiet_year": ("Tyga", "line"),
    "love_yourself_quiet_year": ("Mary J. Blige", "line"),
    "glow_quiet_year": ("Drake", "line"),
    "dat_side_quiet_year": ("CyHi", "line"),
    "top_this_late_registration": ("Bump J", "line"),
    "buy_you_a_drank_remix_graduation": ("T-Pain", "line"),
    "go_late_registration": ("Common", "url"),
    "because_of_you_remix_graduation": ("Ne-Yo", "url"),
    "go2damoon_donda": ("Playboi Carti", "url"),
    "rock_n_roll_donda_2": ("Pusha T", "url"),
}


def filter_csv_by_track_id(path: Path, out_path: Path | None, drop: frozenset[str]) -> int:
    removed = 0
    out_path = out_path or path
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = []
        fn = r.fieldnames
        assert fn and "track_id" in fn, path
        for row in r:
            if row["track_id"] in drop:
                removed += 1
                continue
            rows.append(row)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows)
    return removed


def filter_verses_only(path: Path, meta: list[dict[str, str]]) -> int:
    urls = {m["genius_url"].strip().lower() for m in meta if m.get("genius_url")}
    pairs = {(m["track_title"].strip().lower(), m["album_or_era"].strip().lower()) for m in meta}
    removed = 0
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fn = r.fieldnames
        rows = []
        for row in r:
            gu = (row.get("genius_url") or "").strip().lower()
            tt = (row.get("track_title") or "").strip().lower()
            ae = (row.get("album_or_era") or "").strip().lower()
            if gu and gu in urls:
                removed += 1
                continue
            if (tt, ae) in pairs:
                removed += 1
                continue
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows)
    return removed


def apply_reflang_track_features(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fn = list(r.fieldnames or [])
        if "parser_note" not in fn:
            fn.append("parser_note")
        rows = []
        for row in r:
            tid = row.get("track_id", "")
            if tid in REFLAG:
                primary, kind = REFLAG[tid]
                tpl = NOTE_LINE if kind == "line" else NOTE_URL
                row["track_type"] = "Feature"
                row["parser_note"] = tpl.format(primary=primary)
            else:
                row.setdefault("parser_note", "")
            rows.append({k: row.get(k, "") for k in fn})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows)


def apply_reflang_cleaned(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fn = r.fieldnames
        assert fn
        rows = []
        for row in r:
            tid = row.get("track_id", "")
            if tid in REFLAG:
                primary, kind = REFLAG[tid]
                tpl = NOTE_LINE if kind == "line" else NOTE_URL
                row["track_type"] = "Feature"
                row["parser_note"] = tpl.format(primary=primary)
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fn))
        w.writeheader()
        w.writerows(rows)


def write_exclusions_csv() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = DATA / "phase_4b_exclusions.csv"
    fieldnames = [
        "track_id",
        "track_title",
        "era_clean",
        "year",
        "exclusion_reason",
        "genius_url_at_exclusion",
        "fetch_source",
        "excluded_timestamp",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in EXCLUSION_ROWS:
            w.writerow({**row, "excluded_timestamp": ts})


def main() -> int:
    meta = []
    with (DATA / "kanye_cleaned.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["track_id"] in EXCLUDE_IDS:
                album = (row.get("album_or_era") or row.get("era_clean") or "").strip()
                meta.append(
                    {
                        "track_id": row["track_id"],
                        "track_title": row["track_title"],
                        "album_or_era": album,
                        "genius_url": (row.get("genius_url") or "").strip(),
                    }
                )

    paths_tid = [
        DATA / "kanye_line_features_enriched.csv",
        DATA / "kanye_line_features.csv",
        DATA / "kanye_track_features.csv",
        DATA / "kanye_track_topics.csv",
        DATA / "kanye_cleaned.csv",
        DATA / "kanye_track_performance.csv",
        DATA / "kanye_track_sonic_derived.csv",
        DATA / "phase_3_5_pronoun_features.csv",
        PROJECT_ROOT / "kanye_track_event_proximity.csv",
    ]

    for p in paths_tid:
        n = filter_csv_by_track_id(p, None, EXCLUDE_IDS)
        print(f"filtered {p.relative_to(PROJECT_ROOT)} removed={n}")

    n = filter_verses_only(PROJECT_ROOT / "kanye_verses_only.csv", meta)
    print(f"filtered kanye_verses_only.csv removed={n}")

    apply_reflang_track_features(DATA / "kanye_track_features.csv")
    apply_reflang_cleaned(DATA / "kanye_cleaned.csv")
    print("applied 10 re-flags to kanye_track_features.csv and kanye_cleaned.csv")

    write_exclusions_csv()
    print("wrote data/phase_4b_exclusions.csv (4 rows)")

    # Regenerate ghost audit from cleaned track list (overwrites ghost CSVs).
    py = "/usr/bin/python3"
    subprocess.check_call([py, str(PROJECT_ROOT / "scripts" / "phase_4b_ghost_tracks.py")], cwd=PROJECT_ROOT)
    subprocess.check_call([py, str(PROJECT_ROOT / "scripts" / "phase_4b_ghost_tracks_enriched.py")], cwd=PROJECT_ROOT)
    print("regenerated phase_4b_ghost_tracks*.csv")

    subprocess.check_call([py, str(PROJECT_ROOT / "scripts" / "03_aggregate_and_tfidf.py")], cwd=PROJECT_ROOT)
    subprocess.check_call(
        [
            py,
            str(PROJECT_ROOT / "scripts" / "00c_deep_audit.py"),
            "--md",
            str(PROJECT_ROOT / "discography.md"),
        ],
        cwd=PROJECT_ROOT,
    )
    print("ran 03_aggregate_and_tfidf.py and 00c_deep_audit.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
