#!/usr/bin/env python3
"""
Phase 4b — Audit kanye_track_features.csv against the canonical catalog.

A row is in_canonical_catalog=True if its track_title matches (fuzzy) any of:
  - Studio / collab album tracks (same CANONICAL_ALBUMS dict as 00c_deep_audit.py)
  - Tier 2 singles listed in data/audit_tier2_singles.csv (discography.md-derived)
  - Tier 3 features listed in data/audit_tier3_features.csv
  - Phase 3.95 supplement targets in data/phase_3_95_target_list.csv (when present)

This mirrors the PDF-era catalog scope used elsewhere in the repo (discography.md +
hardcoded album tracklists), not mixtape/leak-only lines excluded from Tier 2/3.

Writes: data/phase_4b_ghost_tracks.csv
Does not delete or modify kanye_track_features.csv.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = PROJECT_ROOT / "data" / "kanye_track_features.csv"
OUT_CSV = PROJECT_ROOT / "data" / "phase_4b_ghost_tracks.csv"

# Same CANONICAL_ALBUMS as scripts/00c_deep_audit.py (studio + collab albums).
CANONICAL_ALBUMS: dict[str, list[str]] = {
    "The College Dropout (2004)": [
        "Intro",
        "We Don't Care",
        "Graduation Day",
        "All Falls Down",
        "I'll Fly Away",
        "Spaceship",
        "Jesus Walks",
        "Never Let Me Down",
        "Get Em High",
        "Workout Plan",
        "The New Workout Plan",
        "Slow Jamz",
        "Breathe In Breathe Out",
        "School Spirit Skit 1",
        "School Spirit",
        "School Spirit Skit 2",
        "Lil Jimmy Skit",
        "Two Words",
        "Through the Wire",
        "Family Business",
        "Last Call",
    ],
    "Late Registration (2005)": [
        "Wake Up Mr. West",
        "Heard 'Em Say",
        "Touch the Sky",
        "Gold Digger",
        "Skit #1",
        "Drive Slow",
        "My Way Home",
        "Crack Music",
        "Roses",
        "Bring Me Down",
        "Addiction",
        "Skit #2",
        "Diamonds from Sierra Leone (Remix)",
        "We Major",
        "Skit #3",
        "Hey Mama",
        "Celebration",
        "Skit #4",
        "Gone",
        "Diamonds from Sierra Leone",
        "Late",
    ],
    "Graduation (2007)": [
        "Good Morning",
        "Champion",
        "Stronger",
        "I Wonder",
        "Good Life",
        "Can't Tell Me Nothing",
        "Barry Bonds",
        "Drunk and Hot Girls",
        "Flashing Lights",
        "Everything I Am",
        "The Glory",
        "Homecoming",
        "Big Brother",
    ],
    "808s & Heartbreak (2008)": [
        "Say You Will",
        "Welcome to Heartbreak",
        "Heartless",
        "Amazing",
        "Love Lockdown",
        "Paranoid",
        "RoboCop",
        "Street Lights",
        "Bad News",
        "See You in My Nightmares",
        "Coldest Winter",
        "Pinocchio Story",
    ],
    "My Beautiful Dark Twisted Fantasy (2010)": [
        "Dark Fantasy",
        "Gorgeous",
        "Power",
        "All of the Lights (Interlude)",
        "All of the Lights",
        "Monster",
        "So Appalled",
        "Devil in a New Dress",
        "Runaway",
        "Hell of a Life",
        "Blame Game",
        "Lost in the World",
        "Who Will Survive in America",
    ],
    "Watch the Throne (2011)": [
        "No Church in the Wild",
        "Lift Off",
        "Niggas in Paris",
        "Otis",
        "Gotta Have It",
        "New Day",
        "That's My Bitch",
        "Welcome to the Jungle",
        "Who Gon Stop Me",
        "Murder to Excellence",
        "Made in America",
        "Why I Love You",
        "Illest Motherfucker Alive",
        "H•A•M",
        "Primetime",
        "The Joy",
    ],
    "Cruel Summer (2012)": [
        "To the World",
        "Clique",
        "Mercy",
        "New God Flow",
        "The Morning",
        "Cold",
        "Higher",
        "Sin City",
        "The One",
        "Creepers",
        "Bliss",
        "Don't Like (Remix)",
    ],
    "Yeezus (2013)": [
        "On Sight",
        "Black Skinhead",
        "I Am a God",
        "New Slaves",
        "Hold My Liquor",
        "I'm In It",
        "Blood on the Leaves",
        "Guilt Trip",
        "Send It Up",
        "Bound 2",
    ],
    "The Life of Pablo (2016)": [
        "Ultralight Beam",
        "Father Stretch My Hands Pt. 1",
        "Pt. 2",
        "Famous",
        "Feedback",
        "Low Lights",
        "Highlights",
        "Freestyle 4",
        "I Love Kanye",
        "Waves",
        "FML",
        "Real Friends",
        "Wolves",
        "Frank's Track",
        "Siiiiiiiiilver Surffffeeeeer Intermission",
        "30 Hours",
        "No More Parties in LA",
        "Facts (Charlie Heat Version)",
        "Fade",
        "Saint Pablo",
    ],
    "ye (2018)": [
        "I Thought About Killing You",
        "Yikes",
        "All Mine",
        "Wouldn't Leave",
        "No Mistakes",
        "Ghost Town",
        "Violent Crimes",
    ],
    "Kids See Ghosts (2018)": [
        "Feel the Love",
        "Fire",
        "4th Dimension",
        "Freeee (Ghost Town Pt. 2)",
        "Reborn",
        "Kids See Ghosts",
        "Cudi Montage",
    ],
    "Jesus Is King (2019)": [
        "Every Hour",
        "Selah",
        "Follow God",
        "Closed on Sunday",
        "On God",
        "Everything We Need",
        "Water",
        "God Is",
        "Hands On",
        "Use This Gospel",
        "Jesus Is Lord",
    ],
    "Donda (2021)": [
        "Donda Chant",
        "Jail",
        "God Breathed",
        "Off the Grid",
        "Hurricane",
        "Praise God",
        "Jonah",
        "Ok Ok",
        "Junya",
        "Believe What I Say",
        "24",
        "Remote Control",
        "Moon",
        "Heaven and Hell",
        "Donda",
        "Keep My Spirit Alive",
        "Jesus Lord",
        "New Again",
        "Tell the Vision",
        "Lord I Need You",
        "Pure Souls",
        "Come to Life",
        "No Child Left Behind",
        "Jail Pt. 2",
        "Ok Ok Pt. 2",
        "Junya Pt. 2",
        "Jesus Lord Pt. 2",
    ],
    "Donda Deluxe (2021)": [
        "Life of the Party",
        "Up From the Ashes",
        "Never Abandon Your Family",
        "Keep My Spirit Alive Pt. 2",
        "Remote Control Pt. 2",
    ],
    "Donda 2 (2022)": [
        "True Love",
        "Broken Road",
        "Get Lost",
        "Too Easy",
        "Flowers",
        "Security",
        "We Did It Kid",
        "Pablo",
        "Louie Bags",
        "Happy",
        "Sci Fi",
        "City of Gods",
        "Lord Lift Me Up",
        "First Time in a Long Time",
        "Selfish",
        "Link Up",
        "Daylight",
        "Possessions",
        "Life",
        "My Life Was Never Eazy",
        "We Made It",
        "Bridge to Lay Down",
    ],
    "Vultures 1 (2024)": [
        "Stars",
        "Keys to My Life",
        "Paid",
        "Talking",
        "Back to Me",
        "Hoodrat",
        "Do It",
        "Paperwork",
        "Burn",
        "Fuk Sumn",
        "Vultures",
        "Carnival",
        "Beg Forgiveness",
        "Good (Don't Die)",
        "Problematic",
        "King",
    ],
    "Vultures 2 (2024)": [
        "Slide",
        "Time Moving Slow",
        "Field Trip",
        "Lifestyle",
        "River",
        "Promotion",
        "Forever Rolling",
        "Husband",
        "Dead",
        "Bomb",
        "530",
        "Forever",
        "Maybe",
        "Fried",
        "Sky City",
        "My Soul",
    ],
    "Bully (2026)": [
        "King",
        "This a Must",
        "Father",
        "All the Love",
        "Sisters and Brothers",
        "Mama's Favorite",
        "Punch Drunk",
        "Whatever Works",
        "I Can't Wait",
        "Bully",
        "Highs and Lows",
        "Preacher Man",
        "Beauty and the Beast",
        "Last Breath",
        "White Lines",
        "Circles",
        "Damn",
        "This One Here",
    ],
}


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower().strip()).strip()


def is_match(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if len(na) > 5 and len(nb) > 5 and (na in nb or nb in na):
        return True
    return False


def find_in_dataset(candidate: str, catalog_titles: list[str]) -> bool:
    return any(is_match(candidate, c) for c in catalog_titles)


def collect_catalog_titles(root: Path) -> list[str]:
    titles: list[str] = []
    for tracks in CANONICAL_ALBUMS.values():
        titles.extend(tracks)

    tier2 = root / "data" / "audit_tier2_singles.csv"
    if tier2.exists():
        with tier2.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                t = (row.get("title") or "").strip()
                if t:
                    titles.append(t)

    tier3 = root / "data" / "audit_tier3_features.csv"
    if tier3.exists():
        with tier3.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                t = (row.get("title") or "").strip()
                if t:
                    titles.append(t)

    p395 = root / "data" / "phase_3_95_target_list.csv"
    if p395.exists():
        with p395.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                t = (row.get("title") or "").strip()
                if t:
                    titles.append(t)

    # Preserve order, drop exact duplicates
    seen: set[str] = set()
    out: list[str] = []
    for t in titles:
        k = normalize(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = ap.parse_args()
    root: Path = args.root

    feats_path = root / "data" / "kanye_track_features.csv"
    if not feats_path.exists():
        print(f"ERROR: {feats_path} not found", file=sys.stderr)
        return 1

    catalog = collect_catalog_titles(root)
    if not catalog:
        print("ERROR: no catalog titles loaded (missing audit CSVs?)", file=sys.stderr)
        return 1

    rows_out: list[dict[str, str]] = []
    with feats_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            title = row.get("track_title") or ""
            inc = find_in_dataset(title, catalog)
            rows_out.append(
                {
                    "track_id": row.get("track_id", ""),
                    "track_title": title,
                    "era_clean": row.get("era_clean", ""),
                    "year": row.get("year", ""),
                    "in_canonical_catalog": "True" if inc else "False",
                }
            )

    out_path = root / "data" / "phase_4b_ghost_tracks.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fnames = ["track_id", "track_title", "era_clean", "year", "in_canonical_catalog"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader()
        w.writerows(rows_out)

    n = len(rows_out)
    ghosts = [r for r in rows_out if r["in_canonical_catalog"] == "False"]
    print(f"Wrote {out_path} ({n} rows). Likely ghosts (in_canonical_catalog=False): {len(ghosts)}")
    for r in ghosts[:50]:
        print(f"  - {r['track_title']!r} ({r['track_id']})")
    if len(ghosts) > 50:
        print(f"  … {len(ghosts) - 50} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
