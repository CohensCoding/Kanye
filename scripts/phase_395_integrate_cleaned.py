"""
Phase 3.95 — merge fetched payloads + debate-stripped Ye vs into data/kanye_cleaned.csv.

Does not print or log Genius tokens.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.5 Safari/605.1.15"
)

YE_VS_URL = "https://genius.com/Kanye-west-ye-vs-the-people-starring-ti-as-the-people-lyrics"
# Line indices AFTER dropping the leading "[Verse: Kanye West & T.I.]" marker line.
# Content-based T.I. attribution (Genius section tags ignored). Includes bars where T.I.
# addresses Kanye as "Ye" in third person (e.g. idx 61).
YE_VS_TI_INDICES = frozenset(
    {
        2,
        3,
        6,
        7,
        10,
        11,
        14,
        15,
        18,
        19,
        22,
        23,
        30,
        31,
        34,
        35,
        43,
        45,
        46,
        47,
        50,
        51,
        54,
        55,
        58,
        59,
        61,
        62,
    }
)

YE_VS_PARSER_NOTE = (
    "Structurally a debate with T.I. as “The People,” not a standard feature; "
    "all T.I. lines removed using recording-voice attribution (not Genius markup). "
    "Third-person “Ye” clauses voiced by T.I. are excluded as non-Kanye."
)

BILLIE_PARSER_NOTE = (
    "106 words from Kanye-tagged intro/outro markers; remix appearance, not a verse contribution."
)

FREAKY_PARSER_NOTE = (
    "Genius URL fallback (API hit scoring missed match): "
    "https://genius.com/Lil-pump-and-kanye-west-i-love-it-freaky-girl-edit-lyrics"
)

THERAFLU_PARSER_NOTE = (
    "Lyrics fetched from kanye-west-and-dj-khaled-cold-lyrics; track also released "
    "under aliases Way Too Cold and Cold."
)


def _word_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z']+", str(text or "")))


def _strip_only_one_blurb(kanye_verses: str) -> str:
    s = str(kanye_verses or "")
    key = "As I lay me down to sleep"
    i = s.find(key)
    return s[i:].lstrip() if i != -1 else s


def _fetch_ye_vs_kanye_verses() -> str:
    r = requests.get(YE_VS_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
    parts = []
    for c in containers:
        for br in c.find_all("br"):
            br.replace_with("\n")
        parts.append(c.get_text())
    text = "\n".join(parts)
    text = re.sub(r"^\d+\s*Contributors?.*?Lyrics", "", text, flags=re.DOTALL)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    start = 0
    for i, ln in enumerate(lines):
        if ln.startswith("[Verse:") and "T.I." in ln:
            start = i + 1
            break
    core = lines[start:]
    out: list[str] = []
    for i, ln in enumerate(core):
        if i in YE_VS_TI_INDICES:
            continue
        if ln.startswith("["):
            continue
        out.append(ln)
    return "\n\n".join(out)


def main() -> int:
    payload_path = PROJECT_ROOT / "data" / "phase_395_fetched_payloads.jsonl"
    cleaned_path = PROJECT_ROOT / "data" / "kanye_cleaned.csv"
    backup_path = PROJECT_ROOT / "data" / "kanye_cleaned.backup_phase395.csv"

    payloads: dict[str, dict] = {}
    with payload_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            payloads[str(obj["catalog_track_title"])] = obj

    df = pd.read_csv(cleaned_path)
    df.to_csv(backup_path, index=False)

    # --- Ye vs. the People ---
    mask_ye = df["track_title"] == "Ye vs. the People"
    if not mask_ye.any():
        print("ERROR: Ye vs. the People missing from kanye_cleaned.csv", file=sys.stderr)
        return 1
    try:
        kv_ye = _fetch_ye_vs_kanye_verses()
    except Exception as e:
        print(f"ERROR: Ye vs Genius fetch failed: {type(e).__name__}", file=sys.stderr)
        return 1
    df.loc[mask_ye, "kanye_verses"] = kv_ye
    df.loc[mask_ye, "parser_note"] = YE_VS_PARSER_NOTE
    df.loc[mask_ye, "extraction_method"] = "debate_ti_stripped"
    df.loc[mask_ye, "kanye_verse_word_count"] = _word_count(kv_ye)
    df.loc[mask_ye, "genius_url"] = YE_VS_URL

    # --- Payload-backed updates ---
    simple_updates = [
        "Impossible",
        "I Love It",
        "Only One",
        "All Day",
        "FourFiveSeconds",
    ]
    for title in simple_updates:
        p = payloads.get(title)
        if not p:
            print(f"WARN: missing payload for {title}", file=sys.stderr)
            continue
        m = df["track_title"] == title
        if not m.any():
            print(f"WARN: {title} not in kanye_cleaned — skipping", file=sys.stderr)
            continue
        kv = p["kanye_verses"]
        if title == "Only One":
            kv = _strip_only_one_blurb(kv)
        df.loc[m, "kanye_verses"] = kv
        df.loc[m, "full_track_lyrics"] = p["full_lyrics"]
        df.loc[m, "lyrics"] = p["full_lyrics"]
        df.loc[m, "genius_url"] = p["genius_url"]
        df.loc[m, "extraction_method"] = f"phase395_{p['extraction_method']}"
        df.loc[m, "lyrics_word_count"] = _word_count(p["full_lyrics"])
        df.loc[m, "original_word_count"] = _word_count(p["full_lyrics"])
        df.loc[m, "kanye_verse_word_count"] = _word_count(kv)

    # --- Theraflu → Therflu/Way Too Cold (Cold lyric body) ---
    p_tf = payloads.get("Theraflu")
    if not p_tf:
        print("ERROR: Theraflu payload missing", file=sys.stderr)
        return 1
    m_tf = df["track_title"] == "Theraflu"
    if not m_tf.any():
        print("ERROR: Theraflu row missing", file=sys.stderr)
        return 1
    df.loc[m_tf, "track_title"] = "Theraflu/Way Too Cold"
    df.loc[m_tf, "kanye_verses"] = p_tf["kanye_verses"]
    df.loc[m_tf, "full_track_lyrics"] = p_tf["full_lyrics"]
    df.loc[m_tf, "lyrics"] = p_tf["full_lyrics"]
    df.loc[m_tf, "genius_url"] = p_tf["genius_url"]
    df.loc[m_tf, "parser_note"] = THERAFLU_PARSER_NOTE
    df.loc[m_tf, "extraction_method"] = f"phase395_{p_tf['extraction_method']}"
    df.loc[m_tf, "lyrics_word_count"] = _word_count(p_tf["full_lyrics"])
    df.loc[m_tf, "original_word_count"] = _word_count(p_tf["full_lyrics"])
    df.loc[m_tf, "kanye_verse_word_count"] = _word_count(p_tf["kanye_verses"])

    # --- Billie Jean (new row) ---
    p_bj = payloads.get("Billie Jean (2008 Kanye West Mix)")
    if not p_bj:
        print("ERROR: Billie Jean payload missing", file=sys.stderr)
        return 1
    if (df["track_title"] == "Billie Jean (2008 Kanye West Mix)").any():
        print("WARN: Billie Jean already present — replacing row", file=sys.stderr)
        df = df[df["track_title"] != "Billie Jean (2008 Kanye West Mix)"]
    amazing = df[df["track_title"] == "Amazing"].iloc[0].to_dict()
    billie = amazing.copy()
    billie.update(
        {
            "track_title": "Billie Jean (2008 Kanye West Mix)",
            "album_or_era": "808s & Heartbreak",
            "release_date": "2008-02-08",
            "track_type": "intro_outro_only",
            "primary_artist": "Kanye West",
            "featured_artists": "Michael Jackson",
            "parser_note": BILLIE_PARSER_NOTE,
            "genius_url": p_bj["genius_url"],
            "lyrics": p_bj["full_lyrics"],
            "lyrics_word_count": _word_count(p_bj["full_lyrics"]),
            "original_word_count": _word_count(p_bj["full_lyrics"]),
            "full_track_lyrics": p_bj["full_lyrics"],
            "kanye_verses": p_bj["kanye_verses"],
            "kanye_verse_word_count": _word_count(p_bj["kanye_verses"]),
            "extraction_method": f"phase395_{p_bj['extraction_method']}",
            "era_clean": "808s & Heartbreak",
            "release_date_clean": "2008-02-08",
            "date_imputed": False,
            "track_id": "billie_jean_2008_kanye_west_mix_808s_heartbreak",
            "analysis_status": "analyze",
            "era_rank": amazing["era_rank"],
            "year": 2008,
            "date_obj": "2008-02-08",
        }
    )
    df = pd.concat([df, pd.DataFrame([billie])], ignore_index=True)

    # --- Freaky Girl Edit ---
    p_fg = payloads.get("Freaky Girl Edit")
    if not p_fg:
        print("ERROR: Freaky Girl Edit payload missing", file=sys.stderr)
        return 1
    if (df["track_title"] == "Freaky Girl Edit").any():
        df = df[df["track_title"] != "Freaky Girl Edit"]
    il_row = df[df["track_title"] == "I Love It"].iloc[0].to_dict()
    freaky = il_row.copy()
    freaky.update(
        {
            "track_title": "Freaky Girl Edit",
            "album_or_era": "Wyoming Sessions (ye / KSG)",
            "release_date": "2018-09-30",
            "track_type": "Solo",
            "primary_artist": "Kanye West",
            "featured_artists": "Lil Pump",
            "parser_note": FREAKY_PARSER_NOTE,
            "genius_url": p_fg["genius_url"],
            "lyrics": p_fg["full_lyrics"],
            "lyrics_word_count": _word_count(p_fg["full_lyrics"]),
            "original_word_count": _word_count(p_fg["full_lyrics"]),
            "full_track_lyrics": p_fg["full_lyrics"],
            "kanye_verses": p_fg["kanye_verses"],
            "kanye_verse_word_count": _word_count(p_fg["kanye_verses"]),
            "extraction_method": f"phase395_{p_fg['extraction_method']}",
            "era_clean": il_row["era_clean"],
            "era_rank": il_row["era_rank"],
            "release_date_clean": "2018-09-30",
            "date_imputed": False,
            "track_id": "freaky_girl_edit_wyoming_sessions_ye_ksg",
            "analysis_status": "analyze",
            "year": 2018,
            "date_obj": "2018-09-30",
        }
    )
    df = pd.concat([df, pd.DataFrame([freaky])], ignore_index=True)

    df.to_csv(cleaned_path, index=False)
    print(f"Wrote {cleaned_path} (backup {backup_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
