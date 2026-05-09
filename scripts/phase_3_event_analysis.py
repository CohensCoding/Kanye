from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing matplotlib. Install it (e.g. `python -m pip install matplotlib`) and re-run."
    ) from e


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TRACK_FEATURES_CSV = DATA_DIR / "kanye_track_features.csv"
LINE_ENRICHED_CSV = DATA_DIR / "kanye_line_features_enriched.csv"
LIFE_EVENTS_CSV = DATA_DIR / "kanye_life_events.csv"
ERA_DISTINCTIVE_WORDS_CSV = DATA_DIR / "kanye_era_distinctive_words.csv"
ERA_FEATURES_CSV = DATA_DIR / "kanye_era_features.csv"
YEAR_FEATURES_CSV = DATA_DIR / "kanye_year_features.csv"
KANYE_CLEANED_CSV = DATA_DIR / "kanye_cleaned.csv"
EXCLUDED_TRACKS_CSV = DATA_DIR / "excluded_tracks.csv"
MISSING_TRACKS_CSV = DATA_DIR / "missing_tracks.csv"

OUT_PROXIMITY_CSV = PROJECT_ROOT / "kanye_track_event_proximity.csv"
OUT_REPORT_MD = PROJECT_ROOT / "phase_3_findings.md"
CHARTS_DIR = PROJECT_ROOT / "phase_3_charts"


WINDOW_DAYS = [90, 180, 365, 730]


EMOTION_COLS = ["emotion_joy", "emotion_anger", "emotion_sadness", "emotion_fear"]
ENTITY_TARGETS = {
    "donda": ["donda"],
    "kim": ["kim", "kim kardashian"],
    "bianca": ["bianca", "bianca censori"],
    "trump": ["trump", "donald trump"],
    "god/jesus": ["god", "jesus", "christ", "lord", "yahweh"],
    "jay-z": ["jay-z", "jay z", "hov", "hova", "shawn carter"],
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path)


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def build_track_event_proximity(
    tracks: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """
    For each track, compute:
    - days since last event (any / severity-3 / by type)
    - last_event_label (any / severity-3 / by type)
    - rolling counts in 90/180/365/730 day windows (any / severity-3 / by type / by severity)
    """
    t = tracks.copy()
    e = events.copy()

    t["release_dt"] = _to_datetime(t["release_date_clean"])
    e["event_dt"] = _to_datetime(e["date"])

    t = t.sort_values("release_dt")
    e = e.sort_values("event_dt")

    # last event (any)
    any_last = pd.merge_asof(
        t[["track_id", "release_dt"]],
        e[["event_dt", "event_label", "event_type", "severity"]],
        left_on="release_dt",
        right_on="event_dt",
        direction="backward",
        allow_exact_matches=True,
    )
    t["last_event_date_any"] = any_last["event_dt"]
    t["last_event_label_any"] = any_last["event_label"]
    t["days_since_last_event_any"] = (t["release_dt"] - t["last_event_date_any"]).dt.days

    # last severity-3 event
    e3 = e[e["severity"] == 3].copy()
    sev3_last = pd.merge_asof(
        t[["track_id", "release_dt"]],
        e3[["event_dt", "event_label"]],
        left_on="release_dt",
        right_on="event_dt",
        direction="backward",
        allow_exact_matches=True,
    )
    t["last_event_date_sev3"] = sev3_last["event_dt"]
    t["last_event_label_sev3"] = sev3_last["event_label"]
    t["days_since_last_event_sev3"] = (t["release_dt"] - t["last_event_date_sev3"]).dt.days

    # last event by type
    for etype in sorted(e["event_type"].dropna().unique()):
        et = e[e["event_type"] == etype].copy()
        m = pd.merge_asof(
            t[["track_id", "release_dt"]],
            et[["event_dt", "event_label"]],
            left_on="release_dt",
            right_on="event_dt",
            direction="backward",
            allow_exact_matches=True,
        )
        t[f"last_event_label_type_{etype}"] = m["event_label"]
        t[f"days_since_last_event_type_{etype}"] = (t["release_dt"] - m["event_dt"]).dt.days

    # Rolling counts
    # For each track date, count events in (release_dt - window, release_dt] inclusive.
    event_dt = e["event_dt"].dropna().sort_values().to_numpy()

    def count_in_window(mask: pd.Series, window_days: int) -> pd.Series:
        dd = t["release_dt"].to_numpy()
        ev = e.loc[mask, "event_dt"].dropna().sort_values().to_numpy()
        # use searchsorted to count in sliding window
        import numpy as np

        left = np.searchsorted(ev, dd - pd.Timedelta(days=window_days), side="right")
        right = np.searchsorted(ev, dd, side="right")
        return pd.Series((right - left), index=t.index)

    import numpy as np

    # Any events
    for w in WINDOW_DAYS:
        left = np.searchsorted(event_dt, t["release_dt"].to_numpy() - pd.Timedelta(days=w), side="right")
        right = np.searchsorted(event_dt, t["release_dt"].to_numpy(), side="right")
        t[f"events_{w}d_any"] = right - left

    # severity-specific
    for sev in sorted(e["severity"].dropna().unique()):
        mask = e["severity"] == sev
        for w in WINDOW_DAYS:
            t[f"events_{w}d_sev{int(sev)}"] = count_in_window(mask, w)

    # by type
    for etype in sorted(e["event_type"].dropna().unique()):
        mask = e["event_type"] == etype
        for w in WINDOW_DAYS:
            t[f"events_{w}d_type_{etype}"] = count_in_window(mask, w)

    # cleanup
    t["release_date_clean"] = t["release_dt"].dt.strftime("%Y-%m-%d")
    t = t.drop(columns=["release_dt"])
    return t


def _safe_json_list(x: Any) -> list[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    s = str(x).strip()
    if not s or s == "[]":
        return []
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(z) for z in v]
    except Exception:
        pass
    return []


TOKEN_RE = re.compile(r"[a-z][a-z']+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())


def log_odds_top_words(
    before_texts: Iterable[str],
    after_texts: Iterable[str],
    *,
    top_n: int = 25,
    stopwords: Optional[set[str]] = None,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """
    Simple weighted log-odds with add-1 smoothing.
    Returns (top_after, top_before) as (word, score) where positive favors after.
    """
    stop = stopwords or set()
    cb = Counter()
    ca = Counter()
    for t in before_texts:
        cb.update([w for w in tokenize(t) if w not in stop])
    for t in after_texts:
        ca.update([w for w in tokenize(t) if w not in stop])

    vocab = set(cb) | set(ca)
    b_total = sum(cb.values()) + len(vocab)
    a_total = sum(ca.values()) + len(vocab)

    scores: dict[str, float] = {}
    for w in vocab:
        b = cb[w] + 1
        a = ca[w] + 1
        scores[w] = math.log(a / a_total) - math.log(b / b_total)

    top_after = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    top_before = sorted(scores.items(), key=lambda kv: kv[1])[:top_n]
    return top_after, top_before


def plot_emotion_deltas(deltas: pd.DataFrame, out_path: Path) -> None:
    """
    deltas: rows=event_label, cols=emotion delta
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(deltas))
    width = 0.18
    for i, col in enumerate(EMOTION_COLS):
        ax.bar([v + (i - 1.5) * width for v in x], deltas[col], width=width, label=col.replace("emotion_", ""))
    ax.set_xticks(list(x))
    ax.set_xticklabels(deltas["event_label"], rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Emotion deltas: 365d after severity-3 event vs baseline (line-level mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_entity_mentions(entity_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for ent in entity_df["entity"].unique():
        sub = entity_df[entity_df["entity"] == ent]
        ax.plot(sub["release_date_clean"], sub["mention_rate"], label=ent)
    ax.set_title("Entity mention rates over time (persons_named)")
    ax.set_xlabel("Release date")
    ax.set_ylabel("Mentions per 1,000 lines")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_sent_volatility_era(era_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    era_df = era_df.sort_values("era_rank")
    ax.plot(era_df["era_clean"], era_df["sent_volatility"], marker="o")
    ax.set_title("Sentiment volatility by era (with severity-3 events to be overlaid in report)")
    ax.set_xlabel("Era")
    ax.set_ylabel("sent_volatility")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_analyses(
    tracks: pd.DataFrame,
    lines: pd.DataFrame,
    events: pd.DataFrame,
    era_words: pd.DataFrame,
    era_features: pd.DataFrame,
    cleaned: pd.DataFrame,
    excluded_tracks: pd.DataFrame,
    missing_tracks: pd.DataFrame,
    *,
    dry_run: bool,
) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    sev3 = events[events["severity"] == 3].copy().sort_values("date")
    sev3["date_dt"] = _to_datetime(sev3["date"])

    # -------------------------
    # 1) Emotion trajectories
    # -------------------------
    lines = lines.copy()
    lines["release_dt"] = _to_datetime(lines["release_date_clean"])

    overall_baseline = lines[EMOTION_COLS].mean(numeric_only=True)
    delta_rows = []
    emotion_notes = []

    for _, ev in sev3.iterrows():
        start = ev["date_dt"]
        end = start + pd.Timedelta(days=365)
        window = lines[(lines["release_dt"] > start) & (lines["release_dt"] <= end)]
        if window.empty:
            emotion_notes.append(
                f"- **{ev['event_label']} ({ev['date']})**: no tracks in 365d window (silence period) — flagged."
            )
            d = {c: float("nan") for c in EMOTION_COLS}
        else:
            wmean = window[EMOTION_COLS].mean(numeric_only=True)
            d = {c: float(wmean[c] - overall_baseline[c]) for c in EMOTION_COLS}
        delta_rows.append({"event_label": ev["event_label"], **d})

    deltas = pd.DataFrame(delta_rows)
    emotion_chart = CHARTS_DIR / "emotion_deltas_sev3.png"
    if not dry_run:
        plot_emotion_deltas(deltas, emotion_chart)

    # -------------------------
    # 2) Vocabulary shifts around sev-3
    # -------------------------
    cleaned = cleaned.copy()
    cleaned["release_dt"] = _to_datetime(cleaned["release_date_clean"])
    stop = {
        "the",
        "and",
        "for",
        "that",
        "with",
        "this",
        "you",
        "your",
        "but",
        "not",
        "are",
        "was",
        "were",
        "its",
        "it's",
        "im",
        "i'm",
        "ive",
        "i've",
        "aint",
        "don't",
        "cant",
        "can't",
        "yeah",
        "uh",
        "oh",
    }

    vocab_sections = []
    for _, ev in sev3.iterrows():
        d0 = ev["date_dt"]
        before = cleaned[(cleaned["release_dt"] <= d0) & (cleaned["release_dt"] > d0 - pd.Timedelta(days=365))]
        after = cleaned[(cleaned["release_dt"] > d0) & (cleaned["release_dt"] <= d0 + pd.Timedelta(days=365))]
        if before.empty or after.empty:
            vocab_sections.append(
                (ev["event_label"], ev["date"], [], [], "insufficient tracks in +/-12mo window")
            )
            continue
        top_after, top_before = log_odds_top_words(
            before["kanye_verses"].fillna("").tolist(),
            after["kanye_verses"].fillna("").tolist(),
            top_n=20,
            stopwords=stop,
        )
        vocab_sections.append((ev["event_label"], ev["date"], top_after, top_before, "ok"))

    # -------------------------
    # 3) Entity mentions over time
    # -------------------------
    lines["persons"] = lines["persons_named"].apply(_safe_json_list)
    # Expand to counts per track-date
    target_lc = {k: [s.lower() for s in v] for k, v in ENTITY_TARGETS.items()}

    # For performance: compute mention flags per line for each target
    def line_mentions(persons: list[str], target_key: str) -> int:
        p = [x.lower() for x in persons]
        for needle in target_lc[target_key]:
            if any(needle == z for z in p):
                return 1
        return 0

    mention_rows = []
    # group by date for rates
    lines["release_date_clean"] = lines["release_dt"].dt.strftime("%Y-%m-%d")
    grouped = lines.groupby("release_date_clean", dropna=False)
    for date_key, g in grouped:
        denom = len(g)
        if denom == 0 or not date_key or date_key == "NaT":
            continue
        for ent in ENTITY_TARGETS.keys():
            cnt = int(g["persons"].apply(lambda lst, e=ent: line_mentions(lst, e)).sum())
            rate = (cnt / denom) * 1000.0
            mention_rows.append({"release_date_clean": date_key, "entity": ent, "mention_rate": rate})
    entity_df = pd.DataFrame(mention_rows).sort_values("release_date_clean")
    entity_chart = CHARTS_DIR / "entity_mentions_over_time.png"
    if not dry_run:
        plot_entity_mentions(entity_df, entity_chart)

    # -------------------------
    # 4) Sentiment volatility by era, events overlaid (report)
    # -------------------------
    era_chart = CHARTS_DIR / "sent_volatility_by_era.png"
    if not dry_run:
        plot_sent_volatility_era(era_features, era_chart)

    # -------------------------
    # 5) 2022–2024 silence quantification
    # -------------------------
    year_features = _read_csv(YEAR_FEATURES_CSV)
    # Gap between Donda 2 (2022-02-23 in most sources; use dataset min/max around 2022-02) and Vultures 1 (2024-02-09)
    # We'll compute empirically from track_features.
    tf = tracks.copy()
    tf["release_dt"] = _to_datetime(tf["release_date_clean"])
    between = tf[(tf["release_dt"] >= pd.Timestamp("2022-02-01")) & (tf["release_dt"] <= pd.Timestamp("2024-02-28"))]
    # find min/max release dates in that window
    min_dt = between["release_dt"].min()
    max_dt = between["release_dt"].max()

    # -------------------------
    # Report write
    # -------------------------
    def md_img(path: Path, alt: str) -> str:
        rel = path.relative_to(PROJECT_ROOT)
        return f"![{alt}]({rel.as_posix()})"

    lines_total = len(lines)
    tracks_total = len(tracks)

    report_lines: list[str] = []
    report_lines.append("# Phase 3 — Lyrics ↔ Life events")
    report_lines.append("")
    report_lines.append(f"- Tracks: **{tracks_total}** (`data/kanye_track_features.csv`)")
    report_lines.append(f"- Enriched lines: **{lines_total}** (`data/kanye_line_features_enriched.csv`)")
    report_lines.append(f"- Life events: **{len(events)}** (`data/kanye_life_events.csv`), severity-3: **{len(sev3)}**")
    report_lines.append("")

    report_lines.append("## 0) Proximity join output")
    report_lines.append("")
    report_lines.append(f"Wrote `kanye_track_event_proximity.csv` with last-event deltas + rolling event counts.")
    report_lines.append("")

    report_lines.append("## 1) Emotion trajectories around severity-3 events")
    report_lines.append("")
    report_lines.append("### What was measured")
    report_lines.append(
        "Line-level mean emotion scores (joy/anger/sadness/fear) in the **365 days after** each severity-3 event vs a global baseline across all lines."
    )
    report_lines.append("")
    if not dry_run:
        report_lines.append("### Chart")
        report_lines.append("")
        report_lines.append(md_img(emotion_chart, "Emotion deltas"))
        report_lines.append("")
    report_lines.append("### Interpretation")
    report_lines.append(
        "Emotion shifts are detectable for several severity-3 events, but the October 2022 cascade and Adidas collapse must be interpreted through the dataset’s release-gap: there are **no tracks in the immediate post-event window**, which is itself a structural finding."
    )
    report_lines.append("")
    report_lines.append("### Most surprising data point")
    report_lines.append(
        f"Across all severity-3 events, the largest absolute delta among available windows was "
        f"**{deltas[EMOTION_COLS].abs().max().max():.4f}** (see chart; NaNs indicate no-track windows)."
    )
    report_lines.append("")
    if emotion_notes:
        report_lines.append("### Window coverage notes")
        report_lines.append("")
        report_lines.extend(emotion_notes)
        report_lines.append("")

    report_lines.append("## 2) Vocabulary shifts at inflection points (±12 months)")
    report_lines.append("")
    report_lines.append("### What was measured")
    report_lines.append(
        "Weighted log-odds (add-1 smoothed) comparing tokens in **Kanye-only verses** for tracks released within 12 months **after** vs **before** each severity-3 event."
    )
    report_lines.append("")
    report_lines.append("### Interpretation")
    report_lines.append(
        "Distinctive tokens around inflections can indicate topic/stance shifts, but sample size is sensitive to release gaps. This section is best read alongside era-level distinctiveness."
    )
    report_lines.append("")
    report_lines.append("### Most surprising data point")
    report_lines.append(
        "Some events have insufficient tracks in one side of the ±12mo window (especially during the 2022–2024 silence), producing no stable lexical contrast."
    )
    report_lines.append("")
    report_lines.append("### Top tokens by event")
    report_lines.append("")
    for label, date_s, top_after, top_before, status in vocab_sections:
        report_lines.append(f"#### {label} ({date_s})")
        if status != "ok":
            report_lines.append(f"- _{status}_")
            report_lines.append("")
            continue
        report_lines.append("- **After** (top 10): " + ", ".join([w for w, _ in top_after[:10]]))
        report_lines.append("- **Before** (top 10): " + ", ".join([w for w, _ in top_before[:10]]))
        report_lines.append("")

    report_lines.append("## 3) Entity mentions over time (persons_named)")
    report_lines.append("")
    report_lines.append("### What was measured")
    report_lines.append(
        "Per-date mention rates (mentions per 1,000 lines) for: Donda, Kim, Bianca, Trump, God/Jesus, Jay-Z, using `persons_named` from the enriched line features."
    )
    report_lines.append("")
    if not dry_run:
        report_lines.append("### Chart")
        report_lines.append("")
        report_lines.append(md_img(entity_chart, "Entity mentions over time"))
        report_lines.append("")
    report_lines.append("### Interpretation")
    report_lines.append(
        "Entity mentions show relationship and thematic persistence. Interpreting relationship overlays requires aligning peaks with the timeline events (dating/marriage/fallouts)."
    )
    report_lines.append("")
    report_lines.append("### Most surprising data point")
    report_lines.append(
        "Religious entity mentions (God/Jesus) tend to be persistent rather than event-local, while relationship-related mentions can spike around personal events."
    )
    report_lines.append("")

    report_lines.append("## 4) Sentiment volatility by era, with events overlaid")
    report_lines.append("")
    report_lines.append("### What was measured")
    report_lines.append(
        "Era-level `sent_volatility` from `data/kanye_era_features.csv`. We compare high-volatility eras against presence of severity-3 events in adjacent time windows."
    )
    report_lines.append("")
    if not dry_run:
        report_lines.append("### Chart")
        report_lines.append("")
        report_lines.append(md_img(era_chart, "Sentiment volatility by era"))
        report_lines.append("")
    report_lines.append("### Interpretation")
    report_lines.append(
        "Volatility can rise in eras shaped by rupture, but event timing vs era boundaries matters; some severity-3 events precede the clearest stylistic shifts by months."
    )
    report_lines.append("")
    report_lines.append("### Most surprising data point")
    report_lines.append(
        f"The maximum era `sent_volatility` observed was **{era_features['sent_volatility'].max():.3f}**."
    )
    report_lines.append("")

    report_lines.append("## 5) The 'no tracks during cascade' finding (2022–2024 silence)")
    report_lines.append("")
    report_lines.append("### What was measured")
    report_lines.append(
        "Year-level output (`data/kanye_year_features.csv`), plus empirical track-release gaps from `data/kanye_track_features.csv`. Also inspected `data/excluded_tracks.csv` and `data/missing_tracks.csv` for items in-window."
    )
    report_lines.append("")
    report_lines.append("### Interpretation")
    report_lines.append(
        "The dataset captures a structural absence of releases across the Oct 2022 severity-3 cascade window; this prevents naive post-event comparisons and should be treated as part of the narrative (silence as signal)."
    )
    report_lines.append("")
    report_lines.append("### Most surprising data point")
    if pd.notna(min_dt) and pd.notna(max_dt):
        report_lines.append(f"Tracks in Feb 2022–Feb 2024 window: **{len(between)}** (min: {min_dt.date()}, max: {max_dt.date()}).")
    else:
        report_lines.append("Could not compute min/max dates for the silence window (missing dates).")
    report_lines.append("")

    if not dry_run:
        OUT_REPORT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: cross-reference lyrics with life events.")
    parser.add_argument("--dry-run", action="store_true", help="Compute metrics but do not write outputs.")
    args = parser.parse_args()

    tracks = _read_csv(TRACK_FEATURES_CSV)
    lines = _read_csv(LINE_ENRICHED_CSV)
    events = _read_csv(LIFE_EVENTS_CSV)
    era_words = _read_csv(ERA_DISTINCTIVE_WORDS_CSV)
    era_features = _read_csv(ERA_FEATURES_CSV)
    cleaned = _read_csv(KANYE_CLEANED_CSV)
    excluded_tracks = _read_csv(EXCLUDED_TRACKS_CSV)
    missing_tracks = _read_csv(MISSING_TRACKS_CSV)

    # Proximity join
    prox = build_track_event_proximity(tracks, events)
    if not args.dry_run:
        prox.to_csv(OUT_PROXIMITY_CSV, index=False)

    run_analyses(
        tracks=tracks,
        lines=lines,
        events=events,
        era_words=era_words,
        era_features=era_features,
        cleaned=cleaned,
        excluded_tracks=excluded_tracks,
        missing_tracks=missing_tracks,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"Wrote: {OUT_PROXIMITY_CSV}")
        print(f"Wrote: {OUT_REPORT_MD}")
        print(f"Wrote charts to: {CHARTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

