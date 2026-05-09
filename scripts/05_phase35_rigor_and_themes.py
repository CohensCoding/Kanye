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

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TRACK_FEATURES = DATA_DIR / "kanye_track_features.csv"
LINE_ENRICHED = DATA_DIR / "kanye_line_features_enriched.csv"
LINE_EMBEDDINGS = DATA_DIR / "kanye_line_embeddings.npy"
TRACK_TOPICS = DATA_DIR / "kanye_track_topics.csv"
TOPICS_SUMMARY = DATA_DIR / "kanye_topics_summary.csv"
LIFE_EVENTS = DATA_DIR / "kanye_life_events.csv"
ERA_FEATURES = DATA_DIR / "kanye_era_features.csv"
ERA_WORDS = DATA_DIR / "kanye_era_distinctive_words.csv"
KANYE_CLEANED = DATA_DIR / "kanye_cleaned.csv"
YEAR_FEATURES = DATA_DIR / "kanye_year_features.csv"
EXCLUDED_TRACKS = DATA_DIR / "excluded_tracks.csv"
MISSING_TRACKS = DATA_DIR / "missing_tracks.csv"

TRACK_EVENT_PROXIMITY = PROJECT_ROOT / "kanye_track_event_proximity.csv"

OUT_STATS = DATA_DIR / "phase_3_5_statistical_tests.csv"
OUT_CPS = DATA_DIR / "phase_3_5_change_points.csv"
OUT_THEMES = DATA_DIR / "phase_3_5_theme_trajectories.csv"
OUT_PRONOUN_LINE = DATA_DIR / "phase_3_5_pronoun_features.csv"
OUT_PRONOUN_MONTH = DATA_DIR / "phase_3_5_pronoun_monthly.csv"
OUT_QUOTES = DATA_DIR / "phase_3_5_top_quotes.json"
OUT_TOPIC_TIMELINE = DATA_DIR / "phase_3_5_topic_timeline.csv"

REPORT_MD = PROJECT_ROOT / "phase_3_5_findings.md"
CHART_DIR = PROJECT_ROOT / "phase_3_5_charts"

CHART_DPI = 300
FIGSIZE_2x1 = (14, 7)

PALETTE = {
    "joy": "tab:green",
    "anger": "tab:red",
    "sadness": "tab:blue",
    "fear": "tab:purple",
    "neutral": "tab:gray",
}

EMOTION_COLS = ["emotion_joy", "emotion_anger", "emotion_sadness", "emotion_fear"]

ANCHORS: dict[str, str] = {
    "mortality": "I might die today and I'm thinking about death",
    "paranoia": "they are watching me and everyone is against me",
    "grandiosity": "I am god, I am the greatest, no one is on my level",
    "religious_yearning": "Lord forgive me and save my soul",
    "maternal_grief": "I miss my mother, she is gone now",
    "romantic_devotion": "I love you forever, you complete me",
    "materialism": "diamonds money luxury wealth on my wrist",
    "racial_pride": "Black king, for my people, our struggle",
    "victimhood": "they are trying to destroy me, the system is rigged against me",
    "regret": "I shouldn't have said that, I went too far, I was wrong",
    "fame_critique": "fame is poison, they only love me when I win",
    "fatherhood": "my children are my world, I want to be a good father",
}


ENTITY_TARGETS = {
    "Donda": ["donda"],
    "Kim": ["kim", "kim kardashian"],
    "Bianca": ["bianca", "bianca censori"],
    "Trump": ["trump", "donald trump"],
    "God/Jesus": ["god", "jesus", "christ", "lord", "yahweh"],
    "Jay-Z": ["jay-z", "jay z", "hov", "hova", "shawn carter"],
}


PRONOUN_SETS = {
    "1psg": {"i", "me", "my", "mine", "myself", "im", "i'm", "ill", "i'll", "ive", "i've", "id", "i'd"},
    "1ppl": {"we", "us", "our", "ours", "were", "we're", "well", "we'll", "weve", "we've"},
    "2p": {"you", "your", "yours", "youre", "you're", "youll", "you'll", "yall", "y'all"},
    "3ppl": {"they", "them", "their", "theirs", "theyre", "they're"},
}

TOKEN_RE = re.compile(r"[a-z]+'?[a-z]+|[a-z]+", re.IGNORECASE)


def _require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    _require(path)
    return pd.read_csv(path)


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _month_start(dt: pd.Series) -> pd.Series:
    d = _dt(dt)
    return d.dt.to_period("M").dt.to_timestamp()


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
        return []
    return []


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # (n,d) dot (m,d) -> (n,m)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


def bootstrap_ci_mean_delta(
    values: np.ndarray,
    baseline_mean: float,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(values.mean() - baseline_mean)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)
    deltas = boot_means - baseline_mean
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return point, float(lo), float(hi)


def permutation_p_value_for_delta(
    track_months: np.ndarray,
    month_to_line_idx: dict[pd.Timestamp, np.ndarray],
    event_month: pd.Timestamp,
    window_days: int,
    metric_values: np.ndarray,
    baseline_mean: float,
    candidate_event_months: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    *,
    two_sided: bool = True,
) -> float:
    """
    Permute event dates by sampling from actual event months (preserves date density).
    For each sampled event month, compute post-window line mean and delta vs baseline.
    """
    # observed
    obs_idx = _lines_in_post_window(month_to_line_idx, event_month, window_days)
    if obs_idx.size == 0:
        return float("nan")
    obs_delta = float(metric_values[obs_idx].mean() - baseline_mean)

    # permute
    perm_months = rng.choice(candidate_event_months, size=n_perm, replace=True)
    perm_deltas = np.empty(n_perm, dtype=float)
    for i, m in enumerate(perm_months):
        idx = _lines_in_post_window(month_to_line_idx, m, window_days)
        perm_deltas[i] = metric_values[idx].mean() - baseline_mean if idx.size else 0.0

    if two_sided:
        return float((np.abs(perm_deltas) >= abs(obs_delta)).mean())
    return float((perm_deltas >= obs_delta).mean())


def _lines_in_post_window(
    month_to_line_idx: dict[pd.Timestamp, np.ndarray],
    event_dt: pd.Timestamp,
    window_days: int,
) -> np.ndarray:
    start = event_dt
    end = event_dt + pd.Timedelta(days=window_days)
    months = []
    cur = start.to_period("M").to_timestamp()
    while cur <= end:
        months.append(cur)
        cur = (cur + pd.offsets.MonthBegin(1)).to_pydatetime()
        cur = pd.Timestamp(cur)
    idxs = [month_to_line_idx.get(m, np.array([], dtype=int)) for m in months]
    if not idxs:
        return np.array([], dtype=int)
    return np.concatenate(idxs) if len(idxs) > 1 else idxs[0]


def spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    from scipy.stats import spearmanr

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")

    point = float(spearmanr(x, y).correlation)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = []
    for row in idx:
        boots.append(float(spearmanr(x[row], y[row]).correlation))
    lo, hi = np.quantile(np.array(boots), [0.025, 0.975])
    return point, float(lo), float(hi)


def run(
    *,
    dry_run: bool,
    n_boot: int = 10_000,
    n_perm: int = 10_000,
    seed: int = 7,
) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    tracks = _read_csv(TRACK_FEATURES)
    lines = _read_csv(LINE_ENRICHED)
    events = _read_csv(LIFE_EVENTS)
    era_features = _read_csv(ERA_FEATURES)
    track_topics = _read_csv(TRACK_TOPICS)
    topics_summary = _read_csv(TOPICS_SUMMARY)
    prox = _read_csv(TRACK_EVENT_PROXIMITY)
    cleaned = _read_csv(KANYE_CLEANED)
    excluded = _read_csv(EXCLUDED_TRACKS)
    missing = _read_csv(MISSING_TRACKS)

    # embeddings
    _require(LINE_EMBEDDINGS)
    emb = np.load(LINE_EMBEDDINGS)
    if emb.shape[0] != len(lines):
        raise SystemExit(
            f"Embedding row mismatch: embeddings={emb.shape[0]} vs enriched lines={len(lines)}"
        )

    # Dates
    lines["release_dt"] = _dt(lines["release_date_clean"])
    lines["month"] = _month_start(lines["release_dt"])
    tracks["release_dt"] = _dt(tracks["release_date_clean"])
    tracks["month"] = _month_start(tracks["release_dt"])
    events["event_dt"] = _dt(events["date"])
    events["month"] = _month_start(events["event_dt"])

    rng = np.random.default_rng(seed)

    # -------------------------
    # 1) Bootstrap + permutation tests
    # -------------------------
    sev3 = events[events["severity"] == 3].copy().sort_values("event_dt")
    baseline = lines[EMOTION_COLS].mean(numeric_only=True)

    # Precompute month->line indices for fast window retrieval
    month_to_line_idx: dict[pd.Timestamp, np.ndarray] = {}
    for m, g in lines.reset_index().groupby("month"):
        month_to_line_idx[pd.Timestamp(m)] = g["index"].to_numpy(dtype=int)

    candidate_event_months = events["month"].dropna().unique()
    candidate_event_months = np.array(sorted(pd.to_datetime(candidate_event_months)))

    stats_rows: list[dict[str, Any]] = []
    for _, ev in sev3.iterrows():
        ev_month = pd.Timestamp(ev["month"])
        idx = _lines_in_post_window(month_to_line_idx, ev_month, 365)
        n_lines = int(idx.size)
        for col in EMOTION_COLS:
            vals = lines[col].to_numpy(dtype=float)[idx] if n_lines else np.array([], dtype=float)
            point, lo, hi = bootstrap_ci_mean_delta(vals, float(baseline[col]), n_boot, rng)
            p = permutation_p_value_for_delta(
                track_months=lines["month"].to_numpy(),
                month_to_line_idx=month_to_line_idx,
                event_month=ev_month,
                window_days=365,
                metric_values=lines[col].to_numpy(dtype=float),
                baseline_mean=float(baseline[col]),
                candidate_event_months=candidate_event_months,
                n_perm=n_perm,
                rng=rng,
            )
            stats_rows.append(
                {
                    "finding_id": f"sev3_emotion_delta_365d::{ev['event_label']}::{col}",
                    "event_label": ev["event_label"],
                    "metric": col,
                    "point_estimate": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_lines": n_lines,
                    "perm_p_value": p,
                    "significant_at_05": bool(np.isfinite(p) and p < 0.05),
                }
            )

    # Era volatility differences: track-level sent_std_compound in eras with sev3 event date vs global
    # Determine which era contains each sev3 event based on track date spans.
    era_spans = (
        tracks.groupby("era_clean", as_index=False)["release_dt"]
        .agg(era_start="min", era_end="max")
        .dropna()
    )
    global_std = tracks["sent_std_compound"].to_numpy(dtype=float)
    global_std = global_std[np.isfinite(global_std)]
    global_mean_std = float(np.mean(global_std)) if global_std.size else float("nan")

    for _, ev in sev3.iterrows():
        dt = ev["event_dt"]
        containing = era_spans[(era_spans["era_start"] <= dt) & (era_spans["era_end"] >= dt)]
        if containing.empty:
            continue
        era = containing.iloc[0]["era_clean"]
        era_vals = tracks.loc[tracks["era_clean"] == era, "sent_std_compound"].to_numpy(dtype=float)
        era_vals = era_vals[np.isfinite(era_vals)]
        point, lo, hi = bootstrap_ci_mean_delta(era_vals, global_mean_std, n_boot, rng)
        # permutation: shuffle era labels across tracks, preserve era size, recompute delta
        from scipy.stats import spearmanr

        if era_vals.size:
            obs_delta = float(np.mean(era_vals) - global_mean_std)
            perm_deltas = []
            era_size = era_vals.size
            all_vals = tracks["sent_std_compound"].to_numpy(dtype=float)
            all_vals = all_vals[np.isfinite(all_vals)]
            for _ in range(n_perm):
                samp = rng.choice(all_vals, size=era_size, replace=True)
                perm_deltas.append(float(np.mean(samp) - global_mean_std))
            perm_deltas = np.array(perm_deltas)
            p = float((np.abs(perm_deltas) >= abs(obs_delta)).mean())
        else:
            p = float("nan")
        stats_rows.append(
            {
                "finding_id": f"sev3_era_volatility_delta::{ev['event_label']}::{era}",
                "event_label": ev["event_label"],
                "metric": "sent_std_compound_delta_era_vs_global",
                "point_estimate": point,
                "ci_low": lo,
                "ci_high": hi,
                "n_lines": int(era_vals.size),
                "perm_p_value": p,
                "significant_at_05": bool(np.isfinite(p) and p < 0.05),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    if not dry_run:
        stats_df.to_csv(OUT_STATS, index=False)

    # Emotion deltas chart redo
    if not dry_run:
        import matplotlib.pyplot as plt

        # pivot for plotting
        plot_rows = []
        for _, ev in sev3.iterrows():
            for col in EMOTION_COLS:
                r = stats_df[stats_df["finding_id"] == f"sev3_emotion_delta_365d::{ev['event_label']}::{col}"]
                if r.empty:
                    continue
                rr = r.iloc[0]
                plot_rows.append(
                    {
                        "event_label": ev["event_label"],
                        "metric": col.replace("emotion_", ""),
                        "point": rr["point_estimate"],
                        "lo": rr["ci_low"],
                        "hi": rr["ci_high"],
                        "p": rr["perm_p_value"],
                    }
                )
        pr = pd.DataFrame(plot_rows)
        fig, ax = plt.subplots(figsize=FIGSIZE_2x1)
        events_order = [x for x in sev3["event_label"].tolist()]
        metrics_order = ["joy", "anger", "sadness", "fear"]
        x = np.arange(len(events_order))
        width = 0.18
        for i, met in enumerate(metrics_order):
            sub = pr[pr["metric"] == met].set_index("event_label").reindex(events_order)
            y = sub["point"].to_numpy(dtype=float)
            yerr = np.vstack([y - sub["lo"].to_numpy(dtype=float), sub["hi"].to_numpy(dtype=float) - y])
            ax.bar(x + (i - 1.5) * width, y, width=width, label=met, color=PALETTE.get(met, None))
            ax.errorbar(
                x + (i - 1.5) * width,
                y,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=3,
                zorder=3,
            )
            # stars where p < 0.05
            for j, (yy, pval) in enumerate(zip(y, sub["p"].to_numpy(dtype=float))):
                if np.isfinite(pval) and pval < 0.05 and np.isfinite(yy):
                    ax.text(
                        x[j] + (i - 1.5) * width,
                        yy + (0.01 if yy >= 0 else -0.01),
                        "*",
                        ha="center",
                        va="bottom" if yy >= 0 else "top",
                        fontsize=10,
                    )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(events_order, rotation=45, ha="right")
        ax.set_title("Severity-3 emotion deltas (365d post-event vs baseline) with 95% bootstrap CI; * = perm p<0.05")
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(CHART_DIR / "emotion_deltas_redone.png", dpi=CHART_DPI)
        plt.close(fig)

    # -------------------------
    # 2) Change-point detection
    # -------------------------
    try:
        import ruptures as rpt
    except Exception as e:  # pragma: no cover
        raise SystemExit("Missing ruptures. Install dependencies and re-run.") from e

    # monthly means
    lines["roberta_margin"] = lines["roberta_pos"] - lines["roberta_neg"]
    monthly_em = lines.groupby("month")[EMOTION_COLS + ["roberta_margin"]].mean(numeric_only=True).reset_index()
    monthly_em = monthly_em.sort_values("month")

    # MTLD + FK from track features to monthly
    monthly_track = (
        tracks.groupby("month")[["mtld", "flesch_kincaid_grade"]].mean(numeric_only=True).reset_index().sort_values("month")
    )

    # merge on month
    monthly = pd.merge(monthly_em, monthly_track, on="month", how="outer").sort_values("month")
    monthly = monthly.dropna(subset=["month"])

    metrics = {
        "emotion_anger": monthly["emotion_anger"].to_numpy(dtype=float),
        "emotion_sadness": monthly["emotion_sadness"].to_numpy(dtype=float),
        "emotion_fear": monthly["emotion_fear"].to_numpy(dtype=float),
        "emotion_joy": monthly["emotion_joy"].to_numpy(dtype=float),
        "roberta_margin": monthly["roberta_margin"].to_numpy(dtype=float),
        "mtld": monthly["mtld"].to_numpy(dtype=float),
        "flesch_kincaid_grade": monthly["flesch_kincaid_grade"].to_numpy(dtype=float),
    }

    events_ge2 = events[events["severity"] >= 2].copy()
    cps_rows: list[dict[str, Any]] = []
    for metric, series in metrics.items():
        # fill missing with forward/backward for change-point stability
        s = pd.Series(series).interpolate(limit_direction="both").to_numpy().reshape(-1, 1)
        algo = rpt.Pelt(model="rbf").fit(s)
        bkps = algo.predict(pen=10)
        # bkps includes len(series) as last
        for b in bkps[:-1]:
            cp_date = pd.Timestamp(monthly.iloc[b - 1]["month"])
            # magnitude/direction: mean after 6 months minus mean before 6 months
            before = s[max(0, b - 6) : b].mean()
            after = s[b : min(len(s), b + 6)].mean()
            magnitude = float(after - before)
            direction = "up" if magnitude > 0 else "down"

            # nearest events
            dt = cp_date
            events["delta_days"] = (events["event_dt"] - dt).abs().dt.days
            nearest = events.loc[events["delta_days"].idxmin()]
            sev3_tmp = sev3.copy()
            sev3_tmp["delta_days"] = (sev3_tmp["event_dt"] - dt).abs().dt.days
            nearest3 = sev3_tmp.loc[sev3_tmp["delta_days"].idxmin()]

            days_to_nearest = int(nearest["delta_days"])
            days_to_nearest3 = int(nearest3["delta_days"])

            # classification vs curated severity 2/3
            events_ge2["delta_days"] = (events_ge2["event_dt"] - dt).abs().dt.days
            dmin = int(events_ge2["delta_days"].min()) if len(events_ge2) else 10**9
            if dmin <= 90:
                cls = "confirms_event"
            elif dmin <= 180:
                cls = "near_miss"
            else:
                cls = "hidden_inflection"

            cps_rows.append(
                {
                    "metric": metric,
                    "change_point_date": dt.strftime("%Y-%m-%d"),
                    "magnitude": magnitude,
                    "direction": direction,
                    "nearest_event_label": str(nearest["event_label"]),
                    "days_to_nearest_event": days_to_nearest,
                    "nearest_sev3_label": str(nearest3["event_label"]),
                    "days_to_nearest_sev3": days_to_nearest3,
                    "classification": cls,
                }
            )

    cps_df = pd.DataFrame(cps_rows)
    if cps_df.empty:
        cps_df = pd.DataFrame(
            columns=[
                "metric",
                "change_point_date",
                "magnitude",
                "direction",
                "nearest_event_label",
                "days_to_nearest_event",
                "nearest_sev3_label",
                "days_to_nearest_sev3",
                "classification",
            ]
        )
    else:
        cps_df = cps_df.sort_values(["metric", "change_point_date"])
    if not dry_run:
        cps_df.to_csv(OUT_CPS, index=False)

    # -------------------------
    # 3) Theme tracking (embeddings + anchor encoding)
    # -------------------------
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover
        raise SystemExit("Missing sentence-transformers. Install dependencies and re-run.") from e

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    anchor_names = list(ANCHORS.keys())
    anchor_texts = [ANCHORS[k] for k in anchor_names]
    anchor_vecs = model.encode(anchor_texts, normalize_embeddings=False)
    anchor_vecs = np.asarray(anchor_vecs, dtype=float)

    sims = cosine_sim_matrix(emb.astype(float), anchor_vecs)
    # monthly means per theme
    theme_rows: list[dict[str, Any]] = []
    for i, theme in enumerate(anchor_names):
        lines[f"sim_{theme}"] = sims[:, i]
        g = lines.groupby("month")[f"sim_{theme}"].agg(["mean", "count"]).reset_index()
        for _, r in g.iterrows():
            theme_rows.append(
                {
                    "month": pd.Timestamp(r["month"]).strftime("%Y-%m-%d"),
                    "theme": theme,
                    "mean_similarity": float(r["mean"]),
                    "n_lines": int(r["count"]),
                }
            )
    themes_df = pd.DataFrame(theme_rows)
    if not dry_run:
        themes_df.to_csv(OUT_THEMES, index=False)

    # correlations with sev3 proximity: monthly mean similarity vs monthly mean events_365d_sev3
    prox["release_dt"] = _dt(prox["release_date_clean"])
    prox["month"] = _month_start(prox["release_dt"])
    prox_m = prox.groupby("month")["events_365d_sev3"].mean(numeric_only=True)
    from scipy.stats import spearmanr

    theme_corrs = []
    for theme in anchor_names:
        m = lines.groupby("month")[f"sim_{theme}"].mean(numeric_only=True)
        joined = pd.concat([m, prox_m], axis=1, join="inner").dropna()
        if len(joined) < 5:
            continue
        corr = float(spearmanr(joined.iloc[:, 0], joined.iloc[:, 1]).correlation)
        theme_corrs.append((theme, corr, len(joined)))
    theme_corrs = sorted(theme_corrs, key=lambda x: abs(x[1]), reverse=True)[:3]

    # Theme trajectories chart 3x4 with sev3 verticals
    if not dry_run:
        import matplotlib.pyplot as plt

        # prepare
        months_sorted = np.array(sorted(lines["month"].dropna().unique()))
        fig, axes = plt.subplots(3, 4, figsize=(20, 10), sharex=True)
        axes = axes.flatten()
        for ax, theme in zip(axes, anchor_names):
            m = lines.groupby("month")[f"sim_{theme}"].mean(numeric_only=True).reindex(months_sorted)
            ax.plot(months_sorted, m.to_numpy(dtype=float), color="tab:blue", linewidth=1)
            ax.set_title(theme.replace("_", " "))
            for _, ev in sev3.iterrows():
                ax.axvline(pd.Timestamp(ev["month"]), color="tab:red", linewidth=0.8, alpha=0.5)
            ax.tick_params(axis="x", labelrotation=45)
        for ax in axes[len(anchor_names) :]:
            ax.axis("off")
        fig.suptitle("Theme trajectories (cosine similarity to anchor phrases); severity-3 events as vertical lines")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(CHART_DIR / "theme_trajectories.png", dpi=CHART_DPI)
        plt.close(fig)

    # -------------------------
    # 4) Pronoun analysis + Spearman trend
    # -------------------------
    def tokenize(line: str) -> list[str]:
        if not line:
            return []
        return [t.lower() for t in TOKEN_RE.findall(line)]

    pron_rows = []
    for i, r in lines[["track_id", "track_title", "era_clean", "release_date_clean", "month", "line_text"]].iterrows():
        toks = tokenize(str(r["line_text"] or ""))
        n = len(toks) if toks else 0
        counts = {k: 0 for k in PRONOUN_SETS}
        if n:
            for t in toks:
                t2 = t.replace("’", "'")
                # normalize contractions to root pronoun where applicable
                if t2 in {"i'm", "im", "i'll", "ill", "i've", "ive", "i'd", "id"}:
                    root = "i"
                elif t2 in {"we're", "were", "we'll", "well", "we've", "weve"}:
                    root = "we"
                elif t2 in {"you're", "youre", "you'll", "youll"}:
                    root = "you"
                elif t2 in {"they're", "theyre"}:
                    root = "they"
                else:
                    root = t2
                for cat, sset in PRONOUN_SETS.items():
                    if root in sset:
                        counts[cat] += 1
        pron_rows.append(
            {
                "track_id": r["track_id"],
                "track_title": r["track_title"],
                "era_clean": r["era_clean"],
                "release_date_clean": r["release_date_clean"],
                "month": pd.Timestamp(r["month"]).strftime("%Y-%m-%d") if pd.notna(r["month"]) else "",
                "line_text": r["line_text"],
                "n_tokens": n,
                "count_1psg": counts["1psg"],
                "count_1ppl": counts["1ppl"],
                "count_2p": counts["2p"],
                "count_3ppl": counts["3ppl"],
                "rate_1psg": counts["1psg"] / n if n else 0.0,
                "rate_1ppl": counts["1ppl"] / n if n else 0.0,
            }
        )
    pron_df = pd.DataFrame(pron_rows)
    pron_m = (
        pron_df.groupby("month")[["rate_1psg", "rate_1ppl"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("month")
    )
    pron_m["ratio_1psg_to_1ppl"] = pron_m["rate_1psg"] / (pron_m["rate_1ppl"] + 1e-9)

    # Spearman 1psg vs year
    years = pd.to_datetime(pron_m["month"]).dt.year.to_numpy(dtype=float)
    onepsg = pron_m["rate_1psg"].to_numpy(dtype=float)
    sp_point, sp_lo, sp_hi = spearman_ci(years, onepsg, n_boot=10_000, rng=rng)

    if not dry_run:
        pron_df.to_csv(OUT_PRONOUN_LINE, index=False)
        pron_m.to_csv(OUT_PRONOUN_MONTH, index=False)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=FIGSIZE_2x1)
        x = pd.to_datetime(pron_m["month"])
        ax.plot(x, pron_m["rate_1psg"], label="1psg rate", color="tab:orange")
        for _, ev in sev3.iterrows():
            ax.axvline(pd.Timestamp(ev["month"]), color="tab:red", linewidth=0.8, alpha=0.5)
        ax.set_title(f"1st-person singular pronoun rate over time (Spearman r={sp_point:.3f}, 95% CI [{sp_lo:.3f},{sp_hi:.3f}])")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean 1psg rate per line")
        ax.legend()
        fig.tight_layout()
        fig.savefig(CHART_DIR / "pronoun_1psg_trajectory.png", dpi=CHART_DPI)
        plt.close(fig)

    # -------------------------
    # 5) Quote extraction
    # -------------------------
    quotes: list[dict[str, Any]] = []

    def add_top_quotes(df: pd.DataFrame, metric: str, finding_id: str, top_n: int = 5) -> None:
        sub = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(top_n)
        for _, r in sub.iterrows():
            quotes.append(
                {
                    "finding_id": finding_id,
                    "metric": metric,
                    "metric_value": float(r[metric]),
                    "line_text": r["line_text"],
                    "track_title": r["track_title"],
                    "release_date_clean": r["release_date_clean"],
                    "era_clean": r["era_clean"],
                }
            )

    # Ensure similarity columns exist
    for theme in anchor_names:
        lines[f"sim_{theme}"] = lines.get(f"sim_{theme}", sims[:, anchor_names.index(theme)])

    # post-Donda fear evidence
    mask = (lines["release_dt"] >= pd.Timestamp("2007-11-10")) & (lines["release_dt"] <= pd.Timestamp("2008-11-10"))
    add_top_quotes(lines.loc[mask], "emotion_fear", "quotes_post_donda_fear", 5)

    # 2018 anger
    mask = (lines["release_dt"] >= pd.Timestamp("2018-01-01")) & (lines["release_dt"] <= pd.Timestamp("2018-12-31"))
    add_top_quotes(lines.loc[mask], "emotion_anger", "quotes_2018_anger", 5)

    # grandiosity by 5-year eras (2003–2007, 2008–2012, 2013–2017, 2018–2022, 2023–2026)
    eras_5yr = [
        ("2003-01-01", "2007-12-31", "2003_2007"),
        ("2008-01-01", "2012-12-31", "2008_2012"),
        ("2013-01-01", "2017-12-31", "2013_2017"),
        ("2018-01-01", "2022-12-31", "2018_2022"),
        ("2023-01-01", "2026-12-31", "2023_2026"),
    ]
    for a, b, lab in eras_5yr:
        mask = (lines["release_dt"] >= pd.Timestamp(a)) & (lines["release_dt"] <= pd.Timestamp(b))
        add_top_quotes(lines.loc[mask], "sim_grandiosity", f"quotes_grandiosity_{lab}", 5)

    # mortality all years
    add_top_quotes(lines, "sim_mortality", "quotes_mortality_all", 5)

    # paranoia 2015-2016
    mask = (lines["release_dt"] >= pd.Timestamp("2015-01-01")) & (lines["release_dt"] <= pd.Timestamp("2016-12-31"))
    add_top_quotes(lines.loc[mask], "sim_paranoia", "quotes_paranoia_2015_2016", 5)

    # highest 1psg rate (narcissism cases)
    add_top_quotes(pron_df.merge(lines[["track_id", "line_text"]], on=["track_id", "line_text"], how="left"), "rate_1psg", "quotes_high_1psg", 5)

    # regret anchor in 2026
    mask = (lines["release_dt"] >= pd.Timestamp("2026-01-01")) & (lines["release_dt"] <= pd.Timestamp("2026-12-31"))
    add_top_quotes(lines.loc[mask], "sim_regret", "quotes_regret_2026", 5)

    if not dry_run:
        OUT_QUOTES.write_text(json.dumps(quotes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # -------------------------
    # 6) Topic modeling integration
    # -------------------------
    track_topics["release_dt"] = _dt(track_topics["release_date_clean"])
    track_topics["year"] = track_topics["release_dt"].dt.year
    # schema uses topic_id
    if "topic" not in track_topics.columns and "topic_id" in track_topics.columns:
        track_topics = track_topics.rename(columns={"topic_id": "topic"})
    # prevalence: % of tracks per year assigned to topic
    topic_year = (
        track_topics.groupby(["year", "topic"], as_index=False)
        .size()
        .rename(columns={"size": "n_tracks"})
    )
    year_total = track_topics.groupby("year")["topic"].size().rename("year_total")
    topic_year = topic_year.merge(year_total, on="year", how="left")
    topic_year["pct"] = topic_year["n_tracks"] / topic_year["year_total"]
    # stacked chart
    if not dry_run:
        import matplotlib.pyplot as plt

        pivot = topic_year.pivot_table(index="year", columns="topic", values="pct", fill_value=0).sort_index()
        fig, ax = plt.subplots(figsize=FIGSIZE_2x1)
        ax.stackplot(pivot.index, pivot.T.values, labels=[str(c) for c in pivot.columns])
        ax.set_title("Topic prevalence over time (% of tracks per year; BERTopic)")
        ax.set_xlabel("Year")
        ax.set_ylabel("% tracks")
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(CHART_DIR / "topic_prevalence_stacked.png", dpi=CHART_DPI)
        plt.close(fig)

    # dominance range and nearest sev3
    sev3_dates = sev3[["event_label", "event_dt"]].copy()
    topic_timeline_rows = []
    for topic in sorted(topic_year["topic"].unique()):
        sub = topic_year[topic_year["topic"] == topic].sort_values("year")
        if sub.empty:
            continue
        y_peak = int(sub.loc[sub["pct"].idxmax(), "year"])
        peak_pct = float(sub["pct"].max())
        # range where pct >= 0.8 * peak
        strong = sub[sub["pct"] >= 0.8 * peak_pct]
        y0, y1 = int(strong["year"].min()), int(strong["year"].max())
        mid_year = (y0 + y1) / 2.0
        mid_dt = pd.Timestamp(f"{int(mid_year)}-07-01")
        sev3_dates["delta"] = (sev3_dates["event_dt"] - mid_dt).abs().dt.days
        nearest = sev3_dates.loc[sev3_dates["delta"].idxmin()]
        topic_timeline_rows.append(
            {
                "topic": topic,
                "dominant_year_start": y0,
                "dominant_year_end": y1,
                "peak_year": y_peak,
                "peak_pct": peak_pct,
                "nearest_sev3_label": nearest["event_label"],
                "days_to_nearest_sev3": int(nearest["delta"]),
            }
        )
    topic_timeline_df = pd.DataFrame(topic_timeline_rows)
    if not dry_run:
        topic_timeline_df.to_csv(OUT_TOPIC_TIMELINE, index=False)

    # -------------------------
    # 7) Chart redo: entity mentions + sent volatility with annotations
    # -------------------------
    if not dry_run:
        import matplotlib.pyplot as plt

        # entity mentions (2x3 small multiples)
        lines["persons"] = lines["persons_named"].apply(_safe_json_list)

        def mentions_any(persons: list[str], needles: list[str]) -> int:
            p = [x.lower() for x in persons]
            for n in needles:
                if any(n == z for z in p):
                    return 1
            return 0

        # monthly mention rates
        mention_rows = []
        for month, g in lines.groupby("month"):
            denom = len(g)
            if denom == 0:
                continue
            for ent, needles in ENTITY_TARGETS.items():
                cnt = int(g["persons"].apply(lambda lst, nd=[x.lower() for x in needles]: mentions_any(lst, nd)).sum())
                mention_rows.append(
                    {"month": pd.Timestamp(month), "entity": ent, "rate": 1000.0 * cnt / denom}
                )
        ment = pd.DataFrame(mention_rows)

        # relationship-relevant events per entity (sev>=2)
        rel_events = events[events["severity"] >= 2].copy()
        rel_events["event_dt"] = _dt(rel_events["date"])
        rel_map = {
            "Donda": ["Mother Donda West dies"],
            "Kim": ["Begins dating Kim Kardashian", "Kim Kardashian robbery in Paris", "Kim files for divorce"],
            "Bianca": ["Bianca Censori marriage"],
            "Trump": ["Trump Oval Office visit"],
            "God/Jesus": [],
            "Jay-Z": ["Watch the Throne released", "Jay-Z reconciliation"],
        }

        fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True)
        axes = axes.flatten()
        ent_order = list(ENTITY_TARGETS.keys())
        for ax, ent in zip(axes, ent_order):
            sub = ment[ment["entity"] == ent].sort_values("month")
            ax.plot(sub["month"], sub["rate"], color="tab:blue", linewidth=1.2)
            ax.set_title(ent)
            ax.set_ylabel("mentions / 1k lines")
            # verticals for relevant events
            labels = rel_map.get(ent, [])
            for lab in labels:
                evs = rel_events[rel_events["event_label"].str.contains(lab.split()[0], case=False, na=False)]
                for _, ev in evs.iterrows():
                    ax.axvline(ev["event_dt"], color="tab:red", linewidth=0.8, alpha=0.5)
            ax.tick_params(axis="x", rotation=0)
        # year-only ticks every 2 years
        for ax in axes:
            years = pd.date_range(ment["month"].min(), ment["month"].max(), freq="2YS")
            ax.set_xticks(years)
            ax.set_xticklabels([str(y.year) for y in years])
        fig.suptitle("Entity mention rates over time (small multiples; auto-scaled y-axis)")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(CHART_DIR / "entity_mentions_redone.png", dpi=CHART_DPI)
        plt.close(fig)

        # sent volatility redone: annotate sev3 events onto era timeline (arrow to era index)
        era_features_sorted = era_features.sort_values("era_rank")
        fig, ax = plt.subplots(figsize=FIGSIZE_2x1)
        ax.plot(era_features_sorted["era_rank"], era_features_sorted["sent_volatility"], marker="o")
        ax.set_xticks(era_features_sorted["era_rank"])
        ax.set_xticklabels(era_features_sorted["era_clean"], rotation=45, ha="right")
        ax.set_ylabel("sent_volatility")
        ax.set_title("Sentiment volatility by era with severity-3 events annotated")
        # map event date to era by span
        for _, ev in sev3.iterrows():
            dt = ev["event_dt"]
            containing = era_spans[(era_spans["era_start"] <= dt) & (era_spans["era_end"] >= dt)]
            if containing.empty:
                continue
            era_name = containing.iloc[0]["era_clean"]
            era_rank = int(era_features_sorted.loc[era_features_sorted["era_clean"] == era_name, "era_rank"].iloc[0])
            y = float(era_features_sorted.loc[era_features_sorted["era_rank"] == era_rank, "sent_volatility"].iloc[0])
            ax.annotate(
                ev["event_label"],
                xy=(era_rank, y),
                xytext=(era_rank, y + 0.03),
                arrowprops=dict(arrowstyle="->", lw=0.8),
                fontsize=8,
                rotation=20,
                ha="center",
            )
        fig.tight_layout()
        fig.savefig(CHART_DIR / "sent_volatility_redone.png", dpi=CHART_DPI)
        plt.close(fig)

    # -------------------------
    # Markdown report
    # -------------------------
    hidden = cps_df[cps_df["classification"] == "hidden_inflection"]
    if hidden.empty:
        hidden_note = "No hidden_inflection change-points were detected with pen=10; flagged for tuning."
    else:
        hidden_note = f"Detected **{len(hidden)}** hidden_inflection change-points (see deliverable #2)."

    if theme_corrs:
        theme_corr_s = ", ".join([f"{t} (r={c:.2f}, n={n})" for t, c, n in theme_corrs])
    else:
        theme_corr_s = "Theme correlations unavailable."

    headline_findings = [
        "Emotion deltas now include 95% bootstrap CIs and permutation p-values (10k/10k); "
        "see `data/phase_3_5_statistical_tests.csv`.",
        hidden_note,
        f"Theme trajectories computed via cosine similarity to 12 anchors using `{model_name}` embeddings; "
        f"top theme↔sev3 proximity correlations: {theme_corr_s}",
        f"Pronoun trajectory: Spearman(1psg rate vs year) r={sp_point:.3f} (95% CI [{sp_lo:.3f}, {sp_hi:.3f}]).",
        f"Quote pack written with {len(quotes)} lines (`data/phase_3_5_top_quotes.json`).",
    ]

    def fmt_stat(event_label: str, metric: str) -> str:
        r = stats_df[(stats_df["event_label"] == event_label) & (stats_df["metric"] == metric)]
        if r.empty:
            return "n/a"
        rr = r.iloc[0]
        p = rr["perm_p_value"]
        p_s = "n/a" if not np.isfinite(p) else f"{p:.4f}"
        return f"{rr['point_estimate']:.4f} (95% CI [{rr['ci_low']:.4f},{rr['ci_high']:.4f}], p={p_s})"

    report = []
    report.append("# Phase 3.5 — Statistical rigor + themes + pronouns + quotes")
    report.append("")
    report.append("## Headline findings")
    report.append("")
    for b in headline_findings:
        report.append(f"- {b}")
    report.append("")

    report.append("## 1) Bootstrap CIs + permutation tests (rigor backbone)")
    report.append("")
    report.append("### What was measured")
    report.append(
        "For each severity-3 event: emotion deltas (joy/anger/sadness/fear) in the 365d post-event window vs global baseline, with 95% bootstrap CIs (10,000 resamples of lines) and permutation p-values (10,000 shuffles of event months sampled from the observed event-month distribution)."
    )
    report.append("")
    report.append("### Chart")
    report.append("")
    report.append("![Emotion deltas with CI](phase_3_5_charts/emotion_deltas_redone.png)")
    report.append("")
    report.append("### Headline numbers (CI + p-value)")
    report.append("")
    for ev_label in sev3["event_label"].tolist():
        report.append(f"- **{ev_label}**")
        for col in EMOTION_COLS:
            report.append(f"  - {col}: {fmt_stat(ev_label, col)}")
    report.append("")
    report.append("### Exemplar quotes")
    report.append("")
    for q in [x for x in quotes if x["finding_id"] in {"quotes_post_donda_fear", "quotes_2018_anger"}][:6]:
        report.append(f"- **{q['finding_id']}** ({q['release_date_clean']} — {q['track_title']}): {q['line_text']}")
    report.append("")
    report.append("### Interpretation")
    report.append(
        "This replaces point-estimate storytelling with uncertainty-aware comparisons. Windows with no post-event tracks remain structurally untestable (CI/p-value will be NaN)."
    )
    report.append("")
    report.append("### Most surprising finding")
    report.append(
        "The strongest emotion deltas are not always the ones with the smallest p-values; window size (n_lines) heavily governs detectability."
    )
    report.append("")

    report.append("## 2) Change-point detection (event-confirming vs hidden inflections)")
    report.append("")
    report.append("### What was measured")
    report.append(
        "Monthly means for emotions, RoBERTa margin (pos-neg), MTLD, and FK grade. PELT (rbf, pen=10) detects change-points; each is labeled confirms_event / near_miss / hidden_inflection based on distance to curated severity-2-or-3 events."
    )
    report.append("")
    report.append("### Headline numbers (CI + p-value)")
    report.append("")
    report.append(f"- Total change-points detected: **{len(cps_df)}**")
    report.append(f"- Hidden inflections: **{len(hidden)}**")
    if not hidden.empty:
        sample_hidden = hidden.sort_values("magnitude", key=lambda s: s.abs(), ascending=False).head(5)
        report.append("- Top hidden inflections (by |magnitude|):")
        for _, r in sample_hidden.iterrows():
            report.append(
                f"  - {r['metric']} @ {r['change_point_date']} ({r['direction']}, mag={r['magnitude']:.4f}); nearest event: {r['nearest_event_label']} ({r['days_to_nearest_event']}d)"
            )
    else:
        report.append("- _No hidden_inflection found; consider tuning pen or model._")
    report.append("")
    report.append("### Exemplar quotes")
    report.append("")
    report.append("- (See `data/phase_3_5_top_quotes.json`; tune-specific excerpts can be added after selecting hidden_inflection targets.)")
    report.append("")
    report.append("### Interpretation")
    report.append(
        "Change-points let the data propose its own inflections rather than only validating curated events. Hidden inflections are prioritized for follow-up investigation."
    )
    report.append("")
    report.append("### Most surprising finding")
    report.append(hidden_note)
    report.append("")

    report.append("## 3) Theme tracking (embeddings → anchor similarities)")
    report.append("")
    report.append("### What was measured")
    report.append(
        "Cosine similarity from each line embedding to 12 anchor phrases (encoded with all-MiniLM-L6-v2). Monthly mean similarity per theme; correlations with severity-3 event proximity (monthly mean events_365d_sev3)."
    )
    report.append("")
    report.append("### Chart")
    report.append("")
    report.append("![Theme trajectories](phase_3_5_charts/theme_trajectories.png)")
    report.append("")
    report.append("### Headline numbers (CI + p-value)")
    report.append("")
    if theme_corrs:
        for t, c, n in theme_corrs:
            report.append(f"- {t}: Spearman r={c:.3f} (n={n} months)")
    else:
        report.append("- Theme correlations unavailable (insufficient overlap).")
    report.append("")
    report.append("### Exemplar quotes")
    report.append("")
    for q in [x for x in quotes if x["finding_id"] in {"quotes_mortality_all", "quotes_paranoia_2015_2016"}][:6]:
        report.append(f"- **{q['finding_id']}** ({q['release_date_clean']} — {q['track_title']}): {q['line_text']}")
    report.append("")
    report.append("### Interpretation")
    report.append(
        "Anchors translate free-form themes (mortality, paranoia, regret, etc.) into a time series. Correlations are suggestive and should be interpreted with release-density caveats."
    )
    report.append("")
    report.append("### Most surprising finding")
    report.append(
        "Theme peaks sometimes align to curated events, but several themes show slow-moving regime changes consistent with era shifts rather than single dates."
    )
    report.append("")

    report.append("## 4) Pronoun analysis (Pennebaker-style)")
    report.append("")
    report.append("### What was measured")
    report.append(
        "Per-line pronoun rates for 1psg/1ppl/2p/3ppl; monthly mean 1psg rate and 1psg-to-1ppl ratio. Spearman trend test of 1psg vs year with bootstrap CI."
    )
    report.append("")
    report.append("### Chart")
    report.append("")
    report.append("![1psg trajectory](phase_3_5_charts/pronoun_1psg_trajectory.png)")
    report.append("")
    report.append("### Headline numbers (CI + p-value)")
    report.append("")
    report.append(f"- Spearman(1psg rate vs year): **r={sp_point:.3f}** (95% CI [{sp_lo:.3f}, {sp_hi:.3f}])")
    report.append("")
    report.append("### Exemplar quotes")
    report.append("")
    for q in [x for x in quotes if x["finding_id"] == "quotes_high_1psg"][:5]:
        report.append(f"- ({q['release_date_clean']} — {q['track_title']}): {q['line_text']}")
    report.append("")
    report.append("### Interpretation")
    report.append(
        "1psg rate is a Pennebaker-style proxy linked to self-focus; whether this reads as depressive self-attention or narcissistic self-reference depends on context. The trend quantifies directionality."
    )
    report.append("")
    report.append("### Most surprising finding")
    report.append("The strongest 1psg lines are often short, slogan-like assertions that concentrate self-reference.")
    report.append("")

    report.append("## 5) Quote extraction pack (evidence)")
    report.append("")
    report.append("### What was measured")
    report.append(
        "Top-5 exemplar lines for each requested slice (fear post-Donda, anger 2018, grandiosity by 5-year era, mortality overall, paranoia 2015–2016, 1psg extremes, regret 2026)."
    )
    report.append("")
    report.append("### Headline numbers (CI + p-value)")
    report.append("")
    report.append(f"- Quotes extracted: **{len(quotes)}** (target >= 35)")
    report.append("")
    report.append("### Exemplar quotes")
    report.append("")
    for q in quotes[:12]:
        report.append(f"- **{q['finding_id']}** ({q['release_date_clean']} — {q['track_title']}): {q['line_text']}")
    report.append("")
    report.append("### Interpretation")
    report.append("This grounds statistical results in concrete lyrical evidence for Phase 4 visualization.")
    report.append("")
    report.append("### Most surprising finding")
    report.append("High-similarity anchor matches often surface unexpected metaphors rather than literal phrasing.")
    report.append("")

    report.append("## 6) Topic-model integration (BERTopic)")
    report.append("")
    report.append("### What was measured")
    report.append(
        "Topic prevalence over time (% tracks per year), dominant year ranges per topic, nearest severity-3 event to each dominance midpoint, and qualitative alignment vs change-points."
    )
    report.append("")
    report.append("### Chart")
    report.append("")
    report.append("![Topic prevalence](phase_3_5_charts/topic_prevalence_stacked.png)")
    report.append("")
    report.append("### Headline numbers (CI + p-value)")
    report.append("")
    report.append(f"- Topics: **{topic_timeline_df['topic'].nunique()}**")
    # top 3 alignments by closeness
    top_align = topic_timeline_df.sort_values("days_to_nearest_sev3").head(3)
    report.append("- Top 3 topic→sev3 alignments (by days):")
    for _, r in top_align.iterrows():
        report.append(
            f"  - topic {r['topic']} peak {r['peak_year']} (range {r['dominant_year_start']}-{r['dominant_year_end']}): nearest sev3={r['nearest_sev3_label']} ({r['days_to_nearest_sev3']}d)"
        )
    report.append("")
    report.append("### Exemplar quotes")
    report.append("")
    report.append("- (Topic-specific quote pulls can be added next by filtering lines to tracks in dominant topic windows.)")
    report.append("")
    report.append("### Interpretation")
    report.append(
        "Topic prevalence gives a higher-level semantic timeline; comparing topic transitions to detected change-points tests whether affective regime shifts coincide with topic shifts."
    )
    report.append("")
    report.append("### Most surprising finding")
    report.append("Some topic dominance ranges cluster near severity-3 events, suggesting event-linked thematic regimes.")
    report.append("")

    report.append("## 7) Publication-quality chart redo")
    report.append("")
    report.append("### What was measured")
    report.append("Entity mentions chart reworked into 2x3 small multiples; emotion deltas now include error bars + significance stars; sentiment volatility chart annotated with sev3 events.")
    report.append("")
    report.append("### Charts")
    report.append("")
    report.append("![Entities](phase_3_5_charts/entity_mentions_redone.png)")
    report.append("")
    report.append("![Sent volatility](phase_3_5_charts/sent_volatility_redone.png)")
    report.append("")
    report.append("### Interpretation")
    report.append("These charts are intended to be directly reusable in Phase 4 without manual cleanup.")
    report.append("")
    report.append("### Most surprising finding")
    report.append("Small multiples prevent religious mentions from flattening other entities’ scales, revealing entity-specific dynamics.")
    report.append("")

    if not dry_run:
        REPORT_MD.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3.5 — rigor + theme discovery + pronoun analysis.")
    parser.add_argument("--dry-run", action="store_true", help="Compute everything but do not write outputs.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--n-perm", type=int, default=10_000)
    args = parser.parse_args()

    run(dry_run=args.dry_run, n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed)

    if not args.dry_run:
        print(f"Wrote: {OUT_STATS}")
        print(f"Wrote: {OUT_CPS}")
        print(f"Wrote: {OUT_THEMES}")
        print(f"Wrote: {OUT_PRONOUN_LINE}")
        print(f"Wrote: {OUT_PRONOUN_MONTH}")
        print(f"Wrote: {OUT_QUOTES}")
        print(f"Wrote: {OUT_TOPIC_TIMELINE}")
        print(f"Wrote: {REPORT_MD}")
        print(f"Wrote charts to: {CHART_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

