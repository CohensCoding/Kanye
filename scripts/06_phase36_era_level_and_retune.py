#!/usr/bin/env python3
"""
Phase 3.6 — Era-level rigor + change-point retune + fear deep dive + quote pack v2.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TRACK_FEATURES = DATA_DIR / "kanye_track_features.csv"
LINE_ENRICHED = DATA_DIR / "kanye_line_features_enriched.csv"
ERA_FEATURES = DATA_DIR / "kanye_era_features.csv"
LIFE_EVENTS = DATA_DIR / "kanye_life_events.csv"
LINE_EMBEDDINGS = DATA_DIR / "kanye_line_embeddings.npy"
PRONOUN_LINE = DATA_DIR / "phase_3_5_pronoun_features.csv"
PHASE35_STATS = DATA_DIR / "phase_3_5_statistical_tests.csv"

OUT_ERA_EMOTION = DATA_DIR / "phase_3_6_era_level_tests.csv"
OUT_ERA_PRONOUN = DATA_DIR / "phase_3_6_era_pronouns.csv"
OUT_CP_FULL = DATA_DIR / "phase_3_6_change_points_full_sweep.csv"
OUT_CP_CRED = DATA_DIR / "phase_3_6_change_points_credible.csv"
OUT_FEAR_DIVE = DATA_DIR / "phase_3_6_fear_deep_dive.csv"
OUT_QUOTES_V2 = DATA_DIR / "phase_3_6_top_quotes_v2.json"

REPORT_MD = PROJECT_ROOT / "phase_3_6_findings.md"
CHART_DIR = PROJECT_ROOT / "phase_3_6_charts"

CHART_DPI = 300
FIGSIZE_7ROW = (14, 21)

EMOTION_COLS_7 = [
    "emotion_joy",
    "emotion_anger",
    "emotion_sadness",
    "emotion_fear",
    "emotion_disgust",
    "emotion_surprise",
    "emotion_neutral",
]

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

MODEL_NAME = "all-MiniLM-L6-v2"

ERA_LEVEL_EXPORT_COLS = [
    "era_clean",
    "era_rank",
    "n_lines",
    "metric",
    "era_mean",
    "ci_low",
    "ci_high",
    "global_mean",
    "effect_size_z",
    "exclude_baseline",
]


def _require(p: Path) -> None:
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")


def _read_csv(p: Path) -> pd.DataFrame:
    _require(p)
    return pd.read_csv(p)


def _dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _df_to_markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    """Markdown pipe table without optional pandas/tabulate dependency."""
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows)
    cols = [str(c) for c in view.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines_body = []
    for _, row in view.iterrows():
        cells = []
        for c in view.columns:
            v = row[c]
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, (float, np.floating)) and np.isfinite(v):
                cells.append(f"{float(v):.6g}")
            else:
                s = str(v).replace("|", "\\|").replace("\n", " ")
                cells.append(s[:500])
        lines_body.append("| " + " | ".join(cells) + " |")
    tail = ""
    if len(df) > max_rows:
        tail = f"\n\n_Showing first {max_rows} of {len(df)} rows._\n"
    return "\n".join([header, sep] + lines_body) + tail


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Returns point_mean, ci_low, ci_high (line-level resampling)."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(values.mean())
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return point, float(lo), float(hi)


def bootstrap_p_global_null(
    era_vals: np.ndarray,
    global_vals: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 10_000,
) -> float:
    """Two-sided p: compare |era_mean - global_mean| to draws of same n from global."""
    era_vals = era_vals[np.isfinite(era_vals)]
    global_vals = global_vals[np.isfinite(global_vals)]
    if era_vals.size == 0 or global_vals.size == 0:
        return float("nan")
    g_mean = float(global_vals.mean())
    obs = abs(float(era_vals.mean()) - g_mean)
    n = era_vals.size
    draws = rng.choice(global_vals, size=(n_perm, n), replace=True)
    samp_means = draws.mean(axis=1)
    diffs = np.abs(samp_means - g_mean)
    return float((diffs >= obs).mean())


def exclude_baseline(ci_low: float, ci_high: float, global_mean: float) -> bool:
    if not (np.isfinite(ci_low) and np.isfinite(ci_high) and np.isfinite(global_mean)):
        return False
    return global_mean < ci_low or global_mean > ci_high


def flag_substantive(excl: bool, z: float, thresh: float = 0.3) -> bool:
    return excl and np.isfinite(z) and abs(z) > thresh


def era_emotion_analysis(
    lines: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    era_order = lines.groupby(["era_rank", "era_clean"]).size().reset_index()[["era_rank", "era_clean"]].sort_values(
        "era_rank"
    )
    rows = []
    for metric in EMOTION_COLS_7:
        global_vals = lines[metric].to_numpy(dtype=float)
        global_vals = global_vals[np.isfinite(global_vals)]
        global_mean = float(global_vals.mean())
        global_std = float(global_vals.std(ddof=0))
        if global_std == 0:
            global_std = 1e-12

        for _, er in era_order.iterrows():
            era = er["era_clean"]
            rank = int(er["era_rank"])
            sub = lines.loc[lines["era_clean"] == era, metric].to_numpy(dtype=float)
            n_lines = int(np.sum(np.isfinite(sub)))
            era_mean, ci_lo, ci_hi = bootstrap_mean_ci(sub, n_boot, rng)
            z = (era_mean - global_mean) / global_std if np.isfinite(era_mean) else float("nan")
            excl = exclude_baseline(ci_lo, ci_hi, global_mean)
            p_val = bootstrap_p_global_null(sub, global_vals, rng)
            rows.append(
                {
                    "era_clean": era,
                    "era_rank": rank,
                    "n_lines": n_lines,
                    "metric": metric,
                    "era_mean": era_mean,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "global_mean": global_mean,
                    "effect_size_z": z,
                    "exclude_baseline": excl,
                    "bootstrap_p_two_sided": p_val,
                    "substantive_signature": flag_substantive(excl, z),
                }
            )
    return pd.DataFrame(rows)


def era_pronoun_analysis(
    pron: pd.DataFrame,
    lines: pd.DataFrame,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Columns rate_1psg, rate_1ppl; derive rate_2p, rate_3ppl."""
    pron = pron.copy()
    if len(pron) == len(lines) and "era_rank" not in pron.columns:
        pron["era_rank"] = lines["era_rank"].to_numpy()
    if "rate_2p" not in pron.columns:
        pron["rate_2p"] = pron["count_2p"] / pron["n_tokens"].replace(0, np.nan)
    if "rate_3ppl" not in pron.columns:
        pron["rate_3ppl"] = pron["count_3ppl"] / pron["n_tokens"].replace(0, np.nan)

    cats = ["rate_1psg", "rate_1ppl", "rate_2p", "rate_3ppl"]
    metric_labels = ["1psg", "1ppl", "2p", "3ppl"]

    era_rank_map = pron.groupby("era_clean", as_index=False)["era_rank"].first().sort_values("era_rank")

    rows = []
    for metric, label in zip(cats, metric_labels):
        vals_all = pron[metric].to_numpy(dtype=float)
        vals_all = vals_all[np.isfinite(vals_all)]
        global_mean = float(vals_all.mean())
        global_std = float(vals_all.std(ddof=0)) or 1e-12

        for _, er in era_rank_map.iterrows():
            era = er["era_clean"]
            rank = int(er["era_rank"])
            sub = pron.loc[pron["era_clean"] == era, metric].to_numpy(dtype=float)
            n_lines = int(np.sum(np.isfinite(sub)))
            era_mean, ci_lo, ci_hi = bootstrap_mean_ci(sub, n_boot, rng)
            z = (era_mean - global_mean) / global_std if np.isfinite(era_mean) else float("nan")
            excl = exclude_baseline(ci_lo, ci_hi, global_mean)
            p_val = bootstrap_p_global_null(sub, vals_all, rng)
            rows.append(
                {
                    "era_clean": era,
                    "era_rank": rank,
                    "n_lines": n_lines,
                    "metric": label,
                    "era_mean": era_mean,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "global_mean": global_mean,
                    "effect_size_z": z,
                    "exclude_baseline": excl,
                    "bootstrap_p_two_sided": p_val,
                    "substantive_signature": flag_substantive(excl, z),
                }
            )
    return pd.DataFrame(rows)


def build_monthly_series_monthly_ffill(
    lines: pd.DataFrame,
    tracks: pd.DataFrame,
) -> pd.DataFrame:
    lines = lines.copy()
    lines["month"] = _dt(lines["release_date_clean"]).dt.to_period("M").dt.to_timestamp()

    lines["roberta_margin_row"] = lines["roberta_pos"] - lines["roberta_neg"]
    monthly_em = (
        lines.groupby("month")[
            ["emotion_anger", "emotion_sadness", "emotion_fear", "emotion_joy", "roberta_margin_row"]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"roberta_margin_row": "roberta_margin"})
    )

    tracks = tracks.copy()
    tracks["month"] = _dt(tracks["release_date_clean"]).dt.to_period("M").dt.to_timestamp()
    mtld_m = tracks.groupby("month")["mtld"].mean(numeric_only=True).reset_index()
    fk_m = tracks.groupby("month")["flesch_kincaid_grade"].mean(numeric_only=True).reset_index()
    monthly = monthly_em.merge(mtld_m, on="month", how="outer").merge(fk_m, on="month", how="outer")
    monthly = monthly.sort_values("month").set_index("month")

    if monthly.index.size == 0:
        return monthly.reset_index()

    full_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_idx).ffill()
    monthly = monthly.reset_index()
    c0 = monthly.columns[0]
    monthly = monthly.rename(columns={c0: "month"})
    return monthly


def build_quarterly_series(monthly_ffill: pd.DataFrame) -> pd.DataFrame:
    if monthly_ffill.empty or "month" not in monthly_ffill.columns:
        return pd.DataFrame()
    m = monthly_ffill.set_index("month").sort_index()
    q = m.resample("QE").mean(numeric_only=True).reset_index()
    c0 = q.columns[0]
    q = q.rename(columns={c0: "quarter_end"})
    return q


def run_pelt(series: np.ndarray, pen: float, model: str) -> list[int]:
    import ruptures as rpt

    sig = np.asarray(series, dtype=float).reshape(-1, 1)
    sig = np.nan_to_num(sig, nan=np.nanmedian(sig))
    algo = rpt.Pelt(model=model).fit(sig)
    return algo.predict(pen=pen)


def change_point_sweep(
    lines: pd.DataFrame,
    tracks: pd.DataFrame,
    events: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    penalties = [3, 5, 8, 10]
    models = ["rbf", "l2"]
    aggs = ["monthly_ffill", "quarterly"]

    metric_cols_cfg = [
        ("emotion_anger", "emotion_anger"),
        ("emotion_sadness", "emotion_sadness"),
        ("emotion_fear", "emotion_fear"),
        ("emotion_joy", "emotion_joy"),
        ("roberta_margin", "roberta_margin"),
        ("mtld", "mtld"),
        ("flesch_kincaid_grade", "flesch_kincaid_grade"),
    ]

    monthly = build_monthly_series_monthly_ffill(lines, tracks)
    quarterly = build_quarterly_series(monthly)

    full_rows: list[dict[str, Any]] = []

    for agg_name in aggs:
        if agg_name == "monthly_ffill":
            df_agg = monthly.copy()
            date_col = "month"
        else:
            df_agg = quarterly.copy()
            date_col = "quarter_end"

        for model in models:
            for pen in penalties:
                for metric_key, col in metric_cols_cfg:
                    if col not in df_agg.columns:
                        continue
                    y = df_agg[col].to_numpy(dtype=float)
                    if np.sum(np.isfinite(y)) < 6:
                        continue
                    bkps = run_pelt(y, float(pen), model)
                    dates = df_agg[date_col].to_numpy()
                    for b in bkps[:-1]:
                        if b <= 0 or b >= len(dates):
                            continue
                        cp_date = pd.Timestamp(dates[b])
                        before = y[max(0, b - 2) : b]
                        after = y[b : min(len(y), b + 2)]
                        mag = float(np.nanmean(after) - np.nanmean(before))
                        direction = "up" if mag > 0 else "down"

                        if events.empty:
                            nearest_lab = ""
                            days_nearest = 999999
                            nearest3 = ""
                            days3 = 999999
                        else:
                            ev_dt = _dt(events["date"])
                            deltas = (ev_dt - cp_date).abs().dt.days
                            j = int(np.nanargmin(deltas.to_numpy()))
                            nearest_lab = str(events.iloc[j]["event_label"])
                            days_nearest = int(deltas.iloc[j])

                            sev3 = events[events["severity"] == 3].copy()
                            if sev3.empty:
                                nearest3 = ""
                                days3 = 999999
                            else:
                                ev_dt3 = _dt(sev3["date"])
                                deltas3 = (ev_dt3 - cp_date).abs().dt.days
                                j3 = int(np.nanargmin(deltas3.to_numpy()))
                                nearest3 = str(sev3.iloc[j3]["event_label"])
                                days3 = int(deltas3.iloc[j3])

                        if days3 <= 90:
                            classification = "near_severity3_90d"
                        elif days_nearest <= 90:
                            classification = "near_any_event_90d"
                        else:
                            classification = "distant_from_catalog_events"

                        full_rows.append(
                            {
                                "metric": metric_key,
                                "pen": pen,
                                "model": model,
                                "agg": agg_name,
                                "change_point_date": cp_date.strftime("%Y-%m-%d"),
                                "magnitude": mag,
                                "direction": direction,
                                "nearest_event": nearest_lab,
                                "days_to_nearest": days_nearest,
                                "nearest_sev3": nearest3,
                                "days_to_nearest_sev3": days3,
                                "classification": classification,
                                "setting_id": f"{agg_name}|{model}|{pen}",
                            }
                        )

    full_df = pd.DataFrame(full_rows)

    cred_cols = [
        "metric",
        "change_point_date",
        "n_settings_support",
        "support_settings",
        "magnitude",
        "nearest_event",
        "days_to_nearest",
        "nearest_sev3",
        "days_to_nearest_sev3",
    ]

    # Credible: appears in >=3 distinct settings within ±60 days (same metric), dedupe
    cred_rows = []
    window_days = 60
    if full_df.empty:
        return full_df, pd.DataFrame(columns=cred_cols)

    for metric in full_df["metric"].unique():
        sub = full_df[full_df["metric"] == metric].copy()
        sub["cp_dt"] = pd.to_datetime(sub["change_point_date"])
        sub = sub.sort_values("cp_dt")
        used = set()
        dates_sorted = sub["cp_dt"].sort_values().unique()
        for cp in dates_sorted:
            if pd.Timestamp(cp) in used:
                continue
            cluster = sub[
                (sub["cp_dt"] >= pd.Timestamp(cp) - pd.Timedelta(days=window_days))
                & (sub["cp_dt"] <= pd.Timestamp(cp) + pd.Timedelta(days=window_days))
            ]
            settings = cluster["setting_id"].unique()
            if len(settings) < 3:
                continue
            rep_date = cluster["cp_dt"].median()
            used.update(cluster["cp_dt"].tolist())
            cred_rows.append(
                {
                    "metric": metric,
                    "change_point_date": pd.Timestamp(rep_date).strftime("%Y-%m-%d"),
                    "n_settings_support": len(settings),
                    "support_settings": "|".join(sorted(settings)),
                    "magnitude": float(cluster["magnitude"].median()),
                    "nearest_event": str(cluster.loc[cluster["days_to_nearest"].idxmin(), "nearest_event"]),
                    "days_to_nearest": int(cluster["days_to_nearest"].min()),
                    "nearest_sev3": str(cluster.loc[cluster["days_to_nearest_sev3"].idxmin(), "nearest_sev3"]),
                    "days_to_nearest_sev3": int(cluster["days_to_nearest_sev3"].min()),
                }
            )

    cred_df = pd.DataFrame(cred_rows)
    return full_df, cred_df


def fear_deep_dive(lines: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    lines = lines.copy()
    lines["month"] = _dt(lines["release_date_clean"]).dt.to_period("M").dt.to_timestamp()
    monthly_fear = lines.groupby("month")["emotion_fear"].mean(numeric_only=True).reset_index()
    monthly_fear = monthly_fear.sort_values("emotion_fear", ascending=False)

    sev3 = events[events["severity"] == 3].copy()
    sev3["dt"] = _dt(sev3["date"])

    rows_out = []
    top5 = monthly_fear.head(5)
    for _, r in top5.iterrows():
        m = r["month"]
        tracks_in_month = (
            lines.loc[lines["month"] == m, ["track_title", "release_date_clean"]]
            .drop_duplicates()
            .values.tolist()
        )
        if sev3.empty:
            dmin = 9999
        else:
            days_min = [int(abs((pd.Timestamp(m) - ev["dt"]).days)) for _, ev in sev3.iterrows()]
            dmin = min(days_min) if days_min else 9999
        hidden = dmin > 180
        month_str = pd.Timestamp(m).strftime("%Y-%m-%d")
        rows_out.append(
            {
                "month": month_str,
                "mean_emotion_fear": float(r["emotion_fear"]),
                "tracks_released": "; ".join([t[0] for t in tracks_in_month[:15]])
                + ("..." if len(tracks_in_month) > 15 else ""),
                "days_to_nearest_sev3": dmin,
                "hidden_fear_inflection": hidden,
                "hidden_inflection_name": (
                    f"Hidden fear spike ({month_str}, Δdays≥181 from nearest severity-3)"
                    if hidden
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows_out)


def line_quote_filters(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 5:
        return False
    if len(set(words)) < 4:
        return False
    alnum = sum(1 for c in text if c.isalnum())
    if alnum < 15:
        return False
    return True


def build_quotes_v2(lines: pd.DataFrame, emb: np.ndarray, pron: pd.DataFrame, _rng: np.random.Generator) -> list[dict]:
    from sentence_transformers import SentenceTransformer

    if emb.shape[0] != len(lines):
        raise SystemExit(f"Embeddings {emb.shape[0]} vs lines {len(lines)}")
    model = SentenceTransformer(MODEL_NAME)
    anchors = list(ANCHORS.keys())
    texts = [ANCHORS[k] for k in anchors]
    avec = np.asarray(model.encode(texts, normalize_embeddings=False), dtype=float)
    emb_f = emb.astype(np.float64)
    emb_n = emb_f / (np.linalg.norm(emb_f, axis=1, keepdims=True) + 1e-12)
    avec_n = avec / (np.linalg.norm(avec, axis=1, keepdims=True) + 1e-12)
    sims = emb_n @ avec_n.T

    lines = lines.reset_index(drop=True)
    lines["release_dt"] = _dt(lines["release_date_clean"])

    base_scores: dict[str, np.ndarray] = {}
    for i, k in enumerate(anchors):
        base_scores[f"sim_{k}"] = sims[:, i]

    base_scores["emotion_fear"] = lines["emotion_fear"].to_numpy(dtype=float)
    base_scores["emotion_anger"] = lines["emotion_anger"].to_numpy(dtype=float)
    lines["rate_1psg"] = np.nan
    if len(pron) == len(lines):
        lines["rate_1psg"] = pron["rate_1psg"].to_numpy(dtype=float)

    # Masks for time slices
    m_donda = (lines["release_dt"] >= "2007-11-10") & (lines["release_dt"] <= "2008-11-10")
    m_2018 = lines["release_dt"].dt.year == 2018
    m_1516 = (lines["release_dt"] >= "2015-01-01") & (lines["release_dt"] <= "2016-12-31")
    m_2026 = lines["release_dt"].dt.year == 2026

    eras_5 = [
        ("2003-01-01", "2007-12-31", "era_2003_2007"),
        ("2008-01-01", "2012-12-31", "era_2008_2012"),
        ("2013-01-01", "2017-12-31", "era_2013_2017"),
        ("2018-01-01", "2022-12-31", "era_2018_2022"),
        ("2023-01-01", "2026-12-31", "era_2023_2026"),
    ]

    cat_specs: list[tuple[str, np.ndarray, str]] = []
    cat_specs.append(("quotes_donda_fear_top5", m_donda.to_numpy(), "emotion_fear"))
    cat_specs.append(("quotes_2018_anger", m_2018.to_numpy(), "emotion_anger"))
    cat_specs.append(("quotes_paranoia_2015_16", m_1516.to_numpy(), "sim_paranoia"))
    cat_specs.append(("quotes_regret_2026", m_2026.to_numpy(), "sim_regret"))
    cat_specs.append(("quotes_mortality_all", np.ones(len(lines), dtype=bool), "sim_mortality"))

    for a, b, lab in eras_5:
        mask = (lines["release_dt"] >= a) & (lines["release_dt"] <= b)
        cat_specs.append((f"quotes_grandiosity_{lab}", mask.to_numpy(), "sim_grandiosity"))

    for theme in anchors:
        cat_specs.append((f"quotes_theme_{theme}", np.ones(len(lines), dtype=bool), f"sim_{theme}"))

    cat_specs.append(("quotes_high_1psg", np.ones(len(lines), dtype=bool), "rate_1psg"))

    def raw_score_row(i: int, score_col: str) -> float:
        if score_col == "rate_1psg":
            v = lines.iloc[i]["rate_1psg"]
            return float(v) if np.isfinite(v) else float("nan")
        arr = base_scores[score_col]
        return float(arr[i])

    # Deep candidates per category (filtered lines only), before cross-category assignment
    cand: dict[str, list[tuple[int, float]]] = {}
    for cat_id, mask, score_col in cat_specs:
        scores = base_scores.get(score_col)
        if scores is None and score_col == "rate_1psg":
            scores = lines["rate_1psg"].to_numpy(dtype=float)
        if scores is None:
            cand[cat_id] = []
            continue
        idx = np.where(mask & np.isfinite(scores))[0]
        if idx.size == 0:
            cand[cat_id] = []
            continue
        order = idx[np.argsort(scores[idx])[::-1][:400]]
        pairs: list[tuple[int, float]] = []
        for j in order:
            txt = str(lines.iloc[int(j)]["line_text"])
            if not line_quote_filters(txt):
                continue
            pairs.append((int(j), float(scores[j])))
            if len(pairs) >= 120:
                break
        cand[cat_id] = pairs

    line_scores_by_cat: dict[int, dict[str, float]] = defaultdict(dict)
    cat_abs_max: dict[str, float] = {}
    for cat_id, pairs in cand.items():
        if not pairs:
            cat_abs_max[cat_id] = 1.0
            continue
        mx = max(abs(sc) for _, sc in pairs)
        cat_abs_max[cat_id] = mx if mx > 0 else 1.0
        for i, sc in pairs:
            line_scores_by_cat[i][cat_id] = abs(sc) / cat_abs_max[cat_id]

    assigned_cat: dict[int, str] = {}
    for i, per_cat in line_scores_by_cat.items():
        assigned_cat[i] = max(per_cat.keys(), key=lambda k: per_cat[k])

    quotes_out: list[dict[str, Any]] = []
    seen_text: set[str] = set()

    def assemble(max_per_cat: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen_local: set[str] = set()
        for cat_id, mask_arr, score_col in cat_specs:
            mask_arr = np.asarray(mask_arr, dtype=bool)
            cand_indices = {i for i, _ in cand.get(cat_id, [])}
            ranked: list[tuple[int, float]] = []
            for i in cand_indices:
                if assigned_cat.get(i) != cat_id:
                    continue
                if i >= len(lines) or not mask_arr[i]:
                    continue
                rs = raw_score_row(i, score_col)
                if not np.isfinite(rs):
                    continue
                ranked.append((i, rs))
            ranked.sort(key=lambda x: x[1], reverse=True)
            picked = 0
            for i, mv in ranked:
                txt = str(lines.iloc[i]["line_text"]).strip()
                if txt in seen_local:
                    continue
                if not line_quote_filters(txt):
                    continue
                seen_local.add(txt)
                out.append(
                    {
                        "finding_id": cat_id,
                        "metric_value": mv,
                        "line_text": txt,
                        "track_title": str(lines.iloc[i]["track_title"]),
                        "release_date_clean": str(lines.iloc[i]["release_date_clean"]),
                        "era_clean": str(lines.iloc[i]["era_clean"]),
                        "score_column": score_col,
                    }
                )
                picked += 1
                if picked >= max_per_cat:
                    break
        return out

    max_per = 5
    while max_per <= 40:
        quotes_try = assemble(max_per)
        quotes_try.sort(key=lambda q: (q["finding_id"], -float(q["metric_value"])))
        deduped: list[dict[str, Any]] = []
        seen_text.clear()
        for q in quotes_try:
            t = q["line_text"].strip()
            if t in seen_text:
                continue
            seen_text.add(t)
            deduped.append(q)
        quotes_out = deduped
        if len(quotes_out) >= 35:
            break
        max_per += 3

    # Tie-break ordering: Donda quotes first in JSON for report convenience
    quotes_out.sort(key=lambda q: (0 if q["finding_id"] == "quotes_donda_fear_top5" else 1, q["finding_id"]))
    return quotes_out


def plot_era_emotions(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = EMOTION_COLS_7
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 22), sharex=False)
    if len(metrics) == 1:
        axes = [axes]
    df = df.sort_values("era_rank")
    eras = df["era_clean"].unique()
    era_rank_order = df.groupby("era_clean")["era_rank"].first().sort_values().index.tolist()
    x = np.arange(len(era_rank_order))
    for ax, met in zip(axes, metrics):
        sub = df[df["metric"] == met].set_index("era_clean").reindex(era_rank_order)
        g_mean = float(sub["global_mean"].iloc[0])
        y = sub["era_mean"].to_numpy(dtype=float)
        lo = sub["ci_low"].to_numpy(dtype=float)
        hi = sub["ci_high"].to_numpy(dtype=float)
        err = np.vstack([y - lo, hi - y])
        ax.bar(x, y, yerr=err, capsize=3, color="steelblue", alpha=0.85, ecolor="black")
        ax.axhline(g_mean, color="tab:red", linestyle="--", linewidth=1.2, label="Global mean")
        ax.set_title(met.replace("emotion_", ""))
        ax.set_xticks(x)
        ax.set_xticklabels(era_rank_order, rotation=55, ha="right", fontsize=7)
        sig = sub["substantive_signature"].to_numpy()
        lo_a = sub["ci_low"].to_numpy(dtype=float)
        hi_a = sub["ci_high"].to_numpy(dtype=float)
        for i, (xi, s) in enumerate(zip(x, sig)):
            if s:
                span = hi_a[i] - lo_a[i] if np.isfinite(hi_a[i]) and np.isfinite(lo_a[i]) else 0.05
                y_star = hi_a[i] + 0.05 * max(span, 1e-6)
                ax.text(float(xi), y_star, "*", ha="center", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Era-level emotion means ± 95% bootstrap CI (* = substantive signature)", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=CHART_DPI)
    plt.close(fig)


def plot_era_pronouns(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = sorted(df["metric"].unique())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 14))
    df = df.sort_values(["metric", "era_rank"])
    era_rank_order = df.groupby("era_clean")["era_rank"].first().sort_values().index.tolist()
    x = np.arange(len(era_rank_order))
    for ax, met in zip(axes, metrics):
        sub = df[df["metric"] == met].set_index("era_clean").reindex(era_rank_order)
        g_mean = float(sub["global_mean"].iloc[0])
        y = sub["era_mean"].to_numpy(dtype=float)
        lo = sub["ci_low"].to_numpy(dtype=float)
        hi = sub["ci_high"].to_numpy(dtype=float)
        err = np.vstack([y - lo, hi - y])
        ax.bar(x, y, yerr=err, capsize=3, color="darkorange", alpha=0.85, ecolor="black")
        ax.axhline(g_mean, color="tab:red", linestyle="--", linewidth=1.2)
        ax.set_title(f"Pronoun rate: {met}")
        ax.set_xticks(x)
        ax.set_xticklabels(era_rank_order, rotation=55, ha="right", fontsize=7)
    fig.suptitle("Era-level pronoun rates ± 95% bootstrap CI", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=CHART_DPI)
    plt.close(fig)


def plot_credible_cps(monthly: pd.DataFrame, credible: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = ["emotion_anger", "emotion_sadness", "emotion_fear", "emotion_joy", "roberta_margin", "mtld", "flesch_kincaid_grade"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 21))
    mdf = monthly.sort_values("month")
    for ax, met in zip(axes, metrics):
        if met not in mdf.columns:
            ax.axis("off")
            continue
        ax.plot(mdf["month"], mdf[met], color="tab:blue", lw=1)
        _ymin, ymax = ax.get_ylim()
        sub = credible[credible["metric"] == met] if len(credible) else pd.DataFrame()
        for _, r in sub.iterrows():
            ts = pd.Timestamp(r["change_point_date"])
            ax.axvline(ts, color="tab:red", linestyle="--", alpha=0.8)
            ax.text(ts, ymax, str(r["change_point_date"]), rotation=90, fontsize=6, va="top", clip_on=False)
        ax.set_title(met)
    fig.suptitle("Monthly series (forward-filled) with credible change-points", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=CHART_DPI)
    plt.close(fig)


def plot_fear_trajectory(lines: pd.DataFrame, events: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    lines = lines.copy()
    lines["month"] = _dt(lines["release_date_clean"]).dt.to_period("M").dt.to_timestamp()
    m = lines.groupby("month")["emotion_fear"].mean(numeric_only=True).reset_index()
    sev3 = events[events["severity"] == 3]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(m["month"], m["emotion_fear"], color="purple", lw=1.5)
    _ymin, ymax = ax.get_ylim()
    for _, ev in sev3.iterrows():
        ax.axvline(pd.Timestamp(ev["date"]), color="tab:red", alpha=0.7, ls="--")
    for _, ev in sev3.iterrows():
        ax.text(
            pd.Timestamp(ev["date"]),
            ymax,
            str(ev["event_label"])[:40],
            rotation=90,
            fontsize=6,
            va="top",
            clip_on=False,
        )
    ax.set_title("Monthly mean emotion_fear with severity-3 events")
    ax.set_xlabel("Month")
    ax.set_ylabel("emotion_fear")
    fig.tight_layout()
    fig.savefig(out_path, dpi=CHART_DPI)
    plt.close(fig)


def write_report(
    era_em: pd.DataFrame,
    era_pr: pd.DataFrame,
    credible: pd.DataFrame,
    fear_df: pd.DataFrame,
    quotes: list[dict],
    phase35_stats: pd.DataFrame,
    lines: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    donda = phase35_stats[
        (phase35_stats["event_label"].str.contains("Mother Donda", na=False))
        & (phase35_stats["metric"] == "emotion_fear")
    ]
    donda_row = donda.iloc[0] if len(donda) else None

    sig_em = era_em[(era_em["exclude_baseline"]) & (era_em["effect_size_z"].abs() > 0.3)]
    top_z_em = era_em.loc[era_em["effect_size_z"].abs().nlargest(8).index]

    pr_1psg = era_pr[era_pr["metric"] == "1psg"]
    hi_era = pr_1psg.loc[pr_1psg["effect_size_z"].idxmax()] if len(pr_1psg) else None
    lo_era = pr_1psg.loc[pr_1psg["effect_size_z"].idxmin()] if len(pr_1psg) else None

    headline = []
    if donda_row is not None:
        headline.append(
            f"Donda-window fear spike (Phase 3.5 event test): Δ={float(donda_row['point_estimate']):.4f}, "
            f"95% CI [{float(donda_row['ci_low']):.4f}, {float(donda_row['ci_high']):.4f}], "
            f"permutation p={float(donda_row['perm_p_value']):.4f}, n_lines={int(donda_row['n_lines'])}."
        )
    if len(sig_em) >= 3:
        headline.append(
            f"Era-level substantive signatures (exclude baseline & |z|>0.3): **{len(sig_em)}** emotion-era pairs."
        )
    elif len(sig_em) == 0:
        headline.append(
            "Era-level substantive signatures (exclude baseline & |z|>0.3): **0** pairs — "
            "no era×emotion combinations cleared both gates at this threshold."
        )
    else:
        headline.append(
            f"Era-level substantive signatures (exclude baseline & |z|>0.3): **{len(sig_em)}** pairs "
            f"(below the Phase 3.6 review threshold of 3; contrasts versus the global mean are sparse at |z|>0.3)."
        )
    if hi_era is not None:
        headline.append(
            f"Highest 1psg deviation era: **{hi_era['era_clean']}** (z={float(hi_era['effect_size_z']):.3f}, "
            f"95% CI [{float(hi_era['ci_low']):.4f},{float(hi_era['ci_high']):.4f}], p={float(hi_era['bootstrap_p_two_sided']):.4f})."
        )
    if lo_era is not None:
        headline.append(
            f"Lowest 1psg deviation era: **{lo_era['era_clean']}** (z={float(lo_era['effect_size_z']):.3f}, "
            f"95% CI [{float(lo_era['ci_low']):.4f}, {float(lo_era['ci_high']):.4f}], "
            f"p={float(lo_era['bootstrap_p_two_sided']):.4f})."
        )
    headline.append(
        f"Credible change-points (≥3 settings, ±60d cluster): **{len(credible)}** "
        + ("(none robust)" if credible.empty else "(see section 4).")
    )
    headline.append(f"Quote pack v2: **{len(quotes)}** curated lines (filtered + deduped).")

    sec_quotes_donda = [q for q in quotes if q["finding_id"] == "quotes_donda_fear_top5"]

    lines_md = ["# Phase 3.6 — Era-level rigor + change-point retune\n"]
    lines_md.append("## HEADLINE FINDINGS\n")
    for h in headline:
        lines_md.append(f"- {h}")
    lines_md.append("\n---\n")

    lines_md.append("## (1) Donda fear finding\n")
    lines_md.append(
        "### What was measured\nPhase 3.5 severity-3 window test (365d post-event vs baseline) for "
        "`emotion_fear` after Mother Donda West dies (2007-11-10).\n"
    )
    lines_md.append("### Key numbers\n")
    if donda_row is not None:
        lines_md.append(
            f"- Δ fear (post − baseline): **{float(donda_row['point_estimate']):.4f}** "
            f"(95% bootstrap CI [{float(donda_row['ci_low']):.4f}, {float(donda_row['ci_high']):.4f}]); "
            f"permutation **p = {float(donda_row['perm_p_value']):.4f}** (n={int(donda_row['n_lines'])} lines).\n"
        )
    lines_md.append(f"![Fear trajectory](phase_3_6_charts/fear_trajectory.png)\n")
    lines_md.append("### Quotes\n")
    for q in sec_quotes_donda[:5]:
        lines_md.append(f"- *{q['track_title']}* ({q['release_date_clean']}): {q['line_text'][:280]}")
    lines_md.append("\n### Interpretation\n")
    lines_md.append(
        "This is the only curated severity-3 window that survived Phase 3.5 multiple-testing reality; "
        "era-level analyses below recover broader structure.\n"
    )

    lines_md.append("\n## (2) Era-level emotion profiles\n")
    lines_md.append("### Chart\n![Era emotions](phase_3_6_charts/era_emotion_profiles.png)\n")
    lines_md.append("### Significant combinations (exclude_baseline & |z|>0.3)\n")
    for _, r in sig_em.sort_values("effect_size_z", key=np.abs, ascending=False).head(20).iterrows():
        lines_md.append(
            f"- **{r['era_clean']}** × `{r['metric']}`: mean={float(r['era_mean']):.4f}, "
            f"CI [{float(r['ci_low']):.4f},{float(r['ci_high']):.4f}], z={float(r['effect_size_z']):.3f}, "
            f"p={float(r['bootstrap_p_two_sided']):.4f}"
        )
    q_pick = [q for q in quotes if "emotion" in q.get("score_column", "")][:3]
    lines_md.append("\n### Quotes\n")
    for q in q_pick[:3]:
        lines_md.append(f"- ({q['finding_id']}) {q['line_text'][:220]}")
    lines_md.append("\n### Interpretation\n")
    if len(sig_em) == 0:
        lines_md.append(
            "No era×emotion pairs cleared exclude_baseline with |z|>0.3—the catalog shows little concentrated "
            "era-wise deviation from global emotion means at this bar.\n"
        )
    elif len(sig_em) < 3:
        lines_md.append(
            f"Only **{len(sig_em)}** pairs cleared both gates (Phase 3.6 targeted ≥3 for a strong era-emotion read); "
            "aggregation raises power versus 365-day windows, but line-level emotions mostly hug the global mean.\n"
        )
    else:
        lines_md.append(
            "Era aggregation concentrates sufficient n_lines for stable emotion contrasts versus global baselines.\n"
        )

    lines_md.append("\n## (3) Era-level pronoun analysis\n")
    lines_md.append("![Pronouns](phase_3_6_charts/era_pronoun_profiles.png)\n")
    if hi_era is not None and lo_era is not None:
        lines_md.append(
            f"- Highest 1psg deviation: **{hi_era['era_clean']}** (z={float(hi_era['effect_size_z']):.3f}, "
            f"95% CI [{float(hi_era['ci_low']):.4f}, {float(hi_era['ci_high']):.4f}], "
            f"p={float(hi_era['bootstrap_p_two_sided']):.4f}).\n"
            f"- Lowest 1psg deviation: **{lo_era['era_clean']}** (z={float(lo_era['effect_size_z']):.3f}, "
            f"95% CI [{float(lo_era['ci_low']):.4f}, {float(lo_era['ci_high']):.4f}], "
            f"p={float(lo_era['bootstrap_p_two_sided']):.4f}).\n"
        )
    pq = [q for q in quotes if "1psg" in q.get("score_column", "")][:3]
    lines_md.append("### Quotes\n")
    for q in pq:
        lines_md.append(f"- {q['line_text'][:220]}")
    lines_md.append("\n### Interpretation\nFlat catalog-wide year trends can coexist with strong era heterogeneity in self-focus.\n")

    lines_md.append("\n## (4) Change-point credible set\n")
    lines_md.append(f"![Credible CPs](phase_3_6_charts/change_points_credible.png)\n")
    if credible.empty:
        lines_md.append("**No robust change-points** passed the ≥3-settings / ±60d credibility filter.\n")
    else:
        lines_md.append(_df_to_markdown_table(credible, max_rows=40))
        if len(credible) > 40:
            lines_md.append(f"\n_Full table: `data/phase_3_6_change_points_credible.csv` ({len(credible)} rows)._")
    lines_md.append("\n### Interpretation\nParameter sweep exposes instability at pen=10-only defaults; consensus CPs are rarer.\n")

    lines_md.append("\n## (5) Fear deep dive\n")
    lines_md.append(_df_to_markdown_table(fear_df))
    lines_md.append("\n### Interpretation\nTop fear months outside curated-event proximity flag candidate hidden fear inflections.\n")

    lines_md.append("\n## (6) Appendix — Phase 3.5 non-significant event windows\n")
    ns = phase35_stats[
        (phase35_stats["finding_id"].str.startswith("sev3_emotion_delta"))
        & (phase35_stats["significant_at_05"] == False)
        & (phase35_stats["metric"].isin(EMOTION_COLS_7))
    ]
    lines_md.append(f"{len(ns)} emotion-window tests were not significant at α=0.05 (retained for transparency).\n")

    REPORT_MD.write_text("\n".join(lines_md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3.6 era-level rigor + CP sweep + quotes v2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    lines = _read_csv(LINE_ENRICHED)
    tracks = _read_csv(TRACK_FEATURES)
    events = _read_csv(LIFE_EVENTS)
    pron = _read_csv(PRONOUN_LINE)
    phase35_stats = _read_csv(PHASE35_STATS)

    _require(LINE_EMBEDDINGS)
    emb = np.load(LINE_EMBEDDINGS)

    era_em = era_emotion_analysis(lines, args.n_boot, rng)
    era_pr = era_pronoun_analysis(pron, lines, args.n_boot, rng)

    monthly_ffill = build_monthly_series_monthly_ffill(lines, tracks)
    full_cp, cred_cp = change_point_sweep(lines, tracks, events, rng)

    fear_df = fear_deep_dive(lines, events)

    quotes = build_quotes_v2(lines, emb, pron, rng)

    if args.dry_run:
        print(
            f"Dry-run summary: era_emotion rows={len(era_em)}, era_pronoun rows={len(era_pr)}, "
            f"cp_full={len(full_cp)}, cp_credible={len(cred_cp)}, fear_rows={len(fear_df)}, quotes={len(quotes)}"
        )
        return 0

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    era_em[ERA_LEVEL_EXPORT_COLS].to_csv(OUT_ERA_EMOTION, index=False)
    era_pr[ERA_LEVEL_EXPORT_COLS].to_csv(OUT_ERA_PRONOUN, index=False)
    full_cp.to_csv(OUT_CP_FULL, index=False)
    cred_cp.to_csv(OUT_CP_CRED, index=False)
    fear_df.to_csv(OUT_FEAR_DIVE, index=False)
    OUT_QUOTES_V2.write_text(json.dumps(quotes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plot_era_emotions(era_em, CHART_DIR / "era_emotion_profiles.png")
    plot_era_pronouns(era_pr, CHART_DIR / "era_pronoun_profiles.png")
    plot_credible_cps(monthly_ffill, cred_cp, CHART_DIR / "change_points_credible.png")
    plot_fear_trajectory(lines, events, CHART_DIR / "fear_trajectory.png")

    write_report(era_em, era_pr, cred_cp, fear_df, quotes, phase35_stats, lines, rng)

    print(f"Substantive era×emotion count: {(era_em['substantive_signature']).sum()}")
    print(f"Credible CP rows: {len(cred_cp)}")
    print(f"Quotes v2: {len(quotes)}")
    print(f"Wrote {REPORT_MD}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
