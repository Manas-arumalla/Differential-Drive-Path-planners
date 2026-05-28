"""Benchmark analytics dashboard.

Reads benchmarks/results.csv (produced by ``python -m navstack.benchmark``) and
renders a multi-panel dashboard: a radar of normalized planner trade-offs plus
success-rate and planning-time bars.

    python -m navstack.analytics      # -> media/benchmark_dashboard.png
"""
import csv
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_ROOT, "benchmarks", "results.csv")
OUT_PATH = os.path.join(_ROOT, "media", "benchmark_dashboard.png")

# (axis label, csv column, higher_is_better)
METRICS = [
    ("Optimality", "length_m", False),
    ("Smoothness", "smoothness_deg", False),
    ("Clearance", "min_clear_m", True),
    ("Speed", "time_s", False),
]


def _load(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _aggregate(rows):
    planners = []
    for r in rows:
        if r["planner"] not in planners:
            planners.append(r["planner"])
    agg = {}
    for p in planners:
        runs = [r for r in rows if r["planner"] == p]
        ok = [r for r in runs if r["success"] == "1"]
        agg[p] = {
            "success": len(ok) / len(runs) if runs else 0.0,
            "vals": {col: np.mean([float(r[col]) for r in ok]) if ok else np.nan
                     for _, col, _ in METRICS},
        }
    return planners, agg


def _normalize(planners, agg):
    """Map each metric to a 0..1 score where 1 is best across planners."""
    scores = {p: [] for p in planners}
    for _, col, higher in METRICS:
        vals = np.array([agg[p]["vals"][col] for p in planners], float)
        finite = vals[np.isfinite(vals)]
        lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
        for i, p in enumerate(planners):
            v = vals[i]
            if not np.isfinite(v) or hi - lo < 1e-9:
                s = 0.5
            elif higher:
                s = (v - lo) / (hi - lo)
            else:
                s = (hi - v) / (hi - lo)
            scores[p].append(max(0.05, s))  # floor so every polygon is visible
    return scores


def make_dashboard(csv_path=CSV_PATH, out_path=OUT_PATH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load(csv_path)
    planners, agg = _aggregate(rows)
    scores = _normalize(planners, agg)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 6), facecolor="#1A202C")
    cmap = plt.cm.tab10(np.linspace(0, 1, len(planners)))

    # --- radar of trade-offs ---
    labels = [m[0] for m in METRICS]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    ax1 = fig.add_subplot(1, 3, 1, polar=True)
    ax1.set_facecolor("#1A202C")
    for i, p in enumerate(planners):
        vals = scores[p] + scores[p][:1]
        ax1.plot(angles, vals, color=cmap[i], lw=2, label=p)
        ax1.fill(angles, vals, color=cmap[i], alpha=0.08)
    ax1.set_xticks(angles[:-1]); ax1.set_xticklabels(labels, color="white", fontsize=10)
    ax1.set_yticklabels([]); ax1.set_title("Planner trade-offs (outer = better)", color="white", pad=20)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), facecolor="#2D3748", labelcolor="white", fontsize=8)

    # --- success rate ---
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.bar(planners, [agg[p]["success"] * 100 for p in planners], color=cmap)
    ax2.set_title("Success rate (%)", color="white"); ax2.set_ylim(0, 105)
    ax2.tick_params(axis="x", rotation=45, colors="white"); ax2.grid(True, axis="y", alpha=0.2)

    # --- planning time (log) ---
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.bar(planners, [agg[p]["vals"]["time_s"] for p in planners], color=cmap)
    ax3.set_yscale("log"); ax3.set_title("Avg planning time (s, log)", color="white")
    ax3.tick_params(axis="x", rotation=45, colors="white"); ax3.grid(True, axis="y", alpha=0.2)

    fig.suptitle("navstack — Planner Benchmark Dashboard", color="white", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor="#1A202C", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dashboard to {out_path}")
    return out_path


if __name__ == "__main__":
    make_dashboard()
