"""Plots for an evolution run: best controller's altitude response and the fitness curve.

Can also be run on its own to re-plot a finished run:
    python plot_results.py results/example
"""
from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402

DECIMATE = 6  # plot at 20 Hz to keep files small
SVG_DECIMATE = 30  # 4 Hz if an SVG copy is requested
plt.rcParams["svg.fonttype"] = "none"  # keep SVG text as text, not glyph paths
plt.rcParams["svg.hashsalt"] = "flight_sim"  # reproducible SVG ids


def plot_response(out, summary, scenarios, fmt=("png",)):
    for ext in fmt:
        _plot_response(out, summary, scenarios, ext, SVG_DECIMATE if ext == "svg" else DECIMATE)


def _plot_response(out, summary, scenarios, ext, dec):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.3, 1.3]})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, sc in enumerate(scenarios):
        r = sim.simulate(summary["gains"], sc, record=True)
        tr = {k: v[::dec] for k, v in r["trace"].items()}
        label = "calm" if i == 0 else f"wind {((sc.wind_north_fps**2 + sc.wind_east_fps**2) ** 0.5):.0f} ft/s, gusts"
        c = colors[i % len(colors)]
        axes[0].plot(tr["t"], tr["h"], color=c, lw=1.4, label=f"best, scenario {i} ({label}): cost {r['cost']:.3f}")
        axes[1].plot(tr["t"], tr["theta"], color=c, lw=1)
        axes[2].plot(tr["t"], tr["elevator"], color=c, lw=1)
        if i == 0:
            axes[0].step(tr["t"], tr["target"], where="post", color="k", ls="--", lw=1.2, label="target altitude")
    # generation-0 best on the calm scenario, for contrast
    r0 = sim.simulate(summary["initial_best_gains"], scenarios[0], record=True)
    tr0 = {k: v[::dec] for k, v in r0["trace"].items()}
    axes[0].plot(tr0["t"], tr0["h"], color="0.6", lw=1, ls=":",
                 label=f"gen-0 best, scenario 0: cost {r0['cost']:.3f} ({r0['status']})")
    axes[0].set_ylabel("altitude MSL (ft)")
    axes[0].set_title(f"JSBSim {sim.AIRCRAFT}: evolved altitude-hold response "
                      f"(mean cost {summary['best_cost']:.3f}, gen-0 best {summary['initial_best_cost']:.3f})")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].set_ylabel("pitch (deg)")
    axes[2].set_ylabel("elevator cmd (norm)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out, f"best_response.{ext}"), ext)


def plot_fitness(out, history, fmt=("png",)):
    g = [h["generation"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(g, [h["best_cost"] for h in history], "o-", label="best")
    ax.plot(g, [h["median_cost"] for h in history], "s-", label="median")
    ax.set_yscale("log")
    ax.set_xlabel("generation")
    ax.set_ylabel("cost (lower is better, log scale)")
    ax.set_title("GA progress (failures cost >= 1000)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    for ext in fmt:
        _save(fig, os.path.join(out, f"fitness_curve.{ext}"), ext, close=False)
    plt.close(fig)


def _save(fig, path, ext, close=True):
    meta = {"Date": None} if ext == "svg" else {}
    fig.savefig(path, dpi=110, metadata=meta)
    if close:
        plt.close(fig)


def plot_all(out, summary, history, scenarios, fmt=("png",)):
    """Write best_response.<fmt> and fitness_curve.<fmt> (pass fmt=("png", "svg") for SVG too)."""
    plot_response(out, summary, scenarios, fmt)
    plot_fitness(out, history, fmt)


def _load_history(path):
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/run"
    with open(os.path.join(out, "best_gains.json")) as f:
        summary = json.load(f)
    scenarios = [sim.Scenario(**{**s, "steps": [tuple(x) for x in s["steps"]]}) for s in summary["scenarios"]]
    plot_all(out, summary, _load_history(os.path.join(out, "fitness_history.csv")), scenarios)
    print(f"wrote plots to {out}")
