"""Figures for the README, regenerated from committed results. Never hand-edited.

    uv run python analysis/figures.py

Four figures, one story each:
  1. pass_cost_staircase   the 7B pass cost vs tokens per pass: flat, ramp, 32-token treads
  2. speedup_two_nights    same pair, same prompts, Sep 4 vs Sep 9, with the zero-parameter model
  3. acceptance_by_domain  acceptance per domain for three drafts against the 7B target
  4. batch_scaling         one-token pass and verification cost vs batch size
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spec_decode_advisor.model import MeasuredCostModel  # noqa: E402

R = ROOT / "results"
OUT = ROOT / "figures"

# ---- palette: validated categorical slots 1-3 (light), chrome and ink tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "medium",
    "axes.titlelocation": "left", "axes.titlepad": 12,
    "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "savefig.dpi": 170, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
})
MARK = dict(marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5)


def _load(name):
    return json.loads((R / name).read_text())


def fig_pass_cost_staircase():
    d = _load("05_batch_scaling.json")
    fine = {int(t): v for t, v in d["target_pass_ms_vs_total_tokens_b1"].items()}
    grid = [(int(b) * int(n), ms) for b, row in d["target_latency_ms"].items() for n, ms in row.items() if int(b) > 1]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    xs = sorted(fine)
    ax.plot(xs, [fine[x] for x in xs], color=BLUE, label="batch 1, T positions in one pass", **MARK, zorder=3)
    ax.plot([g[0] for g in grid], [g[1] for g in grid], linestyle="none", marker="o", markersize=6.5,
            markerfacecolor=SURFACE, markeredgecolor=ORANGE, markeredgewidth=2,
            label="batch B × n positions, same total T", zorder=2)
    for x in (32, 64):
        ax.axvline(x + 0.5, color=AXIS, linewidth=0.8, zorder=1)
    ax.text(2.5, 118, "flat: extra tokens hide\nunder the weight read", color=INK2, fontsize=9, ha="center")
    ax.text(8.5, 215, "ramp", color=INK2, fontsize=9, ha="center")
    ax.text(45.5, 372, "32-token treads,\n~145 ms each", color=INK2, fontsize=9, ha="center")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64, 96]))
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "96"])
    ax.set_xlabel("tokens per forward pass, T")
    ax.set_ylabel("pass time, ms")
    ax.set_ylim(0, 520)
    ax.set_title("A 7B forward pass costs the same for 1 to 4 tokens, then climbs in 32-token steps")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98))
    fig.text(0.01, -0.02, "Qwen2.5-7B-Instruct-4bit on an M4, plugged in, idle. Median of 15 passes with a warm KV cache. results/05_batch_scaling.json",
             color=MUTED, fontsize=8)
    fig.savefig(OUT / "pass_cost_staircase.png")
    plt.close(fig)


def fig_speedup_two_nights():
    ac = _load("06_power_state.json")
    bat = _load("02_draft_comparison.json")["sessions"]["0.5B"]
    zp = _load("07_zero_parameter_model.json")["measured_now"]
    model = MeasuredCostModel.from_latency(zp["c_draft"], {int(t): v for t, v in zp["target_pass_ms"].items()})

    ks_ac = sorted(int(k) for k in ac["by_depth"])
    ks_bat = sorted(int(k) for k in bat["by_depth"])
    y_ac = [ac["by_depth"][str(k)]["measured_speedup"] for k in ks_ac]
    y_bat = [bat["by_depth"][str(k)]["measured_speedup"] for k in ks_bat]
    y_model = [model.speedup(ac["by_depth"][str(k)]["pooled_p"], k) for k in ks_ac]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(1.0, color=AXIS, linewidth=0.8, zorder=1)
    ax.text(0.72, 1.012, "no speculation", color=MUTED, fontsize=8.5, va="bottom")
    ax.plot(ks_ac, y_model, color=BLUE, linewidth=1.4, linestyle=(0, (4, 3)), label="model, nothing fitted (Sep 9 timings)", zorder=2)
    ax.plot(ks_ac, y_ac, color=BLUE, label="Sep 9: plugged in, idle", **MARK, zorder=4)
    ax.plot(ks_bat, y_bat, color=ORANGE, label="Sep 4: battery, after a 49-min run, memory pressure", **MARK, zorder=4)
    i = max(range(len(y_ac)), key=lambda j: y_ac[j])
    ax.annotate(f"{y_ac[i]:.2f}×", (ks_ac[i], y_ac[i]), textcoords="offset points", xytext=(0, 9), ha="center", color=INK, fontsize=9.5)
    j = max(range(len(y_bat)), key=lambda m: y_bat[m])
    ax.annotate(f"{y_bat[j]:.2f}×", (ks_bat[j], y_bat[j]), textcoords="offset points", xytext=(0, -15), ha="center", color=INK, fontsize=9.5)
    ax.set_xlabel("draft depth, k")
    ax.set_ylabel("speedup over plain decoding")
    ax.set_xlim(0.6, 6.9)
    ax.set_ylim(0.7, 2.0)
    ax.set_xticks(ks_ac)
    ax.set_title("Same models, same prompts, same code: the machine's state set the speedup")
    ax.legend(loc="lower left")
    fig.text(0.01, -0.02, "Qwen2.5-7B target, 0.5B draft, 30 prompts, greedy. Acceptance identical to 3 decimals on both nights. results/02, 06, 07",
             color=MUTED, fontsize=8)
    fig.savefig(OUT / "speedup_two_nights.png")
    plt.close(fig)


def fig_acceptance_by_domain():
    s = _load("02_draft_comparison.json")["sessions"]
    drafts = [("0.5B", BLUE), ("1.5B", ORANGE), ("3B", AQUA)]
    domains = sorted(s["0.5B"]["by_domain"], key=lambda d: -s["0.5B"]["by_domain"][d]["pooled_p"])

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    width = 0.14
    for i, (name, color) in enumerate(drafts):
        xs = [j + (i - 1) * (width + 0.03) for j in range(len(domains))]
        ps = [s[name]["by_domain"][d]["pooled_p"] for d in domains]
        lo = [p - s[name]["by_domain"][d]["pooled_p_ci"][0] for p, d in zip(ps, domains)]
        hi = [s[name]["by_domain"][d]["pooled_p_ci"][1] - p for p, d in zip(ps, domains)]
        ax.bar(xs, ps, width=width, color=color, label=f"{name} draft", zorder=3)
        ax.errorbar(xs, ps, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=4)
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("acceptance per draft position, p")
    ax.set_title("Acceptance depends on the task more than on the draft: 0.53 to 0.89 across domains")
    ax.legend(loc="upper right", ncol=3)
    fig.text(0.01, -0.02, "Qwen2.5-7B target, 30 prompts, greedy, depths 1-4 pooled. Whiskers: 95% bootstrap CI over prompts. results/02_draft_comparison.json",
             color=MUTED, fontsize=8)
    fig.savefig(OUT / "acceptance_by_domain.png")
    plt.close(fig)


def fig_batch_scaling():
    d = _load("05_batch_scaling.json")
    bs = sorted(int(b) for b in d["per_batch"])
    pass_ms = [d["per_batch"][str(b)]["target_pass_ms"] for b in bs]
    c_verify = [d["per_batch"][str(b)]["c_verify"] for b in bs]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(bs))
        ax.set_xticklabels([str(b) for b in bs])
        ax.set_xlabel("batch size, rows per decode step")
    a1.plot(bs, pass_ms, color=BLUE, **MARK, zorder=3)
    a1.set_ylim(0, 180)
    a1.set_ylabel("one-token pass, ms")
    a1.set_title("Flat to batch 4, then compute-bound")
    a1.annotate(f"{pass_ms[-1]:.0f} ms", (bs[-1], pass_ms[-1]), textcoords="offset points", xytext=(-6, 6), ha="right", color=INK, fontsize=9)
    a2.plot(bs, c_verify, color=BLUE, **MARK, zorder=3)
    a2.set_ylim(0, 0.7)
    a2.set_ylabel("extra cost per verified position,\nin one-token passes")
    a2.set_title("Verification stops being free")
    a2.annotate(f"{c_verify[0]:.2f}", (bs[0], c_verify[0]), textcoords="offset points", xytext=(8, -4), color=INK, fontsize=9)
    a2.annotate(f"{c_verify[-1]:.2f}", (bs[-1], c_verify[-1]), textcoords="offset points", xytext=(-6, 6), ha="right", color=INK, fontsize=9)
    fig.text(0.01, -0.03, "Qwen2.5-7B-Instruct-4bit on an M4, plugged in, idle. Linearised slope over 1-5 positions per row. results/05_batch_scaling.json",
             color=MUTED, fontsize=8)
    fig.tight_layout(w_pad=3)
    fig.savefig(OUT / "batch_scaling.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_pass_cost_staircase()
    fig_speedup_two_nights()
    fig_acceptance_by_domain()
    fig_batch_scaling()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
