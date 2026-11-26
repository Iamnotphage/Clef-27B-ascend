#!/usr/bin/env python3
"""Generate the five publication figures for the Clef-27B A3 report.

All numbers come from committed evidence:
  - fig1: docs/optimization.md section 4 ablation table
  - fig2: logs/profile-tier2-20261126/aggregate.json (W1-W4 device unions)
  - fig3: logs/final-eval/od-s2/od-s2-b16-summary.csv and ERQA S2 summaries
  - fig4: logs/profile-tier2-20261126/fig4_data.json (prefix audit + handover)
  - fig5: S1 roofline budget derived in docs/profile-tier2-2026-11-26.md

Run with /Users/chenclay/Documents/code/projects/figures4papers/.venv/bin/python.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
IMGS = ROOT / "imgs"
LOGS = ROOT / "logs/profile-tier2-20261126"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

PUBLICATION_RCPARAMS = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 18,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2.5,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#272727",
    "xtick.color": "#272727",
    "ytick.color": "#272727",
}


def style(ax, fontsize=18):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", labelsize=fontsize - 3, width=2)
    ax.yaxis.label.set_size(fontsize)
    ax.xaxis.label.set_size(fontsize)


def annotate(ax, bars, values, fmt="{:.1f}", fontsize=14, rotation=0):
    for bar, value in zip(bars, values):
        if value is None:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(value),
                ha="center", va="bottom", fontsize=fontsize,
                rotation=rotation, color="#272727")


def save(fig, name):
    IMGS.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        path = IMGS / f"{name}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06)
        print("wrote", path)


def fig1():
    stages = [
        ("Baseline", 217.04, 5.715, 323.48, 3.957, PALETTE["red_strong"]),
        ("+BIC /\nACLNN / NZ", 166.00, 17.110, None, None, PALETTE["neutral"]),
        ("+ConvQKV /\nChunk128", 155.86, 17.475, None, None, PALETTE["green_2"]),
        ("+TQ2 / 48-AIV /\nMM-A2", 154.55, 17.193, 244.38, 8.240, PALETTE["green_3"]),
        ("+ConvQKV\nVarlen", 154.11, 18.039, 244.66, 8.627, PALETTE["blue_secondary"]),
        ("+Tier 2 Cache\n(Final)", 118.54, 21.583, 122.37, 21.439, PALETTE["blue_main"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
    x = np.arange(len(stages))
    width = 0.38

    for ax, metric, ylabel in ((axes[0], 1, "S1 Tail Latency p95 (ms)  ↓"),
                                (axes[1], 2, "S2 Max Throughput (decisions/s)  ↑")):
        od_values = [s[metric] for s in stages]
        od_bars = ax.bar(x - width / 2, od_values, width,
                         color=[s[5] for s in stages], edgecolor="black",
                         linewidth=1.5, label="OpenDecisions", zorder=3)
        er_values = [s[metric + 2] for s in stages]
        xs = [xi + width / 2 for xi, value in zip(x, er_values) if value is not None]
        ys = [value for value in er_values if value is not None]
        er_bars = ax.bar(xs, ys, width, color=[s[5] for s, v in zip(stages, er_values)
                                                if v is not None],
                         edgecolor="black", linewidth=1.5, hatch="//",
                         label="ERQA (multi-image)", zorder=3)
        style(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in stages], fontsize=11)
        ax.set_ylabel(ylabel)
        top = max([v for v in od_values if v] + [v for v in ys if v])
        ax.set_ylim(0, top * 1.30)
        annotate(ax, od_bars, od_values, fontsize=11, rotation=0)
        annotate(ax, er_bars, ys, fontsize=11)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)

    axes[0].text(0.01, 0.97, "OpenDecisions 2,352 · ERQA 400",
                 transform=axes[0].transAxes, fontsize=11, va="top",
                 color="#4D4D4D")
    axes[1].text(0.02, 0.96,
                 "OD: −45.4% latency, 3.78× throughput\n"
                 "ERQA: −62.2% latency, 5.42× throughput",
                 transform=axes[1].transAxes, fontsize=11, ha="left",
                 va="top", color=PALETTE["blue_main"], linespacing=1.35)
    handles = [Patch(facecolor="#FFFFFF", edgecolor="black", label="OpenDecisions"),
               Patch(facecolor="#FFFFFF", edgecolor="black", hatch="//",
                     label="ERQA (multi-image)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=1.6, rect=(0, 0, 1, 0.96))
    save(fig, "fig1_optimization_ablation")
    plt.close(fig)


def fig2():
    path = LOGS / "aggregate.json"
    if not path.exists():
        print("skip fig2: missing", path)
        return
    data = json.loads(path.read_text())
    specs = [
        ("W1_od_cold", "W2_od_hit", "OpenDecisions  ~967 tokens"),
        ("W3_erqa_cold", "W4_erqa_hit", "ERQA 4-image  ~1,260 tokens"),
    ]
    categories = ["ViT Vision", "MatMul (Cube)", "GDN Core (FLA)",
                  "Fused Conv+QKV", "Full-Attn (FIA)", "Norm/Elementwise",
                  "Other", "HCCL Non-Overlap"]
    colors = [PALETTE["violet"], PALETTE["blue_main"], PALETTE["teal"],
              PALETTE["green_3"], PALETTE["red_strong"], PALETTE["green_2"],
              PALETTE["neutral"], PALETTE["red_2"]]

    def split(window):
        rank = data["windows"][window]["ranks"][0]
        c = rank["category_union_ms"]
        n = max(1, data["windows"][window]["requests"])
        values = {
            "ViT Vision": c.get("vit", 0.0) / n,
            "MatMul (Cube)": c.get("matmul", 0.0) / n,
            "GDN Core (FLA)": c.get("gdn", 0.0) / n,
            "Fused Conv+QKV": 0.0,
            "Full-Attn (FIA)": c.get("attention", 0.0) / n,
            "Norm/Elementwise": (c.get("norm", 0.0) + c.get("elementwise", 0.0)) / n,
            "Other": c.get("other", 0.0) / n,
            "HCCL Non-Overlap": rank["communication_nonoverlapped_ms"] / n,
        }
        # Split the fused Conv+QKV kernel out of the GDN bucket.
        for op in rank["top_ops"]:
            if "ClefGdnConvQkv" in op["name"]:
                values["Fused Conv+QKV"] = op["union_ms"] / n
                values["GDN Core (FLA)"] -= op["union_ms"] / n
                break
        return values

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (cold, hit, title) in zip(axes, specs):
        cold_values = split(cold)
        hit_values = split(hit)
        labels = ["Full forward\n(Populate)", "Tier 2 hit\n(Suffix resume)"]
        y = np.arange(len(labels))
        left = np.zeros(len(labels))
        for category, color in zip(categories, colors):
            widths = np.array([cold_values[category], hit_values[category]])
            if widths.max() <= 0:
                continue
            ax.barh(y, widths, left=left, height=0.55, color=color,
                    edgecolor="black", linewidth=1.2, label=category, zorder=3)
            left += widths
        style(ax)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel("Device union time per request (ms) ↓")
        ax.set_title(title, fontsize=15, pad=8)
        for yi, total in zip(y, left):
            ax.text(total + 2, yi, f"{total:.0f} ms", va="center", fontsize=12,
                    color="#272727")
        ax.set_xlim(0, max(left) * 1.22)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.8, zorder=0)

    handles = [Patch(facecolor=color, edgecolor="black", linewidth=1.2,
                     label=category) for category, color in zip(categories, colors)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(pad=1.2, rect=(0, 0.07, 1, 1))
    save(fig, "fig2_operator_breakdown")
    plt.close(fig)


def fig3():
    csv_path = ROOT / "logs/final-eval/od-s2/od-s2-b16-summary.csv"
    rows = []
    for line in csv_path.read_text().strip().splitlines()[1:]:
        parts = line.split(",")
        rows.append({"c": int(parts[0]), "dec_s": float(parts[1]),
                     "p95": float(parts[4])})
    erqa_c8 = [json.loads(p.read_text()) for p in
               sorted((ROOT / "logs/final-eval/erqa-s2").glob("erqa-s2-c8-*-summary.json"))]
    erqa_c16 = [json.loads(p.read_text()) for p in
                sorted((ROOT / "logs/final-eval/erqa-s2").glob("erqa-s2-c16-*-summary.json"))]
    erqa = [{"c": 8, "dec_s": np.mean([r["throughput"]["decisions_per_second"] for r in erqa_c8]),
             "p95": np.mean([r["latency_ms"]["p95"] for r in erqa_c8])},
            {"c": 16, "dec_s": np.mean([r["throughput"]["decisions_per_second"] for r in erqa_c16]),
             "p95": np.mean([r["latency_ms"]["p95"] for r in erqa_c16])}]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    style(ax)
    xs = [r["c"] for r in rows]
    ys = [r["dec_s"] for r in rows]
    ax.plot(xs, ys, "-o", color=PALETTE["blue_main"], linewidth=2.6,
            markersize=7, markeredgecolor="black", markeredgewidth=1.2,
            label="OpenDecisions (Tier 2)", zorder=3)
    ax.plot([r["c"] for r in erqa], [r["dec_s"] for r in erqa], "s--",
            color=PALETTE["teal"], linewidth=2.4, markersize=8,
            markeredgecolor="black", markeredgewidth=1.2,
            label="ERQA (Tier 2)", zorder=3)
    ax.axhline(18.039, color=PALETTE["red_strong"], linestyle=":", linewidth=2.0,
               label="OD pre-Tier-2 max 18.04")
    ax.axhline(8.627, color=PALETTE["red_2"], linestyle=":", linewidth=2.0,
               label="ERQA pre-Tier-2 warm 8.63")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.set_xlabel("Client concurrency C")
    ax.set_ylabel("Throughput (decisions/s) ↑")
    ax.set_ylim(0, 24)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(color="#E6E6E6", linewidth=0.8, zorder=0)

    ax = axes[1]
    style(ax)
    ax.plot([r["p95"] for r in rows], [r["dec_s"] for r in rows], "-o",
            color=PALETTE["blue_main"], linewidth=2.6, markersize=7,
            markeredgecolor="black", markeredgewidth=1.2, zorder=3,
            label="OpenDecisions")
    ax.plot([r["p95"] for r in erqa], [r["dec_s"] for r in erqa], "s--",
            color=PALETTE["teal"], linewidth=2.4, markersize=8,
            markeredgecolor="black", markeredgewidth=1.2, zorder=3, label="ERQA")
    for target, text in ((32, "OD C32\n21.54 @ 2.07 s"),
                         (64, "OD C64\n21.58 @ 3.97 s")):
        row = next(r for r in rows if r["c"] == target)
        ax.annotate(text, (row["p95"], row["dec_s"]),
                    textcoords="offset points", xytext=(-8, 16),
                    fontsize=10, ha="right", color=PALETTE["blue_main"])
    row = next(r for r in erqa if r["c"] == 16)
    ax.annotate("ERQA C16\n21.44 @ 0.82 s", xy=(row["p95"], row["dec_s"]),
                xytext=(1250, 14.8), fontsize=10, ha="left",
                color=PALETTE["teal"],
                arrowprops=dict(arrowstyle="-", color=PALETTE["teal"], lw=1.2))
    ax.set_xlabel("p95 latency (ms) ↓")
    ax.set_ylabel("Throughput (decisions/s) ↑")
    ax.set_xlim(0, 4400)
    ax.set_ylim(0, 28)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(color="#E6E6E6", linewidth=0.8, zorder=0)
    fig.tight_layout(pad=1.6)
    save(fig, "fig3_s2_concurrency_pareto")
    plt.close(fig)


def fig4():
    path = LOGS / "fig4_data.json"
    if not path.exists():
        print("skip fig4: missing", path)
        return
    data = json.loads(path.read_text())
    groups = data["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax = axes[0]
    style(ax)
    x = np.arange(len(groups))
    prefix = np.array([g["L"] for g in groups])
    suffix = np.array([g["S"] for g in groups])
    ax.bar(x, prefix, 0.58, color=PALETTE["blue_main"], edgecolor="black",
           linewidth=1.5, label="Cached prefix L (128-aligned)", zorder=3)
    ax.bar(x, suffix, 0.58, bottom=prefix, color=PALETTE["green_3"],
           edgecolor="black", linewidth=1.5, hatch="//",
           label="Computed suffix S = T − L", zorder=3)
    for xi, g in zip(x, groups):
        ax.text(xi, g["T"] + 35, f"T={g['T']}", ha="center", fontsize=10)
        ax.text(xi, g["L"] / 2, f"{g['L']}", ha="center", va="center",
                fontsize=10, color="white")
        ax.text(xi, g["L"] + g["S"] / 2, f"{g['S']}", ha="center", va="center",
                fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([g["label"] for g in groups], fontsize=10)
    ax.set_ylabel("Input tokens")
    ax.set_ylim(0, max(g["T"] for g in groups) * 1.18)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)

    ax = axes[1]
    style(ax)
    handover = data["handover"]
    labels = ["window=3000\n(stale pool)", "window=500\nR1", "window=500\nR2"]
    values = [handover["old"], handover["new_r1"], handover["new_r2"]]
    colors = [PALETTE["red_strong"], PALETTE["blue_secondary"], PALETTE["blue_main"]]
    bars = ax.bar(np.arange(3), values, 0.55, color=colors, edgecolor="black",
                  linewidth=1.5, zorder=3)
    annotate(ax, bars, values, fmt="{:.2f}", fontsize=13)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("ERQA C16 throughput (decisions/s) ↑")
    ax.set_ylim(0, 28)
    ax.text(0, 7.3, "399 evictions\nfinal misses > 0", ha="center", va="top",
            fontsize=9.5, color="white")
    ax.text(1.5, 24.3, "294 handover evictions\n0 final misses", ha="center",
            va="bottom", fontsize=9.5, color=PALETTE["blue_main"])
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)
    fig.tight_layout(pad=1.6)
    save(fig, "fig4_prefix_handover_analysis")
    plt.close(fig)


def fig5():
    data = json.loads((LOGS / "roofline.json").read_text())
    full = data["full"]
    hit = data["hit"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, budget, title in (
            (axes[0], full, "All 967 tokens compute"),
            (axes[1], hit, "Tier 2 hit (199-token suffix)")):
        style(ax)
        segments = [("Cube FLOPs / peak", budget["cube"], PALETTE["blue_main"]),
                    ("Non-Cube (measured)", budget["noncube"], PALETTE["teal"]),
                    ("HCCL non-overlap", budget["hccl"], PALETTE["red_2"])]
        left = 0.0
        y = [0]
        for label, width, color in segments:
            ax.barh(y, width, left=left, height=0.42, color=color,
                    edgecolor="black", linewidth=1.4, label=label, zorder=3)
            left += width
        device = left
        ax.axvline(budget["host_floor"], color=PALETTE["green_3"], linestyle="--",
                   linewidth=2.6, label="Host dispatch floor", zorder=2)
        ax.axvline(budget["measured_p95"], color="#272727", linestyle=":",
                   linewidth=2.4, label="Measured S1 p95", zorder=2)
        ax.set_yticks([0])
        ax.set_yticklabels(["Device budget"], fontsize=12)
        ax.set_xlabel("Latency (ms) ↓")
        ax.set_title(title, fontsize=15, pad=8)
        ax.set_xlim(0, 185)
        ax.text(device - 3, 0, f"{device:.0f} ms", va="center", ha="right",
                fontsize=12, color="#272727")
        ax.text(budget["host_floor"] + 2, 0.30, f"host {budget['host_floor']:.0f} ms",
                fontsize=10, color=PALETTE["blue_main"], va="bottom")
        ax.text(budget["measured_p95"] + 2, -0.34,
                f"measured {budget['measured_p95']:.1f} ms", fontsize=10,
                color="#272727", va="top")
        ax.set_ylim(-0.55, 0.55)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.8, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=1.2, rect=(0, 0.06, 1, 1))
    save(fig, "fig5_s1_roofline")
    plt.close(fig)


def main():
    plt.rcParams.update(PUBLICATION_RCPARAMS)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()


if __name__ == "__main__":
    main()
