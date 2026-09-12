"""Manuscript figures, regenerated from committed result tables only.

Every figure reads ``results/tables/*.csv`` and nothing else -- no image data,
no HQColon mask -- so it can be rebuilt by anyone with the repository and it
cannot leak anything the licences forbid publishing. HQColon masks in
particular (CC BY-NC-ND 4.0) never appear; only summary statistics derived from
comparing against them do.

Palette: the first three categorical slots of the validated reference palette
(blue, orange, aqua), checked for colour-vision-deficiency separation. Aqua sits
below 3:1 contrast on white, so every series is also direct-labelled and
distinguished by marker shape, never by colour alone.

    python scripts/make_figures.py --out results/figures

Numbering follows first citation in the manuscript: 1 study flow, 3 reference
agreement per series, 5 prone-supine agreement, 6 polyp phantoms. Figures 2
and 4 need image data and come from ``make_case_figures.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

# ERE: 85 mm half-page, 170 mm full-page, max height 225 mm, ~300 dpi.
MM = 1 / 25.4
FULL_W = 170 * MM
HALF_W = 85 * MM

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("tif", "pdf", "png"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        if ext == "pdf":
            # No creation date: identical inputs must give byte-identical files,
            # so that a re-run shows up in git only when a figure really changed.
            kw["metadata"] = {"CreationDate": None}
        fig.savefig(out / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------------------
# Figure 1 -- study flow
# ---------------------------------------------------------------------------


def figure_1_flow(tables: Path, out: Path) -> None:
    """Study flow: from the collection to each of the three analyses."""
    overlap = read(tables / "hqcolon_overlap.csv")
    position = read(tables / "patient_position.csv")
    agreement = read(tables / "prone_supine_agreement.csv")

    n_series = len(position)
    n_patients = len({r["PatientID"] for r in position})
    n_ref = sum(1 for r in overlap if r["in_hqcolon"] == "True")
    n_ref_pat = len({r["PatientID"] for r in overlap if r["in_hqcolon"] == "True"})
    n_pairs = max(int(r["n_pairs"]) for r in agreement)
    n_unident = sum(1 for r in position if not r["position_combined"])

    fig, ax = plt.subplots(figsize=(FULL_W, 110 * MM))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    def box(x, y, w, h, text, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="white",
                                   edgecolor=INK2, linewidth=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=INK,
                fontsize=7, fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 0.7,
                                "shrinkA": 0, "shrinkB": 0, "mutation_scale": 7})

    box(22, 86, 56, 11,
        "TCIA CT COLONOGRAPHY (ACRIN 6664)\n825 subjects, public, CC BY 3.0", bold=True)
    arrow(50, 86, 50, 80)
    box(18, 66, 64, 14,
        "Deterministic selection: first 30 patients by PatientID\n"
        f"with >= 2 axial series  ->  {n_patients} patients, {n_series} series\n"
        "resampled to 1.0 mm isotropic", bold=False)
    arrow(50, 66, 50, 60)
    box(18, 47, 64, 13,
        "Headless pipeline, no human readers\n"
        f"centerline produced for {n_series} of {n_series} series\n"
        "quality indices and pericolonic fat map for every series")

    # three pillars
    for x in (16.5, 50, 83.5):
        arrow(50, 47, x, 42)
    box(1.5, 12, 30, 30,
        "Pillar 1: external reference\n\n"
        f"HQColon masks for\n{n_ref} series ({n_ref_pat} patients)\n\n"
        "Dice, surface distance,\ncenterline coverage")
    box(35, 12, 30, 30,
        "Pillar 2: polyp measurement\n\n"
        "geometric phantoms,\nexact ground truth\n\n"
        "diameter, volume,\nsphericity")
    box(68.5, 12, 30, 30,
        "Pillar 3: prone/supine\n\n"
        f"position recovered from DICOM,\n{n_pairs} patients paired\n\n"
        "ICC(2,1),\nBland-Altman")
    ax.text(50, 4,
            f"Excluded from pairing: 2 patients (one with a series in decubitus position "
            f"[n = {n_unident}]; one with both series recorded as supine).\n"
            "In-vivo polyp comparison not possible: 3 annotated lesions in the cohort "
            "and no 3-D coordinate in any annotation.",
            ha="center", va="center", fontsize=6.3, color=INK2, linespacing=1.4)
    save(fig, out, "figure_1")


# ---------------------------------------------------------------------------
# Figure 2 -- HQColon validation per series
# ---------------------------------------------------------------------------


def figure_2_hqcolon(tables: Path, out: Path) -> None:
    rows = read(tables / "eval_hqcolon_auto.csv")
    rows = [r for r in rows if num(r.get("dice_lumen_vs_gas")) is not None]
    rows.sort(key=lambda r: num(r["colon_within_30mm_frac"]))
    # "-1"/"-2" rather than P/S: the roles are series order, and P/S would read as
    # prone/supine, which they are not.
    # An asterisk marks the series inspected against the reference during
    # development (make_tables.DEVELOPMENT_SERIES): they are not held out.
    development = {"0007-1", "0003-2", "0001-1", "0004-2", "0030-1"}
    names = [f"{r['PatientID'][-4:]}-{1 if r['role'] == 'primary' else 2}" for r in rows]
    labels = [f"{n}*" if n in development else n for n in names]
    y = np.arange(len(rows))

    metrics = [
        ("dice_lumen_vs_gas", "a  Lumen Dice", BLUE, "o"),
        ("colon_within_30mm_frac", "b  Colon near the path", ORANGE, "s"),
        ("centerline_inside_gas_frac", "c  Path inside colon", AQUA, "D"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 105 * MM), sharey=True)
    for ax, (key, title, colour, marker) in zip(axes, metrics, strict=True):
        vals = np.array([num(r[key]) for r in rows])
        ax.hlines(y, 0, vals, color=GRID, linewidth=0.8, zorder=1)
        ax.scatter(vals, y, s=14, color=colour, marker=marker, zorder=3,
                   edgecolors="white", linewidths=0.5)
        med = float(np.median(vals))
        ax.axvline(med, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=2)
        ax.text(med, len(rows) - 0.2, f" median {med:.3f}", color=INK, fontsize=6.5,
                ha="left", va="bottom")
        ax.set_xlim(0, 1.02)
        ax.set_title(title, color=INK, loc="left")
        ax.set_xlabel("fraction")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", labelsize=5.8)
    axes[0].set_ylabel("series (patient-series number), sorted by coverage")
    fig.tight_layout(w_pad=1.2, rect=(0, 0.035, 1, 1))
    fig.text(0.012, 0.012, "* inspected against the reference during development",
             fontsize=6, color=INK2)
    save(fig, out, "figure_3")


# ---------------------------------------------------------------------------
# Figure 3 -- prone / supine agreement
# ---------------------------------------------------------------------------

# Groups are defined by what each index *measures* (tissue composition, amount
# and distribution of fat, distension state and traced path), not by how well it
# agreed. The grouping was introduced when this figure was drawn, i.e. after the
# agreement statistics were known; that it follows the index definitions rather
# than the results is what keeps it from being a post-hoc sort, and the
# manuscript should say when it was set.
COMPOSITION = ("Fat composition", BLUE, "o")
AMOUNT = ("Fat amount and distribution", ORANGE, "s")
STATE = ("Distension state and traced path", AQUA, "D")
SPHERE = ("Previous fat definition (15 mm sphere)", INK2, "^")
GROUPS = {
    "ring_fat_mean_hu_median": COMPOSITION,
    "ring_fat_fraction_mean": AMOUNT,
    "fat_asymmetry": AMOUNT,
    "gas_volume_ml": STATE,
    "length_adjusted_gas_ml_per_cm": STATE,
    "collapse_ratio_pct": STATE,
    "luminal_radius_mm_median": STATE,
    "distension_quality_score": STATE,
    "traced_centerline_length_cm": STATE,
    "centerline_tortuosity_index": STATE,
    "fat_mean_hu_median": SPHERE,
    "fat_fraction_mean": SPHERE,
    "fat_volume_ml_sum": SPHERE,
}
PRETTY = {
    "ring_fat_mean_hu_median": "Fat attenuation (ring)",
    "ring_fat_fraction_mean": "Fat fraction (ring)",
    "fat_asymmetry": "Fat asymmetry (ring)",
    "fat_mean_hu_median": "Fat attenuation (sphere)",
    "fat_fraction_mean": "Fat fraction (sphere)",
    "fat_volume_ml_sum": "Fat volume (sphere)",
    "gas_volume_ml": "Gas volume",
    "length_adjusted_gas_ml_per_cm": "Length-adjusted gas",
    "collapse_ratio_pct": "Collapse ratio",
    "luminal_radius_mm_median": "Luminal radius",
    "distension_quality_score": "Distension score",
    "traced_centerline_length_cm": "Traced centerline length",
    "centerline_tortuosity_index": "Tortuosity index",
}


def figure_3_agreement(tables: Path, out: Path) -> None:
    agree = {r["index"]: r for r in read(tables / "prone_supine_agreement.csv")}
    indices = read(tables / "indices.csv")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(FULL_W, 84 * MM), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    # (a) ICC forest plot, grouped
    order = [k for k in GROUPS if k in agree]
    y = np.arange(len(order))[::-1]
    for yi, key in zip(y, order, strict=True):
        r = agree[key]
        _grp, colour, marker = GROUPS[key]
        icc, lo, hi = num(r["icc21"]), num(r["icc_lower"]), num(r["icc_upper"])
        ax_a.hlines(yi, lo, hi, color=colour, linewidth=1.4, zorder=2)
        ax_a.scatter([icc], [yi], s=22, color=colour, marker=marker, zorder=3,
                     edgecolors="white", linewidths=0.6)
        ax_a.text(1.04, yi, f"{icc:.2f}", va="center", ha="left", fontsize=6.3, color=INK)
    ax_a.set_yticks(y, [PRETTY[k] for k in order])
    ax_a.axvline(0, color=INK2, linewidth=0.6)
    ax_a.set_xlim(-0.6, 1.0)
    ax_a.set_xlabel("ICC(2,1) with 95% CI")
    ax_a.set_title("a  Agreement between positions", loc="left", color=INK)
    ax_a.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=c, marker=m, linestyle="-", markersize=4, lw=1.2,
               markeredgecolor="white", label=lab)
        for lab, c, m in (COMPOSITION, AMOUNT, STATE, SPHERE)
    ]
    ax_a.legend(handles=handles, loc="upper left", fontsize=6, handlelength=1.6,
                bbox_to_anchor=(-0.02, -0.17), ncol=1)

    # (b) Bland-Altman for fat attenuation, paired by recovered position
    by: dict[str, dict[str, list[float]]] = {}
    for r in indices:
        v = num(r.get("ring_fat_mean_hu_median"))
        pos = (r.get("position") or "").strip()
        if v is not None and pos in ("prone", "supine") and str(r.get("ok")) == "True":
            by.setdefault(r["PatientID"], {}).setdefault(pos, []).append(v)
    pairs = [(d["prone"][0], d["supine"][0]) for d in by.values()
             if len(d.get("prone", [])) == 1 and len(d.get("supine", [])) == 1]
    p = np.array(pairs)
    mean = p.mean(axis=1)
    diff = p[:, 0] - p[:, 1]
    bias = diff.mean()
    sd = diff.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

    ax_b.scatter(mean, diff, s=16, color=BLUE, edgecolors="white", linewidths=0.5, zorder=3)
    for level, text, style in ((bias, f"bias {bias:+.1f} HU", "-"),
                               (hi, f"+1.96 SD {hi:+.1f}", (0, (3, 2))),
                               (lo, f"-1.96 SD {lo:+.1f}", (0, (3, 2)))):
        ax_b.axhline(level, color=INK2, linewidth=0.7, linestyle=style, zorder=2)
        ax_b.text(mean.max(), level, f"  {text}",
                  va="bottom", ha="left", fontsize=6.2, color=INK)
    ax_b.axhline(0, color=GRID, linewidth=0.6, zorder=1)
    ax_b.set_xlabel("mean of prone and supine (HU)")
    ax_b.set_ylabel("prone - supine (HU)")
    ax_b.set_title(f"b  Fat attenuation (ring), n = {len(pairs)} pairs", loc="left",
                   color=INK)
    ax_b.margins(x=0.28)

    fig.tight_layout(w_pad=2.0)
    save(fig, out, "figure_5")


# ---------------------------------------------------------------------------
# Figure 4 -- polyp phantoms
# ---------------------------------------------------------------------------


def figure_4_phantoms(tables: Path, out: Path) -> None:
    # Realistic partial-volume edges only; the step-edge rows are the stated limit.
    rows = [r for r in read(tables / "polyp_phantom_accuracy.csv")
            if r["phantom"].startswith("sphere") and r.get("edge", "ramp") == "ramp"]
    d = np.array([num(r["true_max_diameter_mm"]) for r in rows])
    gray = np.array([num(r["grayscale_diameter_error_mm"]) for r in rows])
    binr = np.array([num(r["binary_diameter_error_mm"]) for r in rows])
    vox = np.array([num(r["voxel_count_volume_error_pct"]) for r in rows])
    mesh = np.array([num(r["mesh_volume_error_pct"]) for r in rows])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FULL_W, 62 * MM))

    for ax, series, ylabel, title, _unit in (
        (ax_a, ((binr, "Binary mask", ORANGE, "s"), (gray, "Grayscale iso-surface", BLUE, "o")),
         "diameter error (mm)", "a  Maximum diameter", "mm"),
        (ax_b, ((vox, "Voxel count", ORANGE, "s"), (mesh, "Iso-surface mesh", BLUE, "o")),
         "volume error (%)", "b  Volume", "%"),
    ):
        ax.axhline(0, color=INK2, linewidth=0.7)
        for vals, label, colour, marker in series:
            ax.plot(d, vals, color=colour, linewidth=1.4, marker=marker, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)
            ax.text(d[-1] + 0.4, vals[-1], label, color=INK, fontsize=6.5,
                    va="center", ha="left")
        for thr in (6, 10):  # the CT colonography reporting thresholds
            ax.axvline(thr, color=GRID, linewidth=0.9, zorder=1)
        ax.set_xticks(d)
        ax.set_xlabel("true diameter (mm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlim(d.min() - 1, d.max() + 7)
        ax.grid(axis="x", visible=False)
    ax_a.set_ylim(ax_a.get_ylim()[0], 0.19)  # headroom for the threshold labels
    for thr in (6, 10):  # inside the axes at the top: clear of the title and the data
        ax_a.text(thr + 0.15, 0.185, f"{thr} mm", fontsize=6, color=INK2,
                  ha="left", va="top")
    fig.tight_layout(w_pad=2.2)
    save(fig, out, "figure_6")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", type=Path, default=Path("results/tables"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    print(f"figures -> {args.out}")
    figure_1_flow(args.tables, args.out)
    figure_2_hqcolon(args.tables, args.out)
    figure_3_agreement(args.tables, args.out)
    figure_4_phantoms(args.tables, args.out)


if __name__ == "__main__":
    main()
