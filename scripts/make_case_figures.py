"""Illustration figures from image data: the pipeline on one series, and one failure.

Unlike ``make_figures.py`` these read CT volumes and per-case outputs, so they
need the downloaded cohort and the centerline stage. Both cases are chosen by a
stated rule, not by eye:

Figure 2 -- pipeline output
    Among series with an HQColon reference, the one whose centerline used the
    most bridges (ties broken by whole-colon coverage). Shows the gas cast with
    the centerline, one axial station with the fat ring, the VGP unfold (from
    ``make_vgp.py``, run first), the ring fat map and the fat attenuation
    profile, the last three on one arc-length axis.

Figure 4 -- hole-fill failure
    Among the reference series that the three-dimensional fill loses (Dice
    < 0.5 in ``eval_hqcolon_fill_bbox.csv``), the one retaining the least gas.
    Shows the same axial slice under the three-dimensional and the slice-wise
    fill, and the gas each retains. The slice is the one losing the most gas
    inside the lumen the adaptive pipeline traced. The slice-wise total also
    includes gas that is not colon (lung bases, small bowel), which is why it is
    only a fall-back.

Images: TCIA CT COLONOGRAPHY (CC BY 3.0). No HQColon mask is drawn; only its
summary numbers from the committed tables are quoted.

    python scripts/make_case_figures.py --out results/figures
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_figures as mf  # noqa: E402  (shared palette, sizes, rcParams, save)

from ctc_core.centerline import CenterlineConfig  # noqa: E402
from ctc_core.fatmap import (  # noqa: E402
    FatConfig,
    fat_map_polar,
    resample_centerline,
    smooth_centerline,
)
from ctc_core.masks import build_body_air_masks  # noqa: E402

plt = mf.plt
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

# Sequential single-hue ramp for magnitudes (light -> dark blue).
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fb", "#9cc3ec", mf.BLUE, "#0d3a73"])
CT_WINDOW = (-300.0, 200.0)  # wide enough to separate air, fat and soft tissue


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_case(pid: str, role: str, work: Path, cl_root: Path):
    ct = sitk.ReadImage(str(work / pid / role / "ct_lps_iso.nii.gz"), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(ct)
    lumen_img = sitk.ReadImage(str(cl_root / pid / role / "lumen_mask.nii.gz"))
    lumen = sitk.GetArrayFromImage(lumen_img) > 0
    xyz, ijk = [], []
    for r in read_csv(cl_root / pid / role / "centerline_points.csv"):
        xyz.append((float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])))
        ijk.append((int(r["i"]), int(r["j"]), int(r["k"])))
    return ct, arr, lumen, np.array(xyz), np.array(ijk)


def masks(ct: sitk.Image, fill: str):
    cfg = CenterlineConfig.corrected()
    return build_body_air_masks(
        ct, body_threshold=cfg.body_threshold, air_threshold=cfg.air_threshold,
        body_closing_radius=cfg.body_closing_radius, air_closing_radius=cfg.air_closing_radius,
        fill_holes=fill, connectivity=cfg.connectivity,
    )


def panel_label(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=mf.INK, fontsize=8.5)


def ct_gray(ax, img: np.ndarray, extent=None) -> None:
    ax.imshow(img, cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], origin="lower",
              extent=extent, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------


def choose_pipeline_case(tables: Path) -> tuple[str, str]:
    cl = {(r["PatientID"], r["role"]): r for r in read_csv(tables / "centerline_auto.csv")}
    ev = read_csv(tables / "eval_hqcolon_auto.csv")
    cands = [(int(cl[(r["PatientID"], r["role"])]["n_bridges"]),
              float(r["colon_within_30mm_frac"]),
              r["PatientID"], r["role"]) for r in ev if (r["PatientID"], r["role"]) in cl]
    n, cov, pid, role = max(cands)
    return pid, role


def figure_2(tables: Path, work: Path, cl_root: Path, out: Path, vgp_root: Path) -> None:
    from scipy import ndimage

    pid, role = choose_pipeline_case(tables)
    ct, arr, lumen, xyz, ijk = load_case(pid, role, work, cl_root)
    sp = np.array(ct.GetSpacing())  # x, y, z
    body, air, _ = masks(ct, "auto")
    cfg = FatConfig()
    # The ring map exactly as run_indices computes it: smoothed centerline.
    s, pts = resample_centerline(smooth_centerline(xyz, cfg.frame_smooth_sigma_mm), cfg.step_mm)
    fmap = fat_map_polar(ct, arr, body, air, s, pts, cfg)
    ring_n = fmap["ring_samples"].sum(axis=1).astype(float)
    fat_n = fmap["fat_samples"].sum(axis=1).astype(float)
    valid = fat_n >= cfg.min_ring_fat_samples
    ring_hu = np.where(valid, fmap["fat_hu_sum"].sum(axis=1) / np.maximum(fat_n, 1), np.nan)

    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg)))
    k, j, i = ijk[:, 2], ijk[:, 1], ijk[:, 0]
    outside_gas = arr[k, j, i] > -700.0

    vgp_path = vgp_root / pid / role / "unfold_vgp_gpu_cube_rgb.npy"
    vgp = np.load(vgp_path) if vgp_path.exists() else None
    vgp_s = (np.load(vgp_root / pid / role / "unfold_vgp_gpu_cube_s_mm.npy")
             if vgp is not None else None)

    fig = plt.figure(figsize=(mf.FULL_W, 196 * mf.MM))
    gs = fig.add_gridspec(4, 2, height_ratios=[2.3, 1.15, 1.0, 0.8], width_ratios=[1.05, 1],
                          hspace=0.6, wspace=0.12)

    # (a) coronal gas cast: lumen thickness along y, patient's right on the left.
    ax_a = fig.add_subplot(gs[0, 0])
    thick = lumen.sum(axis=1).astype(float) * sp[1]  # (z, x), mm
    thick[thick == 0] = np.nan
    ax_a.set_facecolor("white")
    ax_a.imshow(thick, cmap=ListedColormap(plt.cm.Greys(np.linspace(0.18, 0.55, 64))),
                origin="lower", interpolation="nearest")
    pts2 = np.column_stack([i, k]).astype(float)
    lines = np.stack([pts2[:-1], pts2[1:]], axis=1)
    lc = LineCollection(lines, cmap=SEQ, linewidths=1.6)
    lc.set_array(arc[:-1])
    ax_a.add_collection(lc)
    ax_a.scatter(i[outside_gas], k[outside_gas], s=3, color=mf.ORANGE, zorder=4, linewidths=0)
    ax_a.scatter([i[0], i[-1]], [k[0], k[-1]], s=22, facecolor="white", edgecolor=mf.INK,
                 linewidths=0.8, zorder=5)
    rows = np.where(np.isfinite(thick).any(axis=1))[0]
    cols = np.where(np.isfinite(thick).any(axis=0))[0]
    ax_a.set_xlim(cols.min() - 8, cols.max() + 8)
    ax_a.set_ylim(rows.min() - 8, rows.max() + 8)
    ax_a.set_aspect("equal")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for s_ in ax_a.spines.values():
        s_.set_visible(False)
    ax_a.text(0.01, 0.01, "R", transform=ax_a.transAxes, fontsize=7, color=mf.INK2)
    ax_a.text(0.97, 0.01, "L", transform=ax_a.transAxes, fontsize=7, color=mf.INK2)
    panel_label(ax_a, "a  Gas cast and centerline")
    ax_a.text(0.99, 0.99, f"traced {arc[-1]:.0f} mm", transform=ax_a.transAxes, ha="right",
              va="top", fontsize=6.5, color=mf.INK)
    cb = fig.colorbar(lc, ax=ax_a, fraction=0.035, pad=0.01)
    cb.set_label("arc length (mm)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    # (b) axial slice at the middle station: the ring 5-15 mm beyond the wall,
    # drawn in-plane (the measurement samples it along rays across the tangent).
    st = int(np.argmin(np.abs(s - s[-1] / 2)))
    cx, cy, cz = ct.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in pts[st]))
    kz = int(round(cz))
    half = int(round(45 / sp[0]))
    y0, y1 = int(cy) - half, int(cy) + half
    x0, x1 = int(cx) - half, int(cx) + half
    sl = arr[kz, y0:y1, x0:x1]
    lum_sl = lumen[kz, y0:y1, x0:x1]
    ax_b = fig.add_subplot(gs[0, 1])
    ct_gray(ax_b, sl[::-1])  # anterior at the top
    d_out = ndimage.distance_transform_edt(~lum_sl, sampling=(sp[1], sp[0]))
    ring = (d_out >= cfg.ring_inner_mm) & (d_out <= cfg.ring_outer_mm) & \
        body[kz, y0:y1, x0:x1] & ~air[kz, y0:y1, x0:x1]
    fat = ring & (sl >= cfg.fat_hu_min) & (sl <= cfg.fat_hu_max)
    over = np.zeros(sl.shape + (4,))
    over[fat] = mpl_rgba(mf.AQUA, 0.75)
    ax_b.imshow(over[::-1], origin="lower", interpolation="nearest")
    ax_b.contour(lum_sl[::-1].astype(float), levels=[0.5], colors=[mf.BLUE],
                 linewidths=0.9, origin="lower")
    for level in (cfg.ring_inner_mm, cfg.ring_outer_mm):
        ax_b.contour(d_out[::-1], levels=[level], colors=[mf.ORANGE], linewidths=0.8,
                     linestyles="--", origin="lower")
    ax_b.plot([cx - x0], [(y1 - y0) - 1 - (cy - y0)], marker="+", color="white", ms=7, mew=1.2)
    panel_label(ax_b, f"b  Station at s = {s[st]:.0f} mm")
    if valid[st]:
        ax_b.text(0.02, 0.02, f"ring fat {ring_hu[st]:.0f} HU, fraction "
                  f"{fat_n[st] / ring_n[st]:.2f}", transform=ax_b.transAxes, color="white",
                  fontsize=6.5)
    ax_b.text(0.02, 0.92, "A", transform=ax_b.transAxes, color="white", fontsize=7)

    # (c) VGP unfold, rendered at 0.5 mm (make_vgp.py).
    ax_c = fig.add_subplot(gs[1, :])
    if vgp is not None:
        ax_c.imshow(vgp[::-1], aspect="auto", origin="lower",
                    extent=[float(vgp_s[0]), float(vgp_s[-1]), 0, 360], interpolation="lanczos")
    ax_c.axvline(s[st], color=mf.ORANGE, linewidth=0.9)
    ax_c.set_yticks([0, 90, 180, 270, 360])
    ax_c.set_ylabel("angle from\nanterior (deg)")
    ax_c.grid(False)
    plt.setp(ax_c.get_xticklabels(), visible=False)  # shared axis: hide, don't clear
    panel_label(ax_c, "c  Virtual gross pathology unfold")

    # (d) (s, theta) fat fraction map.
    ax_d = fig.add_subplot(gs[2, :], sharex=ax_c)
    ff = np.ma.masked_invalid(fmap["fat_fraction"].T)
    cmap = SEQ.copy()
    cmap.set_bad("#e6e5e1")
    im = ax_d.imshow(ff, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower",
                     extent=[s[0], s[-1], 0, 360], interpolation="nearest")
    ax_d.axvline(s[st], color=mf.ORANGE, linewidth=0.9)
    ax_d.set_yticks([0, 90, 180, 270, 360])
    ax_d.set_ylabel("angle from\nanterior (deg)")
    ax_d.grid(False)
    panel_label(ax_d, "d  Fat fraction, 5-15 mm beyond the wall")
    cb2 = fig.colorbar(im, ax=ax_d, fraction=0.02, pad=0.01)
    cb2.set_label("fat fraction", fontsize=6.5)
    cb2.ax.tick_params(labelsize=6)
    plt.setp(ax_d.get_xticklabels(), visible=False)

    # (e) ring fat attenuation per station.
    ax_e = fig.add_subplot(gs[3, :], sharex=ax_c)
    ax_e.plot(s, ring_hu, color=mf.BLUE, linewidth=1.0)
    med = float(np.nanmedian(ring_hu))
    ax_e.axhline(med, color=mf.INK2, linewidth=0.7, linestyle=(0, (3, 2)))
    ax_e.text(0.99, 0.95, f"dashed: median {med:.1f} HU", transform=ax_e.transAxes,
              fontsize=6.3, va="top", ha="right", color=mf.INK)
    ax_e.axvline(s[st], color=mf.ORANGE, linewidth=0.9)
    ax_e.set_xlabel("arc length s along the centerline (mm)")
    ax_e.set_ylabel("fat (HU)")
    panel_label(ax_e, "e  Mean fat attenuation in the ring, per station")
    ax_e.set_xlim(s[0], s[-1])
    # The colorbar of (d) narrows its axes; match (c) and (e) to it.
    fig.canvas.draw()
    pd_ = ax_d.get_position()
    for ax in (ax_c, ax_e):
        pos = ax.get_position()
        ax.set_position([pd_.x0, pos.y0, pd_.width, pos.height])
    mf.save(fig, out, "figure_2")
    print(f"    case {pid[-4:]}-{1 if role == 'primary' else 2}: {len(xyz)} points, "
          f"{int(outside_gas.sum())} outside the gas mask, station {st} of {len(s)}, "
          f"ring-valid stations {int(valid.sum())}/{len(s)}, "
          f"VGP {'yes' if vgp is not None else 'MISSING'}")


def mpl_rgba(hex_colour: str, alpha: float) -> tuple[float, float, float, float]:
    from matplotlib.colors import to_rgba
    r, g, b, _ = to_rgba(hex_colour)
    return (r, g, b, alpha)


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------


def choose_failure_case(tables: Path) -> tuple[str, str, float, float]:
    rows = [r for r in read_csv(tables / "eval_hqcolon_fill_bbox.csv")
            if float(r["dice_lumen_vs_gas"]) < 0.5]
    r = min(rows, key=lambda r: float(r["our_air_ml"]))
    return r["PatientID"], r["role"], float(r["our_air_ml"]), float(r["ref_gas_ml"])


def figure_4(tables: Path, work: Path, cl_root: Path, out: Path) -> None:
    pid, role, air3d_ml_table, ref_ml = choose_failure_case(tables)
    ct = sitk.ReadImage(str(work / pid / role / "ct_lps_iso.nii.gz"), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(ct)
    vox_ml = float(np.prod(ct.GetSpacing())) / 1000.0
    body3, air3, _ = masks(ct, "bbox")
    body2, air2, _ = masks(ct, "slicewise")
    lost = air2 & ~air3
    # Choose the slice where the most *colonic* gas is lost: the lumen the
    # adaptive pipeline traced stands in for the colon, so that lung bases,
    # which the slice-wise fill also admits, cannot be picked.
    lumen = sitk.GetArrayFromImage(sitk.ReadImage(str(cl_root / pid / role /
                                                      "lumen_mask.nii.gz"))) > 0
    kz = int(np.argmax((lost & lumen).sum(axis=(1, 2))))

    fig, axes = plt.subplots(2, 2, figsize=(mf.FULL_W, 150 * mf.MM),
                             gridspec_kw={"height_ratios": [1, 1.05]})
    other_rgba = mpl_rgba("#8f8d87", 0.55)  # gas in the mask that is not the traced colon
    for ax, body, air, title in ((axes[0, 0], body3, air3, "a  Three-dimensional fill"),
                                 (axes[0, 1], body2, air2, "b  Slice-by-slice fill")):
        ct_gray(ax, arr[kz][::-1])
        over = np.zeros(arr[kz].shape + (4,))
        over[air[kz] & ~lumen[kz]] = other_rgba
        over[air[kz] & lumen[kz]] = mpl_rgba(mf.AQUA, 0.85)
        if ax is axes[0, 0]:
            over[lost[kz] & lumen[kz]] = mpl_rgba(mf.ORANGE, 0.9)
        ax.imshow(over[::-1], origin="lower", interpolation="nearest")
        ax.contour(body[kz][::-1].astype(float), levels=[0.5], colors=[mf.BLUE],
                   linewidths=0.8, origin="lower")
        panel_label(ax, title)
        ax.text(0.02, 0.93, "A", transform=ax.transAxes, color="white", fontsize=7)

    grey = ListedColormap(plt.cm.Greys(np.linspace(0.12, 0.35, 64)))
    dark = ListedColormap(plt.cm.Greys(np.linspace(0.45, 0.8, 64)))
    for ax, air, title in ((axes[1, 0], air3, "c  Gas retained, three-dimensional fill"),
                           (axes[1, 1], air2, "d  Gas retained, slice-by-slice fill")):
        for part, cmap in ((air & ~lumen, grey), (air & lumen, dark)):
            proj = part.sum(axis=1).astype(float)
            proj[proj == 0] = np.nan
            ax.imshow(proj, cmap=cmap, origin="lower", interpolation="nearest")
        ax.axhline(kz, color=mf.ORANGE, linewidth=0.7, linestyle=(0, (3, 2)))
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_aspect("equal")
        ax.set_xlim(0, air.shape[2])
        ax.set_ylim(0, air.shape[0])
        colon_ml = float((air & lumen).sum()) * vox_ml
        other_ml = float((air & ~lumen).sum()) * vox_ml
        panel_label(ax, title)
        ax.text(0.02, 0.02, f"traced colon {colon_ml:,.0f} mL; other gas {other_ml:,.0f} mL",
                transform=ax.transAxes, fontsize=6.5, color=mf.INK)
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    mf.save(fig, out, "figure_4")
    print(f"    case {pid[-4:]}-{1 if role == 'primary' else 2}: slice {kz}; retained "
          f"{float(air3.sum()) * vox_ml:.1f} mL (table {air3d_ml_table:.1f}) vs "
          f"{float(air2.sum()) * vox_ml:.1f} mL slice-wise; reference {ref_ml:.0f} mL")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=Path("results/tables"))
    ap.add_argument("--work-dir", type=Path, default=Path("data/work"))
    ap.add_argument("--centerline-root", type=Path, default=Path("data/centerline/corrected"))
    ap.add_argument("--vgp-root", type=Path, default=Path("data/vgp"),
                    help="where make_vgp.py wrote the unfold of the Figure 2 case")
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    figure_2(args.tables, args.work_dir, args.centerline_root, args.out, args.vgp_root)
    figure_4(args.tables, args.work_dir, args.centerline_root, args.out)


if __name__ == "__main__":
    main()
