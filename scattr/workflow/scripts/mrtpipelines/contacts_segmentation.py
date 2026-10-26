#!/usr/bin/env python
"""
contacts_segmentation.py

Label a sphere around each SEEG contact centroid in a single NIfTI, with no
overlap between contacts, for use as tractography nodes.

Radius
------
Each contact's safe radius is half the distance to its nearest neighbouring
contact, capped at ``--radius``.  With ``--radius-basis electrode`` (default)
the neighbour is searched on the same electrode only, so one mislocalised or
genuinely close contact on another shaft cannot shrink every sphere.  Spheres
of different electrodes that still come closer than 2r are split voxel by
voxel to the nearer contact, and those contacts are reported.  With
``--radius-basis all`` neighbours on any electrode count, so r_i + r_j <= d_ij
for every pair and no sphere is ever truncated.

``--radius-mode global`` (default) uses the smallest of these radii for every
contact, so all nodes have the same volume and streamline counts can be summed
across contacts without node-size bias.  ``per-contact`` keeps each contact's
own radius.  The run stops if any radius falls below ``--min-radius``, since
that usually means a localisation error that should be fixed first.

Grid
----
Spheres are drawn on the grid of ``--ref`` (e.g. the T1w).  With
``--grid-res`` the grid is supersampled to that isotropic voxel size while
keeping the orientation and field of view of ``--ref``.

Contacts
--------
``--contacts`` takes a 3D Slicer ``.fcsv`` (coordinate system read from its
header) or a ``.tsv`` with ``name``/``x``/``y``/``z`` columns such as a BIDS
``*_electrodes.tsv`` (assumed RAS unless ``--coord-system LPS``).  A TSV is
also saved as an FCSV next to it (same name, ``.fcsv``, RAS) unless that file
already exists.

Usage
-----
    python contacts_segmentation.py \
        --contacts sub-P001_space-T1w_electrodes.fcsv \
        --ref  sub-P001_desc-brain_T1w.nii.gz \
        --radius 1.5 \
        --output-seg sub-P001_space-T1w_desc-AllContacts_dseg.nii.gz \
        --output-tsv sub-P001_space-T1w_desc-AllContacts_dseg.tsv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Contact file parsing (FCSV or TSV)
# ---------------------------------------------------------------------------

def parse_fcsv(fcsv_path: str | Path):
    """Parse a 3D Slicer Markups FCSV file.

    Returns ``(names, coords, declared_system)``; *declared_system* is
    ``"RAS"``, ``"LPS"`` or ``None`` if the header does not say.
    """
    declared = None
    columns = None
    rows = []
    with open(fcsv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                key, _, val = line[1:].partition("=")
                key, val = key.strip().lower(), val.strip()
                if key == "coordinatesystem":
                    declared = {"0": "RAS", "1": "LPS"}.get(val, val.upper())
                elif key == "columns":
                    columns = [c.strip() for c in val.split(",")]
                continue
            rows.append(next(csv.reader([line])))

    if columns:
        ix, iy, iz, il = (columns.index(c) for c in ("x", "y", "z", "label"))
    else:
        ix, iy, iz, il = 1, 2, 3, 11

    names = [r[il] for r in rows]
    coords = np.array([[float(r[ix]), float(r[iy]), float(r[iz])] for r in rows])
    return names, coords, declared


def parse_tsv(tsv_path: str | Path):
    """Parse a tab-separated contact table (e.g. BIDS ``*_electrodes.tsv``).

    Needs a header with ``name`` (or ``label``), ``x``, ``y`` and ``z``
    columns, in any order and case.  Rows whose coordinates are ``n/a`` or
    empty are skipped.  Returns ``(names, coords, None)``: a TSV does not
    declare its coordinate system.
    """
    with open(tsv_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        rows = list(reader)

    name_col = cols.get("name", cols.get("label"))
    missing = [c for c in ("x", "y", "z") if c not in cols]
    if name_col is None or missing:
        raise ValueError(
            f"{tsv_path}: needs 'name' (or 'label'), 'x', 'y', 'z' columns; "
            f"found {list(cols)}"
        )

    names, coords, skipped = [], [], []
    for r in rows:
        vals = [(r[cols[k]] or "").strip() for k in ("x", "y", "z")]
        name = (r[name_col] or "").strip()
        if any(v.lower() in ("", "n/a", "nan") for v in vals):
            skipped.append(name)
            continue
        names.append(name)
        coords.append([float(v) for v in vals])
    if skipped:
        print(f"  Skipped {len(skipped)} row(s) without coordinates: {skipped}")
    return names, np.array(coords), None


def load_contacts(path: str | Path, coord_system: str = "auto"):
    """Read contacts from an FCSV or TSV file, returned in RAS.

    The format is chosen from the extension (``.fcsv`` or ``.tsv``).
    Coordinates are converted to RAS (the NIfTI world convention) when the
    file declares LPS or ``--coord-system LPS`` is given.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".fcsv":
        names, coords, declared = parse_fcsv(path)
    elif suffix == ".tsv":
        names, coords, declared = parse_tsv(path)
    else:
        raise ValueError(f"Unsupported contacts file '{path}': use .fcsv or .tsv")

    if len(names) == 0:
        raise ValueError(f"No contacts found in {path}")

    system = declared if coord_system == "auto" else coord_system
    if system is None:
        print(f"  WARNING: {suffix[1:].upper()} declares no coordinate system; "
              f"assuming RAS (use --coord-system LPS if it is LPS)")
        system = "RAS"
    if system == "LPS":
        coords[:, :2] *= -1
    print(f"  Contacts file: {Path(path).name} ({suffix[1:].upper()}), "
          f"coordinate system {system}"
          + (" (converted to RAS)" if system == "LPS" else ""))
    return names, coords


def write_fcsv(fcsv_path: str | Path, names, coords_ras) -> None:
    """Write contacts as a 3D Slicer Markups FCSV (RAS coordinates)."""
    with open(fcsv_path, "w", newline="") as fh:
        fh.write("# Markups fiducial file version = 4.11\n")
        fh.write("# CoordinateSystem = RAS\n")
        fh.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,"
                 "associatedNodeID\n")
        writer = csv.writer(fh, lineterminator="\n")
        for i, (name, (x, y, z)) in enumerate(zip(names, coords_ras), start=1):
            writer.writerow([f"vtkMRMLMarkupsFiducialNode_{i}",
                             f"{x:.4f}", f"{y:.4f}", f"{z:.4f}",
                             0, 0, 0, 1, 1, 1, 0, name, "", ""])


def electrode_of(name: str) -> str:
    """``LAHc-01`` -> ``LAHc`` (split on the last hyphen)."""
    return name.rsplit("-", 1)[0]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def nearest_neighbours(coords: np.ndarray, groups: np.ndarray):
    """Distance to, and index of, each contact's nearest neighbour in its group.

    Contacts alone in their group get ``inf`` / ``-1``.
    """
    d_nn = np.full(len(coords), np.inf)
    nn_idx = np.full(len(coords), -1)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) < 2:
            continue
        d, j = cKDTree(coords[idx]).query(coords[idx], k=2)
        d_nn[idx] = d[:, 1]
        nn_idx[idx] = idx[j[:, 1]]
    return d_nn, nn_idx


def compute_radii(coords, electrodes, r_max, mode, basis):
    """Radius per contact.  Returns ``(radii, d_nn, nn_idx)``."""
    if len(coords) < 2:
        raise ValueError("Need at least two contacts.")
    d_any, _ = cKDTree(coords).query(coords, k=2)
    if d_any[:, 1].min() < 1e-6:
        raise ValueError("Two contacts share the same coordinates.")

    groups = electrodes if basis == "electrode" else np.zeros(len(coords), int)
    d_nn, nn_idx = nearest_neighbours(coords, groups)

    radii = np.minimum(r_max, d_nn / 2)
    if mode == "global":
        radii = np.full(len(coords), radii.min())
    return radii, d_nn, nn_idx


def cross_electrode_clashes(coords, radii, electrodes):
    """Pairs of contacts on different electrodes whose spheres intersect."""
    tree = cKDTree(coords)
    pairs = []
    for i, j in sorted(tree.query_pairs(2 * radii.max())):
        d = np.linalg.norm(coords[i] - coords[j])
        if electrodes[i] != electrodes[j] and d < radii[i] + radii[j]:
            pairs.append((i, j, d))
    return pairs


def make_grid(ref_img: nib.Nifti1Image, res: float | None):
    """Return ``(shape, affine)`` of the label grid.

    With *res*, the reference grid is resampled to isotropic *res* mm voxels,
    keeping its orientation and the outer edge of its field of view.
    """
    shape = np.array(ref_img.shape[:3])
    affine = ref_img.affine.copy()
    if res is None:
        return tuple(int(n) for n in shape), affine

    zooms = np.array(ref_img.header.get_zooms()[:3], dtype=float)
    scale = res / zooms
    new_shape = np.ceil(shape / scale).astype(int)
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] * scale
    # Keep the outer corner of voxel 0 (index -0.5) in the same place
    new_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ (0.5 * (scale - 1))
    return tuple(int(n) for n in new_shape), new_affine


def rasterise_spheres(coords, radii, shape, affine):
    """Paint sphere *i* with label ``i + 1``.

    Only the voxels in each sphere's bounding box are visited.  A voxel that
    lands in two spheres goes to the nearer contact, so labels never overlap.
    """
    shape = np.array(shape)
    inv_affine = np.linalg.inv(affine)
    dtype = np.int16 if len(coords) < np.iinfo(np.int16).max else np.int32
    seg = np.zeros(shape, dtype=dtype)
    corners = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=float)
    contested = 0

    for i, (p, r) in enumerate(zip(coords, radii)):
        vox = apply_affine(inv_affine, p + r * corners)
        lo = np.maximum(np.floor(vox.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(vox.max(0)).astype(int), shape - 1)
        if np.any(lo > hi):
            continue  # sphere entirely outside the image

        ijk = np.stack(
            np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)],
                        indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        world = apply_affine(affine, ijk)
        dist = np.linalg.norm(world - p, axis=1)
        inside = dist <= r
        ijk, world, dist = ijk[inside], world[inside], dist[inside]

        owner = seg[tuple(ijk.T)]
        keep = owner == 0
        if not keep.all():
            taken = ~keep
            contested += int(taken.sum())
            d_owner = np.linalg.norm(world[taken] - coords[owner[taken] - 1], axis=1)
            keep[taken] = dist[taken] < d_owner

        seg[tuple(ijk[keep].T)] = i + 1

    return seg, contested


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_tsv(tsv_path, names, coords, radii, counts):
    with open(tsv_path, "w") as fh:
        fh.write("index\tname\telectrode\tx\ty\tz\tradius_mm\tn_voxels\n")
        for i, (n, c, r) in enumerate(zip(names, coords, radii)):
            fh.write(f"{i + 1}\t{n}\t{electrode_of(n)}\t{c[0]:.3f}\t{c[1]:.3f}\t"
                     f"{c[2]:.3f}\t{r:.3f}\t{int(counts[i + 1])}\n")


def save_seg(seg, affine, ref_img, path):
    img = nib.Nifti1Image(seg, affine)
    sform_code = int(ref_img.header["sform_code"]) or 1
    qform_code = int(ref_img.header["qform_code"]) or 1
    img.set_sform(affine, code=sform_code)
    img.set_qform(affine, code=qform_code)
    img.header.set_xyzt_units("mm")
    nib.save(img, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Non-overlapping sphere labels around SEEG contacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--contacts", "--fcsv", dest="contacts", required=True,
                   type=Path,
                   help="Contact coordinates: 3D Slicer Markups .fcsv, or a "
                   ".tsv with name, x, y, z columns (e.g. BIDS electrodes.tsv). "
                   "--fcsv is kept as an alias.")
    p.add_argument("--ref", required=True, type=Path,
                   help="Reference NIfTI defining space and grid (e.g. the T1w).")
    p.add_argument("--output-seg", required=True, type=Path,
                   help="Output label NIfTI.")
    p.add_argument("--output-tsv", required=True, type=Path,
                   help="Output label TSV.")
    p.add_argument("--radius", type=float, default=1.5,
                   help="Maximum sphere radius in mm. Default: 1.5")
    p.add_argument("--min-radius", type=float, default=1.0,
                   help="Stop if any radius falls below this (mm); usually a "
                   "localisation error. Default: 1.0")
    p.add_argument("--radius-mode", choices=["global", "per-contact"],
                   default="global",
                   help="'global': one radius for all contacts (equal node "
                   "volumes). 'per-contact': each contact's own safe radius. "
                   "Default: global")
    p.add_argument("--radius-basis", choices=["electrode", "all"],
                   default="electrode",
                   help="Nearest neighbour used for the radius: on the same "
                   "'electrode' (cross-electrode clashes are split by nearest "
                   "contact) or on 'all' electrodes. Default: electrode")
    p.add_argument("--grid-res", type=float, default=None,
                   help="Isotropic voxel size (mm) of the label grid, aligned "
                   "to --ref. Default: the --ref grid itself.")
    p.add_argument("--coord-system", choices=["auto", "RAS", "LPS"],
                   default="auto",
                   help="Coordinate system of the contacts file. Default: read "
                   "from the FCSV header; RAS for TSV.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    print("Contact sphere segmentation")
    names, coords = load_contacts(args.contacts, args.coord_system)

    # A TSV input is also saved as an FCSV next to it (same name, .fcsv), so
    # it can be opened in Slicer.  An existing FCSV is never overwritten.
    if args.contacts.suffix.lower() == ".tsv":
        fcsv_out = args.contacts.with_suffix(".fcsv")
        if fcsv_out.exists():
            print(f"  FCSV already exists, not overwritten: {fcsv_out.name}")
        else:
            write_fcsv(fcsv_out, names, coords)
            print(f"  Saved FCSV from TSV: {fcsv_out}")

    elec_names = [electrode_of(n) for n in names]
    unique_elecs = list(dict.fromkeys(elec_names))
    electrodes = np.array([unique_elecs.index(e) for e in elec_names])
    print(f"  Parsed {len(names)} contacts on {len(unique_elecs)} electrodes")

    # --- radii ----------------------------------------------------------------
    radii, d_nn, nn_idx = compute_radii(
        coords, electrodes, args.radius, args.radius_mode, args.radius_basis
    )

    # Spacing report: nearest-neighbour distance within each electrode
    # (independent of the order contacts are listed in)
    d_within, _ = nearest_neighbours(coords, electrodes)
    for k, e in enumerate(unique_elecs):
        s = d_within[electrodes == k]
        s = s[np.isfinite(s)]
        if len(s):
            print(f"    {e:12s}: nearest-contact spacing median {np.median(s):.2f} mm "
                  f"(range {s.min():.2f}-{s.max():.2f})")

    finite = np.isfinite(d_nn)
    if finite.any():
        i_min = int(np.argmin(np.where(finite, d_nn, np.inf)))
        print(f"  Closest pair ({args.radius_basis}): {names[i_min]} - "
              f"{names[nn_idx[i_min]]} ({d_nn[i_min]:.2f} mm)")
    if args.radius_mode == "global":
        limited = "spacing-limited" if radii[0] < args.radius else "cap"
        print(f"  Radius (global): {radii[0]:.3f} mm ({limited})")
    else:
        n_lim = int(np.sum(radii < args.radius))
        print(f"  Radius (per-contact): {radii.min():.3f}-{radii.max():.3f} mm, "
              f"{n_lim} contact(s) below the {args.radius} mm cap")

    if radii.min() < args.min_radius:
        # Pairs whose spacing forces a radius below the minimum (in global
        # mode these drag every contact down, so report pairs, not contacts)
        bad_pairs = {}
        for i in np.flatnonzero(np.isfinite(d_nn) & (d_nn / 2 < args.min_radius)):
            bad_pairs[tuple(sorted((i, nn_idx[i])))] = d_nn[i]
        print(f"ERROR: {len(bad_pairs)} contact pair(s) closer than "
              f"2 x --min-radius ({2 * args.min_radius:.2f} mm); radius would be "
              f"{radii.min():.3f} mm:", file=sys.stderr)
        for (i, j), d in sorted(bad_pairs.items(), key=lambda kv: kv[1]):
            print(f"    {names[i]} - {names[j]}: {d:.2f} mm", file=sys.stderr)
        print("  Check these coordinates, or lower --min-radius if the spacing "
              "is real.", file=sys.stderr)
        sys.exit(1)

    clashes = cross_electrode_clashes(coords, radii, electrodes)
    if clashes:
        print(f"  NOTE: {len(clashes)} cross-electrode pair(s) closer than their "
              f"two radii; shared voxels go to the nearer contact:")
        for i, j, d in clashes:
            print(f"    {names[i]} - {names[j]}: {d:.2f} mm")

    # --- grid -----------------------------------------------------------------
    ref_img = nib.load(args.ref)
    ref_vox_vol = float(np.prod(ref_img.header.get_zooms()[:3]))
    shape, affine = make_grid(ref_img, args.grid_res)
    vox_vol = abs(float(np.linalg.det(affine[:3, :3])))
    print(f"  Reference grid: {ref_img.shape[:3]}, "
          f"voxel {np.round(ref_img.header.get_zooms()[:3], 3)}")
    print(f"  Label grid:     {shape}, voxel volume {vox_vol:.3f} mm^3")

    expected_vol = 4 / 3 * np.pi * radii ** 3
    lo_v, hi_v = expected_vol.min(), expected_vol.max()
    vol_range = f"{lo_v:.1f}" if np.isclose(lo_v, hi_v) else f"{lo_v:.1f}-{hi_v:.1f}"
    vox_range = (f"{lo_v / vox_vol:.0f}" if np.isclose(lo_v, hi_v)
                 else f"{lo_v / vox_vol:.0f}-{hi_v / vox_vol:.0f}")
    print(f"  Expected sphere volume: {vol_range} mm^3 (~{vox_range} label voxels)")

    # --- rasterise ------------------------------------------------------------
    seg, contested = rasterise_spheres(coords, radii, shape, affine)
    counts = np.bincount(seg.ravel(), minlength=len(names) + 1)
    print(f"  Voxels claimed by two spheres (split to nearer contact): {contested}")

    # --- checks ---------------------------------------------------------------
    ratio = counts[1:] * vox_vol / expected_vol
    empty = [names[i] for i in np.flatnonzero(counts[1:] == 0)]
    clipped = [names[i] for i in np.flatnonzero((counts[1:] > 0) & (ratio < 0.8))]
    if empty:
        # Downstream steps count unique labels and number node pairs 1..N, so
        # a missing label would silently drop the highest contact.
        print(f"ERROR: {len(empty)} contact(s) have 0 voxels (outside the "
              f"image, or wrong coordinate space?): {empty}. Nothing saved.",
              file=sys.stderr)
        sys.exit(1)
    if clipped:
        print(f"  WARNING: {len(clipped)} contact(s) below 80% of the expected "
              f"volume (image edge or a nearby electrode): {clipped}")
    print(f"  Voxel count / expected volume: median {np.median(ratio):.2f} "
          f"(range {ratio.min():.2f}-{ratio.max():.2f})")

    # --- save -----------------------------------------------------------------
    args.output_seg.parent.mkdir(parents=True, exist_ok=True)
    save_seg(seg, affine, ref_img, args.output_seg)
    print(f"  Saved segmentation: {args.output_seg}")

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_tsv, names, coords, radii, counts)
    print(f"  Saved TSV:          {args.output_tsv}")
    print("Done.")


if __name__ == "__main__":
    main()
