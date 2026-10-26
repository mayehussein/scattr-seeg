#!/usr/bin/env python3
"""
yba_circos.py  -  Standardized circos plots on the Yale Brain Atlas (YBA)

One fixed angular skeleton (hemisphere > lobe > region > gyrus > parcel) is built
once and reused for three levels of detail:

    contacts : SEEG contacts as dots, one name per electrode
    parcels  : YBA parcels as dots, one name per gyrus
    regions  : YBA regions

The hemisphere, lobe and region arcs are identical on every plot, including
arcs without any contacts.

Usage
-----
    python yba_circos.py --subject P261 --exp new2
    python yba_circos.py --subject P261 --exp new2 --top-pcts 100 20 --formats pdf png

Sections
--------
    1. CONFIG
    2. DATA LOADING
    3. CONTACT TABLE
    4. LAYOUT              (the standardized skeleton)
    5. AGGREGATION         (contacts -> parcels -> regions)
    6. DRAWING COMPONENTS
    7. FIGURES AND HEATMAPS
    8. MAIN
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import PathPatch, Wedge
from matplotlib.path import Path

plt.rcParams["pdf.fonttype"] = 42      # keep text as real, searchable text
plt.rcParams["svg.fonttype"] = "none"

# ==================================================================
# 1. CONFIG
# ==================================================================
CFG = dict(
    subject_id="P261",
    exp_name="new2",
    base_dir="/nfs/khan/trainees/mkhaled5/scattr-seeg",
    yale_atlas_dir="/nfs/khan/trainees/mkhaled5/Yale_Atlas",
    out_dir=None,                       # default: ./output/<exp>_YBA-updated_sub-<id>

    levels=["contacts", "parcels", "regions"],
    top_pcts=[100, 20],                 # 100 = all edges, 20 = strongest 20 %
    routing="hier",                     # "hier" (hierarchical) or "center"
    formats=["png", "pdf"],
    dpi=200,
    fig_inches=30,
    rasterize_edges=True,               # edges raster, text stays vector in PDF

    # ---- layout --------------------------------------------------
    hemi_order=["Right", "Left"],       # clockwise from the top
    mirror_hemi="Left",                 # reversed so homologous areas face each other
    lobe_order=["Frontal", "Temporal", "Insula", "Parietal", "Occipital",
                "Corpus callosum"],
    gap_hemi=6.0,                       # gap between hemispheres (top and bottom), degrees
    gap_lobe=1.2,
    gap_region=0.4,
    # Extra angular room given to a parcel per localized contact (in units of an
    # empty parcel slot). The room is shared by the two homologous parcels (left and
    # right), so the layout is an exact mirror image. 0 = every slot identical
    # (comparable across subjects); >0 gives contacts room (fixed within a subject).
    slot_weight_per_contact=3.0,
    slot_alpha=(0.9, 0.25),             # parcel slot opacity: with / without contacts

    # ---- style ---------------------------------------------------
    lobe_colors={
        "Frontal": "#7F77DD", "Parietal": "#1D9E75", "Occipital": "#D85A30",
        "Temporal": "#378ADD", "Insula": "#BA7517", "Corpus callosum": "#888780",
    },
    contact_pt=3.0,                     # max contact dot radius (points)
    parcel_pt=(2.0, 6.0),               # parcel dot radius range
    region_pt=(4.0, 10.0),              # region dot radius range
    lw_range=(0.4, 3.0),
    alpha_range=(0.15, 0.75),
    edge_color="blend",                 # "blend": fades from one lobe colour to the other
                                        # "lobe": colour of the first endpoint's lobe
                                        # "neutral": gray
    edge_segments=24,                   # blend mode: segments per connection
    legends="strip",                    # "strip": legend band under the figure; "none": no legend
    bottom_pad=0.03,                    # white space kept under the lowest label (x radius)
    legend_strip_in=1.5,                # height of that band (inches)
    legend_font=20,                     # legend font size (points)
    contact_labels="full",              # "full": every contact named (e.g. LAHc-03)
                                        # "electrode": brackets + one name per electrode run
    contact_label_font=6.5,             # font size of the full contact names (points)
    contact_edges=True,                 # False: contacts level shows nodes + names only
    run_gap_deg=12.0,                   # (electrode mode) a run is split above this gap
    label_all_runs=True,                # (electrode mode) False: name longest run only
    label_empty_gyri=True,              # parcels level: name every gyrus tile
    parcel_label_font=6.5,              # parcels level: parcel name beside each dot (pt)
    header_gap=0.36,                    # hemisphere labels this far below the top edge
)

# Radii (in units of the node circle, r = 1) per level. Angles are shared; only
# the label zones differ so that text fits.
LEVEL_STYLE = {
    "contacts": dict(
        show_slot=False,                # no parcel-slot ring; set True to draw it again
        slot=(1.03, 1.07), bracket0=1.06, label_font=8,
        region_ring=(1.20, 1.24), region_font=9, region_label=1.26,
        lobe_ring=(1.66, 1.70), lobe_font=12, lobe_label=1.72, extent=2.30),
    "parcels": dict(
        show_slot=False,                # no per-parcel tiles
        show_gyrus=True,                # one tile per gyrus (all named)
        slot=(1.03, 1.07), gyrus=(1.20, 1.24), gyrus_font=6.5, gyrus_label=1.26,
        region_ring=(1.62, 1.66), region_font=9, region_label=1.68,
        lobe_ring=(2.06, 2.10), lobe_font=12, lobe_label=2.12, extent=2.70),
    "regions": dict(
        show_slot=False,                # lobe and region rings only
        slot=(1.03, 1.07), region_ring=(1.10, 1.14), region_font=10,
        region_label=1.16,
        lobe_ring=(1.60, 1.64), lobe_font=12, lobe_label=1.66, extent=2.30),
}


# ==================================================================
# 2. DATA LOADING
# ==================================================================
def get_paths(cfg):
    sid, exp = cfg["subject_id"], cfg["exp_name"]
    base, atlas = cfg["base_dir"], cfg["yale_atlas_dir"]
    return dict(
        labels=os.path.join(base, "bids", f"sub-{sid}", "anat",
                            f"sub-{sid}_space-T1w_desc-AllContacts_dseg.tsv"),
        matrix=os.path.join(base, "output", f"scattr_{exp}_{sid}", "mrtrix",
                            f"sub-{sid}", "tractography",
                            f"sub-{sid}_desc-filteredsubcortical_nodeWeights.csv"),
        loc=os.path.join(atlas, "contacts_transformation", "output", f"sub-{sid}",
                         f"sub-{sid}_SEEG_YBA_detailed_localization.csv"),
        yba=os.path.join(atlas, "YBA_696parcels", "YBA_696_parcel_dict.csv"),
    )


def _require(path):
    if not os.path.isfile(path):
        raise SystemExit(f"Error: file not found:\n  {path}")
    return path


def load_inputs(cfg):
    paths = get_paths(cfg)
    print(f"Loading inputs for sub-{cfg['subject_id']} ...")
    names = pd.read_csv(_require(paths["labels"]), sep="\t")["name"].tolist()
    raw = pd.read_csv(_require(paths["matrix"]), header=None).to_numpy(float)
    if raw.shape != (len(names), len(names)):
        raise SystemExit(f"Matrix shape {raw.shape} does not match {len(names)} labels")
    sym = np.maximum(raw, raw.T)                      # symmetrize
    matrix = pd.DataFrame(sym, index=names, columns=names)
    loc = pd.read_csv(_require(paths["loc"]))
    yba = pd.read_csv(_require(paths["yba"]))
    return matrix, loc, yba


# ==================================================================
# 3. CONTACT TABLE
# ==================================================================
_CONTACT_RE = re.compile(r"^(.*?)[-_]?(\d+)$")


def parse_contact(name):
    """'LAHc-01' -> ('LAHc', 1)."""
    m = _CONTACT_RE.match(str(name))
    if not m or not m.group(1):
        return str(name), 0
    return m.group(1), int(m.group(2))


def build_contact_table(loc, matrix, yba, out_dir, subject_id):
    valid = set(yba["Name"])
    keep, dropped = [], []
    for r in loc.itertuples(index=False):
        c = r.contact_name
        if c not in matrix.index:
            continue
        reg = r.radius_region
        if pd.isna(reg) or reg == "Unknown":
            reg = r.center_region
        if pd.isna(reg) or reg == "Unknown" or reg not in valid:
            dropped.append(dict(contact_name=c,
                                dominant_tissue=getattr(r, "dominant_tissue", ""),
                                total_connection_strength=float(matrix.loc[c].sum())))
            continue
        el, idx = parse_contact(c)
        keep.append(dict(contact=c, electrode=el, idx=idx, parcel=reg))

    # tip = lowest contact index of each electrode over ALL contacts in the matrix
    tips = {}
    for c in loc["contact_name"]:
        if c in matrix.index:
            el, idx = parse_contact(c)
            tips[el] = min(idx, tips.get(el, idx))
    contacts = pd.DataFrame(keep)
    contacts["is_tip"] = [i == tips[e] for e, i in zip(contacts.electrode, contacts.idx)]

    if dropped:
        p = os.path.join(out_dir, f"dropped_unknown_contacts_sub-{subject_id}.csv")
        pd.DataFrame(dropped).to_csv(p, index=False)
        print(f"  Dropped {len(dropped)} contacts without a YBA parcel -> {p}")
    print(f"  Contacts kept: {len(contacts)} on "
          f"{contacts.electrode.nunique()} electrodes, "
          f"{contacts.parcel.nunique()} parcels")
    return contacts


# ==================================================================
# 4. LAYOUT  (the standardized skeleton)
# ==================================================================
class Layout:
    """Angles in degrees, clockwise from the top (12 o'clock)."""

    def __init__(self, parcels, regions, lobes, hemis, gyri, deg_per_unit):
        self.parcels, self.regions, self.lobes = parcels, regions, lobes
        self.hemis, self.gyri, self.deg_per_unit = hemis, gyri, deg_per_unit


def _spans(p, keys):
    out = (p.groupby(keys, sort=False)
            .agg(t0=("t0", "min"), t1=("t1", "max"), n=("name", "size"))
            .reset_index())
    out["tmid"] = (out.t0 + out.t1) / 2
    return out


def symmetric_counts(p, hemi_order):
    """Contact count of each parcel and its left/right homologue (the maximum)."""
    n = p["n_contacts"].to_numpy()
    key = p["name"].str.replace(r"^[LR]_", "", regex=True)
    if (key.value_counts() == 2).all():                 # pair by parcel name
        return p.groupby(key)["n_contacts"].transform("max").to_numpy()
    h = p["hemi"].to_numpy()
    a, b = np.where(h == hemi_order[0])[0], np.where(h == hemi_order[1])[0]
    if len(a) == len(b):                                # pair by mirrored position
        print("  NOTE: parcel names do not pair L/R; pairing by mirrored position")
        out = n.copy()
        m = np.maximum(n[a], n[b[::-1]])
        out[a], out[b[::-1]] = m, m
        return out
    print("  WARNING: hemispheres have different parcel counts; layout is not mirrored")
    return n


def build_layout(yba, contacts, cfg):
    d = yba.reset_index(drop=True).copy()
    d["file_idx"] = np.arange(len(d))
    lobes = list(cfg["lobe_order"]) + [l for l in d["Lobe"].unique()
                                       if l not in cfg["lobe_order"]]
    d["lobe_rank"] = d["Lobe"].map({l: i for i, l in enumerate(lobes)})
    # region/gyrus order = order of first appearance in the dictionary file
    d["region_rank"] = d.groupby(["Lobe", "Region"])["file_idx"].transform("min")
    d["gyrus_rank"] = d.groupby(["Lobe", "Region", "Gyrus"])["file_idx"].transform("min")

    parts = []
    for hemi in cfg["hemi_order"]:
        sub = d[d["Hemisphere"] == hemi].sort_values(
            ["lobe_rank", "region_rank", "gyrus_rank", "file_idx"])
        if hemi == cfg["mirror_hemi"]:
            sub = sub.iloc[::-1]
        parts.append(sub)
    p = pd.concat(parts).reset_index(drop=True)
    p = p.rename(columns={"Name": "name", "Hemisphere": "hemi", "Lobe": "lobe",
                          "Region": "region", "Gyrus": "gyrus"})
    counts = contacts.groupby("parcel").size()
    p["n_contacts"] = p["name"].map(counts).fillna(0).astype(int)
    # homologous left/right parcels share the same room -> exact mirror image
    p["n_shared"] = symmetric_counts(p, cfg["hemi_order"])
    p["weight"] = 1.0 + cfg["slot_weight_per_contact"] * p["n_shared"]

    hemi_a, lobe_a, reg_a = p["hemi"].to_numpy(), p["lobe"].to_numpy(), p["region"].to_numpy()
    gaps = np.zeros(len(p))
    for i in range(1, len(p)):
        if hemi_a[i] != hemi_a[i - 1]:
            gaps[i] = cfg["gap_hemi"]
        elif lobe_a[i] != lobe_a[i - 1]:
            gaps[i] = cfg["gap_lobe"]
        elif reg_a[i] != reg_a[i - 1]:
            gaps[i] = cfg["gap_region"]
    dpu = (360.0 - gaps.sum() - cfg["gap_hemi"]) / p["weight"].sum()

    t = cfg["gap_hemi"] / 2
    t0, t1 = np.zeros(len(p)), np.zeros(len(p))
    for i in range(len(p)):
        t += gaps[i]
        t0[i] = t
        t += p["weight"].iat[i] * dpu
        t1[i] = t
    p["t0"], p["t1"] = t0, t1
    ia = np.where(p["hemi"].to_numpy() == cfg["hemi_order"][0])[0]
    ib = np.where(p["hemi"].to_numpy() == cfg["hemi_order"][1])[0]
    if len(ia) == len(ib):
        err = np.abs(t0[ia] - (360.0 - t1[ib[::-1]])).max()
        print(f"  Mirror symmetry: max left/right angular error {err:.4f} deg"
              + ("" if err < 0.01 else "  (!! check parcel structure)"))
    p["tmid"] = (p.t0 + p.t1) / 2
    p["region_key"] = p["hemi"] + "|" + p["lobe"] + "|" + p["region"]

    regions = _spans(p, ["hemi", "lobe", "region"])
    regions["key"] = regions["hemi"] + "|" + regions["lobe"] + "|" + regions["region"]
    regions["n_contacts"] = regions["key"].map(p.groupby("region_key")["n_contacts"].sum())
    lobes_t = _spans(p, ["hemi", "lobe"])
    hemis_t = _spans(p, ["hemi"])
    gyri = _spans(p, ["hemi", "lobe", "region", "gyrus"])
    gyri["n_contacts"] = [
        p[(p.hemi == h) & (p.lobe == l) & (p.region == r) & (p.gyrus == g)]["n_contacts"].sum()
        for h, l, r, g in zip(gyri.hemi, gyri.lobe, gyri.region, gyri.gyrus)]

    print(f"  Layout: {len(p)} parcel slots, {len(gyri)} gyri, {len(regions)} regions, "
          f"{len(lobes_t)} lobe arcs, {dpu:.3f} deg per slot unit")
    return Layout(p, regions, lobes_t, hemis_t, gyri, dpu)


def place_contacts(layout, contacts, cfg):
    """Centre each parcel's contacts in its slot, grouped by electrode, index ascending."""
    p = layout.parcels
    dpu = layout.deg_per_unit
    pos = {n: i for i, n in enumerate(p["name"])}
    c = contacts.copy()
    c["pos"] = c["parcel"].map(pos)
    c = c.sort_values(["pos", "electrode", "idx"]).reset_index(drop=True)
    pm = p.set_index("name")
    s_max = max(cfg["slot_weight_per_contact"], 1.0) * dpu
    theta = np.zeros(len(c))
    for name, grp in c.groupby("parcel", sort=False):
        t0, t1 = pm.at[name, "t0"], pm.at[name, "t1"]
        el = grp["electrode"].to_numpy()
        u = np.zeros(len(grp))
        for k in range(1, len(grp)):                    # half a step between electrodes
            u[k] = u[k - 1] + 1.0 + (0.5 if el[k] != el[k - 1] else 0.0)
        step = min((t1 - t0) * 0.94 / (u[-1] + 1.0), s_max)
        theta[grp.index] = (t0 + t1) / 2 + (u - u[-1] / 2) * step
    c["theta"] = theta
    for col in ["hemi", "lobe", "region", "gyrus"]:
        c[col] = c["parcel"].map(pm[col])
    return c


# ==================================================================
# 5. AGGREGATION
# ==================================================================
def aggregate(M, labels, order):
    """Sum the contact matrix into groups (ordered by `order`); zero the diagonal."""
    idx = {g: i for i, g in enumerate(order)}
    G = np.zeros((len(labels), len(order)))
    G[np.arange(len(labels)), [idx[l] for l in labels]] = 1.0
    A = G.T @ M @ G
    np.fill_diagonal(A, 0.0)
    return A


def edge_table(A):
    i, j = np.triu_indices_from(A, k=1)
    w = A[i, j]
    m = w > 0
    return pd.DataFrame(dict(i=i[m], j=j[m], raw=w[m], w=np.log1p(w[m])))


def build_levels(matrix, contacts, layout):
    names = contacts["contact"].tolist()
    M = np.array(matrix.loc[names, names], dtype=float)
    np.fill_diagonal(M, 0.0)
    levels = {}

    # contacts
    nodes = contacts.copy()
    nodes["strength"] = M.sum(1)
    nodes["label"] = nodes["contact"]
    levels["contacts"] = (nodes, M)

    # contacts with at least one streamline (the same contacts that have lines in the
    # contacts plot); log(1 + w) is zero only where w is zero, so the scaling is irrelevant
    has_conn = pd.Series(M.sum(1) > 0).astype(int)

    # parcels
    p = layout.parcels
    porder = [n for n in p["name"] if n in set(contacts["parcel"])]
    A = aggregate(M, contacts["parcel"].to_numpy(), porder)
    pm = p.set_index("name")
    nodes = pm.loc[porder, ["hemi", "lobe", "region", "gyrus", "tmid", "n_contacts"]].copy()
    nodes = nodes.rename(columns={"tmid": "theta"}).reset_index().rename(columns={"name": "parcel"})
    nodes["strength"] = A.sum(1)
    nodes["n_conn"] = has_conn.groupby(contacts["parcel"].to_numpy()).sum() \
        .reindex(nodes["parcel"]).fillna(0).astype(int).to_numpy()
    nodes["label"] = nodes["parcel"]
    levels["parcels"] = (nodes, A)

    # regions
    r = layout.regions
    rkey = (contacts["hemi"] + "|" + contacts["lobe"] + "|" + contacts["region"]).to_numpy()
    rorder = [k for k in r["key"] if k in set(rkey)]
    A = aggregate(M, rkey, rorder)
    rm = r.set_index("key")
    nodes = rm.loc[rorder, ["hemi", "lobe", "region", "tmid", "n_contacts"]].copy()
    nodes = nodes.rename(columns={"tmid": "theta"}).reset_index()
    nodes["strength"] = A.sum(1)
    nodes["n_conn"] = has_conn.groupby(rkey).sum().reindex(nodes["key"]) \
        .fillna(0).astype(int).to_numpy()
    nodes["label"] = nodes["hemi"].str[0] + " " + nodes["region"]
    levels["regions"] = (nodes, A)
    return levels


# ==================================================================
# 6. DRAWING COMPONENTS
# ==================================================================
def pol(r, th):
    t = np.radians(th)
    return r * np.sin(t), r * np.cos(t)


def text_rot(th):
    th = th % 360
    return (90 - th, "left") if th <= 180 else (270 - th, "right")


def spread_angles(theta, min_gap, iters=400):
    """Push labels apart on the circle until neighbours are >= min_gap degrees."""
    th = np.asarray(theta, float) % 360
    n = len(th)
    if n < 2:
        return th
    order = np.argsort(th)
    a = th[order].copy()
    min_gap = min(min_gap, 0.999 * 360.0 / n)
    for _ in range(iters):
        moved = False
        for i in range(n):
            j = (i + 1) % n
            gap = (a[j] - a[i]) % 360.0
            if gap < min_gap:
                s = (min_gap - gap) / 2 + 1e-6
                a[i] -= s
                a[j] += s
                moved = True
        if not moved:
            break
    out = np.empty(n)
    out[order] = a
    return out


def radial_labels(ax, thetas, texts, r_anchor, r_text, fs, pts, weight="normal",
                  leader=True, colors=None):
    if len(thetas) == 0:
        return
    thetas = np.asarray(thetas, float)
    min_gap = np.degrees(fs * 1.25 / (r_text * pts))
    adj = spread_angles(thetas, min_gap)
    for k, (t0, t1, s) in enumerate(zip(thetas, adj, texts)):
        if leader:
            x0, y0 = pol(r_anchor, t0)
            x1, y1 = pol(r_text - 0.006, t1)
            ax.plot([x0, x1], [y0, y1], lw=0.4, color="0.55", zorder=3, solid_capstyle="butt")
        rot, ha = text_rot(t1)
        x, y = pol(r_text, t1)
        ax.text(x, y, s, rotation=rot, ha=ha, va="center", rotation_mode="anchor",
                fontsize=fs, fontweight=weight, zorder=6,
                color="black" if colors is None else colors[k])


def ring(ax, r0, r1, t0, t1, color, alpha=1.0, z=2):
    ax.add_patch(Wedge((0, 0), r1, 90 - t1, 90 - t0, width=r1 - r0,
                       facecolor=color, edgecolor="none", alpha=alpha, zorder=z))


def draw_skeleton(ax, layout, st, cfg, pts):
    # parcel slots: solid, one colour per lobe; slots without contacts are lighter
    if st.get("show_slot", True):
        patches, cols = [], []
        for r in layout.parcels.itertuples():
            patches.append(Wedge((0, 0), st["slot"][1], 90 - r.t1, 90 - r.t0,
                                 width=st["slot"][1] - st["slot"][0]))
            a = cfg["slot_alpha"][0] if r.n_contacts > 0 else cfg["slot_alpha"][1]
            cols.append((*matplotlib.colors.to_rgb(cfg["lobe_colors"].get(r.lobe, "0.5")), a))
        ax.add_collection(PatchCollection(patches, facecolors=cols, edgecolors=cols,
                                          linewidths=0.3, zorder=2))
    # gyrus tiles (parcels level): solid, darker where the gyrus has contacts
    if st.get("show_gyrus", False):
        for r in layout.gyri.itertuples():
            a = cfg["slot_alpha"][0] if r.n_contacts > 0 else cfg["slot_alpha"][1]
            ring(ax, *st["gyrus"], r.t0 + 0.05, r.t1 - 0.05,
                 cfg["lobe_colors"].get(r.lobe, "0.5"), a)
    # regions
    k = 0
    last = None
    for r in layout.regions.itertuples():
        k = k + 1 if (r.hemi, r.lobe) == last else 0
        last = (r.hemi, r.lobe)
        ring(ax, *st["region_ring"], r.t0 + 0.03, r.t1 - 0.03,
             cfg["lobe_colors"].get(r.lobe, "0.5"), 0.55 if k % 2 == 0 else 0.30)
    rg = layout.regions
    radial_labels(ax, rg["tmid"], rg["region"], st["region_ring"][1], st["region_label"],
                  st["region_font"], pts)
    # lobes
    for r in layout.lobes.itertuples():
        ring(ax, *st["lobe_ring"], r.t0 + 0.03, r.t1 - 0.03, cfg["lobe_colors"].get(r.lobe, "0.5"), 0.95)
    lb = layout.lobes
    radial_labels(ax, lb["tmid"], lb["lobe"], st["lobe_ring"][1], st["lobe_label"],
                  st["lobe_font"], pts, weight="bold",
                  colors=[cfg["lobe_colors"].get(l, "0.3") for l in lb["lobe"]])


def _draw_edges_blend(ax, nodes, edges, cfg, vmax, X, Y, hemi, lobe):
    """Each connection fades from the colour of one endpoint's lobe to the other's.

    A connection is cut into short straight segments, each with its own colour, so a
    cross-lobe chord reads as both lobes; same-lobe chords keep that lobe's colour.
    Strongest connections are drawn last (on top); width and opacity show strength."""
    lw0, lw1 = cfg["lw_range"]
    a0, a1 = cfg["alpha_range"]
    n = int(cfg["edge_segments"])
    e = edges.sort_values("w")
    i, j = e["i"].to_numpy().astype(int), e["j"].to_numpy().astype(int)
    P0 = np.c_[X[i], Y[i]]
    P1 = np.c_[X[j], Y[j]]
    if cfg["routing"] == "hier":
        k = np.where(hemi[i] != hemi[j], 0.12, np.where(lobe[i] != lobe[j], 0.40, 0.78))
    else:
        k = np.zeros(len(e))
    C = k[:, None] * (P0 + P1) / 2
    t = np.linspace(0.0, 1.0, n + 1)[:, None, None]
    pts = (1 - t) ** 2 * P0 + 2 * (1 - t) * t * C + t ** 2 * P1          # (n+1, E, 2)
    segs = np.stack([pts[:-1], pts[1:]], axis=2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    rgb = lambda l: np.array(matplotlib.colors.to_rgb(cfg["lobe_colors"].get(l, "0.4")))
    cA = np.array([rgb(l) for l in lobe[i]])
    cB = np.array([rgb(l) for l in lobe[j]])
    m = ((np.arange(n) + 0.5) / n)[None, :, None]
    col = cA[:, None, :] * (1 - m) + cB[:, None, :] * m                  # (E, n, 3)
    w = np.minimum(e["w"].to_numpy() / vmax, 1.0)
    alpha = np.repeat((a0 + (a1 - a0) * w)[:, None, None], n, axis=1)
    rgba = np.concatenate([col, alpha], axis=2).reshape(-1, 4)
    widths = np.repeat(lw0 + (lw1 - lw0) * w, n)
    ax.add_collection(LineCollection(segs, colors=rgba, linewidths=widths,
                                     capstyle="butt", zorder=1,
                                     rasterized=cfg["rasterize_edges"]))


def draw_edges(ax, nodes, edges, cfg, vmax):
    if edges.empty:
        return
    th = nodes["theta"].to_numpy()
    X, Y = pol(1.0, th)
    hemi, lobe = nodes["hemi"].to_numpy(), nodes["lobe"].to_numpy()
    lw0, lw1 = cfg["lw_range"]
    a0, a1 = cfg["alpha_range"]
    if cfg["edge_color"] == "blend":
        _draw_edges_blend(ax, nodes, edges, cfg, vmax, X, Y, hemi, lobe)
        return
    patches = []
    for i, j, w in edges.sort_values("w")[["i", "j", "w"]].itertuples(index=False):
        if cfg["routing"] == "hier":
            k = 0.12 if hemi[i] != hemi[j] else (0.40 if lobe[i] != lobe[j] else 0.78)
        else:
            k = 0.0
        ctrl = (k * (X[i] + X[j]) / 2, k * (Y[i] + Y[j]) / 2)
        path = Path([(X[i], Y[i]), ctrl, (X[j], Y[j])],
                    [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        n = min(w / vmax, 1.0)
        base = "#555555" if cfg["edge_color"] == "neutral" else cfg["lobe_colors"].get(lobe[int(i)], "0.4")
        rgba = matplotlib.colors.to_rgba(base, a0 + (a1 - a0) * n)
        patches.append(PathPatch(path, facecolor="none", edgecolor=rgba,
                                 linewidth=lw0 + (lw1 - lw0) * n, capstyle="round"))
    ax.add_collection(PatchCollection(patches, match_original=True, zorder=1,
                                      rasterized=cfg["rasterize_edges"]))


def _neighbour_cap(theta, pts):
    """Largest dot radius (pt) that does not touch the neighbouring node."""
    th = np.sort(theta % 360)
    if len(th) < 2:
        return np.full(len(theta), 99.0)
    d = np.diff(np.r_[th, th[0] + 360])
    gap_pt = np.radians(d) * pts
    cap_sorted = 0.48 * np.minimum(gap_pt, np.r_[gap_pt[-1], gap_pt[:-1]])
    order = np.argsort(theta % 360)
    cap = np.empty(len(theta))
    cap[order] = cap_sorted
    return np.maximum(cap, 0.8)


def draw_nodes(ax, nodes, level, cfg, pts, edges):
    th = nodes["theta"].to_numpy()
    x, y = pol(1.0, th)
    face = [cfg["lobe_colors"].get(l, "0.4") for l in nodes["lobe"]]
    cap = _neighbour_cap(th, pts)
    if level == "contacts":
        r = np.minimum(cfg["contact_pt"], cap)
        fc, ec, lw = face, ["white"] * len(face), 0.4
    else:
        lo, hi = cfg["parcel_pt"] if level == "parcels" else cfg["region_pt"]
        # circle area is proportional to the number of contacts with streamlines in the
        # parcel / region (full matrix, so sizes match across figures); a node with no
        # drawn line in this figure gets no circle
        n = nodes["n_conn"].to_numpy(float)
        norm = n / n.max() if n.max() > 0 else n
        r = np.minimum(lo + (hi - lo) * np.sqrt(norm), np.maximum(cap, lo * 0.6))
        drawn = np.zeros(len(nodes), dtype=bool)
        if len(edges):
            drawn[edges["i"].to_numpy().astype(int)] = True
            drawn[edges["j"].to_numpy().astype(int)] = True
        r = np.where((n > 0) & drawn, r, 0.0)
        fc, ec, lw = face, ["white"] * len(face), 0.6
    ax.scatter(x, y, s=(2 * r) ** 2, c=fc, edgecolors=ec, linewidths=lw, zorder=5)


def electrode_runs(nodes, gap_deg):
    """Contiguous runs of one electrode's contacts in circle order.

    Runs of different electrodes never interleave, so brackets never overlap and a
    single bracket track is enough."""
    n = nodes.sort_values("theta")
    th, el = n["theta"].to_numpy(), n["electrode"].to_numpy()
    out, start = [], 0
    for k in range(1, len(n) + 1):
        if k == len(n) or el[k] != el[k - 1] or th[k] - th[k - 1] > gap_deg:
            out.append(dict(electrode=el[start], t0=th[start], t1=th[k - 1], n=k - start))
            start = k
    return pd.DataFrame(out)


def draw_electrode_labels(ax, nodes, st, cfg, pts):
    runs = electrode_runs(nodes, cfg["run_gap_deg"])
    rb = st["bracket0"]
    r_lab = rb + 0.03
    pad = np.degrees(4.0 / (rb * pts))                     # bracket reaches 4 pt past the dots
    cap = 0.007
    for c in runs.itertuples():
        ang = np.linspace(c.t0 - pad, c.t1 + pad, 40)
        x, y = pol(rb, ang)
        ax.plot(x, y, color="0.25", lw=1.0, zorder=4, solid_capstyle="round")
        for a in (c.t0 - pad, c.t1 + pad):
            xa, ya = pol(rb - cap, a)
            xb, yb = pol(rb + cap, a)
            ax.plot([xa, xb], [ya, yb], color="0.25", lw=1.0, zorder=4)
    lab = runs if cfg["label_all_runs"] else runs.loc[runs.groupby("electrode")["n"].idxmax()]
    mids = ((lab["t0"] + lab["t1"]) / 2).to_numpy()
    min_gap = np.degrees(st["label_font"] * 1.25 / (r_lab * pts))
    adj = spread_angles(mids, min_gap)
    for c, m0, m1 in zip(lab.itertuples(), mids, adj):
        x0, y0 = pol(rb + cap, m0)
        x1, y1 = pol(r_lab - 0.004, m1)
        ax.plot([x0, x1], [y0, y1], lw=0.4, color="0.55", zorder=3)
        rot, ha = text_rot(m1)
        x, y = pol(r_lab, m1)
        ax.text(x, y, c.electrode, rotation=rot, ha=ha, va="center",
                rotation_mode="anchor", fontsize=st["label_font"],
                fontweight="bold", zorder=6)
    return len(runs), len(lab), r_lab


def draw_contact_labels(ax, nodes, st, cfg, pts):
    """Full contact name next to every node (spread + leader line where crowded)."""
    fs = cfg["contact_label_font"]
    r_from = st["slot"][1] if st.get("show_slot", True) else 1.015
    r_text = r_from + 0.03
    radial_labels(ax, nodes["theta"].to_numpy(), nodes["contact"].tolist(),
                  r_from, r_text, fs, pts)
    longest = int(nodes["contact"].str.len().max())
    return len(nodes), r_text + longest * 0.62 * fs / pts


def draw_parcel_labels(ax, nodes, st, cfg, pts):
    """Parcel name (dictionary Name) beside every parcel dot."""
    fs = cfg["parcel_label_font"]
    r_text = 1.045
    radial_labels(ax, nodes["theta"].to_numpy(), nodes["parcel"].tolist(),
                  1.015, r_text, fs, pts)
    longest = int(nodes["parcel"].str.len().max())
    return len(nodes), r_text + longest * 0.62 * fs / pts


def draw_gyrus_labels(ax, layout, st, cfg, pts):
    g = layout.gyri
    if not cfg["label_empty_gyri"]:
        g = g[g["n_contacts"] > 0]
    radial_labels(ax, g["tmid"], g["gyrus"], st["gyrus"][1], st["gyrus_label"],
                  st["gyrus_font"], pts)
    return len(g)


def draw_legend_strip(fig, level, nodes, edges, cfg, vmax, strip_in):
    """Legend strip under the figure: lobe colours, ring order, what a dot is, and
    what connection width / colour mean. Drawn in inches on its own axes, so it never
    overlaps the circos."""
    W = cfg["fig_inches"]
    lax = fig.add_axes([0, 0, 1, strip_in / fig.get_figheight()])
    lax.set_xlim(0, W)
    lax.set_ylim(0, strip_in)
    lax.axis("off")
    fs = cfg["legend_font"]
    gray = "0.35"
    rend = fig.canvas.get_renderer()
    ypt = fs / 72.0

    def put(x, y, t, **kw):
        tx = lax.text(x, y, t, fontsize=fs, va="center", ha="left", **kw)
        bb = tx.get_window_extent(rend)
        return x + bb.width / fig.dpi

    y1, y2 = strip_in * 0.70, strip_in * 0.30
    lax.plot([0.6, W - 0.6], [strip_in - 0.04] * 2, color="0.8", lw=1)

    # row 1: lobes, then the ring order
    x = put(0.6, y1, "Lobes:", fontweight="bold") + 0.25
    for lobe in cfg["lobe_order"]:
        lax.add_patch(plt.Rectangle((x, y1 - ypt * 0.45), ypt * 1.4, ypt * 0.9,
                                    color=cfg["lobe_colors"].get(lobe, "0.5"), lw=0))
        x = put(x + ypt * 1.4 + 0.1, y1, lobe) + 0.4
    rings = {"contacts": "Lobe \u2192 Region \u2192 Contact",
             "parcels": "Lobe \u2192 Region \u2192 Gyrus \u2192 Parcel",
             "regions": "Lobe \u2192 Region"}[level]
    x = put(x + 0.3, y1, "Rings, outside in:", fontweight="bold") + 0.2
    put(x, y1, rings)

    # row 2: what the dot, its size and the line width mean (no numbers: they are continuous)
    node = {"contacts": "Contact, labelled as electrode-index",
            "parcels": "Parcel, labelled with its name",
            "regions": None}[level]               # regions: the arc is the region
    if level == "contacts" and cfg["contact_labels"] != "full":
        node = "a contact"
    x = 0.6
    if node:
        lax.scatter([x + ypt * 0.5], [y2], s=(fs * 0.9) ** 2, c=gray, zorder=3)
        x = put(x + ypt * 1.3, y2, "Circle:", fontweight="bold") + 0.15
        x = put(x, y2, node) + 0.6
    if level != "contacts":
        lo, hi = cfg["parcel_pt"] if level == "parcels" else cfg["region_pt"]
        for r in (lo, (lo + hi) / 2, hi):
            x += r / 72.0
            lax.scatter([x], [y2], s=(2 * r) ** 2, c=gray, zorder=3)
            x += r / 72.0 + 0.12
        x = put(x + 0.08, y2, f"Bigger circle = more contributing contacts") + 0.6
    if len(edges):
        lw0, lw1 = cfg["lw_range"]
        for q in (0.0, 0.5, 1.0):
            lax.plot([x, x + 0.6], [y2, y2], color="0.3", lw=lw0 + (lw1 - lw0) * q,
                     solid_capstyle="round")
            x += 0.7
        x = put(x, y2, "Thicker line = more streamlines") + 0.6


def draw_circos(layout, nodes, edges, level, cfg, vmax, title, out_base):
    st = LEVEL_STYLE[level]
    L = st["extent"]
    strip = cfg["legend_strip_in"] if cfg["legends"] == "strip" else 0.0
    W = cfg["fig_inches"]
    fig = plt.figure(figsize=(W, W + strip))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, strip / (W + strip), 1, W / (W + strip)])
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.axis("off")
    pts = cfg["fig_inches"] * 72.0 / (2 * L)          # points per unit radius

    if level == "contacts" and not cfg["contact_edges"]:
        edges = edges.iloc[0:0]
        title = title.split(" - ")[0] + " " + title[title.rindex("("):]
    draw_skeleton(ax, layout, st, cfg, pts)
    right, left = cfg["hemi_order"][0], cfg["hemi_order"][1]      # clockwise from the top
    ax.text(0.08, L - cfg["header_gap"], f"{right} hemisphere", ha="left", va="center",
            fontsize=20, fontweight="bold")
    ax.text(-0.08, L - cfg["header_gap"], f"{left} hemisphere", ha="right", va="center",
            fontsize=20, fontweight="bold")
    draw_edges(ax, nodes, edges, cfg, vmax)
    draw_nodes(ax, nodes, level, cfg, pts, edges)
    extra = ""
    if level == "contacts" and cfg["contact_labels"] == "full":
        n_lab, r_end = draw_contact_labels(ax, nodes, st, cfg, pts)
        extra = f", {n_lab} contact names"
        if r_end > st["region_ring"][0] - 0.01:
            print(f"  WARNING: contact names reach r={r_end:.2f}; lower contact_label_font "
                  f"or raise the region/lobe radii for 'contacts' in LEVEL_STYLE")
    elif level == "contacts":
        n_runs, n_lab, r_lab = draw_electrode_labels(ax, nodes, st, cfg, pts)
        extra = f", {n_runs} electrode runs, {n_lab} electrode labels"
        if r_lab + 0.10 > st["region_ring"][0]:
            print(f"  WARNING: electrode labels reach r={r_lab:.2f}; raise region_ring "
                  f"radii for 'contacts' in LEVEL_STYLE")
    elif level == "parcels":
        n_lab = draw_gyrus_labels(ax, layout, st, cfg, pts)
        n_par, r_end = draw_parcel_labels(ax, nodes, st, cfg, pts)
        extra = f", {n_lab} gyrus labels, {n_par} parcel names"
        if r_end > st["gyrus"][0] - 0.01:
            print(f"  WARNING: parcel names reach r={r_end:.2f}; lower parcel_label_font "
                  f"or raise the gyrus/region/lobe radii for 'parcels' in LEVEL_STYLE")
    ax.text(0, L - 0.05, title, ha="center", va="top", fontsize=24, fontweight="bold")

    # trim the empty band under the circle: cut the axes just below the lowest drawn thing
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    low = []
    for art in list(ax.texts) + list(ax.patches) + list(ax.collections) + list(ax.lines):
        try:
            low.append(art.get_window_extent(rend).transformed(ax.transData.inverted()).y0)
        except Exception:
            pass
    low = [v for v in low if np.isfinite(v)]
    ymin = max(-L, (min(low) if low else -L) - cfg["bottom_pad"] * L)
    H = (L - ymin) * W / (2 * L)                       # axes height, inches
    fig.set_size_inches(W, H + strip)
    ax.set_ylim(ymin, L)
    ax.set_position([0, strip / (H + strip), 1, H / (H + strip)])
    if strip:
        draw_legend_strip(fig, level, nodes, edges, cfg, vmax, strip)

    for fmt in cfg["formats"]:
        fig.savefig(f"{out_base}.{fmt}", dpi=cfg["dpi"], facecolor="white")
    plt.close(fig)
    print(f"  {level}: {len(nodes)} nodes, {len(edges)} edges drawn{extra} -> {out_base}.*")


# ==================================================================
# 7. FIGURES AND HEATMAPS
# ==================================================================
def draw_heatmap(A_log, nodes, level, cfg, vmax, title, out_base):
    n = len(nodes)
    s = float(np.clip(n * 0.17 + 5, 9, 42))
    fig, ax = plt.subplots(figsize=(s, s))
    im = ax.imshow(A_log, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
    key = (nodes["hemi"] + "|" + nodes["lobe"]).to_numpy()
    cuts = [i for i in range(1, n) if key[i] != key[i - 1]]
    for c in cuts:
        ax.axhline(c - 0.5, color="white", lw=1.5)
        ax.axvline(c - 0.5, color="white", lw=1.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fs = 6 if n > 120 else (8 if n > 50 else 10)
    ax.set_xticklabels(nodes["label"], rotation=90, fontsize=fs)
    ax.set_yticklabels(nodes["label"], fontsize=fs)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("log(1 + streamline count)", fontsize=12)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=16)
    fig.tight_layout()
    for fmt in cfg["formats"]:
        fig.savefig(f"{out_base}.{fmt}", dpi=min(cfg["dpi"], 150), facecolor="white")
    plt.close(fig)
    print(f"  heatmap {level} -> {out_base}.*")


# ==================================================================
# 8. MAIN
# ==================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--subject", default=CFG["subject_id"])
    ap.add_argument("--exp", default=CFG["exp_name"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--base-dir", default=CFG["base_dir"])
    ap.add_argument("--atlas-dir", default=CFG["yale_atlas_dir"])
    ap.add_argument("--formats", nargs="+", default=CFG["formats"])
    ap.add_argument("--top-pcts", nargs="+", type=float, default=CFG["top_pcts"])
    ap.add_argument("--levels", nargs="+", default=CFG["levels"],
                    choices=["contacts", "parcels", "regions"])
    ap.add_argument("--routing", default=CFG["routing"], choices=["hier", "center"])
    ap.add_argument("--slot-weight", type=float, default=CFG["slot_weight_per_contact"],
                    help="extra slot width per contact (0 = identical slots)")
    ap.add_argument("--edge-color", default=CFG["edge_color"], choices=["blend", "lobe", "neutral"])
    ap.add_argument("--legends", default=CFG["legends"], choices=["strip", "none"],
                    help="strip: legend band under each circos (default); none: no legend")
    ap.add_argument("--contact-labels", default=CFG["contact_labels"], choices=["full", "electrode"])
    ap.add_argument("--no-contact-edges", action="store_true",
                    help="contacts level: draw nodes and names only, no connections")
    ap.add_argument("--no-heatmaps", action="store_true")
    a = ap.parse_args()

    cfg = dict(CFG)
    cfg.update(subject_id=a.subject, exp_name=a.exp, formats=a.formats,
               base_dir=a.base_dir, yale_atlas_dir=a.atlas_dir,
               top_pcts=a.top_pcts, levels=a.levels, routing=a.routing,
               slot_weight_per_contact=a.slot_weight, edge_color=a.edge_color, legends=a.legends,
               contact_labels=a.contact_labels, contact_edges=not a.no_contact_edges)
    sid = cfg["subject_id"]
    out_dir = a.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output",
        f"{cfg['exp_name']}_YBA-updated_sub-{sid}")
    os.makedirs(out_dir, exist_ok=True)

    matrix, loc, yba = load_inputs(cfg)
    print("Building contact table ...")
    contacts = build_contact_table(loc, matrix, yba, out_dir, sid)
    print("Building the standardized layout ...")
    layout = build_layout(yba, contacts, cfg)
    contacts = place_contacts(layout, contacts, cfg)
    contacts.to_csv(os.path.join(out_dir, f"contact_layout_sub-{sid}.csv"), index=False)
    levels = build_levels(matrix, contacts, layout)

    for level in cfg["levels"]:
        nodes, A = levels[level]
        pd.DataFrame(A, index=nodes["label"], columns=nodes["label"]).to_csv(
            os.path.join(out_dir, f"raw_matrix_{level}_sub-{sid}.csv"))
        edges_all = edge_table(A)
        if edges_all.empty:
            print(f"  {level}: no connections, skipped")
            continue
        vmax = float(np.ceil(edges_all["w"].max()))
        print(f"{level}: {len(nodes)} nodes, {len(edges_all)} non-zero edges, "
              f"colour/width scale 0-{vmax:g}")
        for pct in cfg["top_pcts"]:
            if pct >= 100:
                edges, tag, ttl = edges_all, "", "all connections"
            else:
                thr = np.percentile(edges_all["w"], 100 - pct)
                edges = edges_all[edges_all["w"] >= thr]
                tag, ttl = f"_top{int(pct)}", f"top {int(pct)}% of connections"
            draw_circos(layout, nodes, edges, level, cfg, vmax,
                        f"YBA {level} - {ttl} (sub-{sid})",
                        os.path.join(out_dir, f"circos_{level}{tag}_sub-{sid}"))
        if not a.no_heatmaps:
            draw_heatmap(np.log1p(A), nodes, level, cfg, vmax,
                         f"YBA {level} connections (sub-{sid})",
                         os.path.join(out_dir, f"heatmap_{level}_sub-{sid}"))
    print("Done.")


if __name__ == "__main__":
    main()
