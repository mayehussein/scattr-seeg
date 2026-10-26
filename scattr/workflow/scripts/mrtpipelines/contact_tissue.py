#!/usr/bin/env python
"""
contact_tissue.py

For each node (label) of the segmentation, report the mean fraction of each
5TT tissue type over the node's voxels, and its dominant tissue. Node voxels
are mapped to the 5TT grid by world coordinates (nearest voxel), so the two
images may have different grids. Fractions not covered by any tissue (e.g.
outside the brain) are reported as "outside".
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

TISSUES = ["cGM", "sGM", "WM", "CSF", "pathology"]


def contact_tissue(tt5_path, seg_path, seg_tsv, out_tsv):
    tt5 = nib.load(tt5_path)
    tt5_data = np.asarray(tt5.dataobj, dtype=np.float32)
    seg = nib.load(seg_path)
    labels = np.asarray(seg.dataobj).astype(int)

    ijk = np.argwhere(labels > 0)
    xyz = nib.affines.apply_affine(seg.affine, ijk)
    tt5_ijk = np.rint(
        nib.affines.apply_affine(np.linalg.inv(tt5.affine), xyz)
    ).astype(int)
    in_fov = np.all((tt5_ijk >= 0) & (tt5_ijk < tt5_data.shape[:3]), axis=1)
    frac = np.zeros((len(ijk), len(TISSUES)), dtype=np.float32)
    frac[in_fov] = tt5_data[tuple(tt5_ijk[in_fov].T)]

    df = pd.DataFrame(frac, columns=TISSUES)
    df["index"] = labels[tuple(ijk.T)]
    out = df.groupby("index").mean()
    out["outside"] = (1 - out[TISSUES].sum(axis=1)).clip(lower=0)
    out["n_voxels"] = df.groupby("index").size()
    out["tissue"] = out[TISSUES + ["outside"]].idxmax(axis=1)

    if Path(seg_tsv).exists():
        names = pd.read_csv(seg_tsv, sep="\t").set_index("index")
        out = names.join(
            out.drop(columns=[c for c in out.columns if c in names.columns]),
            how="right",
        )

    out.round(3).to_csv(out_tsv, sep="\t")
    print(out["tissue"].value_counts().to_string())


if __name__ == "__main__":
    with open(snakemake.log[0], "w") as log_fh:
        import sys

        sys.stdout = log_fh
        sys.stderr = log_fh
        contact_tissue(
            tt5_path=snakemake.input.tt5,
            seg_path=snakemake.input.seg,
            seg_tsv=snakemake.params.seg_tsv,
            out_tsv=snakemake.output.tsv,
        )
