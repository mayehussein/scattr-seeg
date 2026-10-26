#!/usr/bin/env python
"""
passthrough_connectome.py

Build a connectome in which a streamline is linked to a node if one of its
endpoints is assigned to the node (endpoint assignments from tck2connectome)
OR it intersects the node (assignments from tck2connectome
-assignment_all_voxels). Each streamline adds its weight to every pair of
distinct nodes it is linked to, so a streamline linked to n nodes contributes
to n*(n-1)/2 edges and the matrix total is not a streamline total.

Both assignment files have one line per streamline (after a "#" header);
a line of "0" means no node.
"""
from itertools import combinations

import nibabel as nib
import numpy as np


def read_weights(path):
    with open(path) as f:
        values = [line for line in f if not line.startswith("#")]
    return np.array(" ".join(values).split(), dtype=np.float64)


def passthrough_connectome(endpoint_path, allvoxels_path, weights_path, seg_path, out_csv):
    weights = read_weights(weights_path)
    n_nodes = int(np.asarray(nib.load(seg_path).dataobj).max())

    rows, cols, vals = [], [], []
    n_streamlines = n_linked = n_passthrough_only = 0
    with open(endpoint_path) as f_ep, open(allvoxels_path) as f_av:
        for line_ep, line_av in zip(f_ep, f_av):
            if line_ep.startswith("#"):
                if not line_av.startswith("#"):
                    raise ValueError("Assignment file headers do not line up")
                continue
            idx = n_streamlines
            n_streamlines += 1
            if line_av == "0\n" and line_ep in ("0 0\n", "0 0"):
                continue
            ends = {int(x) for x in line_ep.split()} - {0}
            nodes = ends | ({int(x) for x in line_av.split()} - {0})
            if len(nodes) < 2:
                continue
            n_linked += 1
            if len(ends) < 2:
                n_passthrough_only += 1
            w = weights[idx]
            for i, j in combinations(sorted(nodes), 2):
                rows.append(i)
                cols.append(j)
                vals.append(w)

    if n_streamlines != len(weights):
        raise ValueError(
            f"{n_streamlines} assignments but {len(weights)} streamline weights"
        )

    mat = np.zeros((n_nodes + 1, n_nodes + 1))
    np.add.at(mat, (np.array(rows, dtype=int), np.array(cols, dtype=int)), vals)
    mat = mat + mat.T
    mat = mat[1:, 1:]
    np.savetxt(out_csv, mat, delimiter=",", fmt="%.6g")

    print(f"Streamlines: {n_streamlines}")
    print(f"Linked to >=2 nodes: {n_linked}")
    print(f"  ... of which not linked by both endpoints: {n_passthrough_only}")
    print(f"Non-zero edges: {int((mat[np.triu_indices(n_nodes, 1)] > 0).sum())}")


if __name__ == "__main__":
    with open(snakemake.log[0], "w") as log_fh:
        import sys

        sys.stdout = log_fh
        sys.stderr = log_fh
        passthrough_connectome(
            endpoint_path=snakemake.input.endpoint_assignment,
            allvoxels_path=snakemake.input.allvoxels_assignment,
            weights_path=snakemake.input.weights,
            seg_path=snakemake.input.subcortical_seg,
            out_csv=snakemake.output.node_weights,
        )
