#!/usr/bin/env python
import nibabel as nib
import numpy as np
import scipy.ndimage
import scipy.spatial


def create_convex_hull(
    binary_seg, out_convex_hull, out_convex_hull_margin, margin_mm
):
    # Load input
    print("Loading data...")
    seg = nib.load(binary_seg)
    seg_affine = seg.affine
    seg_data = seg.get_fdata()

    # Compute convex hull
    print("Computing convex hull...")
    points = np.transpose(np.where(seg_data))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(seg_data.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.ones(seg_data.shape)
    out_img[out_idx] = 0

    # Save convex hull
    print("Saving convex hull...")
    out_hull = nib.Nifti1Image(out_img, seg_affine)
    nib.save(out_hull, out_convex_hull)

    # Hull with margin: 1 = more than margin_mm outside the hull. Distance is
    # computed in mm so the margin is the same along anisotropic voxel axes.
    print(f"Computing convex hull with {margin_mm} mm margin...")
    zooms = seg.header.get_zooms()[:3]
    dist = scipy.ndimage.distance_transform_edt(out_img > 0, sampling=zooms)
    out_margin = (dist > margin_mm).astype(np.float64)
    nib.save(nib.Nifti1Image(out_margin, seg_affine), out_convex_hull_margin)


if __name__ == "__main__":
    create_convex_hull(
        binary_seg=snakemake.input.bin_seg,
        out_convex_hull=snakemake.output.convex_hull,
        out_convex_hull_margin=snakemake.output.convex_hull_margin,
        margin_mm=float(snakemake.params.margin_mm),
    )
