#!/usr/bin/env python
from pathlib import Path

import numpy as np
from snakebids import bids
from snakemake.io import expand
from snakemake.shell import shell


def create_exclude_mask(
    base_dir,
    roi1,
    roi2,
    subcortical_seg,
    num_labels,
    mask_dir,
    out_dir,
    wildcards,
    threads,
    skip_labelmerge=False,
):
    # Create output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Get number of labels
    with open(num_labels) as f:
        num_labels = int(f.read().strip())

    node_pairs = np.triu_indices(num_labels, k=1)

    # Zona incerta labels — only relevant when using the ZonaBB atlas
    # (labelmerge). When using SEEG contacts (skip_labelmerge=True),
    # labels 21/22 are regular contacts, not zona incerta, and should
    # NOT be subtracted from the exclusion mask.
    zi_subtraction = ""
    if not skip_labelmerge:
        lZI = expand(  # noqa: N806
            bids(
                root=base_dir,
                datatype="roi_masks",
                desc="21",
                suffix="mask.mif",
                **wildcards,
            )
        )
        rZI = bids(  # noqa: N806
            root=base_dir,
            datatype="roi_masks",
            desc="22",
            suffix="mask.mif",
            **wildcards,
        )
        zi_subtraction = f"{lZI} -sub {rZI} -sub"

    # Create mask sequentially
    out_mask = expand(
        bids(
            root=base_dir,
            datatype="exclude_mask",
            desc="{desc}",
            suffix="mask.mif",
            **wildcards,
        ),
        desc=[
            f"from{node_pairs[0][idx]+1}-{node_pairs[1][idx]+1}"
            for idx in range(len(node_pairs[0]))
        ],
    )

    # Create masks - run sequentially due to parallelization errors
    for idx, mask in enumerate(out_mask):
        in_roi1, in_roi2 = roi1[idx], roi2[idx]  # (used in shell)
        if zi_subtraction:
            shell(
                "mrcalc -nthreads {threads} "
                "{subcortical_seg} 0 -neq {in_roi1} -sub {in_roi2} -sub "
                "{zi_subtraction} {mask} -force"
            )
        else:
            shell(
                "mrcalc -nthreads {threads} "
                "{subcortical_seg} 0 -neq {in_roi1} -sub {in_roi2} -sub "
                "{mask} -force"
            )


if __name__ == "__main__":
    create_exclude_mask(
        base_dir=snakemake.params.base_dir,
        roi1=snakemake.input.roi1,
        roi2=snakemake.input.roi2,
        subcortical_seg=snakemake.input.subcortical_seg,
        num_labels=snakemake.input.num_labels,
        mask_dir=snakemake.params.mask_dir,
        out_dir=snakemake.output.out_dir,
        wildcards=snakemake.wildcards,
        threads=snakemake.threads,
        skip_labelmerge=snakemake.params.get("skip_labelmerge", False),
    )

