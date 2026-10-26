rule tckgen_wholebrain:
    """
    Whole-brain tractography (no convex-hull constraint, no -include), so
    the tractogram covers the whole brain for SIFT2 and captures long-range
    connections. Seeds are placed randomly across the brain mask, or with
    --act placed dynamically (-seed_dynamic) and tracked with ACT.

    Tournier, J.-D.; Calamante, F. & Connelly, A. Improved probabilistic
    streamlines tractography by 2nd order integration over fibre orientation
    distributions. Proceedings of the International Society for Magnetic
    Resonance in Medicine, 2010, 1670

    Smith, R. E.; Tournier, J.-D.; Calamante, F. & Connelly, A. SIFT2:
    Enabling dense quantitative assessment of brain white matter connectivity
    using streamlines tractography. NeuroImage, 2015, 119, 338-351
    """
    input:
        fod=rules.mtnormalise.output.wm_fod,
        mask=rules.nii2mif.output.mask,
        convex_hull=rules.create_convex_hull.output.convex_hull,
        subcortical_seg=rules.get_num_nodes.input.seg,
        tt5=rules.gen_5tt.output.tt5 if config.get("act") else [],
    params:
        step=config["step"],
        sl=config["sl_count"],
        act_opts=lambda wildcards, input: (
            f"-act {input.tt5} -backtrack -crop_at_gmwmi -maxlength 250"
            if config.get("act")
            else ""
        ),
        seed_opts=lambda wildcards, input: (
            f"-seed_dynamic {input.fod}"
            if config.get("act")
            else f"-seed_image {input.mask}"
        ),
    output:
        tck=bids_tractography_out(
            desc="iFOD2wholebrain",
            suffix="tractography.tck",
        ),
    threads: 32
    resources:
        tmp_dir=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            **wildcards,
        ),
        tmp_tck=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            desc="iFOD2wholebrain",
            suffix="tractography.tck",
            **wildcards,
        ),
        mem_mb=128000,
        time=60 * 24,
    log:
        bids_log(suffix="tckgen_wholebrain.log"),
    container:
        config["singularity"]["scattr"]
    shell:
        """
        mkdir -p {resources.tmp_dir}

        tckgen -nthreads {threads} -algorithm iFOD2 -step {params.step} \\
            -select {params.sl} {params.act_opts} \\
            -mask {input.mask} \\
            {params.seed_opts} {input.fod} {resources.tmp_tck} &> {log}

        rsync -v {resources.tmp_tck} {output.tck} >> {log} 2>&1
        """


# Local tckgen command: seeded within the convex hull interior (default), or
# without the convex hull (--skip_convex_hull)
tckgen_local_hull_shell = """
        mkdir -p {resources.tmp_dir}

        # Hull interior for seeding: 0=inside -> 1, 1=outside -> 0
        mrcalc {input.convex_hull} 0 -eq {resources.interior_mask} \\
            -force &> {log}

        tckgen -nthreads {threads} -algorithm iFOD2 -step {params.step} \\
            -select {params.sl} {params.act_opts} \\
            -include {input.subcortical_seg} \\
            -mask {input.mask} \\
            -exclude {input.convex_hull_margin} \\
            -seed_image {resources.interior_mask} \\
            {input.fod} {resources.tmp_tck} >> {log} 2>&1

        rsync -v {resources.tmp_tck} {output.tck} >> {log} 2>&1
        """

tckgen_local_nohull_shell = """
        mkdir -p {resources.tmp_dir}

        tckgen -nthreads {threads} -algorithm iFOD2 -step {params.step} \\
            -select {params.sl} {params.act_opts} \\
            -include {input.subcortical_seg} \\
            -mask {input.mask} \\
            -seed_image {input.mask} \\
            {input.fod} {resources.tmp_tck} &> {log}

        rsync -v {resources.tmp_tck} {output.tck} >> {log} 2>&1
        """


rule tckgen_local:
    """
    Local tractography seeded within the convex hull interior.
    The convex hull image stores 0=inside and 1=outside, so the interior
    (1=inside) used for seeding is obtained with mrcalc.
    Streamlines may travel anywhere within the brain mask, but are rejected
    (-exclude) if they go more than hull_margin_mm outside the hull, rather
    than being truncated at the hull boundary. The margin keeps streamlines
    that run a few mm past the outermost nodes; a margin of 0 matches the
    exact-hull exclusion of the original SCATTR. With --act, streamlines are
    also tracked with ACT.

    With --skip_convex_hull, the convex hull is not used: seeds are placed
    uniformly within the brain mask and only streamlines that traverse a
    node (-include) are kept, i.e. a whole-brain tractogram restricted to
    streamlines touching the nodes. More streamlines are rejected, so this
    run uses all available cores (it then runs after, not alongside, the
    whole-brain run).

    Tournier, J.-D.; Calamante, F. & Connelly, A. Improved probabilistic
    streamlines tractography by 2nd order integration over fibre orientation
    distributions. Proceedings of the International Society for Magnetic
    Resonance in Medicine, 2010, 1670
    """
    input:
        fod=rules.mtnormalise.output.wm_fod,
        mask=rules.nii2mif.output.mask,
        convex_hull=rules.create_convex_hull.output.convex_hull
        if not config.get("skip_convex_hull")
        else [],
        convex_hull_margin=rules.create_convex_hull.output.convex_hull_margin
        if not config.get("skip_convex_hull")
        else [],
        subcortical_seg=rules.get_num_nodes.input.seg,
        tt5=rules.gen_5tt.output.tt5 if config.get("act") else [],
    params:
        step=config["step"],
        sl=config["sl_count_local"],
        act_opts=lambda wildcards, input: (
            f"-act {input.tt5} -backtrack -crop_at_gmwmi -maxlength 250"
            if config.get("act")
            else ""
        ),
    output:
        tck=bids_tractography_out(
            desc="iFOD2local",
            suffix="tractography.tck",
        ),
    threads: workflow.cores if config.get("skip_convex_hull") else 16
    resources:
        tmp_dir=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            **wildcards,
        ),
        tmp_tck=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            desc="iFOD2local",
            suffix="tractography.tck",
            **wildcards,
        ),
        interior_mask=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            desc="convexHullInterior",
            suffix="mask.mif",
            **wildcards,
        ),
        mem_mb=64000,
        time=60 * 12,
    log:
        bids_log(suffix="tckgen_local.log"),
    container:
        config["singularity"]["scattr"]
    shell:
        tckgen_local_nohull_shell
        if config.get("skip_convex_hull")
        else tckgen_local_hull_shell


rule tckedit_combine:
    """
    Concatenate whole-brain and local tractograms into a single .tck file
    for downstream SIFT2 weighting and connectome construction.
    """
    input:
        tck_wholebrain=rules.tckgen_wholebrain.output.tck,
        tck_local=rules.tckgen_local.output.tck,
    output:
        tck=bids_tractography_out(
            desc="iFOD2",
            suffix="tractography.tck",
        ),
    threads: 4
    resources:
        tmp_dir=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            **wildcards,
        ),
        tmp_tck=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            desc="iFOD2",
            suffix="tractography.tck",
            **wildcards,
        ),
        mem_mb=32000,
        time=60 * 2,
    log:
        bids_log(suffix="tckedit_combine.log"),
    container:
        config["singularity"]["scattr"]
    shell:
        """
        mkdir -p {resources.tmp_dir}

        tckedit {input.tck_wholebrain} {input.tck_local} \\
            {resources.tmp_tck} &> {log}

        rsync -v {resources.tmp_tck} {output.tck} >> {log} 2>&1
        """


rule tcksift2:
    """
    Smith, R. E.; Tournier, J.-D.; Calamante, F. & Connelly, A. The effects 
    of SIFT on the reproducibility and biological accuracy of the structural 
    connectome. NeuroImage, 2015, 104, 253-265
    """
    input:
        tck=rules.tckedit_combine.output.tck,
        fod=rules.mtnormalise.output.wm_fod,
        tt5=rules.gen_5tt.output.tt5 if config.get("act") else [],
    params:
        act_opts=lambda wildcards, input: (
            f"-act {input.tt5}" if config.get("act") else ""
        ),
    output:
        weights=bids_tractography_out(
            desc="iFOD2",
            suffix="tckWeights.txt",
        ),
        mu=bids_tractography_out(desc="iFOD2", suffix="muCoefficient.txt"),
    threads: 8
    resources:
        mem_mb=32000,
        time=60 * 2,
    log:
        bids_log(suffix="tcksift2.log"),
    container:
        config["singularity"]["scattr"]
    shell:
        """
        tcksift2 -nthreads {threads} -out_mu {output.mu} {params.act_opts} \\
            {input.tck} {input.fod} {output.weights} &> {log}
        """


checkpoint create_roi_mask:
    input:
        subcortical_seg=rules.tckgen_wholebrain.input.subcortical_seg,
        num_labels=rules.get_num_nodes.output.num_labels,
    params:
        base_dir=mrtrix_dir,
        subj_wildcards=inputs.subj_wildcards,
    output:
        out_dir=directory(bids_anat_out(datatype="roi_masks")),
    threads: 4
    resources:
        mem_mb=16000,
        time=60,
    group:
        "tract_masks"
    container:
        config["singularity"]["scattr"]
    script:
        "../../scripts/mrtpipelines/create_roi_mask.py"


def aggregate_rois(wildcards):
    """Grab all created roi masks"""
    roi_masks = bids_anat_out(
        datatype="roi_masks",
        desc="{node}",
        suffix="mask.mif",
        **wildcards,
    )
    # Get node label wildcard
    node = glob_wildcards(roi_masks).node

    # Build node pairs
    node_pairs = np.triu_indices(len(node), k=1)

    return {
        "roi1": expand(
            bids_anat_out(
                datatype="roi_masks",
                desc="{node1}",
                suffix="mask.mif",
            ),
            node1=list(node_pairs[0] + 1),
            allow_missing=True,
        ),
        "roi2": expand(
            bids_anat_out(
                datatype="roi_masks",
                desc="{node2}",
                suffix="mask.mif",
            ),
            node2=list(node_pairs[1] + 1),
            allow_missing=True,
        ),
    }


checkpoint create_exclude_mask:
    input:
        unpack(aggregate_rois),
        rules.create_roi_mask.output.out_dir,
        subcortical_seg=rules.tckgen_wholebrain.input.subcortical_seg,
        num_labels=rules.get_num_nodes.output.num_labels,
    params:
        base_dir=mrtrix_dir,
        mask_dir=bids_anat_out(
            datatype="roi_masks",
        ),
        subj_wildcards=inputs.subj_wildcards,
        skip_labelmerge=config.get("skip_labelmerge", False),
    output:
        out_dir=directory(bids_anat_out(datatype="exclude_mask")),
    threads: 4
    resources:
        mem_mb=16000,
        time=60 * 3,
    group:
        "tract_masks"
    container:
        config["singularity"]["scattr"]
    script:
        "../../scripts/mrtpipelines/create_exclude_mask.py"


# TODO (v0.2): ADD OPTION TO OUTPUT TDI MAP


rule tck2connectome:
    """
    Smith, R. E.; Tournier, J.-D.; Calamante, F. & Connelly, A. The 
    effects of SIFT on the reproducibility and biological accuracy of 
    the structural connectome. NeuroImage, 2015, 104, 253-265

    TODO (v0.2): ADD OPTION TO MULTIPLY BY MU COEFFICIENT
    """
    input:
        weights=rules.tcksift2.output.weights,
        tck=rules.tckedit_combine.output.tck,
        subcortical_seg=rules.tckgen_wholebrain.input.subcortical_seg,
    params:
        radius=config["radial_search"],
    output:
        sl_assignment=bids_tractography_out(
            desc="subcortical",
            suffix="nodeAssignment.txt",
        ),
        node_weights=bids_tractography_out(
            desc="subcortical",
            suffix="nodeWeights.csv",
        ),
    threads: 32
    resources:
        tmp_dir=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            **wildcards,
        ),
        tmp_sl_assignment=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            desc="subcortical",
            suffix="nodeAssignment.txt",
            **wildcards,
        ),
        tmp_node_weights=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            desc="subcortical",
            suffix="nodeWeights.csv",
            **wildcards,
        ),
        mem_mb=128000,
        time=60 * 3,
    log:
        bids_log(suffix="tck2connectome.log"),
    group:
        "tractography_update"
    container:
        config["singularity"]["scattr"]
    shell:
        """
        mkdir -p {resources.tmp_dir}

        tck2connectome -nthreads {threads} -zero_diagonal -stat_edge sum \\
        -assignment_radial_search {params.radius} \\
        -tck_weights_in {input.weights} \\
        -out_assignments {resources.tmp_sl_assignment} \\
        -symmetric {input.tck} {input.subcortical_seg} \\
        {resources.tmp_node_weights} &> {log}

        rsync {resources.tmp_sl_assignment} \\
        {output.sl_assignment} >> {log} 2>&1 

        rsync {resources.tmp_node_weights} {output.node_weights} >> {log} 2>&1
        """


checkpoint connectome2tck:
    input:
        node_weights=rules.tcksift2.output.weights,
        sl_assignment=rules.tck2connectome.output.sl_assignment,
        tck=rules.tckedit_combine.output.tck,
        num_labels=rules.get_num_nodes.output.num_labels,
    output:
        output_dir=directory(
            str(
                Path(
                    bids_tractography_out(
                        datatype="unfiltered",
                    )
                ).parent
            )
        ),
    threads: 32
    resources:
        tmp_dir=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            datatype="unfiltered",
            **wildcards,
        ),
        edge_weight_prefix=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            datatype="unfiltered",
            desc="subcortical",
            suffix="tckWeights",
            **wildcards,
        ),
        edge_tck_prefix=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            datatype="unfiltered",
            desc="subcortical",
            suffix="from",
            **wildcards,
        ),
        mem_mb=128000,
        time=60 * 3,
    log:
        bids_log(suffix="connectome2tck.log"),
    group:
        "tractography_update"
    container:
        config["singularity"]["scattr"]
    shell:
        """
        mkdir -p {resources.tmp_dir} {output.output_dir}

        num_labels=$(cat {input.num_labels})

        connectome2tck -nthreads {threads} \\
        -nodes `seq -s, 1 $((num_labels))` \\
        -exclusive -files per_edge -tck_weights_in {input.node_weights} \\
        -prefix_tck_weights_out {resources.edge_weight_prefix} \\
        {input.tck} {input.sl_assignment} \\
        {resources.edge_tck_prefix} &> {log}

        rsync {resources.edge_weight_prefix}*.csv \\
        {output.output_dir}/ >> {log} 2>&1
        rsync {resources.edge_tck_prefix}*.tck \\
        {output.output_dir}/ >> {log} 2>&1
        """
