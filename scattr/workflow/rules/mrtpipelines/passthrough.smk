rule tck2connectome_allvoxels:
    """
    Assign each streamline to every node it intersects along its length
    (-assignment_all_voxels). The assignments are combined with the endpoint
    assignments of tck2connectome in passthrough_connectome; the matrix
    written here counts node pairs intersected by the same streamline.
    """
    input:
        weights=rules.tcksift2.output.weights,
        tck=rules.tckedit_combine.output.tck,
        subcortical_seg=rules.tckgen_wholebrain.input.subcortical_seg,
    output:
        sl_assignment=bids_tractography_out(
            desc="allvoxels",
            suffix="nodeAssignment.txt",
        ),
        node_weights=bids_tractography_out(
            desc="allvoxels",
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
            desc="allvoxels",
            suffix="nodeAssignment.txt",
            **wildcards,
        ),
        tmp_node_weights=lambda wildcards: bids_tractography_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            desc="allvoxels",
            suffix="nodeWeights.csv",
            **wildcards,
        ),
        mem_mb=128000,
        time=60 * 3,
    log:
        bids_log(suffix="tck2connectome_allvoxels.log"),
    container:
        config["singularity"]["scattr"]
    shell:
        """
        mkdir -p {resources.tmp_dir}

        tck2connectome -nthreads {threads} -zero_diagonal -stat_edge sum \\
        -assignment_all_voxels \\
        -tck_weights_in {input.weights} \\
        -out_assignments {resources.tmp_sl_assignment} \\
        -symmetric {input.tck} {input.subcortical_seg} \\
        {resources.tmp_node_weights} -force &> {log}

        rsync {resources.tmp_sl_assignment} \\
        {output.sl_assignment} >> {log} 2>&1

        rsync {resources.tmp_node_weights} {output.node_weights} >> {log} 2>&1
        """


rule passthrough_connectome:
    """
    Connectome in which a streamline is linked to a node if one of its
    endpoints is assigned to the node (radial search, as in tck2connectome)
    OR it passes through the node. Each streamline adds its SIFT2 weight to
    every pair of nodes it is linked to.
    """
    input:
        endpoint_assignment=rules.tck2connectome.output.sl_assignment,
        allvoxels_assignment=rules.tck2connectome_allvoxels.output.sl_assignment,
        weights=rules.tcksift2.output.weights,
        subcortical_seg=rules.tckgen_wholebrain.input.subcortical_seg,
    output:
        node_weights=bids_tractography_out(
            desc="traversal",
            suffix="nodeWeights.csv",
        ),
    threads: 1
    resources:
        mem_mb=32000,
        time=60 * 2,
    log:
        bids_log(suffix="passthrough_connectome.log"),
    container:
        config["singularity"]["scattr"]
    script:
        "../../scripts/mrtpipelines/passthrough_connectome.py"
