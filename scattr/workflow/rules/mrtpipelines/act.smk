rule gen_5tt:
    """
    Generate a 5-tissue-type image from the Freesurfer subject directory for
    anatomically-constrained tractography. Hippocampi and amygdalae are
    always labelled as sub-cortical grey matter by hsvs. The aseg options are
    used because FSL (FIRST) is not available in the container.

    Smith, R. E.; Tournier, J.-D.; Calamante, F. & Connelly, A.
    Anatomically-constrained tractography: Improved diffusion MRI streamlines
    tractography through effective use of anatomical information.
    NeuroImage, 2012, 62, 1924-1938

    Smith, R.; Skoch, A.; Bajada, C.; Caspers, S.; Connelly, A. Hybrid
    Surface-Volume Segmentation for improved Anatomically-Constrained
    Tractography. In Proc OHBM 2020
    """
    input:
        fs_subj_dir=str(Path(bids(root=freesurfer_dir, **inputs.subj_wildcards)).parent),
    params:
        fs_license=fs_license,
    output:
        tt5=bids_anat_out(desc="hsvs", suffix="5tt.nii.gz"),
    threads: 8
    resources:
        tmp_dir=lambda wildcards: bids_anat_out(
            root=os.environ.get("SLURM_TMPDIR")
            if config.get("slurm_tmpdir")
            else "/tmp",
            desc="hsvs",
            **wildcards,
        ),
        mem_mb=16000,
        time=90,
    log:
        bids_log(suffix="5ttgen.log"),
    container:
        config["singularity"]["scattr"]
    shell:
        """
        export FS_LICENSE={params.fs_license}
        mkdir -p {resources.tmp_dir}

        5ttgen hsvs {input.fs_subj_dir} {output.tt5} \\
            -hippocampi aseg -thalami aseg \\
            -nthreads {threads} -scratch {resources.tmp_dir} &> {log}

        5ttcheck {output.tt5} >> {log} 2>&1
        """


rule contact_tissue:
    """Tissue composition of each node (fraction of each 5TT tissue type)"""
    input:
        tt5=rules.gen_5tt.output.tt5,
        seg=rules.get_num_nodes.input.seg,
    params:
        seg_tsv=lambda wildcards, input: input.seg.replace(
            "dseg.nii.gz", "dseg.tsv"
        ),
    output:
        tsv=bids_anat_out(desc="nodes", suffix="tissue.tsv"),
    threads: 1
    resources:
        mem_mb=16000,
        time=30,
    log:
        bids_log(suffix="contactTissue.log"),
    container:
        config["singularity"]["scattr"]
    script:
        "../../scripts/mrtpipelines/contact_tissue.py"
