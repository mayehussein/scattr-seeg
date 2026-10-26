#!/bin/bash
#SBATCH --job-name=scattr_AllContactsACT_P224
#SBATCH --output=batch_logs/scattr_AllContactsACT_P224_%j.out
#SBATCH --error=batch_logs/scattr_AllContactsACT_P224_%j.err
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --partition=hx

mkdir -p batch_logs

# Load required modules
module load apptainer

# Subject to process (also update the three #SBATCH name lines above)
SUBJECT="P224"

# Set experiment name and directories
EXP_NAME="scattr_AllContacts_ACT_${SUBJECT}"
TMP_DIR="/tmp/${EXP_NAME}"
PERSISTENT_DIR="/nfs/khan/trainees/mkhaled5/scattr-seeg/output/${EXP_NAME}"
DATASET_SRC="/nfs/khan/trainees/mkhaled5/scattr-seeg/bids"
SCATTR_SRC="/nfs/khan/trainees/mkhaled5/scattr-seeg/scattr"

# Contacts segmentation parameters
SPHERE_RADIUS=1.5   # mm, maximum sphere radius
MIN_RADIUS=1.0      # mm, stop if contact spacing forces a radius below this
RADIAL_SEARCH=2     # mm, tck2connectome search; capture distance ~ radius + search = ~3.3 mm
HULL_MARGIN=5       # mm, local tckgen rejects streamlines going further than this outside the hull

# Set writable container cache directories
export SINGULARITY_CACHEDIR=/nfs/khan/trainees/mkhaled5/singularity_cache
export APPTAINER_CACHEDIR=/nfs/khan/trainees/mkhaled5/singularity_cache
export SINGULARITY_TMPDIR=/tmp/${EXP_NAME}_singularity_tmp
export APPTAINER_TMPDIR=/tmp/${EXP_NAME}_apptainer_tmp

mkdir -p ${SINGULARITY_CACHEDIR}
mkdir -p ${SINGULARITY_TMPDIR}
mkdir -p ${APPTAINER_TMPDIR}

# Start from a clean copy so no outputs from an earlier (crashed) run are reused
rm -rf ${TMP_DIR}

# Copy SCATTR pipeline and related data to /tmp
echo "Copying experiment to ${TMP_DIR}..."
mkdir -p ${TMP_DIR}
cp -r ${SCATTR_SRC} ${TMP_DIR}

# Sync bids folder preserving symlinks
rsync -a --links ${DATASET_SRC}/ ${TMP_DIR}/bids/

# -----------------------------------------------------------------------
# Step 1: Create sphere contact segmentation from the contacts file
# -----------------------------------------------------------------------
echo "Creating sphere contact segmentation (max R=${SPHERE_RADIUS}mm, min R=${MIN_RADIUS}mm)..."

SEG_SCRIPT="${TMP_DIR}/scattr/workflow/scripts/mrtpipelines/contacts_segmentation.py"
REF="${TMP_DIR}/bids/sub-${SUBJECT}/anat/sub-${SUBJECT}_desc-brain_T1w.nii.gz"
OUT_SEG="${TMP_DIR}/bids/sub-${SUBJECT}/anat/sub-${SUBJECT}_space-T1w_desc-AllContacts_dseg.nii.gz"
OUT_TSV="${TMP_DIR}/bids/sub-${SUBJECT}/anat/sub-${SUBJECT}_space-T1w_desc-AllContacts_dseg.tsv"

if [ ! -f "${SEG_SCRIPT}" ]; then
    echo "ERROR: ${SEG_SCRIPT} not found. Copy contacts_segmentation.py into ${SCATTR_SRC}/workflow/scripts/mrtpipelines/"
    exit 1
fi

# Find the contacts file in the subject's ieeg folder:
# sub-<SUBJECT>*_space-T1w*_electrodes.fcsv, or .tsv if there is no FCSV.
# Stops if none is found, or if more than one file of that type matches.
IEEG_DIR="${TMP_DIR}/bids/sub-${SUBJECT}/ieeg"
CONTACTS=""
shopt -s nullglob
for ext in fcsv tsv; do
    matches=( "${IEEG_DIR}"/sub-${SUBJECT}*_space-T1w*_electrodes.${ext} )
    if [ ${#matches[@]} -gt 1 ]; then
        echo "ERROR: more than one .${ext} contacts file for sub-${SUBJECT}:"
        printf '    %s\n' "${matches[@]}"
        exit 1
    elif [ ${#matches[@]} -eq 1 ]; then
        CONTACTS="${matches[0]}"
        break
    fi
done
shopt -u nullglob
if [ -z "${CONTACTS}" ]; then
    echo "ERROR: no sub-${SUBJECT}*_space-T1w*_electrodes.fcsv or .tsv in ${IEEG_DIR}"
    exit 1
fi
echo "Contacts file: ${CONTACTS}"

# Remove the old segmentation copied in from bids, so it can never
# be used if the new one isn't written
rm -f "${OUT_SEG}" "${OUT_TSV}"

python3 "${SEG_SCRIPT}" \
    --contacts "${CONTACTS}" \
    --ref "${REF}" \
    --output-seg "${OUT_SEG}" \
    --output-tsv "${OUT_TSV}" \
    --radius ${SPHERE_RADIUS} \
    --min-radius ${MIN_RADIUS}

if [ $? -ne 0 ] || [ ! -f "${OUT_SEG}" ]; then
    echo "ERROR: Sphere segmentation failed!"
    exit 1
fi

echo "Sphere segmentation created successfully."

# If the contacts came from a TSV, the script also wrote an FCSV next to it in
# the /tmp ieeg folder; copy it to the persistent ieeg folder (never overwrite)
if [[ "${CONTACTS}" == *.tsv ]]; then
    NEW_FCSV="${CONTACTS%.tsv}.fcsv"
    PERSIST_FCSV="${DATASET_SRC}/sub-${SUBJECT}/ieeg/$(basename "${NEW_FCSV}")"
    if [ -f "${PERSIST_FCSV}" ]; then
        echo "FCSV already in persistent ieeg folder, not overwritten: ${PERSIST_FCSV}"
    elif [ -f "${NEW_FCSV}" ]; then
        cp "${NEW_FCSV}" "${PERSIST_FCSV}"
        echo "Saved FCSV converted from TSV: ${PERSIST_FCSV}"
    fi
fi

# -----------------------------------------------------------------------
# Step 2: Run SCATTR pipeline
# Streamline endpoints are assigned to the nearest contact sphere within
# RADIAL_SEARCH mm: capture distance ~ radius + search (~1.3 + 2 = ~3.3 mm)
# perpendicular to the shaft, limited along the shaft to halfway between
# adjacent contacts.
# -----------------------------------------------------------------------
cd ${TMP_DIR}/scattr

# Resolve symlink targets for Singularity binds
echo "Resolving symlink targets for Singularity binds..."
SYMLINK_BINDS=""
while IFS= read -r -d '' link; do
    target=$(readlink -f "${link}")
    if [ -e "${target}" ]; then
        target_dir=$(dirname "${target}")
        if [[ ":${SYMLINK_BINDS}:" != *":${target_dir}:"* ]]; then
            SYMLINK_BINDS="${SYMLINK_BINDS} --bind ${target_dir}:${target_dir}"
        fi
    fi
done < <(find "${TMP_DIR}/bids" -type l -print0)
echo "Symlink bind mounts: ${SYMLINK_BINDS}"

# Activate virtual environment
source ${HOME}/scattr-venv/bin/activate

echo "Running SCATTR pipeline..."
python run.py ${TMP_DIR}/bids ${TMP_DIR}/output participant \
  --participant-label ${SUBJECT} \
  --force-output \
  --cores all \
  --freesurfer-dir "${TMP_DIR}/bids/derivatives/freesurfer/" \
  --keep-going \
  --rerun-incomplete \
  --fs-license "${HOME}/license.txt" \
  --use-singularity \
  --singularity-args="--bind ${HOME}/license.txt:/usr/local/freesurfer/.license,${HOME}/license.txt:/home/UWO/mkhaled5/license.txt ${SYMLINK_BINDS}" \
  --show-failed-logs \
  --labelmerge-base-dir "${TMP_DIR}/bids/" \
  --labelmerge-base-desc AllContacts \
  --radial-search ${RADIAL_SEARCH} \
  --hull-margin-mm ${HULL_MARGIN} \
  --skip-exclude-filter \
  --act \
  --passthrough-connectome \
  --skip-labelmerge \
  --skip-brainstem \
  --profile default 

# Copy results to persistent storage
echo "Copying results to ${PERSISTENT_DIR}..."
rsync -av \
  --exclude="*/unfiltered/" \
  --exclude="*/roi_masks/" \
  --exclude="*/exclude_mask/" \
  ${TMP_DIR}/output/ ${PERSISTENT_DIR}/

echo "Copying sphere segmentation back to persistent BIDS..."
cp "${OUT_SEG}" "${DATASET_SRC}/sub-${SUBJECT}/anat/"
cp "${OUT_TSV}" "${DATASET_SRC}/sub-${SUBJECT}/anat/"

# Clean up
cd /tmp
rm -rf ${TMP_DIR}

echo "Experiment ${EXP_NAME} completed and cleaned up!"
