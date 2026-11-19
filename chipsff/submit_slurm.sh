#!/bin/bash
# ---------------------------------------------------------------------------
# General SLURM template for CHIPS-FF benchmarks (edit the env-activation line).
#
#   Property table (single job):
#     MODEL=alignn_ff sbatch chipsff/submit_slurm.sh
#
#   WBM relaxation (array job -- one shard per array task):
#     MODEL=alignn_ff TASK=wbm NSHARD=490 \
#         sbatch --array=1-490%100 chipsff/submit_slurm.sh
#
#   WBM scoring + jarvis-leaderboard files (single job, after the array):
#     MODEL=alignn_ff TASK=wbm-score LB=/path/to/jarvis_leaderboard/jarvis_leaderboard \
#         sbatch chipsff/submit_slurm.sh
#
# Knobs (env): MODEL {alignn_ff|uma|matgl|chgnet|mace}, TASK {property|wbm|wbm-score},
#              NSHARD, DEVICE {cpu|cuda}, EXTRA (e.g. "--model_path /dir" or "--wbm_zip /z.zip"),
#              LB (jarvis_leaderboard dir for wbm-score).
# ---------------------------------------------------------------------------
#SBATCH --job-name=chipsff
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=chipsff_%x_%A_%a.out
set -e

MODEL=${MODEL:-alignn_ff}
TASK=${TASK:-property}
NSHARD=${NSHARD:-490}
DEVICE=${DEVICE:-cpu}
EXTRA=${EXTRA:-}
LB=${LB:-}

# --- activate your environment here (EDIT) -------------------------------
# source ~/miniforge3/etc/profile.d/conda.sh && conda activate chipsff
# -------------------------------------------------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} NSHARD=$NSHARD

case "$TASK" in
  property)
    python -m chipsff.run_main --"$MODEL" --device "$DEVICE" $EXTRA ;;
  wbm)
    python -m chipsff.run_main --"$MODEL" --wbm-relax --device "$DEVICE" \
        --nshard "$NSHARD" $EXTRA ;;
  wbm-score)
    python -m chipsff.run_main --"$MODEL" --wbm-score --device "$DEVICE" \
        ${LB:+--leaderboard "$LB"} $EXTRA ;;
  diatomics)
    python -m chipsff.run_main --"$MODEL" --diatomics --device "$DEVICE" $EXTRA ;;
  scaling)
    python -m chipsff.run_main --"$MODEL" --scaling --device "$DEVICE" $EXTRA ;;
  *)
    echo "unknown TASK=$TASK (property|wbm|wbm-score|diatomics|scaling)"; exit 1 ;;
esac
