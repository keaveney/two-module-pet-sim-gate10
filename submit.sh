#!/bin/bash
#SBATCH --job-name=student-pet
#SBATCH --array=0-1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=student-%A_%a.out
#SBATCH --error=student-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from student-two-module}"
source "${GATE_VENV:-/home/people/keaveneyj/venvs/py312-opengate-env}/bin/activate"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/keaveneyj/student-two-module}"
OUTDIR="$OUTPUT_BASE/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SEED=$(( ${BASE_SEED:-12345} + SLURM_ARRAY_TASK_ID ))
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp/$USER}/matplotlib-cache"
python simulate.py --events "${EVENTS_PER_THREAD:-100}" \
  --threads "$SLURM_CPUS_PER_TASK" --seed "$SEED" \
  --pixel-length-mm "${PIXEL_LENGTH_MM:-15}" --output-dir "$OUTDIR"
python analyse.py --input-dir "$OUTDIR" --pde "${PDE:-0.30}" --seed "$SEED"
