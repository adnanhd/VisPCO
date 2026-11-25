# Paths and container shared by the imagelab jobs; sourced by each run/*.slurm.
P=metu__imagelab-denge__kalkan-akbas/scratchbook-symmetry
DATA=/scratch/aharun/data/$P/VisPCO
RES=/scratch/aharun/results/$P/VisPCO
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SIF=/scratch/aharun/data/$P/containers/fastmmoe-a0c6d94.sif
MODEL=$DATA/weights/Qwen2.5-VL-3B-Instruct
EVALJSON=$DATA/eval/eval_max_100_dataset.json
TRAINJSON=$DATA/train/train_10000_uniform_area_single_image.json
RUN=(singularity exec --nv --cleanenv --bind /scratch/aharun --env PYTHONPATH="$DATA/pydeps" --env HF_HOME="$DATA/hf"
     --env PYTHONDONTWRITEBYTECODE=1 --pwd "$REPO" "$SIF")
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || { echo "STOP: tracked changes in $REPO"; exit 1; }
echo "code $REPO ($(git -C "$REPO" rev-parse --short HEAD)); image $SIF"; hostname
