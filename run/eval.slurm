#!/usr/bin/env bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH -J vispco-eval
#SBATCH --output=/scratch/aharun/slurm/metu__imagelab-denge__kalkan-akbas/scratchbook-symmetry/VisPCO/eval-%j.out
# eval_ours.py at 50% budget. One configuration per job: CKPT=<predictor .pth or glob>, FIXED_K=<k>, or R_FILE=<json>.
set -euo pipefail
: "${NAME:?}"
source "$SLURM_SUBMIT_DIR/run/common.sh"
OUT=$RES/eval/$NAME.jsonl
[ ! -e "$OUT" ] || { echo "STOP: $OUT exists"; exit 1; }
mkdir -p "$(dirname "$OUT")"
ARGS=()
if [ -n "${CKPT:-}" ]; then CKPT=$(ls $CKPT | sort -V | tail -1); ARGS=(--trained_predict_pruning_ratio_path "$CKPT")
elif [ -n "${FIXED_K:-}" ]; then ARGS=(--fixed_k "$FIXED_K")
elif [ -n "${R_FILE:-}" ]; then ARGS=(--fixed_r_file "$R_FILE")
else echo "STOP: CKPT, FIXED_K or R_FILE"; exit 1; fi
echo "NAME=$NAME ${ARGS[*]}"; nvidia-smi -L
"${RUN[@]}" python eval_ours.py --model_name qwen2.5vl-3b --model_path "$MODEL" --valdata_path "$EVALJSON" \
    --out_file "$OUT" --ratio 0.5 "${ARGS[@]}"
cat "$OUT"
echo "DONE $(date +%T)"
