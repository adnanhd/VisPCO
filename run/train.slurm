#!/usr/bin/env bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH --gres=gpu:rtxpro6000:2
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH -J vispco-train
#SBATCH --output=/scratch/aharun/slurm/metu__imagelab-denge__kalkan-akbas/scratchbook-symmetry/VisPCO/train-%j.out
# run_training.sh on 2 GPUs: batch 2 per GPU with update_freq 4 keeps its total batch of 16 (8 GPUs x 2).
set -euo pipefail
source "$SLURM_SUBMIT_DIR/run/common.sh"
OUT=$RES/train/
mkdir -p "$OUT"
nvidia-smi -L
"${RUN[@]}" torchrun --nproc_per_node=2 main.py --model_name qwen2.5vl-3b --model_path "$MODEL" \
    --trdata_path "$TRAINJSON" --valdata_path "$EVALJSON" --kernel_type linear --batch_size 2 --num_workers 12 \
    --pin_memory --seed 42 --opt adamw --lr 1e-4 --min_lr 1e-6 --weight_decay 0.0001 --warmup_steps 0.001 \
    --epochs 1000 --device cuda --precision bf16 --eval 1 --update_freq 4 --w 1 --sigma 100 --alpha 5 \
    --epsilon 0.01 --beta 0.5 --log_dir "$OUT" --save_ckpt_path "$OUT"
ls -la "$OUT"*
echo "DONE $(date +%T)"
