#!/usr/bin/env bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH -J vispco-setup
#SBATCH --output=/scratch/aharun/slurm/metu__imagelab-denge__kalkan-akbas/scratchbook-symmetry/VisPCO/setup-%j.out
# Weights, eval TSVs, LLaVA-Instruct + COCO train2017, python extras; eval json (100 per benchmark, seed 42),
# train json (10000 LLaVA samples, seed 42, uniform image area as dataset/uniform_image_area.py).
set -euo pipefail
source "$SLURM_SUBMIT_DIR/run/common.sh"
mkdir -p "$DATA"/{weights,lmudata,llava,coco,eval,train,pydeps,hf}
"${RUN[@]}" python -m pip install -q --no-deps --target "$DATA/pydeps" thop==0.1.1.post2209072238 timm==1.0.15 tensorboard==2.19.0 tensorboard-data-server==0.7.2 werkzeug==3.1.3 markdown==3.8.2 absl-py==2.3.1 grpcio==1.76.0
"${RUN[@]}" python -c "import torch, transformers, torchvision, thop, timm, torch.utils.tensorboard; print('import', torch.__version__, transformers.__version__, torchvision.__version__)"
"${RUN[@]}" python -c "from huggingface_hub import snapshot_download as s, hf_hub_download as h; s('Qwen/Qwen2.5-VL-3B-Instruct', local_dir='$MODEL'); h('liuhaotian/LLaVA-Instruct-150K', 'llava_instruct_150k.json', repo_type='dataset', local_dir='$DATA/llava')"
LMU=/scratch/aharun/data/$P/lmudata
for t in A-OKVQA VizWiz SEEDBench_IMG MMBench_DEV_CN MME; do
    [ -s "$DATA/lmudata/$t.tsv" ] || curl -fsSL -o "$DATA/lmudata/$t.tsv" "https://opencompass.openxlab.space/utils/VLMEval/$t.tsv"
done
for t in ChartQA_TEST OCRBench TextVQA_VAL; do ln -sfn "$LMU/$t.tsv" "$DATA/lmudata/$t.tsv"; done
md5sum "$DATA"/lmudata/*.tsv
[ -d "$DATA/coco/train2017" ] || { curl -fsSL -o "$DATA/coco/train2017.zip" http://images.cocodataset.org/zips/train2017.zip
    (cd "$DATA/coco" && unzip -q train2017.zip && rm train2017.zip); }
ls "$DATA/coco/train2017" | wc -l
"${RUN[@]}" python dataset/build_data.py eval --lmudata "$DATA/lmudata" --out "$DATA/eval" --n 100
"${RUN[@]}" python dataset/build_data.py train --llava_json "$DATA/llava/llava_instruct_150k.json" --coco_dir "$DATA/coco/train2017" --out "$DATA/train/raw_10000.json" --n 10000
"${RUN[@]}" python dataset/uniform_image_area.py --json_input_path "$DATA/train/raw_10000.json" --output_base_dir "$DATA/train/images" --json_output_path "$DATA/train/grouped_10000.json"
"${RUN[@]}" python -c "import json; d=json.load(open('$DATA/train/grouped_10000.json'))['llava']; json.dump(d, open('$TRAINJSON','w')); print(len(d))"
"${RUN[@]}" python -c "
from transformers import AutoProcessor; import json, argparse
from dataset.dataset import preprocess_qwen_visual
pr = AutoProcessor.from_pretrained('$MODEL'); d = json.load(open('$EVALJSON'))
for k, v in d.items():
    ids = preprocess_qwen_visual([v[0]], pr, False)['input_ids'][0]
    print(k, repr(pr.tokenizer.decode(ids[-40:])), '| answer', v[0]['answer'])
"
echo "DONE $(date +%T)"
