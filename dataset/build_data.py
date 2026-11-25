"""Build the evaluation JSON (up to N samples per VLMEvalKit benchmark) and the LLaVA-Instruct training JSON."""
import argparse
import ast
import base64
import csv
import json
import os
import random
import sys

csv.field_size_limit(sys.maxsize)

# json key (the name eval/eval_engine.py scores by) -> VLMEvalKit TSV
BENCHMARKS = {
    'A-OKVQA': 'A-OKVQA.tsv',
    'VizWiz': 'VizWiz.tsv',
    'SEEDBench_IMG': 'SEEDBench_IMG.tsv',
    'MMBench_DEV_CN': 'MMBench_DEV_CN.tsv',
    'MME': 'MME.tsv',
    'ChartQA': 'ChartQA_TEST.tsv',
    'OCRBench': 'OCRBench.tsv',
    'TextVQA_VAL': 'TextVQA_VAL.tsv',
}
CHOICE = {'A-OKVQA', 'MMBench_DEV_CN', 'SEEDBench_IMG'}
MCQ_SUFFIX = "Answer with the option's letter from the given choices directly."
VQA_SUFFIX = "Answer the question using a single word or phrase."


def read_tsv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def prompt_of(name, row):
    q = row['question'].strip()
    if name in CHOICE:
        opts = [c for c in 'ABCDEFGHI' if row.get(c) not in (None, '', 'nan')]
        hint = row.get('hint', '')
        head = f"Hint: {hint}\n" if hint not in (None, '', 'nan') else ''
        return head + q + '\n' + '\n'.join(f"{c}. {row[c]}" for c in opts) + '\n' + MCQ_SUFFIX
    if name == 'MME':
        return q
    return q + '\n' + VQA_SUFFIX


def answer_of(row):
    a = row['answer']
    if a.startswith('['):
        return ast.literal_eval(a)
    return a


def build_eval(lmudata, out_dir, n, seed):
    out = {}
    for name, tsv in BENCHMARKS.items():
        rows = read_tsv(os.path.join(lmudata, tsv))
        by_index = {r['index']: r for r in rows}
        pick = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
        img_dir = os.path.join(out_dir, 'images', name)
        os.makedirs(img_dir, exist_ok=True)
        items = []
        for i in pick:
            row = rows[i]
            img = row['image']
            while len(img) < 64:  # VLMEvalKit stores a repeated image as the index of the row that holds it
                img = by_index[img]['image']
            path = os.path.join(img_dir, f"{row['index']}.jpg")
            with open(path, 'wb') as f:
                f.write(base64.b64decode(img))
            items.append({
                'images': [path],
                'conversations': [{'from': 'human', 'value': '<image>\n' + prompt_of(name, row)},
                                  {'from': 'gpt', 'value': ''}],
                'answer': answer_of(row),
                'index': row['index'],
            })
        out[name] = items
        print(f"{name}: {len(items)} of {len(rows)}", flush=True)
    path = os.path.join(out_dir, f'eval_max_{n}_dataset.json')
    with open(path, 'w') as f:
        json.dump(out, f, ensure_ascii=False)
    print(path)


def build_train(llava_json, coco_dir, out_json, n, seed):
    with open(llava_json) as f:
        data = json.load(f)
    data = [d for d in data if 'image' in d]
    pick = sorted(random.Random(seed).sample(range(len(data)), n))
    items = [{'images': [os.path.join(coco_dir, data[i]['image'])], 'conversations': data[i]['conversations'],
              'id': data[i]['id']} for i in pick]
    with open(out_json, 'w') as f:
        json.dump({'llava': items}, f, ensure_ascii=False)
    print(f"{out_json}: {len(items)} of {len(data)}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('what', choices=['eval', 'train'])
    p.add_argument('--lmudata')
    p.add_argument('--llava_json')
    p.add_argument('--coco_dir')
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.what == 'eval':
        build_eval(a.lmudata, a.out, a.n, a.seed)
    else:
        build_train(a.llava_json, a.coco_dir, a.out, a.n, a.seed)
