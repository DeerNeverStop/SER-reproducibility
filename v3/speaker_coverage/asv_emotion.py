"""Descriptive emotion sensitivity of ASV; never a selector or outcome gate."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from v3.data_design.core_plan import read_metadata, roles, prompt_slots, file_sha, digest, require, write_json
from v3.speaker_coverage.plan import candidate_centroids, load_asv, normalized, neutral_label


def evaluate(meta, embeddings):
    rows = []
    neutral = neutral_label(meta)
    for fold in range(5):
        _, _, pool = roles(meta, 0, fold)
        for rotation in range(6):
            _, _, slots = prompt_slots(meta, rotation)
            prompts = slots['prompt_new']
            candidates = sorted(s for s in pool if all((s, p, c) in meta['cells']
                                                       for p in prompts for c in range(6)))
            centers, _ = candidate_centroids(meta, candidates, prompts, embeddings)
            matrix = np.stack([centers[s] for s in candidates])
            between = np.clip(1-matrix@matrix.T, 0, 2)
            mean_between = float(between[np.triu_indices(len(candidates), 1)].mean())
            class_rows = defaultdict(list)
            for speaker_index, speaker in enumerate(candidates):
                for emotion in range(6):
                    if emotion == neutral:
                        continue  # no self-reference neutral accuracy presented
                    for prompt in prompts:
                        record = meta['cells'][speaker, prompt, emotion]
                        vector = normalized(embeddings[record['relative_path']], record['relative_path'])
                        distance = np.clip(1 - matrix @ vector, 0, 2)
                        class_rows[emotion].append((int(np.argmin(distance) == speaker_index),
                                                   float(distance[speaker_index])))
            for emotion, observations in sorted(class_rows.items()):
                values = np.asarray(observations)
                rows.append({'fold': fold, 'rotation': rotation, 'label_index': emotion,
                             'n_candidate_people': len(candidates), 'n_emotional_queries': len(values),
                             'identity_top1': float(values[:, 0].mean()),
                             'mean_emotion_to_own_neutral_distance': float(values[:, 1].mean()),
                             'mean_between_neutral_centroid_distance': mean_between,
                             'own_emotion_distance_over_between_neutral': float(values[:, 1].mean()/mean_between)})
    require(len(rows) == 150, 'all 30 contexts and five nonneutral labels required')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--asv', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    require(args.repo.resolve() == Path(__file__).resolve().parents[2], 'executed checkout differs from --repo')
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    require(plan['plan_sha256'] == digest({k:v for k,v in plan.items() if k != 'plan_sha256'}), 'plan hash mismatch')
    require(file_sha(args.asv) == plan['input']['asv_sha256'], 'ASV identity mismatch')
    require(plan['source_sha256'].get('v3/speaker_coverage/asv_emotion.py') == file_sha(__file__), 'ASV diagnostic source differs from frozen plan')
    meta = read_metadata(args.repo)
    require(all(meta['input'][k] == plan['input'][k] for k in ('manifest_path','manifest_sha256','demographics_path','demographics_sha256')), 'metadata identity mismatch')
    embeddings, identity = load_asv(args.asv)
    rows = evaluate(meta, embeddings)
    args.out.mkdir(parents=True, exist_ok=True)
    table = args.out / 'emotion_sensitivity.csv'
    with table.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = {'schema':'ser-asv-emotion-diagnostic-1', 'plan_sha256':plan['plan_sha256'],
        'asv':identity, 'source_sha256':file_sha(Path(__file__)), 'table_sha256':file_sha(table),
        'contexts':30, 'query':'one representative per candidate x eight allowed train prompts x five nonneutral classes',
        'reference':'eight neutral representatives; no emotional query used in its centroid',
        'mean_context_class_identity_top1':float(np.mean([r['identity_top1'] for r in rows])),
        'mean_context_class_emotion_over_between':float(np.mean([r['own_emotion_distance_over_between_neutral'] for r in rows])),
        'independent_test':False, 'selection_changed':False, 'ser_scores_read':False,
        'interpretation':'candidate-pool closed-set ASV diagnostic only; overlapping contexts are dependent; no universal cutoff; not SER emotion recognition or causal identity disentanglement'}
    write_json(args.out/'emotion_sensitivity.json', summary)
    print(json.dumps({'status':'complete_descriptive', 'contexts':30, 'selection_changed':False}))


if __name__ == '__main__':
    main()
