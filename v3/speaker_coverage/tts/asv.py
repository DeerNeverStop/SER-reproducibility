"""ASV description-consistency diagnostic for the completed technical probe."""
from pathlib import Path
import argparse, csv, json, math, os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from v3.speaker_coverage.tts.probe import load_manifest, existing_receipt, file_sha, immutable_json, require


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--asv-model-dir',type=Path,required=True)
    ap.add_argument('--asv-pins',type=Path,required=True)
    args=ap.parse_args()
    manifest=load_manifest(args.manifest)
    root=args.manifest.parent
    pins=json.loads(args.asv_pins.read_text())['ecapa']
    require(pins['model_id']=='speechbrain/spkrec-ecapa-voxceleb' and len(pins['revision'])==40,'wrong ASV identity')
    for name,sha in pins['files'].items():
        require(file_sha(args.asv_model_dir/name)==sha,'ASV file mismatch: '+name)
    attempts,receipts=[],[]
    for attempt in manifest['attempts']:
        receipt=existing_receipt(root/'attempts'/attempt['attempt_id'],manifest,attempt)
        require(receipt is not None,'probe still incomplete')
        if receipt['status']=='success': attempts.append(attempt); receipts.append(receipt)
    import torch, soundfile as sf
    from scipy.signal import resample_poly
    from speechbrain.inference.speaker import EncoderClassifier
    torch.set_num_threads(4); torch.manual_seed(20260906); np.random.seed(20260906)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model=EncoderClassifier.from_hparams(source=str(args.asv_model_dir),savedir=str(args.asv_model_dir),
        run_opts={'device':'cuda'},overrides={'pretrained_path':str(args.asv_model_dir)})
    model.eval()
    vectors=[]
    for start in range(0,len(attempts),16):
        waves=[]
        for attempt in attempts[start:start+16]:
            wave,sr=sf.read(root/'attempts'/attempt['attempt_id']/'audio.wav',dtype='float32',always_2d=True)
            wave=wave.mean(axis=1)
            g=math.gcd(sr,16000)
            if sr!=16000: wave=resample_poly(wave,16000//g,sr//g).astype(np.float32)
            waves.append(torch.from_numpy(wave.copy()))
        batch=torch.nn.utils.rnn.pad_sequence(waves,batch_first=True)
        lengths=torch.tensor([len(w)/batch.shape[1] for w in waves],dtype=torch.float32)
        with torch.inference_mode():
            values=model.encode_batch(batch.cuda(),lengths.cuda(),normalize=False).squeeze(1).float().cpu().numpy()
        require(np.isfinite(values).all() and (np.linalg.norm(values,axis=1)>0).all(),'invalid ASV output')
        vectors.append(values)
    values=np.concatenate(vectors)
    norm=values/np.linalg.norm(values,axis=1,keepdims=True)
    anchor_indices=[i for i,a in enumerate(attempts) if a['role']=='anchor']
    anchors=[attempts[i]['voice_id'] for i in anchor_indices]
    rows=[]
    for i,attempt in enumerate(attempts):
        if attempt['role']=='anchor' or attempt['voice_id'] not in anchors: continue
        similarities=norm[anchor_indices]@norm[i]
        target=anchors.index(attempt['voice_id'])
        other=np.delete(similarities,target)
        rows.append({'attempt_id':attempt['attempt_id'],'voice_id':attempt['voice_id'],'emotion':attempt['emotion'],
            'retrieved_anchor_voice_id':anchors[int(similarities.argmax())],
            'anchor_retrieval_matches_description':int(int(similarities.argmax())==target),
            'own_anchor_cosine':float(similarities[target]),'best_other_anchor_cosine':float(other.max()) if len(other) else None,
            'own_minus_other_margin':float(similarities[target]-other.max()) if len(other) else None})
    out=root/'asv_diagnostic'; out.mkdir(exist_ok=True)
    np.savez_compressed(out/'embeddings.npz',attempt_ids=np.array([a['attempt_id'] for a in attempts]),embeddings=values)
    with (out/'anchor_retrieval.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    report={'schema':'ser-tts-asv-diagnostic-1','manifest_sha256':manifest['manifest_sha256'],
        'asv_model':pins,'asv_pins_file_sha256':file_sha(args.asv_pins),'source_sha256':file_sha(__file__),
        'preprocessing':'soundfile float32 mono mean; polyphase resample 16k; no trim/no loudness change; batches16; seed20260906',
        'n_successful_audio':len(attempts),'n_anchors':len(anchors),'n_nonanchor_with_own_anchor':len(rows),
        'description_anchor_retrieval_agreement':float(np.mean([r['anchor_retrieval_matches_description'] for r in rows])),
        'by_emotion':{emo:float(np.mean([r['anchor_retrieval_matches_description'] for r in rows if r['emotion']==emo])) for emo in sorted({r['emotion'] for r in rows})},
        'mean_own_minus_other_cosine_margin':float(np.mean([r['own_minus_other_margin'] for r in rows])),
        'human_raters_completed':0,'emotion_validity_established':False,'formal_E3_gate_passed':False,
        'limitations':['one neutral anchor per described voice','ASV is emotion/prosody sensitive',
                      'agreement with intended description is not verified human identity','no equivalence to 24 independent synthetic speakers'],
        'files':{n:file_sha(out/n) for n in ('embeddings.npz','anchor_retrieval.csv')}}
    immutable_json(out/'summary.json',report)
    print(json.dumps({k:report[k] for k in ('n_successful_audio','n_anchors','description_anchor_retrieval_agreement','formal_E3_gate_passed')}))


if __name__=='__main__': main()
