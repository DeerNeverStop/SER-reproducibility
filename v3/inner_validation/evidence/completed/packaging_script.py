"""Copy completed small evidence for Git; large original results remain on D:."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

A=Path('D:/SER-dual-validation-20260906')
R=Path(__file__).resolve().parents[1]/'SER-speaker-execution-20260906'
V=R/'v3/inner_validation'
E=V/'evidence/completed'
O=V/'reports/results'
F=V/'reports/figures'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    analysis=A/'analysis'
    p=read(analysis/'pipeline.json')
    assert p['pass_all_steps'] is True and p['state']=='completed'
    assert len(p['steps'])==6 and all(s['exit_code']==0 for s in p['steps'])
    gate=read(analysis/'formal_gate.json');replay=read(analysis/'independent_replay.json')
    assert gate['pass'] is True and gate['units']==480 and replay['pass'] is True
    assert read(A/'ops/stop.json')['status']=='EXITED'
    assert all(not d.exists() for d in (E,O,F))
    for d in (E,O,F):d.mkdir(parents=True)
    files={}
    def put(data,dest,source=None,transform='byte-identical copy'):
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as out:out.write(data)
        assert dest.read_bytes()==data
        files[str(dest.relative_to(V)).replace('\\','/')]={'bytes':len(data),'sha256':sha(data),
            'source':str(source) if source else None,'transformation':transform}
    def copy(source,dest):
        data=source.read_bytes();put(data,dest,source)
        assert sha(source.read_bytes())==sha(data)
    for name in ('formal_gate.json','pilot_gate.json','independent_replay.json','pipeline.json'):
        copy(analysis/name,E/name)
    for name in ('pred_metrics.csv','selected_episodes.csv','draws.csv','results.json'):
        copy(analysis/'scores'/name,O/name)
    for path in sorted((analysis/'figures').iterdir()):
        assert path.is_file() and path.suffix in ('.png','.svg','.md','.json')
        copy(path,F/path.name)
    resource=analysis/'resource_summary.json';raw=resource.read_bytes()
    zipped=gzip.compress(raw,mtime=0);assert gzip.decompress(zipped)==raw
    put(zipped,E/'resource_summary.json.gz',resource,'gzip; decompressed bytes identical to source')
    obj=json.loads(raw)
    compact={k:v for k,v in obj.items() if k not in ('unit_resources','metadata_file_sha256','report_sha256')}
    compact.update(schema='ser-dual-validation-resource-extract-1',full_report_file_sha256=sha(raw),
        full_report_compressed_file='resource_summary.json.gz',
        omitted_fields=['unit_resources','metadata_file_sha256','report_sha256'],
        extract_note='Display extract. The full original JSON, including its own semantic hash, is preserved by gzip.')
    put((json.dumps(compact,ensure_ascii=False,indent=2)+'\n').encode(),E/'resource_summary.compact.json',resource,'documented display extract')
    ops_names=('start.json','stop.json','provider_before_stop.json','provider_stop_confirmation.json',
        'billing_at_completion.json','closeout.json','formal_driver_finished.json','final_source_provenance.json',
        'local_input_gate.json','pilot_resource_gate.json','restore_first_summary.json','transport_probe.json')
    for name in ops_names:copy(A/'ops'/name,E/'operations'/name)
    for i in range(60):
        src=A/'ops'/f'batch_{i:02d}_backed_up.json';inv=read(src)
        assert src.read_bytes()==Path(inv['backup_receipt']).read_bytes()
        copy(src,E/'batches'/src.name)
        copy(Path(inv['cleanup_receipt']),E/'batches'/f'batch_{i:02d}_cleanup.json')
    for path in sorted((A/'ops/restore_first').glob('*.json')):copy(path,E/'restore_first'/path.name)
    for phase in ('formal','pilot'):copy(A/'runs'/phase/'ledger.jsonl',E/'ledgers'/(phase+'.jsonl'))
    driver=Path(read(A/'ops/active_driver.json')['path'])
    for name in ('launch.json','exit.json'):copy(driver/name,E/'driver'/name)
    snap=A/'ops/cloud_snapshots/20260907T054616Z/snapshot_inventory.json'
    copy(snap,E/'operations/final_cloud_snapshot_inventory.json')
    for path in sorted(analysis.glob('*.log')):copy(path,E/'analysis_logs'/path.name)
    copy(Path(__file__),E/'packaging_script.py')
    manifest=dict(schema='ser-dual-validation-delivery-files-1',created_at=datetime.now(timezone.utc).isoformat(),
        archive_root=str(A),frozen_scientific_commit='de39f2ae5bad4540da9d13ab4cc6196bec786400',
        plan_sha256=gate['plan_sha256'],files=files,
        scope='Small completed evidence, score tables and figures. Excludes this manifest itself, later prose reports, raw audio and 125.12 GB of per-fit original payloads. Source paths identify the retained original archive; files remain valid after repository relocation.')
    with (E/'FILE_SHA256.json').open('x',encoding='utf-8',newline='\n') as out:
        json.dump(manifest,out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps({'files':len(files),'bytes':sum(v['bytes'] for v in files.values()),'manifest':str(E/'FILE_SHA256.json')}))
if __name__=='__main__':main()
