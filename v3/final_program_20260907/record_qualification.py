"""Archive existing technical pilot evidence without computing scientific scores."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path(__file__).resolve().parents[2]
    archive = Path('D:/SER-final-program-20260907')
    phase = archive/'pilot_v1'
    target = Path(__file__).resolve().parent/'qualification'
    target.mkdir(exist_ok=False)
    gate = json.loads((phase/'COMPLETE_GATE.json').read_text(encoding='utf8'))
    replay_path = archive/'qualification/pilot_fresh_process_replay.json'
    replay = json.loads(replay_path.read_text(encoding='utf8'))
    assert gate['pass'] and gate['verified_units'] == 4
    assert replay['pass'] and replay['units_checked'] == 4 and not replay['scientific_scores_computed']
    assert replay['lock_sha256'] == gate['lock_sha256']
    units = []
    for uid, expected in gate['done_sha256'].items():
        folder = phase/'units'/uid
        assert sha(folder/'DONE') == expected
        done = json.loads((folder/'DONE').read_text(encoding='utf8'))
        receipt_path = folder/done['attempt']/'receipt.json'
        assert sha(receipt_path) == done['artifacts'][receipt_path.relative_to(folder).as_posix()]
        receipt = json.loads(receipt_path.read_text(encoding='utf8'))
        units.append(dict(unit_id=uid, wall_seconds=receipt['info']['wall_seconds'],
                          peak_allocated_bytes=receipt['peak_allocated_bytes'],
                          peak_reserved_bytes=receipt['peak_reserved_bytes'],
                          environment=receipt['environment'], done_sha256=expected,
                          receipt_sha256=sha(receipt_path),
                          checkpoint_bytes=(folder/done['attempt']/'checkpoint.pt').stat().st_size))
    estimate_seconds = sum((24 if u['unit_id'].endswith('_B') else 120)*u['wall_seconds'] for u in units)
    for source, name in [(phase/'COMPLETE_GATE.json', 'pilot_complete_gate.json'),
                         (phase/'SOURCE_LOCK.json', 'pilot_source_lock.json'),
                         (replay_path, 'pilot_fresh_process_replay.json')]:
        (target/name).write_bytes(source.read_bytes())
    result = dict(status='four technical pilots completed and actually restored in a separate CUDA process',
                  program='SER26-FINAL-CHECKPOINT-PROGRAM-1', units=units,
                  plan_sha256=gate['plan_sha256'], pilot_lock_sha256=gate['lock_sha256'],
                  source_commit=replay['source_commit'], fresh_process_pid=replay['pid'],
                  restored_epoch_group_comparisons=replay['saved_epoch_group_comparisons'],
                  restored_max_abs_diff=replay['max_abs_diff'],
                  planned_formal_fits=384, weighted_fit_hours=estimate_seconds/3600,
                  planning_total_hours=[5, 7], route='local NVIDIA GeForce RTX 5070',
                  incremental_cloud_rental_cost=0, formal_scientific_scores_computed=False,
                  pilot_scores_used_to_choose_scientific_configuration=False,
                  free_bytes={drive:shutil.disk_usage(drive+'/').free for drive in ('C:', 'D:', 'E:')},
                  archive_sha256={p.name:sha(p) for p in target.iterdir()},
                  limitations=['One technical observation per corpus/arm; elapsed-time range is a planning estimate, not a confidence interval.',
                               'The 384-fit runtime adds a strict full-ledger validator after pilot qualification; scientific plan and engine stay byte-identical.',
                               'This records technical replay only and is not a completed formal experiment.'])
    (target/'QUALIFICATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    remote = Path(__file__).resolve().parent/'remote_reviews'
    remote.mkdir(exist_ok=False)
    manifest = []
    for ref, source, name in [
        ('0abc182', 'v3/CLAUDE_EXPERIMENT_REVIEW_20260907.md', 'CLAUDE_EXPERIMENT_REVIEW_V2.md'),
        ('0abc182', 'v3/CLAUDE_RESPONSE_REVIEW_20260907.md', 'CLAUDE_RESPONSE_REVIEW.md'),
        ('cd91b91', 'paper/submission-20260906/CLAUDE_REVIEW.md', 'CLAUDE_PAPER_REVIEW_CORRECTED.md')]:
        content = subprocess.check_output(['git', 'show', ref+':'+source], cwd=repo)
        destination = remote/name
        destination.write_bytes(content)
        manifest.append(dict(commit=subprocess.check_output(['git','rev-parse',ref],cwd=repo,text=True).strip(),
                             source_path=source, imported_file=name, bytes=len(content), sha256=sha(destination)))
    (remote/'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n',encoding='utf8')
    print(json.dumps({'qualified_pilots':len(units), 'weighted_fit_hours':estimate_seconds/3600,
                      'restored_comparisons':replay['saved_epoch_group_comparisons'],
                      'new_cloud_rental_cost':0, 'remote_review_files':len(manifest)}))


if __name__ == '__main__':
    main()
