"""Package exact reviewed paper inputs/PDFs and light evidence; never training data."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import zipfile

def pin(path):
    data=path.read_bytes()
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--extract',type=Path,required=True)
    a=p.parse_args(); repo=a.repo.resolve(); root=repo/'paper/final-20260907'
    assert not a.out.exists() and not a.extract.exists(), 'Use fresh archive/extraction paths'
    assert json.loads((root/'qa/structural.json').read_text(encoding='utf8'))['machine_checks_pass']
    assert json.loads((root/'qa/figure_geometry.json').read_text(encoding='utf8'))['pass']
    assert json.loads((root/'qa/visual_review.json').read_text(encoding='utf8'))['pass']
    files=set()
    for name in ('english','chinese','tools','qa'):
        files.update(p for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for rel in ('README.md','RESEARCH_REPORT_中文.md','review/FINAL_SCIENCE_INTERPRETATION.md',
                'review/ACTUAL_MANUSCRIPT_SCIENCE_REVIEW.md','prepared/figure_qa/geometry_qa.py','figures/README.md'):
        files.add(root/rel)
    figdir=root/'figures'
    fm=json.loads((figdir/'FIGURE_MANIFEST.json').read_text(encoding='utf8'))
    files.add(figdir/'FIGURE_MANIFEST.json')
    for rel,v in fm['files'].items():
        assert pin(figdir/rel)==v
        files.add(figdir/rel)
    reports=repo/'v3/final_program_20260907/reports'
    files.update(p for p in reports.rglob('*') if p.is_file())
    for rel in ('SCIENCE_DESIGN.md','PORTABLE_BUNDLE.md','RELATED_WORK_AUDIT.md','TTS_SCOPE_DECISION_AUDIT.md'):
        files.add(repo/'v3/final_program_20260907'/rel)
    content={p.relative_to(repo).as_posix():pin(p) for p in sorted(files)}
    for rel in content:
        assert Path(rel).suffix.lower() not in ('.wav','.npz','.pt','.pth','.zip')
    manifest={'schema':'ser-final-paper-source-bundle-1','created_at':datetime.now(timezone.utc).isoformat(),
              'scope':'Exact paper inputs/PDFs, build and QA tools, light evidence. Not a full training checkout.',
              'scientific_source_commit':'976297d824ea98816bdc0948befd704e612a54c1',
              'fonts_and_compiler_not_bundled':True,'files':content}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for rel in content:
            assert pin(repo/rel)==content[rel], rel
            z.write(repo/rel,rel)
        z.writestr('SOURCE_BUNDLE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    a.extract.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(a.out) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        assert set(z.namelist())==set(content)|{'SOURCE_BUNDLE_MANIFEST.json'}
        for info in z.infolist():
            target=(a.extract/info.filename).resolve()
            assert target.is_relative_to(a.extract.resolve())
            data=z.read(info)
            if info.filename in content:
                assert {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}==content[info.filename]
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f: f.write(data)
    assert all(pin(repo/rel)==v and pin(a.extract/rel)==v for rel,v in content.items())
    result={'archive':str(a.out.resolve()),**pin(a.out),'payload_files':len(content),
            'verified_extraction':str(a.extract.resolve()),'source_inputs_unchanged':True,
            'rebuild_pending':True}
    report=a.out.with_suffix('.zip.acceptance.json')
    with report.open('x',encoding='utf8') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
