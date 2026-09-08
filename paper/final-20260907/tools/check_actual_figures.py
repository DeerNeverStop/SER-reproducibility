"""Check actual curve embedding without using the synthetic demo's CLI gate.

Retains its fully composed PDF font-em measurement and 9 physical-pt minimum.
Actual required text replaces only the demonstration's SYNTHETIC watermark.
"""
from pathlib import Path
import argparse
import importlib.util
import json
from datetime import datetime, timezone

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--english',type=Path,required=True)
    p.add_argument('--chinese',type=Path,required=True)
    p.add_argument('--curve',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    helper=root/'prepared/figure_qa/geometry_qa.py'
    spec=importlib.util.spec_from_file_location('figure_measurement',helper)
    g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    g.REQUIRED=('All 15 epochs','CREMA-D','SUBESCO','RAVDESS','Seen validation',
                'Unseen validation','Outer test','Epoch','Native-task UAR')
    manifest=a.curve.parent/'FIGURE_MANIFEST.json'
    m=json.loads(manifest.read_text(encoding='utf8'))
    assert m['files'][a.curve.name]['sha256']==g.sha(a.curve)
    assert m['files'][a.curve.name]['bytes']==a.curve.stat().st_size
    paths={'native':a.curve,'english':a.english,'chinese':a.chinese}
    scans={k:g.inspect_pdf(v) for k,v in paths.items()}
    width=scans['native']['page_boxes'][0]['crop'][2]-scans['native']['page_boxes'][0]['crop'][0]
    comparisons={k:g.compare_embedding(scans['native'],scans[k],w/width)
                 for k,w in (('english',g.EN_W),('chinese',g.CN_W))}
    checks={k:{'required_actual_text':all(s['required_text'].values()),
               'text_at_least_9_physical_pt':s['all_figure_text_at_least_9pt'],
               'orthogonal':s['all_orthogonal'],'uniform_scale':s['all_uniform_scale'],
               'horizontal_scale_one':s['all_text_horizontal_scaling_one'],
               'rotated_axis_present':s['rotated_character_count']>0,
               'one_figure_page':len({r['page'] for r in s['records']})==1}
            for k,s in scans.items()}
    passed=all(all(v.values()) for v in checks.values()) and all(
        v['glyph_identity_and_order_equal'] and v['pass_scale'] for v in comparisons.values())
    result={'schema':'ser-actual-figure-geometry-1','created_at':datetime.now(timezone.utc).isoformat(),
            'scope':'Actual completed-program curve and full bilingual manuscript PDFs; geometry only.',
            'script':g.pin(Path(__file__)),'measurement_helper':g.pin(helper),
            'figure_manifest':g.pin(manifest),'checks':checks,'comparisons':comparisons,
            'pdfs':scans,'pass':passed,'visual_review_performed_by_script':False}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'pass':passed,'minima':{k:s['minimum_figure_em_pt'] for k,s in scans.items()},
                      'output':str(a.out)}))
    return 0 if passed else 1

if __name__=='__main__':
    raise SystemExit(main())
