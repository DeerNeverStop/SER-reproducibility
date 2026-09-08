"""SYNTHETIC LAYOUT ONLY: vector-placement/font geometry, never research scoring.

Run with the bundled Python. Inputs are the declared synthetic curve PDF and
its actual spconf English embedding. Output must be a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pdfminer
import pypdf
from pdfminer.converter import PDFLayoutAnalyzer
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pypdf import PdfReader, PdfWriter, Transformation

SCOPE = 'SYNTHETIC LAYOUT ONLY; no formal data or research results accessed'
A4_W, A4_H = 595.2755905511812, 841.8897637795277
CN_W, EN_W = A4_W - 108.0, 178.0 * 72.0 / 25.4
REQUIRED = ('SYNTHETIC', 'CREMA-D', 'SUBESCO', 'RAVDESS', 'Seen validation',
            'Unseen validation', 'Outer test', 'Epoch', 'Native-task UAR')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def norm(text):
    return ''.join(c.casefold() for c in text if c.isalnum())


class EmCollector(PDFLayoutAnalyzer):
    """Inspect raw Tf fontsize and the fully composed text/page/Form CTM.

    Do not use LTChar.size: rotated text's axis-aligned bbox height is not
    its em height. PDFMiner already includes the embedding transform here.
    """
    def __init__(self, manager):
        super().__init__(manager)
        self.rows = []
        self.page_number = 0

    def begin_page(self, page, ctm):
        self.page_number += 1
        return super().begin_page(page, ctm)

    def render_char(self, matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate):
        advance = super().render_char(matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate)
        item = self.cur_item._objs[-1]
        a, b, c, d, e, f = (float(x) for x in matrix)
        sx, sy = math.hypot(a, b), math.hypot(c, d)
        valid = sx > 0 and sy > 0
        dot = (a*c + b*d)/(sx*sy) if valid else None
        aspect = abs(sx-sy)/max(sx, sy) if valid else None
        self.rows.append(dict(page=self.page_number, text=item.get_text(), font=str(font.fontname),
                              raw_tf_fontsize_pt=float(fontsize), matrix=[a,b,c,d,e,f],
                              em_pt=float(fontsize)*sy, bbox_height_pt=float(item.height),
                              normalized_axis_dot=dot, relative_axis_scale_difference=aspect,
                              orthogonal=valid and abs(dot) <= 1e-6,
                              uniform_scale=valid and aspect <= 1e-6,
                              text_horizontal_scaling=float(scaling),
                              rotated=abs(b) > 1e-7 or abs(c) > 1e-7))
        return advance


def inspect_pdf(path):
    reader = PdfReader(path)
    manager = PDFResourceManager()
    device = EmCollector(manager)
    interpreter = PDFPageInterpreter(manager, device)
    with path.open('rb') as stream:
        for page in PDFPage.get_pages(stream):
            interpreter.process_page(page)
    rows = [r for r in device.rows if 'dejavu' in r['font'].casefold() and r['text'].strip()]
    whole = norm(''.join(r['text'] for r in rows))
    required = {word: norm(word) in whole for word in REQUIRED}
    rotated = [r for r in rows if r['rotated']]
    valid_sizes = [r['em_pt'] for r in rows]
    return dict(input=pin(path), page_count=len(reader.pages),
                page_boxes=[dict(media=list(map(float, p.mediabox)), crop=list(map(float, p.cropbox)),
                                 rotation=int(p.get('/Rotate',0)), user_unit=float(p.get('/UserUnit',1)))
                            for p in reader.pages],
                figure_character_count=len(rows), fonts=sorted({r['font'] for r in rows}),
                required_text=required, minimum_figure_em_pt=min(valid_sizes) if rows else None,
                maximum_figure_em_pt=max(valid_sizes) if rows else None,
                rotated_character_count=len(rotated),
                rotated_minimum_em_pt=min(r['em_pt'] for r in rotated) if rotated else None,
                rotated_minimum_bbox_height_pt=min(r['bbox_height_pt'] for r in rotated) if rotated else None,
                all_orthogonal=bool(rows) and all(r['orthogonal'] for r in rows),
                all_uniform_scale=bool(rows) and all(r['uniform_scale'] for r in rows),
                all_text_horizontal_scaling_one=bool(rows) and all(abs(r['text_horizontal_scaling']-1) <= 1e-6 for r in rows),
                all_figure_text_at_least_9pt=bool(rows) and all(r['em_pt'] >= 9.0-1e-6 for r in rows),
                records=rows)


def compare_embedding(native, embedded, expected_scale):
    first, second = native['records'], embedded['records']
    identity = len(first) == len(second) and [(r['font'],r['text']) for r in first] == [(r['font'],r['text']) for r in second]
    differences = [abs(b['em_pt']/a['em_pt'] - expected_scale) for a,b in zip(first,second)] if identity else []
    native_box = native['page_boxes'][0]['crop']
    native_width = native_box[2]-native_box[0]
    ratios = [b['em_pt']/a['em_pt'] for a,b in zip(first,second)] if identity else []
    return dict(glyph_identity_and_order_equal=identity, expected_uniform_scale=expected_scale,
                observed_minimum_scale=min(ratios) if identity else None,
                observed_maximum_scale=max(ratios) if identity else None,
                maximum_scale_difference=max(differences) if differences else None,
                expected_width_pt=native_width*expected_scale,
                observed_minimum_width_pt=native_width*min(ratios) if ratios else None,
                observed_maximum_width_pt=native_width*max(ratios) if ratios else None,
                maximum_width_difference_pt=native_width*max(differences) if differences else None,
                width_tolerance_pt=0.1,
                # Physical placement tolerance only. The >=9 pt em check is unchanged.
                # First stricter attempt and its source are retained separately.
                pass_scale=bool(differences) and native_width*max(differences) <= 0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curve', type=Path, required=True)
    parser.add_argument('--english', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pdftoppm', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output must be a new directory; nothing is overwritten')
    if 'SYNTHETIC' not in str(args.curve) or args.curve.parent.resolve() != args.english.parent.resolve():
        parser.error('Only co-located explicitly SYNTHETIC inputs are accepted')
    source_path = args.curve.parent/'LAYOUT_SOURCE.json'
    source = json.loads(source_path.read_text(encoding='utf8'))
    if 'SYNTHETIC LAYOUT ONLY' not in source.get('scope',''):
        parser.error('Missing explicit synthetic source declaration')
    if source['files'][args.curve.name]['sha256'] != sha(args.curve):
        parser.error('Curve bytes differ from declared synthetic source')
    pins = {str(p.resolve()):sha(p) for p in (args.curve,args.english,source_path,Path(__file__))}
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='ser-synthetic-figure-geometry-qa-1', scope=SCOPE,
                  created_at=datetime.now(timezone.utc).isoformat(),
                  versions=dict(python=sys.version, pypdf=pypdf.__version__, pdfminer=pdfminer.__version__),
                  script=pin(Path(__file__)), inputs_sha256=pins, scientific_tests=0,
                  actual_manuscript_qa=False, visual_review_performed_by_script=False,
                  method='raw Tf fontsize multiplied by norm of fully composed font vertical em axis; no bbox-height or double scaling',
                  point_unit='PDF point = 1/72 inch; only DejaVu figure text is thresholded',
                  failures=[], outputs={}, renders={})
    try:
        native_reader = PdfReader(args.curve)
        if len(native_reader.pages) != 1:
            raise ValueError('Synthetic curve must have exactly one page')
        source_page = native_reader.pages[0]
        if int(source_page.get('/Rotate',0)) != 0 or float(source_page.get('/UserUnit',1)) != 1:
            raise ValueError('Unexpected source rotation/UserUnit')
        left,bottom,right,top = map(float,source_page.cropbox)
        width,height = right-left, top-bottom
        scale = CN_W/width
        if height*scale > A4_H-108:
            raise ValueError('Curve does not fit CN geometry frame')
        writer = PdfWriter()
        page = writer.add_blank_page(width=A4_W, height=A4_H)
        y = A4_H-54-height*scale
        page.merge_transformed_page(source_page, Transformation(ctm=(scale,0,0,scale,54-left*scale,y-bottom*scale)))
        writer.add_metadata({'/Title':'SYNTHETIC LAYOUT ONLY - CN geometry, not a complete manuscript', '/Subject':SCOPE})
        cn = args.out/'chinese_geometry.pdf'
        with cn.open('xb') as stream:
            writer.write(stream)
        report['cn_placement'] = dict(scope='SYNTHETIC CN GEOMETRY ONLY; not the Chinese builder or a complete manuscript',
                                      page_width_pt=A4_W,page_height_pt=A4_H,x_pt=54,y_pt=y,
                                      target_width_pt=CN_W,actual_width_pt=width*scale,
                                      actual_height_pt=height*scale,uniform_scale=scale,source_crop_box=[left,bottom,right,top])
        scans = {name:inspect_pdf(path) for name,path in (
            ('native_curve',args.curve),('english_embedded',args.english),('chinese_geometry',cn))}
        report['pdfs'] = scans
        report['embedding_comparison'] = dict(
            english=compare_embedding(scans['native_curve'],scans['english_embedded'],EN_W/width),
            chinese=compare_embedding(scans['native_curve'],scans['chinese_geometry'],CN_W/width))
        for name,scan in scans.items():
            checks = dict(one_page=scan['page_count']==1, required_text_present=all(scan['required_text'].values()),
                          text_at_least_9pt=scan['all_figure_text_at_least_9pt'], orthogonal=scan['all_orthogonal'],
                          uniform_scale=scan['all_uniform_scale'], no_text_horizontal_compression=scan['all_text_horizontal_scaling_one'],
                          rotated_axis_text_present=scan['rotated_character_count']>0,
                          ordinary_user_units_and_rotation=all(b['user_unit']==1 and b['rotation']==0 for b in scan['page_boxes']))
            scan['checks'] = checks
            report['failures'].extend(f'{name}: {k}' for k,v in checks.items() if not v)
        for name,result in report['embedding_comparison'].items():
            if not result['glyph_identity_and_order_equal'] or not result['pass_scale']:
                report['failures'].append(name+': embedded glyph/scale mismatch')
        report['outputs']['chinese_geometry.pdf'] = pin(cn)
        for name,path in (('english',args.english),('chinese_geometry',cn)):
            prefix = args.out/(name+'-page')
            command = [str(args.pdftoppm),'-png','-r','150',str(path),str(prefix)]
            proc = subprocess.run(command,capture_output=True,text=True,timeout=60)
            images = sorted(args.out.glob(name+'-page-*.png'))
            report['renders'][name] = dict(command=command,exit_code=proc.returncode,
                                          stdout=proc.stdout,stderr=proc.stderr,images=[pin(p) for p in images])
            if proc.returncode != 0 or len(images) != 1:
                report['failures'].append(name+': Poppler render failed or unexpected page count')
        report['input_bytes_unchanged'] = all(sha(Path(p))==value for p,value in pins.items())
        if not report['input_bytes_unchanged']:
            report['failures'].append('input/source bytes changed during inspection')
    except BaseException as exc:
        report['failures'].append(type(exc).__name__+': '+str(exc))
        report['exception'] = traceback.format_exc()
    report['pass_geometry'] = not report['failures']
    destination = args.out/'GEOMETRY_QA.json'
    destination.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(dict(scope=SCOPE,report=pin(destination),pass_geometry=report['pass_geometry'],
                         failures=report['failures'],
                         minima={k:v['minimum_figure_em_pt'] for k,v in report.get('pdfs',{}).items()}),ensure_ascii=False))
    return 0 if report['pass_geometry'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
