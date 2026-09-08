"""Render the curve revision from exact, complete, independently audited scores.

Only graphical appearance changes. Frozen numerical results and the original
release remain untouched. Use a new output directory for every render.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    plot_path = repo / 'v3/final_program_20260907/plot_results.py'
    spec = importlib.util.spec_from_file_location('publication_plot', plot_path)
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    result, tables, pins = plot.load(args.results.resolve(), args.audit.resolve())
    pins[str(Path(__file__).resolve())] = plot.sha(__file__)
    out = args.out.resolve()
    assert not out.exists(), 'Use a fresh figure output directory'
    for protected in (args.results.resolve(), args.audit.resolve().parent,
                      Path(result['inputs']['run_dir']).resolve()):
        assert not out.is_relative_to(protected) and not protected.is_relative_to(out)
    out.mkdir(parents=True, exist_ok=False)
    plot.style()
    files, coordinates = {}, {}
    original_save = plot.save

    def checked_save(figure, destination, name, inventory):
        assert name == 'complete_epoch_curves' and len(figure.axes) == 3
        for corpus, axis in zip(plot.CORPORA, figure.axes):
            rows = [r for r in tables['curves'] if r['corpus'] == corpus]
            assert len(rows) == 1800 and len(axis.lines) == 123
            assert tuple(axis.get_ylim()) == (0.0, 100.0)
            lookup = {(int(r['draw']), int(r['fold']), int(r['epoch'])): r for r in rows}
            for index, line in enumerate(axis.lines[:120]):
                draw, fold = divmod(index, 5)
                assert np.array_equal(line.get_xdata(), np.arange(1, 16))
                assert np.array_equal(line.get_ydata(), [float(lookup[draw, fold, e]['outer_uar'])
                                                       for e in range(1, 16)])
            means = {line.get_label(): line for line in axis.lines[120:]}
            coordinates[corpus] = {}
            for label, column in [('Seen validation', 'seen_uar'),
                                  ('Unseen validation', 'unseen_uar'),
                                  ('Outer test', 'outer_uar')]:
                expected = [math.fsum(float(lookup[d, f, e][column])
                                      for d in range(24) for f in range(5)) / 120
                            for e in range(1, 16)]
                assert np.array_equal(means[label].get_xdata(), np.arange(1, 16))
                assert np.array_equal(means[label].get_ydata(), expected)
                coordinates[corpus][column] = expected
        original_save(figure, destination, name, inventory)

    plot.save = checked_save
    plot.curves_plot(tables['curves'], out, files)
    assert all(plot.sha(path) == expected for path, expected in pins.items())
    manifest = dict(schema='ser-curve-appearance-revision-1',
        created_at=datetime.now(timezone.utc).isoformat(), program=result['program'],
        plan_sha256=result['plan_sha256'], result_sha256=result['result_sha256'],
        inputs=pins, files=files, new_fits=0, new_tests=0,
        appearance='Black outer line below orange dotted unseen line; hollow circles at every unseen epoch. Original blue seen line, legend order, canvas and 0-100 shared scale retained.',
        coordinate_audit=dict(pass_exact=True, individual_outer_points_checked=5400,
                              mean_points_checked=135, means=coordinates),
        visual_review_required=True, visual_review_performed_by_script=False)
    (out/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(out=str(out), coordinate_pass=True, files=files), ensure_ascii=False))


if __name__ == '__main__':
    main()
