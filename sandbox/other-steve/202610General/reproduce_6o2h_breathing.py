#!/usr/bin/env python3
"""Reproduce the synthetic breathing-mode study on 6o2h (two GNM domains).

Put in one directory: this script, rigid_body_diffuse_general.ipynb, mbam_multibody.ipynb,
6o2h.cif and 6o2h-sf.cif. Then:

    python reproduce_6o2h_breathing.py                 # the study as reported (~20 min, one core)
    python reproduce_6o2h_breathing.py --quick         # a fast check with smaller data (~5 min)
    python reproduce_6o2h_breathing.py --mbam-steps 8  # also run 8 steps of the MBAM loop (slow)

For each ground truth -- 'rigid' (the two domains welded: 10^4 x their prior internal spring) and
'breathing' (joined by a spring 5 x its prior) -- it runs
  1. rigid_body_diffuse_general.ipynb: the two-domain model of 6o2h, the synthetic data, lambda
     by cross-validation, the refinement, the comparison with the truth, and the MBAM bundle;
  2. mbam_multibody.ipynb on that bundle: the unregularized start with its fit-driven limits and
     the welding test (and, with --mbam-steps N, N steps of the MBAM loop).
Each notebook's code cells run in order in one namespace, with every override below inserted
right after the notebook's own assignment of that name, as in the reported runs. Requires the
notebooks' packages plus nbformat. Logs: study_<truth>.log and mbam_<truth>.log; figures are
written by the notebooks; a summary is printed at the end.
"""
import argparse, contextlib, io, re, time, traceback
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close('all')

GENERAL, MBAM = 'rigid_body_diffuse_general.ipynb', 'mbam_multibody.ipynb'
STUDY = {'PDB_ID': '"6o2h"', 'DATA_SOURCE': "'synthetic'", 'BODY_METHOD': "'gnm'",
         'SYN_N_NEAR': '240', 'SYN_N_MID': '1500', 'LAMBDA_SCAN': '[1.0, 10.0, 100.0]',
         'LSQ_MAX_NFEV': '100', 'FORCE_RESCAN': 'True'}
TRUTHS = {'rigid': {'SYN_INTERNAL': "'rigid'"},
          'breathing': {'SYN_INTERNAL': "'spring'", 'SYN_INTERNAL_SCALE': '5.0'}}
QUICK = {'SYN_N_NEAR': '90', 'SYN_N_MID': '450', 'LAMBDA_SCAN': '[1.0, 10.0]', 'LSQ_MAX_NFEV': '20',
         'D_MIN_TRANSFORM': '1.6', 'SYN_D_MIN': '2.4'}
QUICK_MBAM = {'FIT_MAX_NFEV': '50', 'FIT_ROUNDS': '1'}
UNTIL, ALSO = 'THETA_TRUE = Lmap_to_theta', 'if EXPORT_MBAM_BUNDLE'   # general notebook: up to the
# truth-vs-fit cell, plus the bundle cell (the analysis in between is not needed for the study)


def apply_overrides(cells, overrides):
    """Insert 'NAME = value' after the first top-level assignment of NAME in the cells."""
    done, out = set(), []
    for idx, src in cells:
        lines, new, i = src.split('\n'), [], 0
        while i < len(lines):
            line = lines[i]; new.append(line)
            m = re.match(r'^(\s*)([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*=(?!=)', line)
            if m:
                ind, names = m.group(1), [n.strip() for n in m.group(2).split(',')]
                code = line.split('#')[0]
                depth = sum(code.count(c) for c in '([{') - sum(code.count(c) for c in ')]}')
                while depth > 0 and i + 1 < len(lines):           # multi-line assignment
                    i += 1; new.append(lines[i]); code = lines[i].split('#')[0]
                    depth += sum(code.count(c) for c in '([{') - sum(code.count(c) for c in ')]}')
                for n in names:
                    if n in overrides and n not in done:
                        new.append(f"{ind}{n} = {overrides[n]}   # reproduce_6o2h_breathing.py"); done.add(n)
            i += 1
        out.append((idx, '\n'.join(new)))
    missing = set(overrides) - done
    assert not missing, f"overrides not found in the notebook: {missing}"
    return out


def run_notebook(path, overrides, log_path, until=None, also=None):
    nb = nbformat.read(path, as_version=4)
    code = [(i, c.source) for i, c in enumerate(nb.cells) if c.cell_type == 'code']
    if until:
        k = next(j for j, (_, s) in enumerate(code) if until in s[:400])
        code = code[:k + 1] + ([c for c in code[k + 1:] if also in c[1][:400]] if also else [])
    ns, t_all = {'__name__': '__main__'}, time.time()
    with open(log_path, 'w') as log:
        for idx, src in apply_overrides(code, overrides):
            t0, buf = time.time(), io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(src, f'<{path} cell {idx}>', 'exec'), ns)
            except BaseException:
                log.write(buf.getvalue() + f'\n=== CELL {idx} FAILED\n'); traceback.print_exc(file=log)
                raise SystemExit(f"{path}: cell {idx} failed; see {log_path}")
            log.write(f'\n=== cell {idx} ok ({time.time() - t0:.1f}s)\n' + buf.getvalue()); log.flush()
            print(f"   {path} cell {idx:3d} ok ({time.time() - t0:6.1f} s)", flush=True)
    print(f"   done in {time.time() - t_all:.0f} s; log {log_path}")
    return ns


def blocks(log_path, keys):
    parts = re.split(r'\n=== cell \d+ ok \([\d.]+s\)\n', open(log_path).read())
    return [p.strip() for p in parts if any(k in p for k in keys)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true', help='smaller data and fewer iterations (plumbing check)')
    ap.add_argument('--mbam-steps', type=int, default=0, help='steps of the MBAM loop after the welding test')
    ap.add_argument('--truths', default='rigid,breathing')
    a = ap.parse_args()
    for t in a.truths.split(','):
        print(f"\n### truth '{t}': {GENERAL}")
        run_notebook(GENERAL, {**STUDY, **TRUTHS[t], **(QUICK if a.quick else {})}, f'study_{t}.log', UNTIL, ALSO)
        tag = 'rigid' if t == 'rigid' else 'spring'
        print(f"### truth '{t}': {MBAM}")
        run_notebook(MBAM, {'BUNDLE': f"'synthetic_6o2h_gnm_{tag}_mbam_bundle.npz'", 'OUT': f"'mbam_6o2h_{t}_'",
                            'MBAM_MAX_STEPS': str(a.mbam_steps), **(QUICK_MBAM if a.quick else {})}, f'mbam_{t}.log')
    for t in a.truths.split(','):
        print(f"\n{'=' * 30} truth '{t}' {'=' * 30}")
        for b in blocks(f'study_{t}.log', ('internal motion in the truth', "rule '1se'", 'prior→truth')):
            print(b + '\n')
        for b in blocks(f'mbam_{t}.log', ('after the unregularized start', 'FULL WELD', 'MBAM steps')):
            print(b + '\n')


if __name__ == '__main__':
    main()
