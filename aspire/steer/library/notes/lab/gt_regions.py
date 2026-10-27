"""ANALYSIS ONLY (never imported by programs): region centres from BDDL files, to score perception offline."""
import os, re, sys; sys.path.insert(0, '.')
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from steer.rollout import load_task
def regions(suite, task):
    txt = open(load_task(suite, task).bddl).read()
    out = {}
    for m in re.finditer(r"\((\w+)\s*\(:target main_table\)\s*\(:ranges \(\s*\(([-\d.e ]+)\)", txt):
        x0, y0, x1, y1 = map(float, m.group(2).split()); out[m.group(1)] = ((x0 + x1) / 2, (y0 + y1) / 2)
    init = dict(re.findall(r"\(On (\w+) main_table_(\w+)\)", txt))
    return out, init
if __name__ == "__main__":
    for s, t in [("libero_spatial", 0), ("libero_spatial_swap", 0), ("libero_spatial_swap", 1), ("libero_spatial_swap", 5)]:
        r, i = regions(s, t); print(s, t, {k: r.get(v) for k, v in i.items()})
