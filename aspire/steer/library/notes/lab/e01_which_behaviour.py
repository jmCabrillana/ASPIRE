"""E01: which trained behaviour does a paraphrase trigger? Chunk under the paraphrase vs under other trained goal instructions."""
import sys, json; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A
policy, lab, _ = A.load()
GOAL = {0: "open the middle drawer of the cabinet", 1: "put the bowl on the stove", 2: "put the wine bottle on top of the cabinet",
        3: "open the top drawer and put the bowl inside", 4: "put the bowl on top of the cabinet", 5: "push the plate to the front of the stove",
        6: "put the cream cheese in the bowl", 8: "put the bowl on the plate", 9: "put the wine bottle on the rack"}
task = int(sys.argv[1]); eps = [int(x) for x in sys.argv[2].split(',')]
for e in eps:
    ep = lab.load(f"outputs/programs/none/libero_goal_lan__{task}/episode_{e}.npz")
    o = ep.obs(0)
    para = lab.chunk(o, ep.instruction)
    seeds = [lab.chunk(o, ep.instruction, seed=s) for s in (1, 2)]
    print(f"ep{e} success={ep.success} '{ep.instruction}' {A.summary(para)}  noise-floor {np.mean([A.dist(para, s) for s in seeds]):.1f}cm")
    rows = []
    for k, ins in GOAL.items():
        if k == task: continue
        c = lab.chunk(o, ins)
        rows.append((A.dist(para, c), k, ins, A.summary(c)))
    for d, k, ins, s in sorted(rows)[:9]:
        print(f"   {d:6.1f}cm  goal{k} '{ins}' {s}")
