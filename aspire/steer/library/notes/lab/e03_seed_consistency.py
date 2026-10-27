"""E03: is the policy more self-consistent (across flow-matching noise seeds) on instructions it understands?
Spread = mean pairwise path distance (cm) between chunks from 6 seeds, on goal-scene observations (lan7 frozen episodes)."""
import sys, itertools; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A
policy, lab, _ = A.load()
KNOWN = ["put the bowl on the stove", "put the wine bottle on top of the cabinet", "put the bowl on top of the cabinet",
         "put the cream cheese in the bowl", "put the bowl on the plate", "put the wine bottle on the rack"]
PARA = ["place the bowl on the stove", "place the wine bottle on top of the cabinet", "place the bowl on top of the cabinet",
        "place the cream cheese inside the bowl", "place the bowl on top of the plate", "place the wine bottle on the rack"]
NONSENSE = ["blick the bowl on the stove", "frob the wine bottle on the rack", "zorp the cabinet", "do something"]
TARGETS = ["switch on the stove", "move the plate to the stove’s front", "slide open the top drawer and place the bowl in it",
           "open the cabinet’s middle drawer"]
def spread(o, ins, seeds=range(6)):
    cs = [lab.chunk(o, ins, seed=s) for s in seeds]
    return np.mean([A.dist(a, b) for a, b in itertools.combinations(cs, 2)])
obs = []
for e in (0, 1, 2):
    ep = lab.load(f"outputs/programs/none/libero_goal_lan__7/episode_{e}.npz")
    obs += [ep.obs(0), ep.obs(2)]
for name, group in [("known", KNOWN), ("paraphrase", PARA), ("nonsense", NONSENSE), ("targets", TARGETS)]:
    print(name)
    for ins in group:
        s = [spread(o, ins) for o in obs]
        print(f"   {np.mean(s):6.2f}  {np.round(s, 1)}  {ins}")
