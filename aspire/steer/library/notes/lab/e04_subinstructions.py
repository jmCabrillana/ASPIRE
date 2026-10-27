"""E04: where do candidate sub-instructions for lan3 send the arm (first chunk, 3 episodes)?"""
import sys; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A
policy, lab, _ = A.load()
C = ["slide open the top drawer and place the bowl in it", "open the top drawer and put the bowl", "open the top drawer of the cabinet",
     "open the top drawer", "open the middle drawer of the cabinet", "put the bowl in the top drawer", "put the bowl on the stove",
     "put the black bowl in the top drawer of the cabinet"]
for e in (0, 1, 2):
    ep = lab.load(f"outputs/programs/none/libero_goal_lan__3/episode_{e}.npz")
    o = ep.obs(0)
    ref = lab.chunk(o, C[4])
    print(f"ep{e} eef {np.round(o['robot0_eef_pos'], 3)}")
    for c in C:
        ch = lab.chunk(o, c)
        print(f"   {A.dist(ch, ref):5.1f}cm from 'open middle'  end {A.summary(ch)['end_cm']} close_at {A.summary(ch)['close_at']}  {c}")
