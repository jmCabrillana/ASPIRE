"""Tune steerlib.canon on my own paraphrases of trained instructions (library/tools/canon_devset.json), each scored against its
source with the source excluded from the dialect (as for a real target). Reports exact match and word-level F1."""
import sys, json; sys.path[:0] = ['library', '.']
from collections import Counter
from steerlib import canon
dev = json.load(open('library/tools/canon_devset.json'))
tr = json.load(open('library/notes/lab/trained_instructions.json'))
def f1(a, b):
    a, b = canon.words(a), canon.words(b); c = sum((Counter(a) & Counter(b)).values())
    return 0 if c == 0 else 2 * c / (len(a) + len(b))
import os
if os.environ.get("SCORER"): canon.SCORER["name"] = os.environ["SCORER"]
VER = os.environ.get("VER", "v3")
if os.environ.get("PRIOR"): canon.PRIOR["kind"] = os.environ["PRIOR"]
if os.environ.get("MULTI"): canon.MULTI["on"] = True
if os.environ.get("COMP"): canon.MULTI["compound"] = True
if os.environ.get("COMPL"): canon.MULTI["complete"] = True
if os.environ.get("BEAM"): canon.MULTI["beam"] = int(os.environ["BEAM"])
for w in [float(x) for x in sys.argv[1].split(",")]:
    em = f = f0 = 0
    for x, s, k in dev:
        y = tr[s][k]; out = canon.canonical(x, exclude=(s, k), w=w, version=VER)
        em += out == " ".join(canon.words(y)); f += f1(out, y); f0 += f1(x, y)
        if len(sys.argv) > 2: print(f"   {f1(x,y):.2f}->{f1(out,y):.2f} {x!r} -> {out!r}   [{y}]")
    print(f"w={w}: exact {em}/{len(dev)}  F1 {f/len(dev):.3f} (input {f0/len(dev):.3f})", flush=True)
