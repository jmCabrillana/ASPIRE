"""Save a grid of agentview frames (chunk starts) of saved episodes as one PNG: frames.py out.png ep1.npz [ep2.npz ...]"""
import sys
import numpy as np
from PIL import Image
rows = []
for f in sys.argv[2:]:
    d = np.load(f)
    ims = [im[::-1, ::-1] for im in d["agentview_image"]][:8]      # rotate 180: upright view
    ims += [np.zeros_like(ims[0])] * (8 - len(ims))
    rows.append(np.concatenate(ims, 1))
Image.fromarray(np.concatenate(rows, 0)).resize((8 * 160, len(rows) * 160)).save(sys.argv[1])
