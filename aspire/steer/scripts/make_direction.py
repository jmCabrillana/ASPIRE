#!/usr/bin/env python3
"""Store a steering direction (all layers of one site) from recorded rollouts (scripts/record.py).

  .venv/bin/python scripts/make_direction.py --name goal3_success --record libero_goal__3 --kind success --where vlm
  .venv/bin/python scripts/make_direction.py --name goal1_unswap --record libero_goal_swap__1 --kind shift:libero_goal__1 --where vlm

success: successful minus failed episodes of the recording.   shift:<recording>: that recording minus this one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from steer.directions import shift_direction, success_direction  # noqa: E402
from steer.library import save_direction  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--record", required=True, help="outputs/record/<this>.pt")
    p.add_argument("--kind", default="success")
    p.add_argument("--where", default="vlm", choices=("vlm", "expert"))
    a = p.parse_args()
    rec = HERE / "outputs" / "record"
    data = torch.load(rec / f"{a.record}.pt")
    if a.kind == "success":
        unit, spread, n_ok, n_fail = success_direction(data, a.where)
        source = f"success-minus-failure of {a.record} ({n_ok} vs {n_fail} episodes)"
    elif a.kind.startswith("shift:"):
        unit, spread = shift_direction(torch.load(rec / f"{a.kind[6:]}.pt"), data, a.where)
        source = f"{a.kind[6:]} minus {a.record}"
    else:
        raise SystemExit(f"unknown kind {a.kind}")
    print(save_direction(a.name, a.where, unit, spread, source), source)


if __name__ == "__main__":
    main()
