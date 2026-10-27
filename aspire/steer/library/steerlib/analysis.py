"""Offline analysis helpers on top of steer.lab (no simulator).

Prefix layout of SmolVLA (lerobot/smolvla_libero), checked by recording attention shapes: always 177 tokens = 64 agentview,
64 wrist, 48 language positions (the processor pads to tokenizer_max_length; the instruction's n real tokens, newline included,
are 128..128+n-1, the rest are masked padding), 1 state token at 176. `spans()` gives the real-token slice ("lang"), the padding
("pad") and the state token.
"""

from __future__ import annotations

import numpy as np
import torch

_TOK = {}


def tokenizer():
    if "t" not in _TOK:
        from transformers import AutoTokenizer
        _TOK["t"] = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    return _TOK["t"]


def tokens(instruction: str) -> list[str]:
    """The instruction's tokens as the policy sees them (the processor appends a newline)."""
    t = tokenizer()
    text = instruction if instruction.endswith("\n") else instruction + "\n"
    return [t.decode([i]) for i in t(text)["input_ids"]]


def spans(instruction: str) -> dict[str, slice]:
    n = len(tokens(instruction))
    return {"image": slice(0, 64), "image2": slice(64, 128), "lang": slice(128, 128 + n), "pad": slice(128 + n, 176),
            "state": slice(176, 177)}


def path(chunk: np.ndarray, scale: float = 0.0095) -> np.ndarray:
    """Cumulative end-effector displacement (m) of an unnormalised (T, 7) chunk under OSC delta control (approximation:
    measured 0.0084-0.0102 m per action unit per step by regressing executed chunks on eef displacement; experiments E01-E10
    used 0.05, which scales all their cm numbers by ~5x, relative comparisons unaffected). Returns (T, 3)."""
    return np.cumsum(np.clip(chunk[:, :3], -1, 1) * scale, axis=0)


def summary(chunk: np.ndarray) -> dict:
    """Where a chunk goes: final displacement (cm), first step the gripper command closes (<0), mean gripper command."""
    p = path(chunk)
    close = np.where(chunk[:, 6] > 0)[0]          # LIBERO: +1 closes the gripper, -1 opens
    return {"end_cm": np.round(p[-1] * 100, 1).tolist(), "close_at": int(close[0]) if len(close) else None,
            "grip_mean": round(float(chunk[:, 6].mean()), 2)}


def dist(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two chunks: mean over steps of the displacement-path difference, in cm."""
    return float(np.linalg.norm(path(a) - path(b), axis=1).mean() * 100)


def load(n_action_steps: int | None = None):
    """(policy, lab, neurons) on the GPU."""
    from steer.lab import Lab, Neurons
    from steer.rollout import Policy
    policy = Policy(n_action_steps=n_action_steps)
    lab = Lab(policy)
    return policy, lab, Neurons(lab)
