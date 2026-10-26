"""Conditions a program can use to decide when to intervene (observation, episode memory, VLA internals)."""

from __future__ import annotations


def gripper_closed(obs, threshold: float = 0.03) -> bool:
    """Gripper finger position below threshold: closed (on an object or on nothing)."""
    return float(obs["robot0_gripper_qpos"][0]) < threshold


def phase(obs, memory: dict, threshold: float = 0.03) -> str:
    """'reach' until the gripper first closes, then 'hold' while closed, 'after' once it reopens. Keeps state in memory."""
    closed = gripper_closed(obs, threshold)
    if closed:
        memory["closed_once"] = True
        return "hold"
    return "after" if memory.get("closed_once") else "reach"


def attention_share(internals, layer: int, segment: str) -> float | None:
    """Share of the action tokens' attention on a prefix segment (image, image2, lang, state, actions) in the previous chunk."""
    if not internals:
        return None
    return internals["attention"].get(layer, {}).get(segment)
