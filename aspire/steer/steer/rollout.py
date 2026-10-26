"""Roll out the frozen SmolVLA (lerobot/smolvla_libero) in LIBERO / LIBERO-PRO, the way lerobot's LIBERO eval does.

Observation: agentview and wrist images 256x256, rotated 180 degrees; state = eef position, eef axis-angle, gripper qpos (8).
Episode: init state `episode % n_init_states`, 10 no-op steps, then policy actions until success or the suite's step limit.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")   # LIBERO init-state files are pickled numpy arrays
from libero import benchmark
from libero.envs import OffScreenRenderEnv
from libero.utils import get_libero_path

CHECKPOINT = "lerobot/smolvla_libero"
MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
SIZE = 256


def max_steps(suite: str) -> int:
    base = next((k for k in sorted(MAX_STEPS, key=len, reverse=True) if suite == k or suite.startswith(k + "_")), None)
    return MAX_STEPS.get(base, 500)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite quaternion (x, y, z, w) to axis-angle, as lerobot's LiberoProcessorStep."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(max(1.0 - w * w, 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(w) / den


def bddl_language(path: str, default: str) -> str:
    """The instruction written in the BDDL file (LIBERO-PRO puts its language perturbations there)."""
    m = re.search(r"\(:language\s+([^)]*)\)", Path(path).read_text())
    return m.group(1).strip() if m else default


@dataclass
class Task:
    suite: str
    task_id: int
    name: str
    instruction: str
    bddl: str
    init_states: np.ndarray | None = field(repr=False)


def load_task(suite: str, task_id: int) -> Task:
    ts = benchmark.get_benchmark_dict()[suite]()
    t = ts.get_task(task_id)
    bddl = str(Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file)
    try:
        init = ts.get_task_init_states(task_id)
    except (FileNotFoundError, OSError):
        init = None
    # The policy was trained on the benchmark's task language; only the *_lan suites perturb it (in the BDDL file).
    instruction = bddl_language(bddl, t.language) if suite.endswith("_lan") else t.language
    return Task(suite, task_id, t.name, instruction, bddl, init)


class Policy:
    """SmolVLA with lerobot's pre/post processors. `hooks` is filled by steer.hooks."""

    def __init__(self, checkpoint: str = CHECKPOINT, device: str = "cuda"):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=checkpoint)
        self.device = device

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def act(self, obs: dict, instruction: str) -> np.ndarray:
        def img(key):
            a = np.ascontiguousarray(obs[key][::-1, ::-1])           # rotate 180 degrees
            return torch.from_numpy(a).permute(2, 0, 1).float().div(255).unsqueeze(0)
        state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
        batch = {"observation.images.image": img("agentview_image"), "observation.images.image2": img("robot0_eye_in_hand_image"),
                 "observation.state": torch.from_numpy(state).float().unsqueeze(0), "task": [instruction]}
        action = self.policy.select_action(self.pre(batch))
        return self.post(action).squeeze(0).float().cpu().numpy()


def make_env(task: Task) -> OffScreenRenderEnv:
    env = OffScreenRenderEnv(bddl_file_name=task.bddl, camera_heights=SIZE, camera_widths=SIZE, control_freq=20)
    env.reset()
    return env


def run_episode(env: OffScreenRenderEnv, policy: Policy, task: Task, episode: int, on_step=None) -> dict:
    """One episode. on_step(t, obs, action) may return a replacement action (for action-level programs)."""
    policy.reset()
    env.seed(episode)
    obs = env.reset()
    if task.init_states is not None and len(task.init_states):
        obs = env.set_init_state(task.init_states[episode % len(task.init_states)])
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    for robot in env.robots:
        robot.controller.use_delta = True
    t0, success, steps = time.time(), False, 0
    for steps in range(1, max_steps(task.suite) + 1):
        action = policy.act(obs, task.instruction)
        if on_step is not None:
            replaced = on_step(steps, obs, action)
            action = action if replaced is None else replaced
        obs, _, done, _ = env.step(action.tolist())
        if env.check_success():
            success = True
            break
        if done:
            break
    return {"episode": episode, "success": success, "steps": steps, "seconds": round(time.time() - t0, 1)}
