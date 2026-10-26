"""Offline dissection of SmolVLA on saved observations: no simulator, one forward pass per experiment (well under a second).

Saved observations come from `scripts/run_program.py --save-obs` (outputs/programs/<p>/<suite>__<task>/episode_<n>.npz).
Typical uses: how the predicted action chunk changes under an intervention; activation patching or swapping between two
observations (success vs failure, trained vs perturbed scene, two instructions); which tokens or layers matter (ablation).

    lab = Lab(policy)
    ep = lab.load(".../episode_3.npz")
    a0 = lab.chunk(ep.obs(0), ep.instruction)                           # (50, 7) actions, fixed noise seed
    acts = lab.capture(ep.obs(0), ep.instruction, "vlm", [8])          # {8: (tokens, 960)} per-token MLP outputs
    with lab.replace("vlm", 8, acts_from_other_obs[8], tokens=lab.span("lang")):
        a1 = lab.chunk(ep.obs(0), ep.instruction)                      # instruction tokens of layer 8 swapped in
    print(np.abs(a1 - a0).mean(0))
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from steer.hooks import layers

OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
PREFIX = (("image", 64), ("image2", 64), ("lang", 48), ("state", 1))   # token layout of SmolVLA's prefix


@dataclass
class Episode:
    data: dict
    instruction: str
    success: bool

    def __len__(self) -> int:
        return len(self.data["t"])

    def obs(self, i: int) -> dict:
        """The raw observation at the start of chunk i."""
        return {k: self.data[k][i] for k in OBS_KEYS}

    def actions(self, i: int, horizon: int = 50) -> np.ndarray:
        """The actions executed from the start of chunk i (up to `horizon`)."""
        t0 = int(self.data["t"][i]) - 1
        return self.data["actions"][t0:t0 + horizon]


class Lab:
    def __init__(self, policy):
        self.policy = policy

    @staticmethod
    def load(path) -> Episode:
        d = dict(np.load(path, allow_pickle=False))
        return Episode(d, str(d.pop("instruction")), bool(d.pop("success")))

    @staticmethod
    def span(segment: str) -> slice:
        """Token positions of a prefix segment (image, image2, lang, state) in VLM-layer activations. The instruction length
        varies with the instruction: use slice(128, -1) for "lang" and -1 for "state" on a captured tensor."""
        if segment == "lang":
            return slice(128, -1)
        if segment == "state":
            return slice(-1, None)
        start = 0
        for name, n in PREFIX:
            if name == segment:
                return slice(start, start + n)
            start += n
        raise ValueError(segment)

    @torch.no_grad()
    def chunk(self, obs: dict, instruction: str, seed: int = 0) -> np.ndarray:
        """Predicted (50, 7) action chunk, unnormalised, with a fixed flow-matching noise seed (so differences are causal)."""
        p = self.policy
        batch = p.pre(p.batch(obs, instruction))
        torch.manual_seed(seed)
        p.policy.reset()
        actions = p.policy.predict_action_chunk(batch)                  # (1, 50, 7) normalised
        return np.stack([p.post(actions[:, i]).squeeze(0).float().cpu().numpy() for i in range(actions.shape[1])])

    @torch.no_grad()
    def capture(self, obs: dict, instruction: str, where: str, which: list[int], seed: int = 0) -> dict[int, torch.Tensor]:
        """Per-token MLP outputs. vlm: {layer: (prefix tokens, 960)}; expert: {layer: (denoising steps, 50, 720)}."""
        out = {i: [] for i in which}
        handles = [layers(self.policy, where)[i].mlp.register_forward_hook(
            lambda m, inp, o, i=i: out[i].append(o.detach()[0].float().cpu())) for i in which]
        try:
            self.chunk(obs, instruction, seed)
        finally:
            for h in handles:
                h.remove()
        return {i: (v[0] if where == "vlm" else torch.stack(v)) for i, v in out.items()}

    @contextmanager
    def edit(self, where: str, layer: int, fn):
        """Apply fn(output) -> output to one layer's MLP output during the block."""
        h = layers(self.policy, where)[layer].mlp.register_forward_hook(lambda m, inp, o: fn(o))
        try:
            yield
        finally:
            h.remove()

    @contextmanager
    def replace(self, where: str, layer: int, source: torch.Tensor, tokens: slice | None = None):
        """Swap in activations captured elsewhere (all tokens, or a token span). For the expert, `source` is the
        (steps, 50, width) capture and is replayed step by step."""
        state = {"step": 0}

        def fn(o):
            src = source if where == "vlm" else source[min(state["step"], len(source) - 1)]
            state["step"] += 1
            o = o.clone()
            sl = tokens if tokens is not None else slice(None)
            o[0, sl] = src[sl].to(o.device, o.dtype)
            return o

        with self.edit(where, layer, fn):
            yield


# ---- neurons: residual norms, attribution, dosed pushes ---------------------------------------------------------------

def down_proj(policy, where: str, layer: int):
    return layers(policy, where)[layer].mlp.down_proj


def neuron_width(policy, where: str) -> int:
    return down_proj(policy, where, 0).in_features


class Neurons:
    """MLP neurons (inputs of down_proj). A dosed push follows the recipe used for LLM neuron steering:

        added activation = dose * median(||residual leaving layer L||, tokens 1:) / ||W_down[:, n]||

    so a dose of 1 moves the residual stream by about one typical residual norm along that neuron's output direction.
    """

    def __init__(self, lab: Lab):
        self.lab, self.policy = lab, lab.policy
        self._median: dict[tuple[str, int], float] = {}

    @torch.no_grad()
    def residual_median(self, obs: dict, instruction: str, where: str, layer: int) -> float:
        """Median norm of the residual stream leaving `layer` (input of the next layer's input norm, or of the final norm)."""
        key = (where, layer)
        if key in self._median:
            return self._median[key]
        m = self.policy.policy.model.vlm_with_expert
        stack = layers(self.policy, where)
        target = stack[layer + 1].input_layernorm if layer + 1 < len(stack) else (
            m.get_vlm_model().text_model.norm if where == "vlm" else m.lm_expert.norm)
        norms = []
        h = target.register_forward_pre_hook(lambda mod, inp: norms.append(inp[0].detach()[0, 1:].float().norm(dim=-1).cpu()))
        try:
            self.lab.chunk(obs, instruction)
        finally:
            h.remove()
        self._median[key] = float(torch.cat(norms).median())
        return self._median[key]

    def dose_to_activation(self, obs, instruction, where: str, layer: int, neuron: int, dose: float) -> float:
        w = down_proj(self.policy, where, layer).weight[:, neuron].float().norm()
        return dose * self.residual_median(obs, instruction, where, layer) / float(w)

    @contextmanager
    def push(self, pushes: list[tuple[str, int, int, float]]):
        """Add activations to neurons during the block: pushes = [(where, layer, neuron, added activation)] (all tokens)."""
        handles = []
        by_layer: dict[tuple[str, int], list[tuple[int, float]]] = {}
        for where, layer, n, value in pushes:
            by_layer.setdefault((where, layer), []).append((n, value))
        for (where, layer), items in by_layer.items():
            idx = torch.tensor([n for n, _ in items])
            val = torch.tensor([v for _, v in items])

            def pre(mod, inp, idx=idx, val=val):
                x = inp[0].clone()
                x[..., idx.to(x.device)] += val.to(x.device, x.dtype)
                return (x,)

            handles.append(down_proj(self.policy, where, layer).register_forward_pre_hook(pre))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def attribute(self, obs: dict, instruction: str, target, where: str, which: list[int], seed: int = 0) -> dict[int, torch.Tensor]:
        """Attribution patching: activation x gradient of target(normalised action chunk (50, 7)) per neuron, summed over tokens
        (and denoising steps). Positive: raising the neuron raises the target (first-order estimate; check with push())."""
        p = self.policy
        acts: dict[int, list[torch.Tensor]] = {i: [] for i in which}

        def keep(i):
            def pre(mod, inp):
                x = inp[0]
                if x.requires_grad:
                    x.retain_grad()
                    acts[i].append(x)
            return pre

        handles = [down_proj(p, where, i).register_forward_pre_hook(keep(i)) for i in which]
        try:
            batch = p.policy._prepare_batch(p.pre(p.batch(obs, instruction)))
            torch.manual_seed(seed)
            with torch.enable_grad():
                for prm in p.policy.parameters():
                    prm.requires_grad_(False)
                emb = p.policy.model.vlm_with_expert.get_vlm_model().text_model.get_input_embeddings()
                h = emb.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
                try:
                    actions = p.policy._get_action_chunk(batch, None)[0]
                finally:
                    h.remove()
                target(actions).backward()
        finally:
            for hh in handles:
                hh.remove()
        return {i: sum((x * x.grad).detach()[0].float().sum(dim=0).cpu() for x in acts[i] if x.grad is not None)
                for i in which}
