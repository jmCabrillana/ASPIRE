"""Classic interpretability on SmolVLA: an activation bank over many observations, neuron selectivity, max-activating examples,
and linear probes. Everything runs offline on saved episodes (scripts/run_program.py --save-obs).

    bank = Bank.build(lab, files, layers=[4, 8, 12], where="vlm")       # neuron activations (down_proj inputs) per chunk
    bank.labels["phase"], bank.labels["success"], bank.labels["dz"] ...  # per-chunk labels (see Bank.build)
    bank.selective("vlm8", "success")                                    # neurons whose activation separates the label (AUROC)
    bank.top_examples("vlm8", 1234)                                      # which chunks / tokens activate a neuron most
    bank.probe("vlm8", "phase")                                          # cross-validated linear probe (grouped by episode)
    word_neurons(lab, obs, "turn on the stove", "vlm", 8, "stove")       # neurons that fire on one instruction word

Per chunk, VLM neurons are summarised per prefix segment (mean over image, image2, lang, and the state token); expert neurons
are averaged over the 50 action tokens and the 10 denoising steps. Keys are "<where><layer>:<segment>" (segment "all" for the
expert), e.g. "vlm8:lang".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from steer.lab import PREFIX, Lab, down_proj


def neuron_acts(lab: Lab, obs: dict, instruction: str, where: str, which: list[int]) -> dict[int, torch.Tensor]:
    """Neuron activations (inputs of down_proj). vlm: {layer: (177 tokens, 2560)}; expert: {layer: (steps, 50, n)}."""
    out = {i: [] for i in which}
    hs = [down_proj(lab.policy, where, i).register_forward_pre_hook(
        lambda m, inp, i=i: out[i].append(inp[0].detach()[0].float().cpu())) for i in which]
    try:
        lab.chunk(obs, instruction)
    finally:
        for h in hs:
            h.remove()
    return {i: (v[0] if where == "vlm" else torch.stack(v)) for i, v in out.items()}


def segments(x: torch.Tensor) -> dict[str, torch.Tensor]:
    out, start = {}, 0
    for name, n in PREFIX:
        out[name] = x[start:start + n].mean(0)
        start += n
    return out


@dataclass
class Bank:
    acts: dict[str, np.ndarray] = field(default_factory=dict)     # key -> (chunks, neurons)
    labels: dict[str, np.ndarray] = field(default_factory=dict)   # name -> (chunks,)
    meta: list[dict] = field(default_factory=list)                 # per chunk: file, chunk index, t, instruction

    @classmethod
    def build(cls, lab: Lab, files, layers: list[int], where: str = "vlm", instruction: str | None = None) -> "Bank":
        """Labels per chunk: success (episode), episode (group id), phase (0 open / 1 closed gripper), t, eef x/y/z, and the
        executed chunk's mean dx/dy/dz and gripper command (from the saved actions)."""
        acts: dict[str, list] = {}
        labels: dict[str, list] = {k: [] for k in ("success", "episode", "phase", "t", "x", "y", "z", "dx", "dy", "dz", "grip")}
        meta = []
        for e_id, f in enumerate(files):
            ep = lab.load(f)
            text = instruction or ep.instruction
            for i in range(len(ep)):
                o = ep.obs(i)
                a = neuron_acts(lab, o, text, where, layers)
                for l in layers:
                    if where == "vlm":
                        for seg, v in segments(a[l]).items():
                            acts.setdefault(f"vlm{l}:{seg}", []).append(v.numpy())
                    else:
                        acts.setdefault(f"expert{l}:all", []).append(a[l].mean(dim=(0, 1)).numpy())
                act = ep.actions(i)
                labels["success"].append(float(ep.success)); labels["episode"].append(e_id)
                labels["phase"].append(float(o["robot0_gripper_qpos"][0] < 0.03)); labels["t"].append(float(ep.data["t"][i]))
                for k, v in zip("xyz", o["robot0_eef_pos"]):
                    labels[k].append(float(v))
                for k, j in (("dx", 0), ("dy", 1), ("dz", 2), ("grip", 6)):
                    labels[k].append(float(act[:, j].mean()) if len(act) else 0.0)
                meta.append({"file": str(f), "chunk": i, "t": int(ep.data["t"][i]), "instruction": text})
        return cls({k: np.stack(v) for k, v in acts.items()}, {k: np.array(v) for k, v in labels.items()}, meta)

    def save(self, path) -> None:
        np.savez_compressed(path, **{f"a::{k}": v for k, v in self.acts.items()}, **{f"l::{k}": v for k, v in self.labels.items()},
                            meta=np.array([str(m) for m in self.meta]))

    def _key(self, key: str) -> str:
        return key if ":" in key else f"{key}:lang" if key.startswith("vlm") else f"{key}:all"

    def selective(self, key: str, label: str, top: int = 20, threshold: float | None = None):
        """Neurons ranked by how well their activation separates a label: AUROC for binary labels (|AUROC - 0.5|), Pearson r
        otherwise. Returns [(neuron, score)] (score > 0: higher activation with the label)."""
        x, y = self.acts[self._key(key)], self.labels[label]
        if threshold is not None:
            y = (y > threshold).astype(float)
        if set(np.unique(y)) <= {0.0, 1.0}:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y)) < 2:
                raise ValueError(f"label {label} has one class only")
            s = np.array([roc_auc_score(y, x[:, n]) - 0.5 for n in range(x.shape[1])])
        else:
            xc, yc = x - x.mean(0), y - y.mean()
            s = (xc * yc[:, None]).sum(0) / (np.linalg.norm(xc, axis=0) * np.linalg.norm(yc) + 1e-8)
        idx = np.argsort(-np.abs(s))[:top]
        return [(int(n), round(float(s[n]), 4)) for n in idx]

    def top_examples(self, key: str, neuron: int, k: int = 8):
        """The chunks where a neuron is most active, with their labels."""
        x = self.acts[self._key(key)][:, neuron]
        idx = np.argsort(-x)[:k]
        return [{"act": round(float(x[i]), 3), **self.meta[i], **{l: round(float(v[i]), 3) for l, v in self.labels.items()}}
                for i in idx]

    def probe(self, key: str, label: str, threshold: float | None = None, folds: int = 5) -> dict:
        """Cross-validated linear probe (logistic for binary labels, ridge otherwise), folds grouped by episode so the probe
        cannot memorise an episode. Returns the score, chance level, and the unit direction of a probe fit on all data."""
        from sklearn.linear_model import LogisticRegression, RidgeCV
        from sklearn.model_selection import GroupKFold, cross_val_score
        x, y, g = self.acts[self._key(key)], self.labels[label], self.labels["episode"]
        if threshold is not None:
            y = (y > threshold).astype(int)
        binary = set(np.unique(y)) <= {0, 1}
        model = LogisticRegression(max_iter=2000, C=0.1) if binary else RidgeCV(alphas=np.logspace(-2, 4, 13))
        k = min(folds, len(np.unique(g)))
        scores = cross_val_score(model, x, y, groups=g, cv=GroupKFold(n_splits=k)) if k >= 2 else np.array([np.nan])
        model.fit(x, y)
        w = np.ravel(model.coef_)
        chance = max(np.mean(y), 1 - np.mean(y)) if binary else 0.0
        return {"metric": "accuracy" if binary else "R2", "score": round(float(np.mean(scores)), 3), "chance": round(float(chance), 3),
                "direction": w / (np.linalg.norm(w) + 1e-8), "n": len(y)}


def word_neurons(lab: Lab, obs: dict, instruction: str, where: str, layer: int, word: str, top: int = 20):
    """Neurons most selective for the tokens of `word` within the instruction (activation on its tokens minus the other
    instruction tokens), on one observation. Returns [(neuron, contrast)]."""
    tok = lab.policy.pre.steps[3].tokenizer if hasattr(lab.policy.pre, "steps") else None
    a = neuron_acts(lab, obs, instruction, where, [layer])[layer]
    lang = a[Lab.span("lang")]
    ids = (tok or _tokenizer())(instruction + "\n", add_special_tokens=False)["input_ids"]
    pieces = [(tok or _tokenizer()).decode([i]) for i in ids]
    mask = torch.tensor([word.lower() in p.lower() for p in pieces] + [False] * (len(lang) - len(pieces)))[:len(lang)]
    if not mask.any():
        raise ValueError(f"{word!r} not among the tokens {pieces}")
    contrast = lang[mask].mean(0) - lang[~mask][:len(pieces) - int(mask.sum())].mean(0)
    idx = torch.argsort(-contrast)[:top]
    return [(int(n), round(float(contrast[n]), 3)) for n in idx]


_TOK = {}


def _tokenizer():
    if "t" not in _TOK:
        from transformers import AutoTokenizer
        _TOK["t"] = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    return _TOK["t"]


# ---- groups and interactions -------------------------------------------------------------------------------------------

def clusters(bank: Bank, key: str, n_groups: int = 32, min_std: float = 1e-3) -> dict[int, list[int]]:
    """Groups of neurons that co-activate across the bank's chunks (k-means on standardised activation profiles).
    Returns {group: [neurons]}; neurons that never vary are left out."""
    from sklearn.cluster import KMeans
    x = bank.acts[bank._key(key)]
    keep = np.where(x.std(0) > min_std)[0]
    z = (x[:, keep] - x[:, keep].mean(0)) / x[:, keep].std(0)
    lab_ = KMeans(n_clusters=min(n_groups, len(keep)), n_init=4, random_state=0).fit_predict(z.T)
    groups: dict[int, list[int]] = {}
    for n, g in zip(keep, lab_):
        groups.setdefault(int(g), []).append(int(n))
    return groups


def interaction(lab: Lab, obs: dict, instruction: str, push_a: list, push_b: list, metric) -> dict:
    """Effect of two neuron pushes (lists for Neurons.push) alone and together on metric(chunk_with, chunk_without), and the
    synergy effect(A+B) - effect(A) - effect(B). Non-zero synergy means the groups interact (gate, cancel or reinforce)."""
    from steer.lab import Neurons
    nr = Neurons(lab)
    base = lab.chunk(obs, instruction)

    def run(pushes):
        with nr.push(pushes):
            return metric(lab.chunk(obs, instruction), base)

    ea, eb, eab = run(push_a), run(push_b), run(push_a + push_b)
    return {"A": ea, "B": eb, "A+B": eab, "synergy": eab - ea - eb}


def downstream(policy, where: str, layer: int, neuron: int, to_layer: int, top: int = 20):
    """Direct (virtual-weight) connections from a neuron to the neurons of a later layer through the residual stream:
    W_down[l][:, n] projected on the gate and up rows of layer `to_layer` (ignores the norms in between).
    Returns [(neuron, gate weight, up weight)], strongest |gate| first. Edit a connection with ctx.edit_weights on those rows."""
    from steer.hooks import layers
    w_out = layers(policy, where)[layer].mlp.down_proj.weight[:, neuron].float()
    mlp = layers(policy, where)[to_layer].mlp
    g = (mlp.gate_proj.weight.float() @ w_out).detach().cpu()
    u = (mlp.up_proj.weight.float() @ w_out).detach().cpu()
    idx = torch.argsort(-g.abs())[:top]
    return [(int(m), round(float(g[m]), 4), round(float(u[m]), 4)) for m in idx]
