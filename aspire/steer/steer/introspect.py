"""Read-only instrumentation of SmolVLA: what the frozen policy is doing inside, summarised per action chunk.

Per chunk (one VLM prefill on the current observation, then `num_steps` flow-matching denoising passes of the action expert):
  segments   token counts of the prefix: image (agentview), image2 (wrist), lang, state; then the action tokens
  attention  for every expert layer that attends to the prefix: the share of the action tokens' attention that goes to each
             segment, averaged over heads, action tokens and denoising steps        {layer: {segment: share}}
  mlp_norm   norm of each layer's mean MLP output on the prefix (vlm) and on the action tokens (expert)
  proj       projection of the mean MLP output on named unit directions            {name: value}
The summaries are small on purpose: a steering program reads them, and they go into the episode trace.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from steer.hooks import layers

SEGMENTS = ("image", "image2", "lang", "state", "actions")


class Introspect:
    """Install once per policy; call begin_chunk()/end_chunk() around each chunk (Policy does it when attached)."""

    def __init__(self, policy, directions: dict[str, tuple[str, int, torch.Tensor]] | None = None, attention: bool = True):
        self.policy, self.directions, self.attention = policy, dict(directions or {}), attention
        self.m = policy.policy.model.vlm_with_expert
        self.sizes: list[int] = []          # prefix pieces in call order: image, image2, lang
        self.layer = -1
        self.current: dict | None = None
        self.chunks: list[dict] = []
        self._install()

    # ---- wiring ------------------------------------------------------------------------------------------------------
    def _install(self) -> None:
        m = self.m
        embed_image, embed_lang = m.embed_image, m.embed_language_tokens

        def image(img):
            out = embed_image(img)
            self.sizes.append(out.shape[1])
            return out

        def lang(tokens):
            out = embed_lang(tokens)
            self.sizes.append(out.shape[1])
            return out

        m.embed_image, m.embed_language_tokens = image, lang
        for name in ("forward_attn_layer", "forward_cross_attn_layer"):
            original = getattr(m, name)

            def wrapped(model_layers, inputs_embeds, layer_idx, *args, _original=original, **kwargs):
                self.layer = layer_idx
                return _original(model_layers, inputs_embeds, layer_idx, *args, **kwargs)

            setattr(m, name, wrapped)
        attend = m.eager_attention_forward

        def attention(attention_mask, batch_size, head_dim, q, k, v):
            if self.current is not None and self.attention:
                try:
                    self._attention(attention_mask, head_dim, q, k)
                except RuntimeError as e:            # never break a rollout for a log
                    self.attention = False
                    print(f"introspect: attention capture disabled ({e})")
            return attend(attention_mask, batch_size, head_dim, q, k, v)

        m.eager_attention_forward = attention
        for where in ("vlm", "expert"):
            for i, layer in enumerate(layers(self.policy, where)):
                layer.mlp.register_forward_hook(self._mlp(where, i))

    def _attention(self, mask, head_dim, q, k) -> None:
        """Share of the action tokens' attention on each segment (only for passes whose queries are the action tokens)."""
        chunk = self.policy.policy.config.chunk_size
        if q.shape[1] != chunk or len(self.sizes) < 3:
            return
        groups = q.shape[2] // k.shape[2]
        kk = k.repeat_interleave(groups, dim=2)                                  # (B, K, H, D)
        logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), kk.float()) * head_dim ** -0.5
        logits = logits.masked_fill(~mask[:, None, -q.shape[1]:, :k.shape[1]], torch.finfo(logits.dtype).min)
        probs = F.softmax(logits, dim=-1).mean(dim=(0, 1, 2))                    # (K,)
        bounds, start = {}, 0
        for name, n in zip(SEGMENTS, self.sizes[-3:][:2] + [self.sizes[-1], 1]):
            bounds[name] = (start, start + n)
            start += n
        bounds["actions"] = (start, k.shape[1])
        acc = self.current["attention"][self.layer]
        for name, (a, b) in bounds.items():
            acc[name].append(float(probs[a:b].sum()) if b > a else 0.0)

    def _mlp(self, where: str, i: int):
        def fn(module, inputs, output):
            if self.current is None:
                return
            mean = output.detach().float().mean(dim=1)[0]
            self.current["mlp_norm"][f"{where}{i}"].append(float(mean.norm()))
            for name, (w, layer, unit) in self.directions.items():
                if w == where and layer == i:
                    self.current["proj"][name].append(float(mean @ unit.to(mean.device)))
        return fn

    # ---- per chunk ---------------------------------------------------------------------------------------------------
    def begin_chunk(self, t: int) -> None:
        self.sizes = []
        self.current = {"t": t, "attention": defaultdict(lambda: defaultdict(list)), "mlp_norm": defaultdict(list),
                        "proj": defaultdict(list)}

    def end_chunk(self) -> dict:
        c, self.current = self.current, None
        sizes = self.sizes[:2] + self.sizes[-1:] if len(self.sizes) >= 3 else self.sizes
        summary = {"t": c["t"], "segments": dict(zip(("image", "image2", "lang"), sizes)),
                   "attention": {l: {s: round(sum(v) / len(v), 4) for s, v in d.items()} for l, d in c["attention"].items()},
                   "mlp_norm": {k: round(sum(v) / len(v), 3) for k, v in c["mlp_norm"].items()},
                   "proj": {k: round(sum(v) / len(v), 4) for k, v in c["proj"].items()}}
        self.chunks.append(summary)
        return summary

    def reset(self) -> None:
        self.chunks = []
