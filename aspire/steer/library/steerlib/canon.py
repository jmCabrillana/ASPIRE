"""Instruction canonicalisation: translate an instruction into the policy's training dialect (noisy channel).

Finding (JOURNAL E02): the fine-tuned 16-layer VLM of smolvla_libero has lost general English. Paraphrase words it never saw
in training ("switch", "slide", "move", sentence-initial "place", possessive "’s") get representations near unrelated object
nouns or the end-of-instruction token, so the action expert gets no usable verb. The base SmolVLM2 (all 32 layers + lm_head)
still knows English. This module rewrites the instruction c* = argmax_c  w * log P_dialect(c) + log P_LM(x | c) + log P_LM(c | x)
  P_dialect: add-k bigram model over the trained instructions (all 4 trained suites), with the target's twin task excluded
             (for a *_lan / *_swap / ... suite and task id k, the base suite's task k is dropped, so the original wording of the
             target is never used); every word of c must belong to the dialect vocabulary.
  P_LM:      the base SmolVLM2 language model, paraphrase prompt in both directions (semantic equivalence).
Search: hill climbing over single-word edits (substitute / delete / insert a vocabulary word), batched on the GPU.
General: valid for any instruction and any suite; needs only the instruction string and the list of trained instructions.

    from steerlib import canon
    new = canon.canonical("switch on the stove", exclude=("libero_goal", 7))      # cached in library/notes/canon_cache.json
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINED = ROOT / "notes" / "lab" / "trained_instructions.json"
CACHE = ROOT / "notes" / "canon_cache.json"
LM_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
_LM = {}


def words(s: str) -> list[str]:
    s = s.lower().replace("’", "'").replace("-", " and ")
    s = re.sub(r"[^a-z' ]", " ", s)
    return s.split()


def possessive(ws: list[str]) -> list[str]:
    """General English rule: 'the A's B C <prep|and|end>' -> 'the B C of the A'."""
    STOP = {"on", "in", "into", "inside", "to", "and", "of", "from", "at", "onto", "with", "next", "between"}
    for i, w in enumerate(ws):
        if w.endswith("'s") and len(w) > 2:
            owner = w[:-2]
            j = i + 1
            while j < len(ws) and ws[j] not in STOP:
                j += 1
            det_start = i
            while det_start > 0 and ws[det_start - 1] not in STOP and ws[det_start - 1] not in {"open", "put", "push", "move"} \
                    and det_start > i - 3 and ws[det_start - 1] in {"the", "a"}:
                det_start -= 1
            return possessive(ws[:det_start] + ["the"] + ws[i + 1:j] + ["of", "the", owner] + ws[j:])
    return ws


class Dialect:
    """Add-k bigram model over trained instructions (sentence start/end markers)."""

    def __init__(self, exclude: tuple[str, int] | None = None, k: float = 0.1):
        tr = json.loads(TRAINED.read_text())
        self.sents = [words(s) for suite, l in tr.items() for i, s in enumerate(l) if (suite, i) != tuple(exclude or ())]
        self.vocab = sorted({w for s in self.sents for w in s})
        self.uni, self.bi = Counter(), Counter()
        for s in self.sents:
            seq = ["<s>"] + s + ["</s>"]
            self.uni.update(seq[:-1])
            self.bi.update(zip(seq[:-1], seq[1:]))
        self.k, self.V = k, len(self.vocab) + 1

    def unseen_n(self, ws: list[str], n: int) -> int:
        """Number of n-grams of ws (with sentence markers) never seen in the trained instructions."""
        key = f"ng{n}"
        if not hasattr(self, key):
            c = Counter()
            for s in self.sents:
                seq = ["<s>"] + s + ["</s>"]
                c.update(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
            setattr(self, key, c)
        c = getattr(self, key)
        seq = ["<s>"] + ws + ["</s>"]
        return sum(c[tuple(seq[i:i + n])] == 0 for i in range(len(seq) - n + 1))

    def seen(self, ngram: list[str]) -> bool:
        """True if the word sequence occurs contiguously in some trained instruction (no sentence markers)."""
        n = len(ngram)
        return any(s[i:i + n] == list(ngram) for s in self.sents for i in range(len(s) - n + 1))

    def unseen3(self, ws: list[str]) -> int:
        """Number of trigrams of ws (with sentence markers) never seen in the trained instructions."""
        if not hasattr(self, "tri"):
            self.tri = Counter()
            for s in self.sents:
                seq = ["<s>"] + s + ["</s>"]
                self.tri.update(zip(seq[:-2], seq[1:-1], seq[2:]))
        seq = ["<s>"] + ws + ["</s>"]
        return sum(self.tri[t] == 0 for t in zip(seq[:-2], seq[1:-1], seq[2:]))

    def unseen(self, ws: list[str]) -> int:
        """Number of bigrams of ws (with sentence markers) never seen in the trained instructions."""
        seq = ["<s>"] + ws + ["</s>"]
        return sum(self.bi[(a, b)] == 0 for a, b in zip(seq[:-1], seq[1:]))

    def logp(self, ws: list[str]) -> float:
        seq = ["<s>"] + ws + ["</s>"]
        return sum(math.log((self.bi[(a, b)] + self.k) / (self.uni[a] + self.k * self.V)) for a, b in zip(seq[:-1], seq[1:]))


SCORER = {"name": LM_NAME}     # set canon.SCORER["name"] = "NousResearch/Llama-2-7b-hf" for the stronger scorer (v4)


def _lm():
    name = SCORER["name"]
    if _LM.get("name") != name:
        _LM.clear()
        torch.cuda.empty_cache()
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        if "SmolVLM" in name:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
            lm, head = model.model.text_model, model.lm_head
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
            lm, head = model.model, model.lm_head
        _LM.update(name=name, tok=tok, lm=lm, head=head)
    return _LM["tok"], _LM["lm"], _LM["head"]


@torch.no_grad()
def lm_logp(pairs: list[tuple[str, str]], batch: int = 128) -> list[float]:
    """log P(target | prompt) for each (prompt, target), batched with right padding."""
    tok, lm, head = _lm()
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = []
    for b0 in range(0, len(pairs), batch):
        chunk = pairs[b0:b0 + batch]
        seqs = []
        for p, t in chunk:
            pi = tok(p, add_special_tokens=False)["input_ids"]
            full = tok(p + t, add_special_tokens=False)["input_ids"]      # tokenise jointly (sentencepiece merges at the seam)
            k = len(pi)
            while k > 0 and full[:k] != pi[:k]:
                k -= 1
            seqs.append((full[:k], full[k:]))
        n = max(len(p) + len(t) for p, t in seqs)
        ids = torch.full((len(seqs), n), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for r, (p, t) in enumerate(seqs):
            ids[r, :len(p) + len(t)] = torch.tensor(p + t)
            mask[r, :len(p) + len(t)] = 1
        h = lm(input_ids=ids.cuda(), attention_mask=mask.cuda()).last_hidden_state
        for r, (p, t) in enumerate(seqs):
            pos = torch.arange(len(p) - 1, len(p) + len(t) - 1, device=h.device)
            lp = torch.log_softmax(head(h[r, pos]).float(), -1)
            out.append(float(lp[torch.arange(len(t)), torch.tensor(t, device=h.device)].sum()))
    return out


PROMPT = "Robot command: {}\nThe same command in other words: "


PRIOR = {"kind": "bigram"}     # "bigram": w * log P_bigram (frequency-weighted); "unseen": -w * number of unseen bigrams (v5)


def scores(x: str, cands: list[list[str]], dialect: Dialect, w: float) -> list[float]:
    texts = [" ".join(c) for c in cands]
    a = lm_logp([(PROMPT.format(c), x) for c in texts])
    b = lm_logp([(PROMPT.format(x), c) for c in texts])
    if PRIOR["kind"] == "unseen":
        prior = lambda c: -dialect.unseen(c)
    elif PRIOR["kind"] == "unseen23":          # v6: unseen bigrams + 0.5 x unseen trigrams
        prior = lambda c: -dialect.unseen(c) - 0.5 * dialect.unseen3(c)
    elif PRIOR["kind"] == "frac234":          # v10: length-normalised (fraction of unseen 2/3/4-grams) x 10
        def prior(c):
            n2, n3, n4 = len(c) + 1, len(c), max(len(c) - 1, 1)
            return -10.0 * (dialect.unseen(c) / n2 + 0.5 * dialect.unseen_n(c, 3) / n3 + 0.5 * dialect.unseen_n(c, 4) / n4)
    elif PRIOR["kind"] == "unseen234":        # v7: unseen bigrams + 0.5 x unseen trigrams + 0.5 x unseen 4-grams
        prior = lambda c: -dialect.unseen(c) - 0.5 * dialect.unseen_n(c, 3) - 0.5 * dialect.unseen_n(c, 4)
    else:
        prior = dialect.logp
    return [w * prior(c) + ai + bi for c, ai, bi in zip(cands, a, b)]


# Closed-class English words (articles, pronouns, prepositions, particles, conjunctions): general grammar, not task knowledge.
FUNCTION = {"the", "a", "an", "it", "them", "up", "down", "of", "on", "onto", "in", "into", "inside", "to", "from", "at",
            "beside", "next", "near", "by", "between", "and", "then", "over", "off", "out", "with"}


PARTICLES = {"up", "on", "off", "down", "out"}
ARTICLES = {"the", "a", "an"}
PREPOSITIONS = {"of", "on", "onto", "in", "into", "inside", "to", "from", "at", "beside", "near", "by", "between", "over",
                "with", "next"}


def verb_slots(ws: list[str]) -> list[int]:
    """Positions of the main verbs of an imperative instruction: the first word and every word right after 'and'."""
    return [i for i, w in enumerate(ws) if i == 0 or ws[i - 1] == "and"]


def verb_phrase(ws: list[str], i: int) -> int:
    """Length of the verb phrase at slot i (verb + particle, e.g. 'pick up'); a particle followed by an article belongs to it."""
    return 2 if i + 1 < len(ws) and ws[i + 1] in PARTICLES and (i + 2 >= len(ws) or ws[i + 2] in ARTICLES | {"both"} or
                                                                   ws[i + 2] not in PREPOSITIONS) else 1


def dialect_verbs(d: "Dialect") -> list[tuple[str, ...]]:
    """Verb phrases the trained instructions use in verb slots (data-driven), e.g. ('pick', 'up'), ('put',), ('turn', 'on')."""
    out = set()
    for s in d.sents:
        for i in verb_slots(s):
            if s[i] not in FUNCTION:
                out.add(tuple(s[i:i + verb_phrase(s, i)]))
    return sorted(out)


def edits(cur: list[str], verbs: list[tuple[str, ...]], func: list[str]) -> list[list[str]]:
    """Single edits that keep content words: replace a verb phrase in a verb slot (or drop a verb followed by another verb,
    'slide open' -> 'open'), insert or delete an article, insert 'it'/'up', swap a preposition for another."""
    out = []
    preps = [p for p in func if p in PREPOSITIONS]
    for i in verb_slots(cur):
        n = verb_phrase(cur, i)
        out += [cur[:i] + list(v) + cur[i + n:] for v in verbs if tuple(cur[i:i + n]) != v]
        if i + 1 < len(cur) and any(cur[i + 1] == v[0] for v in verbs):
            out.append(cur[:i] + cur[i + 1:])
    for i in range(len(cur) + 1):
        for v in ("the", "it", "up"):
            out.append(cur[:i] + [v] + cur[i:])
        if i < len(cur):
            if cur[i] in ARTICLES:
                out.append(cur[:i] + cur[i + 1:])
            if cur[i] in PREPOSITIONS:
                out += [cur[:i] + [p] + cur[i + 1:] for p in preps if p != cur[i]]
    if MULTI.get("compound"):                  # v7: verb phrase + article in one edit (escapes 'lift black bowl' local optima)
        for i in verb_slots(cur):
            n = verb_phrase(cur, i)
            if i + n < len(cur) and cur[i + n] not in ARTICLES | PREPOSITIONS:
                out += [cur[:i] + list(v) + ["the"] + cur[i + n:] for v in verbs if tuple(cur[i:i + n]) != v]
    if MULTI.get("complete") and COMPLETIONS.get("d") is not None:   # v8: dialect noun-phrase completion
        left, right = COMPLETIONS["d"]
        for i, w in enumerate(cur):
            if w in left and (i == 0 or cur[i - 1] != left[w]):
                out.append(cur[:i] + [left[w]] + cur[i:])
            if w in right and tuple(cur[i + 1:i + 1 + len(right[w])]) != right[w]:
                out.append(cur[:i + 1] + list(right[w]) + cur[i + 1:])
    if MULTI["on"]:
        for i in range(len(cur)):
            for a, b in MULTIWORD:
                for x, y in ((a, b), (b, a)):
                    if tuple(cur[i:i + len(x)]) == x:
                        out.append(cur[:i] + list(y) + cur[i + len(x):])
    return [c for c in out if c]


# Multi-word prepositions and their one-word equivalents (general English); swapped as one edit when MULTI["on"] (v6)
MULTIWORD = [(("on", "top", "of"), ("on",)), (("next", "to"), ("beside",)), (("in", "front", "of"), ("before",)),
             (("into",), ("in",)), (("onto",), ("on",)), (("inside",), ("in",))]
MULTI = {"on": False}
COMPLETIONS = {"d": None}


def completions(d: "Dialect", min_frac: float = 0.5, min_count: int = 3):
    """Dialect noun-phrase completions: a left modifier m with count(m w) / count(w) > min_frac (e.g. 'black' before 'bowl'),
    and a right phrase 'of the X' / 'of the X Y' with count(w of the X..) / count(w) > min_frac (e.g. 'drawer of the cabinet')."""
    uni, bi = Counter(), Counter()
    right = Counter()
    for s in d.sents:
        uni.update(s); bi.update(zip(s[:-1], s[1:]))
        for i, w in enumerate(s):
            if i + 3 < len(s) + 1 and s[i + 1:i + 3] == ["of", "the"] and i + 3 < len(s):
                right[(w, tuple(s[i + 1:i + 4]))] += 1
                if i + 4 < len(s) and s[i + 4] not in FUNCTION:
                    right[(w, tuple(s[i + 1:i + 5]))] += 1
    left = {}
    for (a, b), n in bi.items():
        if a not in FUNCTION and b not in FUNCTION and n >= min_count and n / uni[b] > min_frac:
            left[b] = a
    rt = {}
    for (w, ph), n in right.items():
        if n >= 2 and n / uni[w] > min_frac and len(ph) > len(rt.get(w, ())):
            rt[w] = ph
    return left, rt


def edits_v2(cur: list[str], verbs: list[str], func: list[str]) -> list[list[str]]:
    """v2 edit set (single words): change a verb-slot word, insert/delete/substitute any function word, drop a verb followed by
    a known verb. Used for the first goal_lan screens (lan7 0/10 -> 10/10)."""
    out = []
    slots = set(verb_slots(cur))
    for i in range(len(cur) + 1):
        for v in func:
            out.append(cur[:i] + [v] + cur[i:])
        if i < len(cur):
            w = cur[i]
            if i in slots:
                out += [cur[:i] + [v] + cur[i + 1:] for v in verbs if v != w]
            if w in FUNCTION:
                out.append(cur[:i] + cur[i + 1:])
                out += [cur[:i] + [v] + cur[i + 1:] for v in func if v != w]
    for i in slots:
        if i + 1 < len(cur) and cur[i + 1] in verbs:
            out.append(cur[:i] + cur[i + 1:])
    return [c for c in out if c]


def canonical(instruction: str, exclude: tuple[str, int] | None = None, w: float = 2.0, iters: int = 10,
              use_cache: bool = True, verbose: bool = False, version: str = "v3") -> str:
    """Rewrite `instruction` into the trained dialect. Content words (nouns, adjectives) are never changed; only verb slots and
    function words are edited. exclude: the (trained suite, task id) whose wording must not be used (the target's twin)."""
    key = json.dumps([version + ("" if SCORER["name"] == LM_NAME else "@" + SCORER["name"]) + ("" if PRIOR["kind"] == "bigram" else "#" + PRIOR["kind"]) + ("+multi" if MULTI["on"] else "") + ("+comp" if MULTI.get("compound") else "") + ("+compl" if MULTI.get("complete") else "") + (f"+beam{MULTI['beam']}" if MULTI.get("beam", 1) > 1 else ""), instruction, list(exclude) if exclude else None, w, iters])
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if use_cache and key in cache:
        return cache[key]
    d = Dialect(exclude)
    x = " ".join(words(instruction))
    cur = possessive(words(instruction))
    func = sorted(FUNCTION & set(d.vocab))
    COMPLETIONS["d"] = completions(d) if MULTI.get("complete") else None
    if version == "v2":
        vw = sorted({p[0] for p in dialect_verbs(d)})
        step = lambda c: edits_v2(c, vw, func)
    else:
        step = lambda c: edits(c, dialect_verbs(d), func)
    best = scores(x, [cur], d, w)[0]
    B = MULTI.get("beam", 1)
    if B <= 1:
        for it in range(iters):
            cands = step(cur)
            if not cands:
                break
            s = scores(x, cands, d, w)
            j = max(range(len(cands)), key=lambda j: s[j])
            if s[j] <= best + 1e-6:
                break
            cur, best = cands[j], s[j]
            if verbose:
                print(f"  it{it}: {' '.join(cur)}  ({best:.2f})")
    else:                                   # v9: beam search over edit sequences (escapes multi-edit local optima)
        beam, seen = [(best, cur)], {tuple(cur)}
        for it in range(iters):
            cands = []
            for _, c in beam:
                for e in step(c):
                    if tuple(e) not in seen:
                        seen.add(tuple(e)); cands.append(e)
            if not cands:
                break
            s = scores(x, cands, d, w)
            pool = sorted(beam + list(zip(s, cands)), key=lambda z: -z[0])[:B]
            if [tuple(c) for _, c in pool] == [tuple(c) for _, c in beam]:
                break
            beam = pool
            if verbose:
                print(f"  it{it}: {' '.join(beam[0][1])}  ({beam[0][0]:.2f})")
        best, cur = beam[0]
    out = " ".join(cur)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    cache[key] = out
    CACHE.write_text(json.dumps(cache, indent=1) + "\n")
    return out


def snap_tail(instruction_rewrite: str, original: str, exclude=None, margin: float = 0.0) -> str:
    """Clause retrieval (v11 post-step): if the sentence has 'and <clause>', replace the rewrite's tail clause by the trained
    tail clause (twin excluded) most equivalent to the ORIGINAL tail (Llama paraphrase score, both directions), when it beats
    keeping the rewrite's own tail by `margin`. Content-preserving because candidates are scored against the original."""
    d = Dialect(exclude)
    ws, wo = instruction_rewrite.split(), words(original)
    if "and" not in ws or "and" not in wo:
        return instruction_rewrite
    i, j = len(ws) - 1 - ws[::-1].index("and"), len(wo) - 1 - wo[::-1].index("and")
    head, tail_o, tail_r = ws[:i], " ".join(wo[j + 1:]), " ".join(ws[i + 1:])
    cands = sorted({" ".join(s[k + 1:]) for s in d.sents for k, w in enumerate(s) if w == "and"} | {tail_r})
    a = lm_logp([(PROMPT.format(c), tail_o) for c in cands])
    b = lm_logp([(PROMPT.format(tail_o), c) for c in cands])
    sc = dict(zip(cands, [x + y for x, y in zip(a, b)]))
    best = max(cands, key=sc.get)
    return " ".join(head + ["and"] + best.split()) if sc[best] > sc[tail_r] + margin else instruction_rewrite


def _suite_index(exclude):
    tr = json.loads(TRAINED.read_text())
    return [(suite, i) for suite, l in tr.items() for i, _ in enumerate(l) if (suite, i) != tuple(exclude or ())]


def frames(d: "Dialect"):
    """Sentence frames of the dialect: content words (not function words, not verb-slot words) replaced by a slot '_'."""
    out = set()
    for s in d.sents:
        slots = set(verb_slots(s))
        out.add(tuple("_" if (w not in FUNCTION and i not in slots and w not in PARTICLES) else w for i, w in enumerate(s)))
    return sorted(out)


def content_words(ws: list[str]) -> list[str]:
    slots = set(verb_slots(ws))
    return [w for i, w in enumerate(ws) if w not in FUNCTION and i not in slots and w not in PARTICLES]


def frame_fill(original: str, exclude=None, extra: list[str] | None = None, top: int = 1, suite_completions: bool = False,
               prior_w: float = 0.0):
    """v12: fill every dialect frame whose slot count equals the number of the paraphrase's content words (optionally with
    dialect left-completions, e.g. 'black' before 'bowl'), in order; score fills and `extra` candidates by Llama paraphrase
    equivalence with the original (both directions). Returns the best `top` (score, sentence)."""
    key = json.dumps(["frame_fill", SCORER["name"], original, list(exclude) if exclude else None, sorted(extra or []), top]
                     + (["suite_compl"] if suite_completions else []) + ([f"prior{prior_w}"] if prior_w else []))
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if key in cache:
        return [tuple(z) for z in cache[key]]
    d = Dialect(exclude)
    ws = possessive(words(original))
    cw = content_words(ws)
    if suite_completions and exclude:          # completion statistics from the same trained suite only (its own dialect)
        ds = Dialect(exclude)
        ds.sents = [w for (sname, i), w in zip(_suite_index(exclude), ds.sents) if sname == exclude[0]]
        left, _ = completions(ds, min_count=1)
    else:
        left, _ = completions(d)
    cw2 = []
    for w in cw:
        if w in left and (not cw2 or cw2[-1] != left[w]):
            cw2.append(left[w])
        cw2.append(w)
    cands = set(extra or [])
    for fr in frames(d):
        for c in {tuple(cw), tuple(cw2)}:
            if fr.count("_") == len(c):
                it = iter(c)
                cands.add(" ".join(next(it) if w == "_" else w for w in fr))
    cands = sorted(cands)
    x = " ".join(words(original))
    a = lm_logp([(PROMPT.format(c), x) for c in cands])
    b = lm_logp([(PROMPT.format(x), c) for c in cands])
    pr = [(-d.unseen(c.split()) - 0.5 * d.unseen_n(c.split(), 3) - 0.5 * d.unseen_n(c.split(), 4)) * prior_w for c in cands]
    out = sorted(zip([p + q + r for p, q, r in zip(a, b, pr)], cands), reverse=True)[:top]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    cache[key] = out
    CACHE.write_text(json.dumps(cache, indent=1) + "\n")
    return out


POLITE = ["please", "kindly", "could you please", "could you", "can you please", "can you", "would you please", "would you",
          "will you", "i want you to", "i need you to", "i would like you to", "i'd like you to", "go ahead and", "now",
          "you should", "try to"]
SEPARABLE = {"on", "off", "up", "down", "open", "out", "away"}


def normalize(instruction: str) -> str:
    """General English normalisation before the dialect search (v14): strip politeness/modal frames; 'I want/need the X <PP>'
    and 'the X goes/belongs/should go <PP>' -> 'put the X <PP>'; move a separable particle back next to its verb
    ('turn the stove on' -> 'turn on the stove', 'pick the bowl up from ...' -> 'pick up the bowl from ...')."""
    s = " ".join(words(instruction))
    changed = True
    while changed:
        changed = False
        for p in sorted(POLITE, key=len, reverse=True):
            if s.startswith(p + " "):
                s = s[len(p) + 1:]; changed = True
        if s.endswith(" please"):
            s = s[: -len(" please")]; changed = True
    ws = s.split()
    for lead in (["i", "want"], ["i", "need"], ["i", "would", "like"], ["i'd", "like"]):
        if ws[:len(lead)] == lead and len(ws) > len(lead) and ws[len(lead)] in ("the", "a"):
            ws = ["put"] + ws[len(lead):]
    for i in range(1, len(ws) - 1):
        if ws[0] in ("the", "a") and ws[i] in ("goes", "belongs", "go") and ws[i + 1] in ("on", "in", "into", "inside", "onto", "to"):
            ws = ["put"] + ws[:i] + ws[i + 1:]
            break
        if ws[0] in ("the", "a") and ws[i:i + 2] == ["should", "go"]:
            ws = ["put"] + ws[:i] + ws[i + 2:]
            break
    out, i = [], 0
    while i < len(ws):                    # verb NP particle -> verb particle NP (NP = article + up to 4 words, no preposition)
        out.append(ws[i])
        if i in verb_slots(ws) and i + 1 < len(ws) and ws[i + 1] in ("the", "a"):
            for j in range(i + 2, min(i + 7, len(ws))):
                if ws[j] in PREPOSITIONS - SEPARABLE or ws[j] == "and":
                    break
                if ws[j] in SEPARABLE and (j + 1 == len(ws) or ws[j + 1] in PREPOSITIONS | {"and"}):
                    out.append(ws[j]); ws = ws[:j] + ws[j + 1:]
                    break
        i += 1
    return " ".join(out)


def coord_article(sentence: str, exclude=None) -> str:
    """General English normalisation (v15): repeat the determiner in a coordinated noun phrase, 'the A and B' -> 'the A and
    the B', when B is a bare noun. A word counts as a noun if the trained instructions (twin excluded) use it after 'the'; a
    verb after 'and' ('and place it') is never touched. Motivation: E25 (function words move the behaviour like content
    words), spatial_lan 0 ('between the plate and ramekin')."""
    d = Dialect(exclude)
    nouns = {s[i + 1] for s in d.sents for i in range(len(s) - 1) if s[i] == "the"}
    dets = {"the", "a", "an", "both", "this", "that", "its", "their"}
    ws = sentence.split()
    out = []
    for i, w in enumerate(ws):
        out.append(w)
        if w == "and" and 0 < i < len(ws) - 1 and ws[i + 1] in nouns and ws[i + 1] not in dets:
            prev = ws[max(0, i - 3):i]
            if ws[i - 1] in nouns and "the" in prev:
                out.append("the")
    return " ".join(out)
