# Review of `LITERATURE.md` (inference-time steering survey)

Reviewed 2026-10-26 against the cited arXiv abstracts/HTML/PDFs (DiMaS from the PDF text; the others through
fetched HTML, so quotations are close paraphrases), the local LeRobot SmolVLA source
(`lerobot/policies/smolvla/{modeling_smolvla,smolvlm_with_expert}.py`), the `lerobot/smolvla_libero` config, our
`steer/hooks.py`, `steer/directions.py`, `scripts/sweep.py`, the pilot log and the baseline table. Every method in the
shortlist and every number in §1–§4 was checked. Verdict first, details after.

**Overall.** The survey is unusually careful: of roughly 120 checked numbers and formulas, about 100 are confirmed
verbatim. The problems are concentrated in three places: (i) a handful of wrong or unsupported statements (listed in
§1 below); (ii) the SmolVLA-specific notation and recipes contain two errors about the model itself (flow-time
convention; self-attention layer indices); (iii) the shortlist and several recipes assume privileged simulator state
(object poses, depth), which our programs are not allowed to use, so the ranking has to be redone (§3).

---

## 1. Factual accuracy

### 1.1 Errors (wrong as written)

| # | Where | Survey says | Correction | Source |
|---|---|---|---|---|
| E1 | Notation, line 18 | "SmolVLA follows the convention that τ=0 is noise and τ=1 is data." | Wrong for the LeRobot code we run. `forward()` builds `x_t = t*noise + (1-t)*actions`, target `u_t = noise - actions`; `sample_actions()` starts at `time = 1.0` and steps `x_t = x_t + dt*v_t`, `dt = -0.1`, through t = 1.0, 0.9, …, 0.1. So **t = 1 is noise, t = 0 is data**, and the one-step clean estimate is **â = x_t − t·v_t**, not `a_τ + (1−τ)v`. Every formula in §3 that uses â₁ must be rewritten in this convention before anyone codes it. | `modeling_smolvla.py` lines 771–772, 833–864 |
| E2 | §1.6 recipe, §4 item 6 | "patch the attention function in self-attention layers 3–5 (of 0–7)" | Ambiguous and, if read as expert layer indices 3–5, wrong. Self-attention layers are the expert layers with `layer_idx % 2 == 0`, i.e. **expert layers 0, 2, 4, …, 14**; the odd layers are cross-attention only. ACG's "4th–6th of 8" = self-attention layers 3–5 = **expert layers 6, 8, 10**. Also: in SmolVLA's even layers the action tokens attend jointly to the cached prefix keys *and* to each other (`forward_attn_layer` concatenates `past_key_values[layer_idx]` with the action-token keys), so an "identity" attention would also cut the prefix. See spec §4.4. | `smolvlm_with_expert.py` lines 111, 252–254, 427–431 |
| E3 | §1.2, §5.5 | "SmolVLA re-predicts every step in the LIBERO checkpoint" | DiMaS says this about *its* evaluation ("SmolVLA predicts a new chunk at every timestep", and attributes the fragility only "in part" to that). The checkpoint config has `n_action_steps = 50 = chunk_size`, and our `rollout.py` executes the whole chunk unless `n_action_steps` is overridden. Re-prediction frequency is our choice, not a property of the checkpoint. | `smolvla_libero/config.json`; `steer/rollout.py` 77–83 |
| E4 | §1.3 evidence, §4 item 3 | "Failure subspaces are shared across tasks; success subspaces are task-specific, which allows cross-task reuse." | COAST's abstract says this, but its results section (4.4 and the Fig. 5 caption) says the opposite: "Success activations cluster tightly and uniformly across tasks, while failure activations spread into distinct, task-specific regions." The paper is internally inconsistent; the survey should not build a "reuse the failure conceptor across tasks" argument on it. | COAST §4.4 |
| E5 | §1.3 caveat, §5.9 | Treats the π0.5 base of 0.43 on LIBERO-10 as an unexplained protocol difference. | It is explained: COAST deliberately uses **early fine-tuning checkpoints** "that produce the mix of successes and failures needed for contrastive fitting" (App. A.8.3: "20–60% success across LIBERO-10 tasks"). So COAST's gains are measured on a weak checkpoint, which is relevant: it is unknown whether they hold on a converged model such as `smolvla_libero` (84% on libero_goal). | COAST §4.1, A.8.3 |
| E6 | §1.3 evidence | "MetaWorld ML45 with π0.5: 0.57→0.82." | That is a 26-task subset (Table 7). The full-ML45 mean row in Table 1 reads 0.69→0.94, and Table 10(b) gives 0.58→0.80 for the same subset. Say which. | COAST Tables 1, 7, 10 |
| E7 | §1.3 evidence | π0-FAST base 0.65 | Table 1 says 0.65; the text says "from 0.62 to 0.84" (0.62 looks like the SFT column). Flag as the paper's inconsistency. | COAST §4.2 |
| E8 | §1.4 idea | "A VLM grounds the task-relevant objects into 3D keypoints" | The VLM only *names* the task-relevant objects/regions. Keypoints come from SAM masks reprojected with **depth** into point clouds that are then clustered. DINOv2 is not confirmed in the text. This matters for us: VLS needs depth and a segmentation stack, see §2. | VLS §3.1 |
| E9 | §1.4 formula | "RBF repulsion, applied in early steps" | The text says early steps; Algorithm 1 applies it inside the full loop. Paper inconsistency; note it. | VLS Alg. 1 |
| E10 | §1.5 formula | "Optional best-of-N with an external verifier" | Not optional: "Our work assumes a robust verifier that effectively selects among action candidates", and the verifier is an unconditional step of Algorithm 1. The 42.7 PolaRiS number includes it. | RL² §3, Alg. 1 |
| E11 | §1.5 signal | "BridgeV2 for π0, DROID for π0.5" | The paper says it augments BridgeV2 and DROID and trains "on the same offline finetuning dataset used by the VLA"; the explicit pairing is not stated. Mark unverified. | RL² §4 |
| E12 | §1.5 formula | $w\sim\mathcal N(0.5,0.25)$ | σ = 0.25 is a standard deviation ("μ=0.5, σ=0.25"); write $\mathcal N(0.5,0.25^2)$. | RL² §3 |
| E13 | §1.7 evidence | "40% on a behaviour with only 1% of the data" listed under CALVIN articulated parts | That number is the Switch-On data-underrepresentation experiment (§4.4), not ArticulatedParts. | DynaGuide §4.4 |
| E14 | §1.10 evidence | "about +10% real" | Abstract says +10%; Table 3 shows GR00T-N1.5 30.0→48.3 (+18.3). The +8% sim figure is carried by π0 (63.9→72.6); GR00T-N1.6 is +5.1. Also the "formula unverified" note can be resolved: the selection is `argmin JerkRMS` (third-order finite differences) over the top-M candidates by grounding score `R = d_B − d_G` (kNN distances to embeddings of the original vs object-masked observation). | SDN §3, Tables 1–3 |
| E15 | §1.11 title and evidence | "instruction-grounding attention recalibration"; Goal V4 LGS gains reported alone. | The paper calls it **Instruction-Guided** Attention Recalibration. More importantly, the LGS gains come with large success-rate drops on the contradictory tasks (π0 SR 76.4→36.4, OpenVLA-OFT 90.2→58.4) — the survey notes "lower SR is desired" but should state the magnitudes. For π0.5, LGS actually *decreases* (4.0→1.4). Normal-task "−1.5 to +0.5" is the per-model average; per-suite range is −4.0 (π0.5 Object) to +1.4. | IGAR Tables 2–3 |
| E16 | §1.12 CoVer | "+22% ID and +13% OOD on SIMPLER" presented as "vs scaling policy training" | Those are the headline gains vs the base policy. The policy-training comparison (§5.4) is +15% ID / +12% OOD. (The paper's intro also says 18%.) | CoVer §5.4 |
| E17 | §2.1 ActAdd | "LLaMA-3 and OPT" | Abstract does say "models including LLaMA-3 and OPT" (the paper was updated; the original experiments were on GPT-2-XL). Fine as stated but note the v5 revision. | arXiv v5 |
| E18 | §2.3 ITI | "LLaMA-7B: K=48, α=15, True*Info 30.5→42.3" | 42.3 is the mass-mean row at **α=20** in Table 3; the K=48/α=15 headline in Table 1 is 43.5. The "inconsistency" the survey flags is the survey conflating two tables. | ITI Tables 1, 3 |
| E19 | §2.7 Conceptors | "steering form $\beta Ch$ unverified" | Verified: Eq. 4 is exactly $h' = \beta_c C h$ (mean-centred variant $h' = \beta_c C^{mc}(h-\mu)+\mu$). Note this differs from COAST's $M=(1-\beta)I+\beta C$: Postmus & Abreu *replace* h by its projection, COAST interpolates. | Postmus & Abreu Eq. 4 |
| E20 | §2.9 AxBench | "prompting > finetuning > representation methods" | Steering ranking (Table 2): Prompt 0.894 > **LoReFT 0.741** > SFT 0.676 > LoRA 0.615 > ReFT-r1 0.543 > DiffMean 0.239 > SAE 0.165 > … LoReFT is a representation method and beats finetuning; DiffMean/PCA/LAT/Probe are the ones that do badly. Detection: DiffMean 0.942 ≈ Probe 0.940 ≈ ReFT-r1 0.938 (no significant difference). | AxBench Tables 1–2 |
| E21 | §1.1 evidence | "Cohen's d from −0.09 to 1.42"; "Random vectors ≈ no effect"; "α∈{2,…,20}" | The paper reports "effect sizes" −0.091 to 1.419 without naming Cohen's d; says random interventions show "minimal difference" from none; the α grid is {2,4,6,8,10,20}. The paper also says top-**55** tokens while its figure shows top-5 (possible typo in the paper). Minor. | Häon App. |
| E22 | §1.1 and §1.13 | π0-FAST on a UR5 (Häon); frozen **GR00T-N1.5** (object-centric residual RL) | Häon's text says "Pi0" and a UR5; "FAST" was not found. The residual-RL abstract does not name GR00T; the base model name is unverified from what was read. Also the residual's deployment inputs are object poses from **FoundationPose tracking after SAM2 registration**, not simulator state — which is the relevant adaptation for us (§2). | Häon abstract; 2606.18953 §3 |

### 1.2 Venue / date claims

Confirmed on the arXiv page: Häon CoRL 2025 (PMLR v305 page exists and matches); ITPS ICRA 2025; DynaGuide NeurIPS 2025; PASTA ICLR 2024; ITI "NeurIPS 2023 spotlight" (arXiv comment); conceptors "MINT workshop at NeurIPS 2024". Not on the arXiv page, so unsupported by the cited URL (though consistent with the proceedings as far as I know): DSRL "CoRL 2025", Arditi "NeurIPS 2024", AcT "ICLR 2025", Tan "NeurIPS 2024". Wrong or missing: **ACG is "Accepted to ICRA 2026"** (v2, Mar 2026) — the survey gives no venue and its "KAIST" attribution is not on the page; **IGAR "ECCV 2026" is not on the page** (v2 Jul 2026, no venue); **COAST is "Submitted to NeurIPS 2026"**, not published. DiMaS, VLS, RL², SDN, CoVer, CORAL: preprints with no venue, as the survey implies.

### 1.3 URLs

All cited arXiv IDs resolve to the claimed papers. Both Häon GitHub URLs resolve (the `vla-mech-interp` one redirects to `Physical-AI-Safety-Institute/mechanistic-steering-vlas`, "[CoRL 2025] …"); `github.com/pegah-kh/dimas` exists. `vision-language-steering.github.io` and `rl2-vla.github.io` were not fetched. The SAEs of 2603.19233 are released through `action-atlas.com` / `github.com/CWRU-AISM/action-atlas` (weights not confirmed).

### 1.4 Confirmed as written (spot-check summary)

Häon Eq. 3–4, LIBERO-Long protocol, 27.73%/148.54%/p<0.001, 0.007/0.086/0.098, prompt < steering. DiMaS: everything in §1.2 except E3 (quantiles 0.25/0.75, low-rank Sinkhorn, second-to-last layer, per-step maps, ~3,500 samples, `lerobot/smolvla_libero`, 96→90→46→7, L15–17 vs L3–9, separability numbers, "inconsistent" linear baselines and which site each baseline uses). COAST formulas, 15/30 rollouts, Table 1 numbers (0.43→0.80, 0.65→0.84, CAA 0.47/0.71, SAE 0.61/0.76), ~1% subspace, "40%" real-robot (absolute vs relative unclear; "DROID" not confirmed by name). VLS Eqs. 4, 5, 9, 10, 11, LIBERO-PRO 23.13→38.50 / 24.25→35.13 / 23.69→36.81 with 20 episodes per task, CALVIN 94/87, "gradient removal collapses", and the survey is right that λ_max, thresholds, particle count and resampling period are not given (only M_MCMC = 4 diffusion / 1 flow). RL² blend/QAM/LSTM-BCE/conformal (SAFE), 14.3→31.8→33.8→42.7, 57.8→60.3 / 46.5→53.8, +17.3%, "steering when the base succeeds can hurt". ACG Attn=V, λ=3, 4th–6th of 8, 38.8→53.6 with the "+9.6%" text inconsistency (real: the per-benchmark gains in the text match the table; only the average does not), SmolVLA 16.7→22.2 / 23.3→30.0, π0 41.1→53.3 / 33.3→46.7, language-CFG "often unstable", no LIBERO. ACG also states the cost: second pass can be cut to ≈1.5× by caching layers before the first perturbed one. DynaGuide Eqs. and 70%/8.7×/5.4×. ITPS six methods, Eq. 5, Maze2D numbers. DSRL latent MDP, b∈[1,3] online (0.5–1.5 offline), NA distillation, LIBERO-90 ≈20→≈100% in ≈10k samples, 5/20→18/20 and 15/20→19/20. IGAR thresholds (k=5, γ=3, τ=20, ρ=0.4, α=0.01, p=0.6), "first 16 layers" vs "mid-level" inconsistency, L>24 degrades, mass goes to text tokens. CORAL r=16/α=32, <100 ms, 96.9→98.4, 98.6→99.3, 26 MB, no LIBERO joint-training number, 24.5% real; CORAL's π0.5 LIBERO-10 ("Long") base is 92.4. 2607.20771: rank-16 LoRA deltas on the action expert's FFN sublayers, π0 and SmolVLA — the survey's "unverified" can be upgraded to confirmed. 2603.19233: 520+ SAEs, 79 concepts, 99.6% of 1,968, absolute coordinates, 4 of 5 weight instruction below vision, SmolVLA pooling-indifferent (the "goal in VLM / motor in expert" sentence was not found verbatim; it is the survey's reading). Robotic Steering, RTC (soft mask Eq. 5, guidance weight clipped at β=5), TouchGuide (π0.5 action expert, guidance scaled by k/(1−k), late steps only), π*0.6 (advantage indicator "randomly omitted" in training; β>1 is CFG; no dropout rate given). CAA, RepE, mean-centring, Arditi, AcT (per-dimension affine maps confirmed), PASTA (formula, α=0.01, |H|=25/53/86, 67.29→95.46, 44.96→85.22), Tan et al.: all confirmed.

### 1.5 Our-model facts the survey should state

- Prefix layout in `embed_prefix`: [64 image tokens (agentview), 64 (wrist), ≤48 language tokens (`pad_language_to="longest"`, so the count varies per instruction), 1 state token]. With BOS/EOS as tokenised. Image tokens precede language tokens, so "sink tokens are usually the first image tokens or the BOS" (§1.11 recipe) needs checking: there is no BOS at position 0, the first token is an image patch.
- VLM: 15 query heads, 5 KV heads (GQA ×3), head_dim 64, MLP intermediate 2560; `lm_head` is not loaded in the policy (the VLM is truncated to 16 of 32 layers), so Häon's vocabulary projection must load `SmolVLM2-500M-Video-Instruct`'s embedding separately and use layers 0–15 of a 32-layer model: the "late half" in Häon's sense (layers 16–31) no longer exists. This weakens §1.1 further.
- Expert: width 720 = 0.75×960, same 15/5 heads with head_dim 64 (k/v projections are sized to match the VLM's 5×64 so that cached prefix keys can be concatenated). MLP hook output is `(B, 50, 720)` per denoising step; the VLM MLP hook output is `(B, P, 960)` once per chunk, P ≈ 177.
- Noise/actions tensor is `(B, 50, 32)` with the first 7 dims real (6 delta-EE + gripper), normalised by dataset stats; the LIBERO controller is OSC_POSE with `output_max = [0.05 m ×3, 0.5 rad ×3]` per step at 20 Hz.

---

## 2. Applicability under our constraints (images + proprioception only, frozen weights, no original wording)

| Method | Privileged input as described | Adaptation with images + proprio | Keep? |
|---|---|---|---|
| VLS (§1.4) and the survey's #1 recipe ("R from sim object poses") | Sim object poses in the recipe; depth + SAM in the paper. LIBERO's agentview depth is a render we do not receive. | Target 3D point from the **agentview image** via an open-vocabulary detector (OWLv2/Grounding-DINO, or SmolVLM2 itself asked to point) and **back-projection to the table plane** using the fixed agentview camera calibration (a constant of the benchmark, like robot kinematics; the object must rest on the table or the target surface's known height). Refine with the wrist image when close. EE pose from proprio. Reward on the *predicted EE path* `p_k = p_0 + Σ_{j≤k} 0.05·unnorm(a_j)[:3]` (an upper bound; the OSC controller under-tracks). | **Keep, adapted** (my #2). |
| DynaGuide (§1.7) | Goal images from "successful demos of the canonical task" (fine), dynamics model trained on rollouts (fine), but the survey's "lower priority because sim state gives exact rewards" argument is void. | A latent dynamics model on our own rollouts is legitimate; with the rollout budget (hundreds of episodes) it is the expensive route to the same reward as the adapted VLS. | Drop for now. |
| Object-centric residual RL (§1.13) | Survey recipe uses "privileged object poses in LIBERO". The paper itself uses FoundationPose tracking. | Residual MLP on (EE pose, detected target 3D point, VLA chunk) is possible, but it needs *training* on rewards; rewards from images only = success classifier or EE-to-target proxy. | Drop (training-heavy, same perception dependency as #2). |
| DSRL (§1.9) | Online RL with env reward. Success from `env` is simulator ground truth. | Our evaluation rule bans sim state *inside programs*; whether the **outer training loop** may read the success flag is a project decision, not addressed in the survey. If success labels are allowed only as rollout outcomes (as the pilot already uses them for "success directions"), DSRL/best-of-N RL over noise is a per-task adaptation, not zero-shot. | Honourable mention only. |
| COAST (§1.3), CAA/ITI/mean-centring/directional ablation (§2) | Need success/failure labels of own rollouts — allowed in the same sense as above. | No change, but see §3 on why the pilot failed and why action-expert conceptors cannot fix swapped positions. | Keep COAST (my #6); keep mean-diff as a diagnostic baseline only. |
| Instruction canonicalisation (§1.12, #5) | The survey's "nearest-neighbour over the 40 LIBERO training instructions" uses the original wording. | Build the candidate bank from **generic templates filled with scene content** ("put the {obj} on the {place}", "open the {part} of the {obj}", "turn on the {obj}") where `{obj}` comes from a VLM reading of the agentview image plus the paraphrase's nouns. This never reads the benchmark's string, but the user must adjudicate whether "LIBERO-style templates" count as original wording. Rephrase+select needs a selection signal: see spec §4.1. | **Keep, adapted** (my #1). |
| RL² gate / failure detector (§1.5, #2) | Trained on rollouts with success labels. | Fine (labels are outcomes, not sim state inside the program); features must be prefix/expert activations + proprio only. | Keep (my #3). |
| IGAR / PASTA (§1.11, §2.10) | None. | Fine. Head profiling needs labelled rollouts. | Keep (my #5). |
| ACG (§1.6), SDN (§1.10), biased init, best-of-N | None. | Fine. | Keep ACG (my #4); SDN's jerk criterion is a cheap selection signal for best-of-N. |
| CFG with "wrong instruction" (§3 table) | None, but a wrong instruction must not be the original one for paraphrase tasks. | Use a masked/empty instruction or an unrelated template. | Keep as an experiment. |
| CORAL-style LoRA | Needs demonstrations of perturbed tasks. | Only via self-distillation from steered successes; later stage. | Later. |

Also note: §5.6 asks "Is privileged sim state acceptable within our evaluation rules?" — it is not, per the task constraints; the survey should say so rather than leave it open.

---

## 3. Ranking for frozen SmolVLA on LIBERO-PRO

Context from our own numbers (`doc/results_baseline.tex`, 10 episodes/task): libero_goal 0.84, libero_goal_lan 0.33,
**libero_goal_swap 0.00**, i.e. paraphrases halve success and swapped positions zero it, consistent with 2603.19233's
"absolute workspace coordinates" finding and LIBERO-PRO's own 0% result.

**Why the pilot (mean-difference at VLM layers 3/7/11/15) most likely showed nothing.** The direction was built from
`record()`, which averages the MLP output over *all* prefix tokens (128 of ~177 are image tokens that are identical
between the trained and paraphrased runs), then over chunks, then over episodes; the strength was `s × spread`, where
`spread` is the standard deviation of *episode-mean* projections, which is far smaller than a per-token activation
norm; and the resulting constant vector was added to every token, including the image and state tokens. Three
diagnostics before concluding "linear VLM steering fails": (a) report ‖strength·v‖ / median per-token ‖MLP(x)‖ at that
layer (if it is below ~0.1 the intervention was negligible); (b) compute the direction from the instruction-token
positions only and add it only there; (c) verify the sweep actually changed the actions (chunk L2 vs unsteered on the
same initial state). DiMaS's "linear steering inconsistent" result concerns motion *style* labels in the action expert
and does not predict our setting either way.

### My top 6

| Rank | Method | Targets | Expected cost in our pipeline |
|---|---|---|---|
| 1 | Instruction canonicalisation (scene-grounded template rewrite) + rephrase-and-select | paraphrases (0.84→0.33 gap) | One VLM/LLM call per episode (≈0.2–1 s), K≤8 extra prefix passes per chunk if selecting; no training. |
| 2 | Image-grounded action-space guidance: detector → table-plane target → best-of-N / FK resampling, optionally gradient on â | swapped positions (0%), new objects | Detector ≈30–80 ms per chunk on the 5090; N=8 batched denoising adds <10% to an episode (the expert is tiny; episodes are sim-bound). ~0 training. |
| 3 | Failure/verifier probe on own rollouts (gate + best-of-N scorer) | all; prevents hurting the 84% | ≈100–200 labelled episodes once (already produced by baseline/sweeps); logistic probe trains in seconds; inference free. |
| 4 | ACG (masked self-attention guidance in expert layers 6, 8, 10) | jitter, general robustness; validated on SmolVLA | 2× expert per step (≈+5% episode time); no training. |
| 5 | PASTA/IGAR-style attention re-weighting toward instruction tokens (VLM) and toward instruction/target-patch keys (expert cross-attention) | vision-over-language, new objects | Head profiling: ~15×16 single-head sweeps × 10 episodes on 2 tasks ≈ 2–3 GPU-hours once; inference free. |
| 6 | COAST conceptors on expert residuals (per-task, from 15–30 own rollouts), with the mean-diff/CAA on instruction tokens as the cheap baseline | paraphrases where both outcomes occur; *not* swaps | 15–30 rollouts per task (1–5 min), conceptor fit seconds; inference one 720×720 matmul per hook call. |

Reasoning against the survey's order: its #1 (VLS with sim poses) violates the constraints as written and becomes my #2
once grounded in images; its #3 (COAST) drops because (E4, E5) its cross-task-reuse argument is contradicted inside the
paper and its gains are on an early checkpoint, and because action-expert steering transplants trajectories in absolute
coordinates (2603.19233), which is precisely wrong for swapped positions; its #5 (canonicalisation) rises to #1 because
paraphrase is the largest *recoverable* gap and the only fix with zero model-side risk; DiMaS stays out of the top 6
(style, not goal).

---

## 4. Implementation specs

Common notation: `B` batch, `P` prefix length (≈177), `L_ins` ≤ 48 instruction tokens at positions `[128, 128+L_ins)`,
state token at `P−1`; VLM MLP hook output `(B, P, 960)`; expert MLP hook output `(B, 50, 720)` at each of the 10 steps
`t ∈ {1.0, 0.9, …, 0.1}`; `x_t ∈ (B, 50, 32)`, clean estimate `â = x_t − t·v_t`; actions unnormalised by the
post-processor's dataset stats (affine, differentiable).

### 4.1 Instruction canonicalisation + rephrase-and-select
- **Signal.** From the given paraphrase `ℓ` and the agentview image: (i) a VLM lists visible objects (SmolVLM2 itself, or any local VLM); (ii) an LLM maps `ℓ` to `(verb, target, destination)` restricted to that object list; (iii) fill generic templates (`"put the {target} on the {dest}"`, `"open the {part} of the {obj}"`, `"turn on the {obj}"`, `"push the {obj} to the {dir}"`), producing K ∈ [1, 8] candidates. No bank of benchmark strings.
- **Selection** (if K > 1). Run K prefix passes on the same observation (batch them; one VLM pass each), sample one chunk per candidate from a shared noise; score with (a) the failure probe of §4.3 (`p_success` from prefix features), and/or (b) instruction attention share: mean over expert cross-attention layers of the attention mass from action tokens to instruction keys (available from `introspect.py`); pick the argmax; re-select every chunk or freeze after the first.
- **Defaults.** K = 4; freeze the chosen instruction for the episode unless the probe falls below 0.3.
- **Failure mode.** Hallucinated objects or the wrong destination noun silently retarget the task; log the chosen string and check it against the paraphrase's nouns. Compliance: do not let the template bank contain benchmark strings; treat "LIBERO-style template" as a project decision to record in `constraints.txt`.

### 4.2 Image-grounded action-space guidance (VLS-lite)
- **Target point.** Open-vocabulary detector on the 256×256 agentview image with the target/destination noun from §4.1 → pixel `(u, v)` of the box bottom-centre → ray through the fixed agentview camera (intrinsics/extrinsics from the robosuite camera config, constant per suite) intersected with the table plane `z = z_table` (or the known top of the destination object class) → `g ∈ ℝ³`. Confidence `c_det`. Optionally refine `g` from the wrist image when ‖p_EE − g‖ < 10 cm.
- **Predicted path.** `p_k = p_0 + Σ_{j≤k} 0.05 · clip(ã_j[:3], −1, 1)`, with `ã = unnorm(â)`, `p_0` from `robot0_eef_pos`. Shapes: `â (N, 50, 32)` → `p (N, 50, 3)`.
- **Reward (stage-wise).** Reach: `R = −min_k ‖p_k − g‖ − 0.5·‖p_50 − g‖`; grasp: add `−|ã_k[6] − (+1)|` after the approach; transport: same form with the destination point; stage switch on gripper state and ‖p_EE − g‖ (Schmitt thresholds 3 cm / 6 cm). Multiply by `c_det`; if `c_det < 0.3` disable guidance (fall back to the frozen policy).
- **Two tiers.** Tier A (first): best-of-N with N = 8 noises, resample by `softmax(R/τ_R)` after steps 4 and 7 (τ_R = 2 cm), execute the argmax. Tier B: add `λ ∇_{x_t} R(â)` to `v_t` during steps 3–8 with the gradient taken through `â = x_t − t·v_t` treating `v_t` as constant (no backprop through the expert); `λ` so that ‖λ∇R‖ ≤ 0.2‖v_t‖ per step, adaptive `λ_t = λ_max·σ(1 − R_t/R_base)`, λ_max = 1.
- **Failure mode.** Off-manifold chunks (jerky, collide, open gripper mid-transport): monitor chunk jerk and reject candidates above 2× the unsteered median (SDN's criterion); detector picks the wrong instance when two same-class objects exist — in that case use the paraphrase's spatial words or the instruction-attention map over image patches to disambiguate.

### 4.3 Failure/verifier probe and gate
- **Features per chunk.** `f = [mean over instruction tokens of VLM layer-11 MLP output (960); state token residual (960); mean over 50 action tokens of expert layer-14 MLP output at t = 0.1 (720); proprio (EE pos 3, gripper 2); chunk index]`. Standardise per dimension.
- **Labels.** Episode success from rollout outcome; also a "progress" label (did the gripper close within the next 2 chunks) for earlier signal. Train logistic regression (or a 2-layer MLP) on ≈150 episodes × ≈6 chunks; calibrate the threshold on held-out successful episodes so that ≤10% of successful chunks are flagged (RL²'s conformal idea, per chunk index).
- **Use.** Gate every other method (apply only if `p_fail > θ`, θ from calibration); score best-of-N candidates by `p_success` computed from the candidate's own expert features (requires one extra denoising pass per candidate at t = 0.1 only).
- **Failure mode.** Distribution shift of the probe itself on perturbed suites (trained on libero_goal, applied to _lan/_swap): include failures from the perturbed suites in training, and check AUROC per suite before trusting it.

### 4.4 ACG for SmolVLA
- **Where.** Expert self-attention layers 6, 8, 10 (the 4th–6th of the 8 even layers). Patch `eager_attention_forward` only for calls where `query_states.shape[1] == 50` and the layer index is in the set (pass the layer index through a context variable set in `forward_attn_layer`).
- **Perturbed pass, default (masked variant).** Keep the prefix columns of the mask; set the action-token×action-token block of the `(B, 50, P+50)` mask to the diagonal, so each action token sees the prefix and itself. Ablation: the paper's literal `Attn = V` (cuts the prefix too).
- **Guidance.** Run the two passes as a batch of 2B sharing the KV cache: `v = (1+λ) v_θ − λ v_IC`, λ = 1 first, sweep {1, 2, 3}; all 10 steps (ACG uses 16; try disabling in the last 2).
- **Cost.** Expert only; ≈2× expert per step, ≈1.5× if the first 6 layers are cached and reused.
- **Failure mode.** Over-amplified velocities at large λ → overshoot; clip ‖v‖ to 1.5× the unsteered norm per token. It cannot change the goal; expect modest gains on _lan and none on _swap.

### 4.5 Attention re-weighting toward the instruction (PASTA/IGAR)
- **VLM pass.** In the patched attention for the prefix pass (`(B, 15, P, P)` scores), for query rows of the instruction and state tokens add a logit bonus `log(1/α)` to the key columns `G` = instruction tokens naming the target/destination (from §4.1), α = 0.01 only on profiled heads; or IGAR-style: detect sink tokens (per-token max |h| > 20·median across tokens at layers 2–4; expect the first image tokens), scale their attention by p = 0.6 for instruction/state queries and give the freed mass to the non-sink instruction columns, in layers 0–7 (the paper's "first 16 of 18/32" scaled to our 16-layer VLM).
- **Expert cross-attention (odd layers).** Scores `(B, 15, 50, P)`: same logit bonus on the instruction columns (and, once §4.2 provides a target box, on the ≤8 agentview patch tokens covering it; patch grid 8×8 over the 512×512 padded image).
- **Head profiling.** Steer one head at a time on 2 tasks × 10 episodes, rank by success or by the §4.3 probe, take the intersection of top-k across tasks (PASTA: 50–100 of 32×32 heads for a 7B model; for us expect 10–30 of 240 VLM heads).
- **Defaults.** α = 0.01 (PASTA), p = 0.6 (IGAR), bonus only when the instruction-attention share of the action tokens (from `introspect.py`) is below the libero_goal median.
- **Failure mode.** Boosting instruction attention when the instruction tokens carry the wrong binding (paraphrase not understood) amplifies the error; combine with §4.1. Also re-check that redistribution preserves row sums (softmax after the logit bonus does).

### 4.6 COAST conceptors on expert residuals (and the corrected mean-diff baseline)
- **Data.** Per task, 15–30 own rollouts with both outcomes (so: _lan tasks with 0.1–0.8 base, not _swap). Record `h = mean over 50 action tokens of expert layer-ℓ MLP output` `(720,)` per denoising step, ℓ ∈ {10, 12, 14}; keep per-step sets (10 conceptors per class) or pool.
- **Fit.** Centre by the pooled mean μ; `R = X̃ᵀX̃/N`, `C = R(R + α⁻²I)⁻¹`, aperture α ∈ {1, 2, 5, 10}; `C_steer = C_succ ∧ ¬C_fail`; `M = (1−β)I + βC_steer`, β ∈ {0.3, 0.5, 0.8}.
- **Hook.** In the expert MLP hook at layer ℓ, return `MLP(x) + ((h_pre + MLP(x) − μ) M ᵀ − (h_pre + MLP(x) − μ))`, which needs the pre-MLP residual: register a forward pre-hook on `layer.mlp` to capture its input (the normed residual) or, simpler, hook `layer.post_attention_layernorm` input. Shapes `(B, 50, 720) @ (720, 720)`. Gate with §4.3.
- **Mean-diff baseline done right.** Direction from instruction-token activations only (`positions [128, 128+L_ins)`), unit-normalised; strength `κ · median‖MLP(x)‖` with κ ∈ {0.25, 0.5, 1, 2}; added at instruction positions only; and at the VLM, which produces the prefix KV that the expert reads.
- **Failure mode.** With few failures the ¬C_fail term is near identity and the method reduces to projection onto the success subspace, which replays canonical trajectories (fine for paraphrases, wrong for swaps); and success-rate-vs-β is steep (DiMaS: 90→46→7 between α 0.5 and 0.9), so sweep β from the low end.

---

## 5. Smaller edits to the survey

- §1.1 recipe step 1: SmolVLA keeps VLM layers 0–15 of 32; there is no "late half" in Häon's sense, and `lm_head` must be loaded separately.
- §1.6: add "ICRA 2026"; the SO-101 attribution for Table III is inferred, not stated; ACG uses 16 Euler steps.
- §1.11 recipe: replace "BOS" with "the first image tokens" and verify empirically.
- §1.12 CoVer: eval uses 8 rephrases × 5 candidates and an ensemble of 3 verifiers; selection is hierarchical (rephrase by mean score, then best action).
- §2.2 CAA: "layer 13 of 32" — the paper says layer 13 and neighbours; the 32 is the survey's.
- §3 table: rewrite every â₁ with `â = x_t − t·v_t`; RTC's guidance weight is `min(β, (1−τ)/(τ r_τ²))` with β = 5 and a soft mask (now verified, drop "unverified").
- §4 items 1 and 2 and §1.13 recipes: remove "from sim object poses" / "privileged object poses in LIBERO".
- §5.5: rephrase as a choice (`n_action_steps`), and note that our rollouts execute full 50-step chunks.
- §5.6: answer it — privileged sim state is not allowed in programs.
