# Inference-time steering: literature for a frozen SmolVLA

This file lists the intervention methods we could add to the Code-as-Steering library. It was compiled on 2026-10-26.

**How results are reported.** A number appears only if it was read in the source: the abstract, the HTML, or the PDF text. Anything not checked this way is marked **unverified**. Where two sources disagree, the entry says so.

**Notation for our model.**

| Symbol | Meaning |
|---|---|
| $h^{V}_{\ell,p}\in\mathbb{R}^{960}$ | VLM residual stream at layer $\ell\in\{0..15\}$, prefix token $p$ |
| $h^{A}_{\ell,p,m}\in\mathbb{R}^{720}$ | Action-expert residual stream at layer $\ell$, action token $p\in\{1..50\}$, denoising step $m\in\{0..9\}$ |
| $\mathrm{MLP}_\ell(\cdot)$ | MLP output added to the residual (our hook site) |
| $v_\theta(a_\tau,\tau,\mathrm{KV})$ | Velocity of the action expert |
| $a_{\tau+\delta}=a_\tau+\delta\,v_\theta$ | Euler update, $\delta=0.1$, starting from $a_0\sim\mathcal N(0,I)$ |
| $\hat a_1 = a_\tau + (1-\tau)\,v_\theta$ | One-step "clean action" estimate |

SmolVLA follows the convention that $\tau=0$ is noise and $\tau=1$ is data. Check the sign in the code: LeRobot's SmolVLA integrates from $t=1$ to $t=0$ with $x_{t+dt}=x_t+dt\cdot v$ and $dt=-1/10$.

---

## 1. Steering VLAs and robot policies

### 1.1 Mechanistic steering via FFN value vectors (Häon, Stocking, Chuang, Tomlin; CoRL 2025)
- **Sources.** Paper: https://arxiv.org/abs/2509.00328 (PMLR: https://proceedings.mlr.press/v305/haon25a.html). Code: https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas. A mirror or org also exists at https://github.com/vla-mech-interp/mechanistic-steering-vlas; both carry the same README.
- **Idea.** Each FFN neuron $i$ writes a "value vector" $w^{(i)}$ (a column of `down_proj`) into the residual stream. Project each $w^{(i)}$ onto the token unembedding to read its top tokens. Find clusters of neurons whose top tokens share a concept ("slow", "up", "careful"). At inference, clamp those neurons' activations to a constant.
- **Formula** (paper Eq. 3–4):
  - $\tilde f^{(i)}(x)=\alpha$ if $i\in S$, otherwise $\tilde f^{(i)}(x)=f(x)_i$.
  - $\mathrm{FFN}_{\text{steer}}(x)=\sum_i \tilde f^{(i)}(x)\,w^{(i)}$.
  - The induced residual shift is $\Delta x=\sum_{i\in S}(\alpha-f_i(x))\,w^{(i)}$.
  - A variant hook on `gate_proj` behaved similarly.
- **Where it acts.** The residual stream through the FFN, in the VLM backbone (an LLM with a vocabulary).
- **How the signal is obtained.** Unsupervised vocabulary projection, then manual selection or kNN in embedding space. The kNN step embeds each vector as a softmax-weighted average of its top-55 token embeddings and matches it to the concept embedding by cosine. No rollouts or labels are needed.
- **Evidence.**
  - OpenVLA-7B on LIBERO-Long, 10 rollouts/task, with $|S|\in\{10,20\}$ and $\alpha\in\{2,\dots,20\}$. "Fast" clusters gave end-effector displacement 27.73% larger on average than "slow" clusters, up to 148.54%, with $p<0.001$ and Cohen's $d$ from −0.09 to 1.42.
  - Late-layer and full-model injection work; early layers barely do (mean Y-displacement 0.007 vs 0.086 late vs 0.098 full).
  - π0-FAST on a UR5: qualitative low/slow control. Random vectors ≈ no effect. Prompt modification was weaker than steering.
  - **No task-success rates are reported.**
- **Limitations.** Clusters are semantically ambiguous ("slow and careful" vs "slow and stuck"). Meaning drifts across models. Effects on fine-tuned models are unclear. Only pick-and-place was tested.
- **Recipe for SmolVLA.**
  1. Take `down_proj.weight` columns of VLM layers 8–15 (late half). Project them through SmolVLM2's `lm_head`/tied embedding and keep the top-k tokens per neuron.
  2. Build clusters for concepts relevant to LIBERO-PRO: object nouns ("bowl", "plate"), spatial words ("left", "right", "top"), and verbs ("open", "pick", "place").
  3. Hook the MLP: in `act_fn(gate)*up`, set the cluster indices to $\alpha\in[2,20]$ for prefix tokens only (the action expert has no vocabulary).
  4. Caveat: SmolVLA was fine-tuned on LIBERO, so vocabulary alignment of late VLM layers may have degraded (the paper flags the fine-tuning question as open).
  5. Cheap to try. It is unclear whether it helps *success* on perturbed tasks, as opposed to motion style.

### 1.2 DiMaS: distribution-matching (optimal-transport) steering (Khayatan, Meziane, Parekh, Cord; Jul 2026) — tested on SmolVLA + LIBERO
- **Sources.** Paper: https://arxiv.org/abs/2607.14280. Code: https://github.com/pegah-kh/dimas.
- **Idea.** Behavioural features are "linearly decodable but not linearly steerable" in the action expert. So learn an OT map from the "feature-absent" to the "feature-present" activation distribution, and interpolate toward it. The intervention is gated by a linear probe.
- **Formula.**
  - **Contrast sets.** Feature $\phi_i$ of each action, e.g. speed $\sqrt{\Delta x^2+\Delta y^2+\Delta z^2}$. Then $\mathcal D^-=\{h_i:\phi_i\le q_\tau\}$ and $\mathcal D^+=\{h_i:\phi_i>q_{1-\tau}\}$, with $q_-=0.25$ and $q_+=0.75$.
  - **Map.** $\mathcal T_\ell^{(m)}$ solves the discrete Kantorovich problem $\min_{\gamma\in\Pi}\int\|z^--z^+\|^2d\gamma$ via low-rank Sinkhorn.
  - **Test time.** Project to the nearest neighbour $P(h)=\arg\min_{z\in\mathcal D^-}\|z-h\|$. If the probe says $g_\ell^{(m)}(h)=1$ (feature absent), set $h\leftarrow(1-\alpha)h+\alpha\,\mathcal T_\ell^{(m)}(P(h))$; otherwise leave $h$ unchanged.
- **Where it acts.** Residual stream of the **action expert**, at one layer (the second-to-last), for every action token and every denoising step $m$ (a separate map per $m$).
- **How the signal is obtained.** Labels from the policy's own predicted actions, about 3,500 samples per distribution, split by quantiles. Unsupervised in the sense that it needs no reward.
- **Evidence.**
  - Uses `lerobot/smolvla_libero` and π0.5 on LIBERO. DiMaS shifts speed in both directions on both models. Speed changes cost little success; z-displacement changes cost more.
  - π0.5 on LIBERO-10 at $\alpha=0.5$: success 96%→90%. Success falls steeply with larger $\alpha$ (0.8→46%, 0.9→7%).
  - SmolVLA "degrades earlier and less regularly" than π0.5. The paper attributes this to SmolVLA re-predicting a chunk every step.
  - The effect transfers to held-out tasks and across suites, weakest on LIBERO-Goal.
  - Layer ablation on π0.5: late layers (L15–17) are effective; L3–L9 are much weaker.
  - SmolVLA separability analysis: the action expert is ≈100% linearly separable for speed at late denoising steps, and >93% at deep layers even at $m=0$. The VLM peaks at 87%.
- **Disagreement.** DiMaS finds mean-difference and regression linear steering "inconsistent", often shifting the feature the same way for both signs. COAST (1.3) and Häon (1.1) report that linear or neuron steering works. The setups differ: DiMaS's mean-difference baseline sits in the VLM, while its regression baseline sits in both the VLM and the action expert.
- **Recipe for SmolVLA.** Hook the MLP output of action-expert layer 14 (second-to-last) for all 50 tokens and all 10 steps. Store per-step maps; POT's low-rank Sinkhorn is enough. Use $\alpha\approx0.3$–$0.5$. Swap the "speed" label for a success-relevant label, e.g. distance to the target object at chunk end, or a success/failure rollout label.

### 1.3 COAST: Contrastive Conceptor Activation Steering (Miao, Kim, Yang, Ungar; May 2026)
- **Sources.** https://arxiv.org/abs/2605.17144 and https://www.alphaxiv.org/abs/2605.17144
- **Idea.** Fit conceptors (soft projection matrices) to residual activations from successful and failed rollouts. At inference, softly project activations onto "success AND NOT failure".
- **Formula.**
  - Centred activations give $R=\tilde X^\top\tilde X/N$. The conceptor is $C=R(R+\alpha^{-2}I)^{-1}$.
  - Boolean operations: $\neg C=I-C$ and $A\wedge B=(A^{-1}+B^{-1}-I)^{-1}$.
  - $C_{\text{steer}}=C_{\text{succ}}\wedge\neg C_{\text{fail}}$ and $M=(1-\beta)I+\beta C_{\text{steer}}$.
  - Steered activation: $h'=hM^\top$ at every action-generation step.
- **Where it acts.** Residual stream of the **action expert**, at one layer (layer 11 in the analysis figures; per-task layer search). Uses activations mean-pooled across action tokens, one per denoising step.
- **How the signal is obtained.** Rollout success/failure labels: 15 training rollouts per task, 30 held-out test rollouts.
- **Evidence.**
  - LIBERO-10 success, base → COAST per-step: π0.5 0.43→0.80; π0-FAST 0.65→0.84. On the same table, CAA reached 0.47/0.71 and SAE steering 0.61/0.76.
  - MetaWorld ML45 with π0.5: 0.57→0.82.
  - DROID real robot: about +40% average.
  - The steering subspace is about 1% of the hidden dimension. Failure subspaces are shared across tasks; success subspaces are task-specific, which allows cross-task reuse.
  - Caveat: the π0.5 base of 0.43 on LIBERO-10 is far below commonly reported π0.5 LIBERO numbers (≈92–96%, e.g. CORAL's table), so their checkpoint or protocol differs. **Treat the absolute numbers with caution.**
- **Limitations.** Needs both successes and failures, and works best when the two subspaces overlap. Dual-use concern.
- **Recipe for SmolVLA.**
  1. Run about 15 rollouts per perturbed LIBERO-PRO task.
  2. Collect $h^A_{\ell,\cdot,m}$ mean-pooled over the 50 action tokens for layers 8–14, separately for successes and failures.
  3. Fit $C$ with aperture $\alpha\in\{0.5,1,2,5,10\}$.
  4. Apply $M$ as a matrix multiply on the residual. The hook can replace the MLP output with $\mathrm{MLP}(x)+(xM^\top-x)$.
  5. Strong first candidate because it directly targets success.

### 1.4 VLS: Vision-Language Steering of pretrained policies (Liu, Singh, Xu, Duan, Krishna; Feb 2026)
- **Sources.** https://arxiv.org/abs/2602.03973 and https://arxiv.org/html/2602.03973. Project page: https://vision-language-steering.github.io/webpage/
- **Idea.** A VLM grounds the task-relevant objects into 3D keypoints and writes differentiable, stage-wise PyTorch reward functions $R_s(a,\mathcal P)$. Their gradients guide the denoising of a frozen diffusion or flow policy. Batched particles add diversity, and Feynman–Kac resampling plus closed-loop stage switching complete the loop.
- **Formulas.**
  - Flow guidance: $\hat v=v(a^k,(o,l),k)+\lambda\,\nabla_aR_s(a^k)$.
  - Diffusion guidance: $\hat\epsilon=\epsilon-\lambda\sqrt{1-\bar\alpha_k}\,\nabla_aR_s$.
  - RBF repulsion, applied in early steps: $g^k_{\text{RBF}}[i]=\nabla_{a[i]}\sum_{j\ne i}1/(\|a[i]-a[j]\|+\epsilon)$.
  - FK resampling weights: $w_i\propto\exp(R_s(a[i]))$.
  - Adaptive strength: $\lambda_t=\lambda_{\max}\,\sigma(1-R^t_s/R^{\text{base}}_s)$.
  - Schmitt-trigger stage switching with thresholds $R_{\text{low}}$ and $R_{\text{high}}$.
- **Where it acts.** The sampler (action space), plus the instruction-level task decomposition.
- **How the signal is obtained.** VLM-generated code over SAM, DINOv2, and depth keypoints.
- **Evidence.**
  - LIBERO-PRO with π0.5 (LeRobot), 20 episodes/task:
    - task-perturbation average 23.13→38.50;
    - position-perturbation average 24.25→35.13;
    - overall 23.69→36.81.
  - CALVIN: 94% movable and 87% articulated with a diffusion policy. The paper claims +15–25 points over DynaGuide and ITPS.
  - Ablation: removing gradient guidance collapses performance. FK resampling and RBF mainly add stability and efficiency.
- **Limitations.** High latency (batch, MCMC, FK). Many hyperparameters are unstated ($\lambda_{\max}$, thresholds, resampling period).
- **Recipe for SmolVLA.** This is a close template for our coding agent: the agent writes $R_s$. Requirements:
  - Our action space is a delta end-effector, so we need a differentiable map from action chunk to predicted end-effector path. Cumulative sum of deltas from the current end-effector pose from the LIBERO state works.
  - Object positions come from the simulator, with privileged state allowed in sim. Alternatively, use the VLM.
  - Per denoising step, add $\lambda\nabla_a R(\hat a_1)$ to $v$, with $\lambda$ adaptive.
  - Use 4–8 particles with FK resampling by $R$.
  - It directly addresses swapped positions and new objects, because the reward names the target.

### 1.5 RL²-VLA: adaptive RL latent compositional steering (Tan et al.; Jul 2026)
- **Sources.** https://arxiv.org/abs/2607.26991 and https://arxiv.org/html/2607.26991. Project page: rl2-vla.github.io
- **Idea.** Train a small flow policy $\pi_{\text{RL}}(a\mid e_t)$ by offline RL (QAM) on aggregated action-expert hidden states $e_t$. At each flow step, blend its velocity with the frozen VLA's. Turn steering on only when an LSTM failure detector fires.
- **Formulas.**
  - Blending: $v_{\text{comp}}=w\,v_{\text{VLA}}+(1-w)\,v_{\text{RL}}$, with $w\sim\mathcal N(0.5,0.25)$ clipped to $[0,1]$.
  - QAM optimum: $\pi\propto\pi_\beta\exp(\tau Q)$.
  - Failure score: $s_t=f_\psi(e_{0:t})$, trained with BCE.
  - Gate: steer if $s_t>\delta_t$, where $\delta_t$ comes from time-varying conformal prediction calibrated on successful rollouts.
  - Optional best-of-N with an external verifier (CoVer or RoboMonkey).
- **Where it acts.** Sampler velocity, with latents read from the action expert.
- **How the signal is obtained.** Learned by offline RL on rollouts (BridgeV2 for π0, DROID for π0.5) plus a failure detector.
- **Evidence.**
  - PolaRiS, OOD prompts, π0.5: success 14.3 (vanilla) → 31.8 (rephrase) → 33.8 (compose) → 42.7 (adaptive, SAFE trigger).
  - SIMPLER π0: OOD prompts 57.8→60.3 and OOD tasks 46.5→53.8 with the SAFE gate vs always-on.
  - Abstract: up to +17.3% OOD.
  - "Steering when the base already succeeds can hurt", which motivates the gate.
- **Recipe for SmolVLA.** Mean-pool $h^A_{14,\cdot,m=9}$ (or VLM prefix features) per chunk as $e_t$.
  - The cheapest variant: a BC or advantage-weighted small MLP velocity head on $e_t$, blended at each Euler step.
  - The gating idea (an LSTM or logistic probe on $e_t$, predicting failure) is reusable for *every* method in this list.

### 1.6 Action Coherence Guidance, ACG (KAIST; Oct 2025)
- **Source.** https://arxiv.org/abs/2510.22201
- **Idea.** A perturbed-attention-style guidance for flow VLAs. Build an "incoherent" velocity by replacing self-attention maps in some action-expert layers with the identity, so each action token sees only itself. Then extrapolate away from it.
- **Formula.**
  - $\mathrm{Attn}_{IC}(Q,K,V)=IV=V$.
  - $v_{\text{ACG}}=(1+\lambda)v_\theta-\lambda v^{IC}_\theta$, with $\lambda=3$ by default.
  - Default perturbed layers: the 4th–6th of 8 self-attention layers. Middle and late layers are better than early ones.
- **Where it acts.** Attention patch plus sampler.
- **How the signal is obtained.** None (training-free and self-contrastive).
- **Evidence.**
  - GR00T-N1: average 38.8→53.6 over RoboCasa, DexMG, and two real SO-101 tasks. The text says "+9.6%", which does not match the table delta of 14.8 (internal inconsistency).
  - **SmolVLA** on real SO-101: Three Strawberries 16.7→22.2; Tic-Tac-Toe 23.3→30.0.
  - π0: 41.1→53.3 and 33.3→46.7.
  - The paper reports that CFG by dropping or replacing the *language* condition is "often unstable" in VLAs.
- **Recipe for SmolVLA.** SmolVLA's action expert has self-attention in every second layer, so 8 self-attention layers, which matches the paper. Run two velocity passes per Euler step (batch them). In the second pass, patch the attention function in self-attention layers 3–5 (of 0–7) to return $v$ for action-token queries over action-token keys. Use $\lambda\in[1,3]$. Cost is 2× the action expert only; the prefix KV is shared. Very cheap to add.

### 1.7 DynaGuide (Du & Song; NeurIPS 2025)
- **Source.** https://arxiv.org/abs/2506.13922
- **Idea.** An external latent dynamics model $h_\theta(\phi(o_t),a)$ predicts the future DINOv2 embedding of the scene after chunk $a$. Guide denoising toward positive goal images and away from negative ones.
- **Formula.**
  - $d=\log\sum_i e^{-d^+_i/\sigma}-\log\sum_j e^{-d^-_j/\sigma}$, where $d^\pm$ are squared latent distances to the goals.
  - $\hat\epsilon=\epsilon-s\sqrt{1-\bar\alpha_k}\nabla_{a^k}d$.
  - Each step is repeated $M$ times (stochastic sampling, following ITPS).
- **Where it acts.** Sampler.
- **How the signal is obtained.** A learned dynamics model plus goal images.
- **Evidence.** CALVIN articulated parts: 70% average target success (8.7× base). 5.4× over goal-conditioning under low-quality objectives. 40% on a behaviour with only 1% of the data.
- **Recipe for SmolVLA.** This needs training a dynamics model on LIBERO rollouts, which is medium cost. Goal images of the "target object in target place" could come from successful demos of the canonical task. Lower priority than VLS, because sim state gives us exact rewards.

### 1.8 ITPS: Inference-Time Policy Steering through human interaction (Wang et al.; ICRA 2025)
- **Source.** https://arxiv.org/abs/2411.16627
- **Idea.** Compares six ways to bias a frozen generative policy toward a user objective $\xi(\tau,z)$:
  - random sampling;
  - output perturbation;
  - post-hoc ranking (best-of-N);
  - biased initialization (start from a noised user input);
  - guided diffusion;
  - stochastic sampling.
- **Formulas.**
  - GD: $\tau_{i-1}=\alpha_i(\tau_i-\gamma_i(\epsilon_\theta+\beta_i\nabla\xi))+\sigma_i\eta$.
  - SS: $M-1$ annealed-Langevin steps at a fixed noise level $i$ with the same drift, then one GD step.
- **Evidence.** Maze2D: SS has the best alignment (min/avg L2 0.10/0.12) with the lowest collision rate (0.01). GD and BI align but raise collisions to 0.06.
- **Recipe for SmolVLA.** "Biased initialization" and "post-hoc ranking" are trivially available. For flow matching, BI means starting at $a_{\tau_0}=\tau_0\,a_{\text{ref}}+(1-\tau_0)\epsilon$ and integrating from $\tau_0$, an SDEdit-style warm start. Use ITPS's finding that SS > GD when guidance pushes off-manifold.

### 1.9 DSRL: steering diffusion policies via latent-noise RL (Wagenmaker et al.; CoRL 2025)
- **Source.** https://arxiv.org/abs/2506.15799
- **Idea.** Treat the initial noise $w$ as the action of a new MDP, $a=\pi^W_{dp}(s,w)$ with deterministic flow sampling. Learn $\pi^W(s)$ by RL with black-box access only.
- **Formula.**
  - $r^W(s,w)=r(s,\pi^W_{dp}(s,w))$, with $w\in[-b,b]^d$ and $b\in[1,3]$.
  - DSRL-NA (noise aliasing): $Q^A$ is trained by TD on actions, then distilled to $Q^W(s,w)\approx Q^A(s,\pi_{dp}(s,w))$ with $w\sim\mathcal N(0,I)$.
- **Where it acts.** Initial noise.
- **How the signal is obtained.** Online RL with environment reward.
- **Evidence.**
  - π0 on LIBERO-90 pick-and-place (sim): ≈20%→≈100% after about 10k online samples.
  - Real Franka, π0-DROID: toaster 5/20→18/20; spoon 15/20→19/20.
- **Recipe for SmolVLA.** Replace `noise = torch.randn(...)` in SmolVLA's `sample_actions` with $w=\pi^W(\text{features})$. Features can be pooled prefix-KV features. RLinf has a DSRL-for-π0 LIBERO example: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html. The noise is 50×action_dim (padded to 32 in SmolVLA). Per perturbed task, about 10k env steps is affordable in sim, but this is per-task adaptation, not zero-shot.

### 1.10 Selected Diffusion Noise, SDN (Jun 2026)
- **Source.** https://arxiv.org/abs/2606.14084
- **Idea.** Training-free noise selection. Sample noises far from a reference set to reduce spurious-cue reliance, and pick the candidate with the most coherent trajectory.
- **Evidence (abstract only).** π0 and GR00T-N1.5/1.6: about +8% in sim (Google Robot, WidowX) and about +10% real. **The formula is unverified** (only the abstract was read).
- **Recipe.** Sample N noises, run N chunks (batched), and score by smoothness (jerk) or by agreement. This is cheap.

### 1.11 IGAR: instruction-grounding attention recalibration (ECCV 2026)
- **Source.** https://arxiv.org/abs/2603.06001
- **Idea.** VLAs ignore instructions that contradict the visual scene ("linguistic blindness"). Detect attention-sink tokens via massive activations, then move part of the attention mass from sink tokens to non-sink instruction tokens in selected heads.
- **Formula.**
  - Spike ratio $\phi(d)=\max_i|H_{i,d}|/(\mathrm{mean}_i|H_{i,d}|+\epsilon)$. Keep the top-5 dimensions with $\phi>3$. Token $i$ is a sink if $\max_d|H_{i,d}|>20$.
  - Select a head/query if the sink share of its visual attention is ≤0.4 and its total visual attention is ≥0.01.
  - Scale sink attention by $p=0.6$ and redistribute the freed mass $\Omega=(1-p)\sum A_{\text{sink}}$ proportionally: $A'_j=A_j+\Omega A_j/\sum A_{\text{non-sink text}}$.
  - Applied in the first 16 layers. The paper is inconsistent about which layers.
- **Evidence.**
  - ICBench, a LIBERO-based benchmark with contradictory instructions. On Goal V4, the language-grounding score rose for π0 (19.4→59.4) and OpenVLA-OFT (7.8→39.6), with almost no effect on π0.5.
  - Normal-task success changes by −1.5 to +0.5 points.
  - Here a lower SR under contradictions is the desired outcome: the robot stops executing the visually-cued task.
- **Recipe for SmolVLA.** Patch VLM attention (we see q, k, and the mask):
  - Compute the post-softmax $A$.
  - For query rows (instruction, state, and last tokens), rescale columns of sink tokens. These are usually the first image tokens or the BOS.
  - Add the freed mass to the 48 instruction-token columns.
  - Directly targets "new object / swapped position" cases where vision dominates language.

### 1.12 Steering by instruction rewriting and verification
- **CoVer-VLA.** https://arxiv.org/abs/2602.12281
  - A VLM rephrases the instruction several times. The VLA samples actions for each rephrase, and a 1B contrastive instruction–action verifier picks the best pair.
  - Reported vs scaling policy training: +22% in-distribution and +13% OOD on SIMPLER; +14% progress and +9% success on PolaRiS.
  - In RL²'s table, rephrasing alone took π0.5 PolaRiS OOD-prompt success from 14.3 to 31.8, a large effect for zero model access.
- **Prompt baseline.** DiMaS and Häon both found prompt modification weaker than activation steering *for motion style*. That finding does not cover task identity.
- **Recipe.** For LIBERO-PRO paraphrases, map the paraphrase back to the training-distribution instruction. Options: nearest-neighbour over the 40 LIBERO training instructions using SmolVLM2's own pooled embeddings, or an LLM rewrite. Note our project rule: programs cannot see the benchmark task name or use the original wording. The canonical set must therefore come from generic training data, or this falls back to rephrase+select.

### 1.13 Related VLA work (adapters or analysis, not pure inference-time)
- **CORAL** (https://arxiv.org/abs/2603.09298). Frozen backbone with one LoRA expert per task (r=16, α=32, on the attention layers of the VLM and action head). A non-learned "Manager" maps the instruction to an expert and merges its weights in under 100 ms. Backbones: SimVLA and π0.5. LIBERO average: π0.5 96.9→98.4; SimVLA 98.6→99.3. About 26 MB per expert. The abstract says it "substantially outperforms joint training", but the paper gives no LIBERO joint-training numbers, only a real-world 24.5% joint average.
  - **Recipe.** Train a LoRA per *perturbation family* (paraphrase, position, object) and route by a probe on the prefix.
- **Emergent compositional skills in MoE VLAs** (https://arxiv.org/abs/2607.20771). MoE action head; a search snippet says rank-16 LoRA experts on π0 and SmolVLA action experts. **Unverified**: the abstract does not confirm LoRA or the models.
- **Object-centric residual RL** (https://arxiv.org/abs/2606.18953). Residual policy on object poses trained in sim with pose noise and dropout, on top of a frozen GR00T-N1.5. Real Franka success 42%→76% zero-shot sim-to-real, per abstract and project summary. Formula unverified. The standard residual form is $a=a_{\text{VLA}}+\pi_{\text{res}}(\text{object poses},a_{\text{VLA}})$.
  - **Recipe.** We have privileged object poses in LIBERO, so a small residual MLP over (EE pose, target-object pose, VLA chunk) is natural for swapped positions.
- **"Not all features are created equal"** (https://arxiv.org/abs/2603.19233). Over 520 SAEs across six VLAs, including SmolVLA, plus activation injection into the action expert.
  - π0.5 executes task A's motor program in 99.6% of 1,968 episodes when task-A activations are injected while viewing scene B.
  - The behaviour is tied to **absolute workspace coordinates**, which explains brittleness to position perturbations.
  - Goal selection lives in the VLM; motor generation lives in the action expert.
  - Four of the five architectures tested weight instruction below vision.
  - SmolVLA is indifferent to SAE token pooling.
  - Directly relevant: action-expert steering transplants trajectories, not goals. **To change the goal, intervene in the VLM / prefix.**
- **Robotic Steering** (https://arxiv.org/abs/2511.22697). Selects task-relevant attention heads from few-shot demos and fine-tunes only those (beats LoRA on a Franka). A head-selection idea for our LoRA budget.
- **Real-time chunking, RTC** (https://arxiv.org/abs/2506.07339). Training-free inpainting guidance that freezes the already-committed prefix of the next chunk. Useful as a pattern for "constraint on part of the chunk" guidance. The details (soft mask, weight clipping) were not read here: **unverified**.
- **TouchGuide** (https://arxiv.org/abs/2601.20239). Classifier guidance on π0.5's flow action expert from touch feedback, with its own derivation. Details not read.

---

## 2. LLM activation steering that transfers to our hook sites

### 2.1 ActAdd (Turner et al.)
- **Source.** https://arxiv.org/abs/2308.10248
- **Idea.** Steering vector = activation difference of one prompt pair, added in the forward pass. No optimisation.
- **Formula.** $v=h_\ell(x^+)-h_\ell(x^-)$, then $h_\ell\leftarrow h_\ell+c\,v$ at chosen positions.
- **Evidence (abstract).** LLaMA-3 and OPT. Reported state of the art on sentiment shift and detoxification, with off-target performance preserved.
- **Recipe.** $x^\pm$ = the same observation with two instructions, e.g. "put the bowl" vs "put the plate". Take the difference of VLM residuals at layer $\ell$, averaged over the 48 instruction tokens or at the last prefix token. Add $c\,v$ to the instruction-token residuals.

### 2.2 Contrastive Activation Addition, CAA (Rimsky / Panickssery et al.)
- **Source.** https://arxiv.org/abs/2312.06681
- **Formula.** $v_\ell=\frac1N\sum_i[h_\ell(x_i^+)-h_\ell(x_i^-)]$ (mean difference over many pairs). $h_\ell\leftarrow h_\ell+c\,v_\ell$ at all post-prompt positions.
- **Evidence.** Llama 2 Chat on behavioural MC and open-ended evaluations: significant control with minimal capability loss, and it stacks with finetuning or system prompts.
  - Layer and strength heuristics (from the paper body, **unverified here**): middle layers (≈13 of 32 for 7B) work best, and too large $c$ degrades output.
  - VLA evidence: in COAST's table, CAA gave only small gains (π0.5 0.43→0.47 on LIBERO-10).
- **Recipe.** Contrast pairs from LIBERO rollouts:
  - (success, failure) prefix activations, or
  - (canonical instruction, paraphrase) activations on the same scene. Add $c(\bar h_{\text{canon}}-\bar h_{\text{para}})$ to the instruction tokens. This is effectively "paraphrase un-shifting".
  - Sweep $\ell\in\{4,\dots,12\}$ and $c\in\{0.5,1,2,4\}$.

### 2.3 Inference-Time Intervention, ITI (Li et al.)
- **Source.** https://arxiv.org/abs/2306.03341
- **Formula.** $x_{\ell+1}=x_\ell+\sum_h Q^h_\ell\big(\mathrm{Att}^h_\ell(P^h_\ell x_\ell)+\alpha\,\sigma^h_\ell\,\theta^h_\ell\big)$.
  - $\theta=0$ for heads that are not selected.
  - Heads are selected as the top-$K$ by linear-probe accuracy.
  - The best direction is the **mass-mean shift** $\theta=\mu_{\text{true}}-\mu_{\text{false}}$, which beats the probe weight vector.
  - $\sigma$ = std of activations along $\theta$.
- **Evidence.** Alpaca on TruthfulQA: 32.5%→65.1%. LLaMA-7B: $K=48$, $\alpha=15$, True*Info 30.5→42.3. The paper is internally inconsistent: one table uses $\alpha=20$.
- **Where it acts.** Per-head attention outputs, before `o_proj`. We see q/k/v, so we can add to the per-head output.
- **Recipe.** Probe each VLM head's output at the last prefix token for "will succeed" (rollout labels) or for "target-object identity". Take the top-K (≈16 of 16×15) heads and add $\alpha\sigma\theta$.

### 2.4 Representation Engineering, RepE / LAT (Zou et al.)
- **Source.** https://arxiv.org/abs/2310.01405
- **Idea.** Population-level "reading vectors". The abstract confirms only the framing.
- **Method (from the paper body, not re-checked: unverified).** Collect activations on contrastive stimuli. Take differences of paired activations, often randomly paired. The first PCA component is the reading vector $r_\ell$.
  - Control by $h\leftarrow h\pm c\,r_\ell$, by projection removal, or via LoRRA (a LoRA trained to move representations).
- **Recipe.** PCA on $\{h_\ell(x_i^+)-h_\ell(x_i^-)\}$, where pairs share a scene and differ in target object. The top components give an "object-identity" axis to project onto or add.

### 2.5 Mean-centring (Jorgensen et al.)
- **Source.** https://arxiv.org/abs/2312.03813
- **Formula.** $v=\mathbb E_{x\in\mathcal D_{\text{target}}}[h_\ell(x)]-\mathbb E_{x\in\mathcal D_{\text{all}}}[h_\ell(x)]$, which needs no negative set.
- **Evidence (abstract).** Effective for detoxification and genre steering, and improves function vectors "by a significant margin". Models are not named in the abstract.
- **Recipe.** Target = activations from successful rollouts of the task family; all = every rollout. Useful when we only have successes.

### 2.6 Directional ablation (Arditi et al., "Refusal is mediated by a single direction")
- **Source.** https://arxiv.org/abs/2406.11717
- **Formula (standard; the abstract confirms add/erase of a single direction).**
  - Ablation: $h\leftarrow h-\hat r\hat r^\top h$ at every layer and position.
  - Addition: $h\leftarrow h+c\,\hat r$.
  - Weight-orthogonalisation equivalent: $W_{\text{out}}\leftarrow W_{\text{out}}-\hat r\hat r^\top W_{\text{out}}$.
- **Evidence.** 13 chat models up to 72B: refusal is mediated by a 1-D subspace.
- **Recipe.** Find a "distractor/visual-default" direction, i.e. the mean difference between rollouts where the robot goes to the canonical-position object vs the instructed one. Ablate it from the prefix. Also usable to remove a "memorised task" direction (LIBERO-PRO shows models replay memorised trajectories).

### 2.7 Conceptor steering (Postmus & Abreu, NeurIPS 2024 MINT)
- **Source.** https://arxiv.org/abs/2410.16314
- **Formula.** $C=R(R+\alpha^{-2}I)^{-1}$ and $h\leftarrow\beta\,C h$ (soft projection), with Boolean AND/OR/NOT to combine goals. The steering form $\beta Ch$ is **unverified here**; the abstract only confirms soft projection and Boolean combination.
- **Evidence (abstract).** Outperforms additive steering, and Boolean combination outperforms vector addition. COAST (1.3) is the VLA instance.

### 2.8 Affine and optimal-transport steering
- **Activation Transport, AcT** (Rodriguez et al.; https://arxiv.org/abs/2410.23054). An OT framework that generalises activation steering, for LLMs and T2I models. Per-neuron 1-D OT maps are **unverified here**. DiMaS (1.2) is its multivariate VLA analogue.
- **Simple affine (mean+covariance) map.** $h\leftarrow\mu^++\Sigma_+^{1/2}\Sigma_-^{-1/2}(h-\mu^-)$, the Gaussian OT map. A cheap closed-form middle ground between CAA and DiMaS. No VLA source; this is our suggestion.

### 2.9 SAE-feature steering
- **Evidence.**
  - AxBench (https://arxiv.org/abs/2501.17148, Gemma-2 2B/9B) ranks steering methods: prompting > finetuning > representation methods. SAEs are not competitive for steering *or* detection. DiffMean is strong for detection.
  - In COAST's VLA table, SAE steering beat CAA but lost to conceptors.
  - SAEs for SmolVLA exist from "Not all features are created equal" (2603.19233), which reports over 520 SAEs and 79 concepts. Release location: check the paper.
- **Formula.** $h\leftarrow h+c\,W_{\text{dec}}[:,j]$ (add a decoder direction), or clamp $z_j$ and write back $\hat h=W_{\text{dec}}z+b+(h-\mathrm{SAE}(h))$.

### 2.10 Attention steering: PASTA (Zhang et al., ICLR 2024)
- **Source.** https://arxiv.org/abs/2311.02262
- **Formula.** For emphasised token set $G$:
  - $T(A)_{ij}=\alpha A_{ij}/C_i$ for $j\notin G$, and $A_{ij}/C_i$ for $j\in G$, with $C_i=\sum_{j\in G}A_{ij}+\sum_{j\notin G}\alpha A_{ij}$.
  - $\alpha=0.01$ (never 0).
  - Applied only on profiled heads: steer each head alone, rank heads per task, and take the intersection of the top-$k$ (≈50–100 heads for LLaMA-7B).
- **Evidence.** LLaMA-7B average 67.29→95.46 over JSON format, pronoun change, BiasBios, and CounterFact. GPT-J 44.96→85.22.
- **Recipe.** In the VLM, emphasise the instruction tokens naming the target object (or all 48 instruction tokens) for queries at the state and instruction positions. In the action expert, emphasise prefix positions of image patches that cover the target object (found by attention or segmentation). Equivalent logit form: add $\log(1/\alpha)$ to the pre-softmax scores of $G$, which is easy in our attention patch.

### 2.11 Known failure modes and heuristics
- **Input-dependent reliability.** Steering vectors vary a lot per input, are affected by spurious biases, and can fail under reasonable prompt rewording (Tan et al., https://arxiv.org/abs/2407.12404). Paraphrase robustness is exactly our setting.
- **Strength vs success trade-off.** It is steep in VLAs. DiMaS shows π0.5 LIBERO-10 SR 96→90→46→7% for α=0.5→0.8→0.9, and SmolVLA degrades earlier. ITI documents a truthfulness/helpfulness trade-off controlled by α.
- **Layer sensitivity.**
  - Häon: early VLM layers have almost no effect.
  - DiMaS: late action-expert layers work, early/mid do not.
  - IGAR: performance degrades if more than 24 layers are intervened.
  - ACG: middle and late self-attention layers work.
- **Norm blow-up / off-distribution.** Adding fixed-norm vectors can push residuals off-manifold. The common mitigations below are general practice, **not verified in a cited source here**:
  - scale $c$ relative to $\|h\|$, e.g. $c=\kappa\|h\|/\|v\|$;
  - renormalise $h'\leftarrow h'\|h\|/\|h'\|$;
  - gate the intervention (DiMaS probe gate, RL² failure gate).
- **Goal vs motor binding.** Action-expert injections reproduce trajectories in absolute coordinates and ignore the scene (2603.19233). Use action-expert steering for *how* to move; use the VLM, prefix, or sampler guidance for *where*.
- **Gate the steering.** RL² shows always-on steering is worse than failure-gated steering (π0.5 PolaRiS 33.8 vs 42.7).

---

## 3. Guidance for the flow-matching action expert

Let $v_\theta(a_\tau,\tau,c)$ be the velocity, with clean estimate $\hat a_1(a_\tau)=a_\tau+(1-\tau)v_\theta$.

| Method | Formula | Signal | Notes / sources |
|---|---|---|---|
| Classifier-free guidance (CFG) | $v=v_\theta(c_\emptyset)+w\,[v_\theta(c)-v_\theta(c_\emptyset)]$ | Condition-dropout contrast | SmolVLA was not trained with condition dropout. ACG reports language-CFG in VLAs is "often unstable" (https://arxiv.org/abs/2510.22201). π\*0.6 uses CFG with an indicator dropped during training (https://arxiv.org/abs/2511.14759). **Contrastive variant (our idea):** $c_\emptyset$ = the prefix with a *wrong* instruction or with the target object masked. |
| Autoguidance / perturbed-attention guidance (ACG) | $v=(1+\lambda)v_\theta-\lambda v^{\text{bad}}_\theta$ | Self-perturbation | See §1.6. Also the prefix-perturbation variant (our idea): $v^{\text{bad}}$ with instruction tokens masked in cross-attention, which turns into "language guidance" without retraining. |
| Reward/cost gradient guidance | $v\leftarrow v_\theta+\lambda\nabla_{a_\tau}R(\hat a_1(a_\tau))$ | Differentiable reward | VLS Eq. 5, DynaGuide, TouchGuide. Cheaper variant: gradient w.r.t. $\hat a_1$ only, skipping backprop through $v_\theta$. |
| Stochastic / MCMC refinement | Repeat a step at fixed $\tau$ $M$ times, with noise | – | ITPS (best alignment/feasibility trade-off), DynaGuide, VLS ($M=1$ for flow). |
| Particle filtering (FK) / best-of-N | Resample particles by $w_i\propto e^{R}$ | Reward or verifier | VLS; RL²+CoVer; RoboMonkey. |
| Initial-noise steering | $a_0=w$ instead of $\epsilon$; learned (DSRL) or selected (SDN) | RL reward / coherence score | §1.9–1.10. |
| Biased init / SDEdit | Start at $\tau_0$ from $\tau_0 a_{\text{ref}}+(1-\tau_0)\epsilon$ | Reference chunk | ITPS "BI". |
| Inpainting / partial-chunk constraint | Fix part of the chunk, guide the rest | Known actions | RTC (§1.13). |
| Velocity composition | $v=w v_{\text{VLA}}+(1-w)v_{\text{aux}}$ | Auxiliary policy | RL² (§1.5). |

---

## 4. Ranked shortlist: what to implement first for frozen SmolVLA on LIBERO-PRO

The perturbations are paraphrased instructions, swapped object positions, and new objects.

1. **Sampler reward guidance with particles (VLS-style, §1.4).** The agent writes $R(\hat a_1)$ from sim object poses, e.g. EE-path distance to the target object/region, plus a gripper schedule. Add $\lambda\nabla R$ to $v$ and resample 4–8 particles by $e^R$.
   - *Why:* the only method with reported LIBERO-PRO gains (π0.5: 23.7→36.8). It directly fixes swapped positions and new objects, needs no training, and fits a coding agent perfectly.
2. **Failure-gated application of everything (RL² gate, DiMaS probe gate).** A logistic or LSTM probe on pooled prefix or action-expert features predicts failure; steer only then.
   - *Why:* always-on steering hurts already-correct episodes (RL²: 33.8 vs 42.7). It is cheap and multiplies the value of every other method.
3. **Contrastive conceptor gate on action-expert residuals (COAST, §1.3).**
   - *Why:* the largest reported LIBERO gains among activation methods (beats CAA and SAE in the same table). Needs only about 15 rollouts per task with success/failure labels, which our loop already produces. Reusable across tasks via the shared failure subspace.
4. **Instruction-grounding attention re-weighting in the VLM (IGAR / PASTA, §1.11, §2.10).** Remove sink mass and boost the target-object instruction tokens in selected heads, via a pre-softmax logit bonus.
   - *Why:* LIBERO-PRO failures are largely "vision overrides language" or memorisation. The method is training-free, with about ±1 pt cost on normal tasks.
5. **Instruction canonicalisation and rephrase+select (§1.12).** Map a paraphrase to an in-distribution instruction, or sample K rephrasings and pick by a verifier/probe.
   - *Why:* the cheapest fix for paraphrases. In RL²'s table, rephrasing alone took π0.5 from 14.3 to 31.8 OOD-prompt success. Must respect our rule against using the original wording.
6. **Action Coherence Guidance (§1.6).** Identity attention in action-expert self-attention layers 3–5 and $\lambda\approx1$–3.
   - *Why:* already shown on SmolVLA (+5.5 and +6.7 pts on real tasks). Training-free, about 2× the action-expert cost. Reduces jitter, which DiMaS identifies as a SmolVLA weakness.
7. **CAA / mean-difference prefix steering for "paraphrase → canonical" and "target-object identity" (§2.2, §2.5, §2.6), with norm-relative strength and ablation variants.**
   - *Why:* a trivial, universal baseline that the library needs anyway. VLM/prefix is the right site for *goal* changes (2603.19233). Expect modest effects: DiMaS found it inconsistent and COAST found it small, so keep it as a baseline and building block.
8. **DiMaS OT transport at action-expert layer 14 (§1.2).**
   - *Why:* validated on `smolvla_libero` itself, and the most reliable way to modulate motion properties (speed, height) when a perturbation needs slower or higher approaches. Lower rank because it controls *style*, not goal.

**Honourable mentions.**
- DSRL noise-policy RL: strong, but per-task online RL.
- Object-centric residual RL: sim object poses make it easy, but it needs training.
- CORAL-style per-family LoRA routed by a prefix probe: training budget.
- Häon FFN-neuron clamping: cheap, but no success evidence.

---

## 5. Open questions
1. **Where is the goal encoded in SmolVLA?** Only 16 VLM layers are kept, and the action expert reads the prefix KV, not the residual. Prefix-residual steering only matters through the K and V it produces. Should we instead steer the **KV cache** directly ($K,V$ of instruction or target-patch tokens)? No paper tested this for VLAs.
2. **Does the vocabulary projection (Häon) still work** on SmolVLM2's truncated, LIBERO-fine-tuned decoder? Does anything align with object nouns?
3. **Linear vs non-linear.** DiMaS says linear steering fails in the action expert; COAST says conceptors (multiplicative) work. Is the right abstraction "projection/gating" rather than "addition"? Needs a head-to-head on the same SmolVLA checkpoint.
4. **Per-step schedules.** DiMaS fits one map per denoising step. Should guidance strength ramp up or down across the 10 Euler steps? Early steps fix the mode; late steps fix details. VLS applies RBF only early. There is no systematic study for 10-step flow VLAs.
5. **SmolVLA re-predicts every step** in the LIBERO checkpoint, which makes it more fragile to intervention (DiMaS). Does executing more of each chunk (n_action_steps > 1) make steering more stable?
6. **Reward guidance through delta-EE actions.** The gradient through a cumsum map is cheap, but how accurate is "action chunk → future EE path" under the OSC controller? Is privileged sim state acceptable within our evaluation rules?
7. **CFG without dropout training.** Is "wrong-instruction" or "masked-target" contrastive guidance stable for SmolVLA, given ACG's report that language CFG is unstable?
8. **Gating signal.** Which features best predict failure early: VLM last-token, pooled action-expert, or attention entropy over instruction tokens?
9. **Disagreements to resolve empirically.**
   - COAST's base π0.5 LIBERO-10 (0.43) vs commonly reported ≈92–96%.
   - ACG's text ("+9.6%") vs its table (+14.8).
   - ITI α=15 vs α=20.
   - IGAR's layer range (first 16 vs "mid-level").
   - DiMaS (linear steering inconsistent) vs Häon and COAST (steering effective). The sites and labels differ.
