# Variance model and design to detect a library improvement

Date: 2026-10-19. Internal note (not part of `REPORT.md`). The numbers marked *est.* come from our own runs and are rough.

**Correction to my earlier message:** the benchmark has **90 tasks**: 70 for building (the pool) and 20 held out. I had used 70
in total (50 + 20), which was wrong. This note uses 70 + 20.

## 1. What is measured

A library `L` is measured by writing `K` fresh zero-shot programs per task, running each on `S` seeds, and averaging the 0/1
outcomes over the `N` tasks of the measurement set:

    ybar = 1/(N K S) * sum_{i,j,s} y_ijs

Targets: the improvement of the method over its baseline (or over the start library) is at least **0.10** at the end of the
rounds and at least **0.05** in the middle (absolute pass-rate points).

## 2. Model and derivation

Levels: build `b` (one full run of 40 rounds), task `i`, program `j`, seed `s`.

- A build gives a library `L_b,t` at round `t`. Its **true quality** on the task set is theta_b,t = (1/N) sum_i mu_i(L_b,t),
  where mu_i is the mean success probability of a fresh program for task `i` under that library.
- A program has its own success probability q_ij = mu_i + w_ij. The **writer luck** w_ij has mean 0 and variance
  sigma_q,i^2 (the same programs differ a lot in strategy).
- Given q_ij, each seed is a Bernoulli(q_ij) trial. Its variance is q(1-q); let `pi` be its mean over programs and tasks.

**Step 1: variance of one measurement of a fixed library.** The programs and seeds are independent, so by the law of total
variance, for task i: Var(ybar_i | L) = (sigma_q,i^2 + pi_i/S) / K. Averaging `N` independent tasks:

    V_e(N, K, S) = ( sigma_q^2 + pi / S ) / ( N K )          (measurement variance; sigma_q^2, pi are task averages)

**Step 2: build-to-build variance.** The true quality differs between builds (which task was repaired, how good each lesson
was): Var(theta_b,t) = sigma_b^2. It does not depend on N, K or S. So for one build

    Var(ybar_b,t) = sigma_b^2 + V_e

**Step 3: gap between two arms over R paired builds.** For repetition r, D_r = ybar(arm A) - ybar(arm B). The two arms are
independent given the shared round 0, so Var(D_r) = 2 (sigma_b^2 + V_e), and the mean of R repetitions has

    Var(Dbar) = 2 (sigma_b^2 + V_e) / R

**Step 4: repetitions needed.** With a paired t-test on the R differences (df = R - 1), significance level alpha and power
1 - beta, to detect a true gap delta, the normal approximation is

    R  ~=  (z_{1-alpha/2} + z_{1-beta})^2 * 2 (sigma_b^2 + V_e) / delta^2  +  z_{1-alpha/2}^2 / 2

(z sum = 2.80 for alpha 5%, power 80%; the last term is the small-sample correction). The tables below use the exact
non-central t power instead, which is slightly larger for small R. R must be at least 3.

**Step 5: gain of one arm over its own start library** (the same tasks): Var(gain_b) = sigma_bg^2 + V_e(L_t) + V_e(L_0), where
sigma_bg^2 is the build-to-build variance of the true gain. The start library `L_0` (empty) is the same for every build and arm,
so it can be measured once with very many programs, and V_e(L_0) becomes negligible. This helps this reading only; in the
arm gap the start library cancels.

**What the formulas say**
- `N K S` only reduce V_e. Once V_e is below sigma_b^2 (about N K >= sigma_q^2 / sigma_b^2, i.e. 30 to 100 programs) more
  programs, tasks or seeds do **not** help.
- **Only R reduces sigma_b**: required R scales with 1/delta^2, so detecting 0.05 needs about 4 times more repetitions than 0.10.
- More rounds do not reduce noise; they raise delta (the effect grows with the number of repairs).

## 3. Estimates from our runs (est.)

| quantity | value | how |
|---|---|---|
| sigma_q^2 | 0.025 (SD 0.16) | variance of a program's observed pass rate within a task over six empty-library generations (0.0224 at 15 seeds, 0.0296 at 30), minus the seed part |
| pi | 0.04 | mean of q(1-q); the unbiased estimate of q(1-q)/S is p(1-p)/(S-1) |
| check | predicts 0.029 for 27 programs | observed SD between generations: 0.026 |
| sigma_b | **0.03**, range 0 to 0.06 | the 6-program final measurements of reps 1 to 3: `lex` 0.110, 0.104, 0.098; `unsolved` 0.120, 0.101, 0.026. Observed variance of builds minus V_e (0.00049 for 9 tasks x 6 x 30). Pooled over arms. Only 4 degrees of freedom, so very uncertain |

`sigma_b` is the parameter that decides the budget. Reps 4 and 5 (10 seeds) will give more builds; re-estimate it after them:
sigma_b^2 = (sample variance across builds of a 6-program final measurement) - V_e.

## 4. Choosing K and S for the cheapest precision

Measuring has two costs: writing a program (`c_w`, wall-clock and tokens) and running a trial (`c_t`). Fixing the total cost
C = N K (c_w + S c_t) and minimising V_e gives, with f(S) = (sigma_q^2 + pi/S)(c_w + S c_t), the optimum

    S* = sqrt( pi c_w / ( sigma_q^2 c_t ) )

From our logs: a trial costs c_t ~ 0.06 min (16 trials a minute on one GPU with 4 workers); writing a program costs about
0.3 to 0.45 min of wall-clock with 3 writers in parallel, so c_w / c_t ~ 5 to 7 and S* ~ 3 (range 2 to 6). **Few seeds, more
programs** is cheaper for the same precision. I would not go below 5 seeds, because the formula assumes seeds are
independent trials of the same program, and a few extra seeds guard against that. Token cost and rate limits push the other way
(more programs means more writer sessions), which is why K stays near 12.

## 5. Repetitions needed (80% power, 5% level, exact paired t)

Held-out measurement set N = 20. delta = gap of the arms (`lex` minus baseline).

| design (N x K x S) | programs | trials | SD of V_e | sigma_b = 0.015: R at 0.10 / 0.05 | 0.03: R at 0.10 / 0.05 | 0.06: R at 0.10 / 0.05 |
|---|---|---|---|---|---|---|
| 20 x 6 x 20 | 120 | 2400 | 0.015 | 3 / 6 | 5 / 10 | 9 / 27 |
| **20 x 12 x 5** | 240 | 1200 | 0.012 | 3 / 5 | 4 / 9 | 8 / 26 |
| 20 x 12 x 10 | 240 | 2400 | 0.011 | 3 / 5 | 4 / 9 | 8 / 26 |
| 20 x 20 x 5 | 400 | 2000 | 0.009 | 3 / 5 | 4 / 9 | 8 / 26 |

- Going from 120 to 400 programs saves at most one repetition. The choice among rows is therefore a cost choice, and
  **20 x 12 x 5** (half the trials of the first row, twice the programs) is the cheapest row with this precision.
- The mid target (0.05) is the expensive one: 9 repetitions at sigma_b = 0.03, 26 at 0.06. Treat it as secondary unless
  sigma_b turns out near 0.015.
- Reading "gain of one arm over its own start library" (Step 5, start library measured once with very many programs): SD per
  build = sqrt(sigma_bg^2 + V_e + V_e(L_0)); at sigma_bg = 0.03 this needs R = 4 for 0.10 and 6 for 0.05 (6 and 14 at 0.06).

## 6. Recommended design (resource-optimised)

| item | choice | reason |
|---|---|---|
| tasks | 70 build pool, 20 held out | held-out tasks are only measured, never repaired or used for selection |
| rounds | 40 repairs, at most 1 per task | 40 of 70 pool tasks; no repeats needed |
| build-time probe | each round a random **10 candidate tasks** x 3 programs x 5 seeds = 150 trials | `lex` and `unsolved` only need outcomes of the candidates; a full 70-task probe would be 1050 to 6300 trials per round |
| measurement set | the 20 held-out tasks, **K = 12 programs, S = 5 seeds** (1200 trials) | cheapest row of the table above |
| measurement times | rounds 20 and 40 (and 10, 30 with K = 3, S = 5 for the curve only) | linear growth: end points carry the information; the early effect is below the noise (gain at round 5 is about 0.0125) |
| start library | measure once, 30 programs per held-out task | its noise then drops out of the gain reading |
| testing | fixed sequence: round 40 first, then round 20, both at 5% | no multiplicity penalty |
| repetitions | **R = 4 for the 0.10 target; 9 or more for 0.05** | table above, sigma_b = 0.03 |
| test | paired t-test over builds (df = R - 1), or a mixed model with a random build effect on the per-trial data | |

Cost per arm and repetition (cost parameters above, one GPU, 4 workers):

| part | wall-clock |
|---|---|
| build round: write 30 programs (9 min), probe 150 trials (9 min), repair (15 min) | about 33 min |
| 40 rounds | about 22 h |
| two held-out measurements: write 240 programs and run 1200 trials each | about 2.5 h each, 5 h |
| **per arm** | **about 27 h** |
| per repetition (2 arms) | about 54 h |
| 4 repetitions | about 9 days on one GPU (about 2 to 3 days on 4 GPUs) |

The writing load (240 programs per measurement, 30 per round) is what hits subscription rate limits; the driver needs an
automatic resume (the last stall cost 17 hours).

## 7. Caveats

- The model assumes tasks are independent and the same sigma_q^2 applies to the 70 + 20 task set. LIBERO tasks differ a lot
  (some are solved at 0, some at 0.3); the held-out 20 may have a different sigma_q^2. Estimate it on the held-out tasks with a
  first measurement of L_0 before fixing K.
- sigma_b is one number with 4 degrees of freedom today. The required R changes by a factor of 3 between 0.015 and 0.06.
- Seeds are treated as independent trials. Seeds that share a scene layout would make `pi / S` an underestimate.
- The effect is assumed to grow roughly linearly with rounds. If it saturates, the middle target is easier and the end target
  is not helped by more rounds.
- Cheaper build probes (10 candidates, 5 seeds) give `lex` a noisier disagreement signal than the 30-seed probes of reps 1 to 3.
  The effect of that on `lex` itself is untested.
