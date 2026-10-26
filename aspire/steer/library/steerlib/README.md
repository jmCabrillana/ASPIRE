# steerlib index

Every reusable function of the steering library, with the evidence behind it. The agent adds rows when it moves something that
worked (or partly worked) out of a program. "Seed" rows were written by hand to start the library and have no evidence yet.

| function | what | evidence |
|---|---|---|
| `gates.gripper_closed(obs)` | gripper closed | seed |
| `gates.phase(obs, memory)` | reach / hold / after | seed |
| `gates.attention_share(internals, layer, segment)` | attention share of the action tokens on a segment, previous chunk | seed |
| `interventions.push(ctx, direction, layer, sigmas)` | additive steering term from a stored direction | seed |
| `interventions.ablate(ctx, module, unit, scale)` | remove or amplify one direction in a module's output | seed |
| `interventions.scale_output(ctx, module, factor)` | rescale a module's output | seed |
