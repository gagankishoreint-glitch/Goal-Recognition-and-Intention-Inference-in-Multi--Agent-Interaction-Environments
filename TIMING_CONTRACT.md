# Time-Step and Label Contract

Single source of truth for *what a model may observe at time `t`* and *what it is
asked to predict*. Implemented in `src/features/timing.py`; enforced by
`tests/test_timing_contract.py`. Any probe, baseline or sequence model must build
its samples through that module — no ad-hoc feature construction.

## 1. Index convention

Steps are 0-based and match `src/trajectory.generator`. At step `t`:

1. the agent occupies `position_before[t]` — the start cell when `t = 0`,
   otherwise `position[t-1]`;
2. the policy samples the hidden intention `I_t` and an intended action;
3. the environment applies the single-pass symmetric transition, and the step
   record stores `position[t]` (post-transition), `realized_action[t]` (the
   action that produced that transition) and `blocked[t]`.

Consequence: **`realized_action[t]` and `position[t]` are outcomes of step `t`,
not inputs to it.** A model predicting anything about step `t` may not see them.

## 2. The observation

```
O_t = ( focal cell before the step        position_before[t]
        peer cells at the same instant     position_before of every peer at t
        previous realized action           realized_action[t-1]
        previous blocked flag              blocked[t-1]          (Condition A only)
        static map M, candidate targets    constant per scenario )
```

For `t = 0` there is no `t-1`: the previous-behaviour features are an explicit
two-bit "no previous step" flag, never a silent zero that could be confused with
a real `WAIT`.

`O_{1:t}` denotes the sequence `O_1, …, O_t`. It contains `A_{t-1}` but **not**
`A_t`.

Condition B is the same object with the peer cells and the `blocked` flag
removed, by the allowlist projection in `src/features/observations.py`. Nothing
else changes — in particular the timing does not.

## 3. Labels and tasks

| task | label | alignment | question answered |
|---|---|---|---|
| `goal` | `G = focal_goal_id` (episode-level, constant) | one label per prefix | `P(G \| O_{1:t})` |
| `intention` | `I_t` (privileged) | sample `t` ↔ label `t` | `P(I_t \| O_{1:t})` |
| `action` | `A_t = realized_action[t]` | sample `t` ↔ label `t` | `P(A_t \| O_{1:t})` |

Notation bridge to PROJECT_SPEC §2.1, which is 1-based: the `action` task here —
"predict the action taken at the step whose intention is being inferred, from the
observation available before it" — is `P(A_{t+1} | O_{1:t})` in that notation.
Both describe the same object; the 0-based form is what the code implements.

## 4. What is never an input

- `realized_action[t]`, `blocked[t]`, `position[t]` — outcomes of step `t`;
- `I_t` and every `step_intentions` entry;
- `focal_goal_id`, `focal_goal_position`, `other_agent_goals`;
- `intended_action[*]`, `blocked_reason[*]`, `seeds`, `planner_internal_state`;
- any step record with index `> t`.

The frame builder reads only `{"step", "position_before", "agent_id"}` from the
current step and `{"realized_action", "blocked"}` from step `t-1`;
`tests/test_timing_contract.py::test_frame_reads_only_the_allowlisted_keys`
asserts that allowlist, so adding a field to the reader is a test failure until
the contract is updated deliberately.

## 5. How the contract is enforced

1. **Mutation test.** Perturb `realized_action`, `blocked` and `position` of step
   `t` *and of every later step*, then rebuild `O_t`. If the frame changes, the
   builder is reading the future and the test fails.
2. **Prefix test.** `O_t` built from the full episode must equal `O_t` built from
   the episode truncated after `t`.
3. **Canary test.** `timing.leaky_frame_features` deliberately includes step `t`'s
   outcome. A shallow probe on leaky features predicts `A_t` near-perfectly; the
   same probe on contract features must not. If the two accuracies are close, the
   leakage detector is not sensitive and the test fails — i.e. the test verifies
   that it can detect the bug it is guarding against.
4. **Isolation guard.** Model-facing batches pass `validate_batch`, which raises
   on any privileged key, and `episode_tensor` is the last stage before a model.

## 6. Conventions for the M3 sequence model

- A GRU consumes `O_1 … O_T` as a sequence of frames and emits per-step heads for
  `G` (shared across steps), `I_t` and `A_t`. The input at position `t` is
  `frame_features(O_t)`; the targets are the row-`t` labels above. No teacher
  forcing of `A_t` into the input at position `t`.
- Truncation to an observation fraction `α` means keeping frames `0 … ⌊α(T-1)⌋`
  and evaluating the head at that index — the same operation the baseline `α`
  sweep performs, so learned and closed-form models are compared on identical
  prefixes.
- Preprocessing statistics are computed on the training scenarios only.

## 7. The interaction channels

`frame_features(O_t)` is 58 channels for the shipped configuration (K = 4,
3 peer slots): 35 geometry, 7 previous-behaviour, 16 interaction. The
interaction block exists so that RQ2 can be turned on and off without touching
anything else:

- per candidate target `g` — a peer holds a greedy step toward `g`; a peer holds
  *every* greedy step toward `g`; a peer is standing on target `g`;
- global — adjacent peer count, nearest peer distance, "the previous step was
  blocked and a peer is adjacent", peer count.

Every channel is a function of `O_t` and is covered by
`tests/test_interaction.py`, including the requirement that it responds to where
the peers actually are. One candidate channel was rejected on that test: "is a
peer collinear with the focal cell and target `g`" turns out to be a
distance-to-goal proxy — the number of collinear cells grows with the distance —
so a control with randomly relocated peers scored as well as the real thing. It
is not an interaction feature and it is not in the block.

## 8. How to mask interaction (mandatory protocol)

**A masked arm is built by substituting noise, never by zeroing channels.**
Measured on the M2 verification set with a 5-frame logistic probe
(`experiments/multi_agent/interaction_contrast.md`):

| contrast | intention accuracy gap | 95% CI |
|---|---|---|
| A (full) − B (channels zeroed) | +0.0616 | [+0.0203, +0.0995] |
| D (channels = Gaussian noise) − B | +0.0588 | [+0.0254, +0.0890] |
| **A − D** (the valid contrast) | **+0.0027** | **[−0.0101, +0.0142]** |

Zeroing 28 of 290 input channels changes what an L2-regularized model sees, and
on its own it is worth almost the whole apparent effect. Reporting A − B would
have attributed a regularization artifact to peer interaction.

Three rules follow, and they bind the M3 sequence model as well:

1. Mask by substitution (Gaussian noise, or draws from the empirical marginal of
   the masked channels), keeping dimensionality and capacity fixed.
2. Ablate the previous-step `blocked` flag **separately** from the peer
   positions. It is a one-bit lagging trace of a collision, and here it carried
   the entire effect (+0.0643 [+0.0286, +0.0979] accuracy on its own) while the
   peer positions added nothing detectable on top.
3. Report the paired gap with a scenario-cluster bootstrap CI and treat it as
   supported only when the interval excludes zero.
