# Dataset Schema Specification (Version 1.0.0)

## 1. Overview and Principles
This document defines the schema and storage format for behavioral trajectories generated in the multi-agent goal recognition environment.
The schema enforces a strict separation between **Observable State** (available to an external observer / model) and **Privileged Simulator State** (ground truth available only during environment execution and evaluation).

To guarantee reproducibility and zero data leakage:
- Trajectories are partitioned into **Train**, **Validation**, and **Test** splits strictly by **Scenario / Map Layout ID**, with a leakage assertion that no `scenario_id` appears in more than one split. Configurations sharing the same map topology (e.g., multiple target assignments / agent configurations) are grouped consistently.
- Every episode carries an `interaction_mask` flag indicating the Condition A (full interaction observation) vs. Condition B (interaction-masked) variant, evaluated on identical underlying episodes.
- Loaders validate `schema_version` and reject privileged fields in model-facing batches (loud failure, no silent dropping).
- Preprocessing parameters (e.g., feature normalization statistics) are computed solely from the training set.

scenario/configuration level split — not episode level — with
a deterministic `scenario_id` hash of the full structural configuration
(map layout, obstacle placement, target placement, starting agent
positions, agent count/configuration, environment parameters). All
trajectories from one scenario stay in the same split, and
`assert_no_scenario_leakage` must fail on leaky splits.
```

---

## 2. Directory Layout & File Organization
```text
data/
├── metadata.json                 # Global dataset metadata, generation timestamp, split hashes
├── train/
│   ├── scenario_00001.json       # Trajectory episodes generated under Scenario 1
│   └── ...
├── val/
│   ├── scenario_00071.json
│   └── ...
└── test/
    ├── scenario_00086.json
    └── ...
```

---

## 3. Data Structure Definition

### 3.1 Scenario File Level (`scenario_<id>.json`)
Each JSON file contains all episodes executed on a specific environment layout:
```json
{
  "schema_version": "1.0.0",
  "scenario_id": "scen_grid16_obs04_0001",
  "grid_dimensions": [16, 16],
  "static_obstacles": [[4, 4], [4, 5], [4, 6], [10, 11], [10, 12]],
  "candidate_targets": [
    {"target_id": 0, "position": [1, 1], "name": "Target_A"},
    {"target_id": 1, "position": [14, 1], "name": "Target_B"},
    {"target_id": 2, "position": [1, 14], "name": "Target_C"},
    {"target_id": 3, "position": [14, 14], "name": "Target_D"}
  ],
  "episodes": [
    { ... }
  ]
}
```

### 3.2 Episode Level (`episodes[k]`)
Each episode record contains full run history and metadata. The structure below
is exactly what `src.trajectory.serialize.episode_to_schema` writes:

```json
{
  "episode_id": "8a9ef2bcf8aba0c7_ep0000",
  "episode_index": 0,
  "scenario_id": "8a9ef2bcf8aba0c7",
  "seeds": {
    "master_seed": 42, "scenario_seed": 918273, "env_seed": 55123,
    "agent_seed": 77341, "goal_seed": 40921
  },
  "num_agents": 3,
  "focal_agent_id": 0,
  "total_steps": 28,
  "termination_reason": "GOAL_REACHED",

  "privileged_ground_truth": {
    "focal_goal_id": 2,
    "focal_goal_position": [1, 14],
    "focal_initial_intention": "DIRECT",
    "step_intentions": ["DIRECT", "DIRECT", "DETOUR_LEFT", "DETOUR_LEFT", "DIRECT"],
    "intended_actions": ["RIGHT", "RIGHT", "UP", "UP", "RIGHT"],
    "other_agent_goals": [
      {"agent_id": 1, "goal_id": 0, "goal_position": [1, 1]},
      {"agent_id": 2, "goal_id": 3, "goal_position": [14, 14]}
    ],
    "planner_type": "stochastic_policy"
  },

  "observations": [
    {
      "step": 0,
      "focal_agent_state": {
        "position": [8, 8],
        "position_before": [8, 7],
        "velocity": [0, 1]
      },
      "other_agents_state": [
        {
          "agent_id": 1,
          "position": [9, 10],
          "position_before": [9, 9],
          "velocity": [0, 1],
          "realized_action": "DOWN",
          "blocked": false
        }
      ],
      "realized_action": "DOWN",
      "blocked": false
    }
  ]
}
```

Notes:

- `position_before` is the cell occupied when the intention/action for that step
  was chosen. `position` is post-transition. Probes and models predicting
  `I_t` must use `position_before` at step `t` plus step `t-1`'s realized
  action, because `I_t` generates `A_t` (see M2_VERIFICATION_GUIDE §6).
- `seeds` is the addressability metadata required by M2 test 3e: an episode can
  be reconstructed bit-for-bit from `scenario_spec` + `seeds` + `config_hash`
  (`src.trajectory.dataset.reconstruct_episode_record`).
- The scenario file additionally carries `scenario_index`, `config_hash` and the
  full `scenario_spec` (layout, `agent_configs`, `env_params`) so that
  `scenario_id` can be recomputed from the file alone.
- `realized_action` / `blocked` are recorded for peer agents too: peer
  behaviour is observable in Condition A (PROJECT_SPEC §4.2).

---

## 4. Field Semantics and Type Contracts

| Field | Type | Description | Visibility |
|---|---|---|---|
| `episode_id` | `str` | Globally unique identifier `{scenario_id}_{seed}_{idx}` | Observable |
| `scenario_id` | `str` | Deterministic hash of full structural configuration (map, obstacles, targets, initial positions, agent count, agent config, environment parameters) | Observable |
| `step` | `int` | Discrete time index $t \in [0, T-1]$ | Observable |
| `focal_agent_state.position` | `[int, int]` | Discrete coordinates $(x, y) \in [0, W-1] \times [0, H-1]$ | Observable |
| `focal_agent_state.position_before` | `[int, int]` | Cell occupied when this step's intention/action was chosen | Observable |
| `focal_agent_state.velocity` | `[int, int]` | `position - position_before` | Observable |
| `other_agents_state[].realized_action` | `str` | Peer realized action at step $t$ | Observable (Condition A) |
| `other_agents_state` | `list[dict]` | List of observable states for all non-focal agents | Observable |
| `realized_action` | `str` | Realized action at step $t \in \{\text{UP}, \text{DOWN}, \text{LEFT}, \text{RIGHT}, \text{WAIT}\}$ | Observable |
| `blocked` | `bool` | Whether the intended movement was blocked (obstacle/boundary/peer). Condition A only | Observable (Condition A) |
| `intended_action` | `str` | Policy-selected action before transition | **Privileged** (Evaluation Only) |
| `focal_goal_id` | `int` | True target index $g \in \{0, \dots, |\mathcal{G}|-1\}$ | **Privileged** (Evaluation Only) |
| `step_intentions` | `list[str]` | Ground-truth behavioral sub-strategy at step $t$ | **Privileged** (Evaluation Only) |
| `termination_reason` | `str` | Reason why the episode finished | **Privileged** (Evaluation Only) |
| `blocked_reason` | `str` | `none`/`obstacle`/`boundary`/`occupied`/`conflict`; derived from the intended action | **Privileged** (diagnostics only) |
| `seeds` | `dict` | `master_seed`, `scenario_seed`, `env_seed`, `agent_seed`, `goal_seed` | **Privileged** (reproducibility metadata) |
| `scenario_spec`, `config_hash` | `dict`, `str` | Full structural configuration used to derive `scenario_id` | **Privileged** (reproducibility metadata) |

`scenario_id` and `episode_id` are observable identifiers; everything else in
`privileged_ground_truth`, `seeds`, `scenario_spec` and `config_hash` is
quarantined. `src.trajectory.serialize.iter_observations` is the only supported
way to turn a scenario file into a model-facing batch, and it routes through
`src.features.observations.validate_batch`, which raises on any privileged key.
`python -m src.evaluation.validate_dataset` audits a dataset directory for
leakage, bounds violations, checksum mismatches and cross-split scenario
overlap.

---

## 5. Noise and Perturbation Transformations
When evaluating under observational noise (RQ5), a deterministic transformation pipeline is applied strictly at load-time to the observable fields without mutating the raw dataset:
1. **Gaussian Position Jitter**: $\tilde{p}_t = p_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2 I)$
2. **Missing Observations**: With probability $p_{\text{missing}}$, $O_t$ is replaced with an explicit masking flag `is_observed = False`.
3. **Agent Occlusion**: *Removed from the observation model.* The research observer is global; no occlusion or sensing radius is applied to observations. Peer information is removed only in Condition B via the explicit allowlist projection.
