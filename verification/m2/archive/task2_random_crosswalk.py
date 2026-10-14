#!/usr/bin/env python3
"""Random baseline crosswalk for Task 2."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from src.agents.intentions import ALL_INTENTIONS
from src.evaluation.baselines import RandomBaseline
from src.environment.gridworld import GridWorld
from src.trajectory.generator import generate_episode
from src.trajectory.scenarios import ScenarioConfig
from src.agents.policy import StochasticPolicy


def make_basic_config() -> ScenarioConfig:
    """Create a basic scenario configuration for testing."""
    return ScenarioConfig(
        width=8,
        height=8,
        obstacles=((2, 2), (3, 3), (4, 4)),
        targets=((6, 6), (7, 7)),
        agent_starts=((0, 0), (1, 1)),
        num_agents=2,
        agent_configs=(
            {"planner": "stochastic_policy", "policy_sensing_radius": 3,
             "intention": {"min_dwell_steps": 4, "stochastic": True,
                          "blocked_detour_threshold": 2},
             "behavior_weights": {"DIRECT": 0.6, "DETOUR_LEFT": 0.15,
                                 "DETOUR_RIGHT": 0.15, "WAIT": 0.1}},
            {"planner": "goal_seeker"}
        ),
        env_params=()
    )


def create_policies_from_config(config: ScenarioConfig, 
                                env_seeds: Dict[str, int] = None) -> Dict[int, StochasticPolicy]:
    """Create agent policies from scenario configuration."""
    if env_seeds is None:
        env_seeds = {"env_seed": 42, "agent_seed": 123}
    
    policies = {}
    # Create focal agent (stochastic policy)
    focal_rng = random.Random(env_seeds["agent_seed"])
    policies[0] = StochasticPolicy(
        rng=focal_rng,
        min_dwell_steps=config.agent_configs[0]["intention"]["min_dwell_steps"],
        blocked_detour_threshold=config.agent_configs[0]["intention"]["blocked_detour_threshold"],
        temperature=1.0,
        persistence_bonus=1.5,
        wait_bias=0.0,
        detour_boost=0.3
    )
    
    # Create other agents (using stochastic policy for simplicity in testing)
    for i in range(1, config.num_agents):
        agent_rng = random.Random(env_seeds["agent_seed"] + i * 100)
        policies[i] = StochasticPolicy(
            rng=agent_rng,
            min_dwell_steps=config.agent_configs[0]["intention"]["min_dwell_steps"],
            blocked_detour_threshold=config.agent_configs[0]["intention"]["blocked_detour_threshold"],
            temperature=1.0,
            persistence_bonus=1.5,
            wait_bias=0.0,
            detour_boost=0.3
        )
    
    return policies


def generate_episodes_for_baselines(config: ScenarioConfig, num_episodes: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate episodes for baseline evaluation."""
    # Set seeds for reproducibility
    random.seed(seed)
    
    episodes = []
    
    for episode_idx in range(num_episodes):
        seeds = {"env_seed": 42 + episode_idx * 100, 
                 "agent_seed": 123 + episode_idx * 100}
        
        # Create grid world
        grid = GridWorld(
            width=config.width,
            height=config.height,
            obstacles=list(config.obstacles),
            targets=list(config.targets)
        )
        
        policies = create_policies_from_config(config, seeds)
        starts = {0: config.agent_starts[0], 1: config.agent_starts[1]}
        goals = {0: config.targets[0], 1: config.targets[1]}
        
        episode = generate_episode(
            scenario_id=config.scenario_id(),
            grid=grid,
            agents=policies,
            goals=goals,
            starts=starts,
            max_steps=50,
            deadlock_limit=10,
            seed_info=seeds,
            policy_sensing_radius=config.agent_configs[0]["policy_sensing_radius"]
        )
        
        episodes.append(episode)
    
    return episodes


def main():
    config = make_basic_config()
    episodes = generate_episodes_for_baselines(config, num_episodes=15, seed=42)
    
    # Extract the evaluation set (focal agent steps)
    true_intentions = []
    for episode in episodes:
        focal_agent_id = episode["focal_agent_id"]
        steps = [s for s in episode["steps"] if s["agent_id"] == focal_agent_id]
        for step in steps:
            true_intentions.append(step["intention"])
    
    # Compute label histogram
    from collections import Counter
    label_counts = Counter(true_intentions)
    total_labels = len(true_intentions)
    label_hist = {intent: count / total_labels for intent, count in label_counts.items()}
    
    # Random baseline
    targets = [list(t) for t in config.targets]
    intentions = list(ALL_INTENTIONS)
    random_baseline = RandomBaseline(targets=targets, intentions=intentions, rng=random.Random(42))
    
    # Prediction rule: returns a uniform distribution over intentions
    pred_dist = random_baseline.predict_intention()
    # The prediction for each step is the argmax of this distribution (since we use argmax in the test)
    # But note: the distribution is uniform, so the argmax is the first intention in the list (by max's tie-breaking)
    # However, in the test we used: random_pred_intent = max(random_pred_dist, key=random_pred_dist.get)
    # This returns the first intention with the maximum value (since all are equal, it's the first in the dict).
    # The dict is built from the intentions list, so the order is the order of intentions in the list.
    # We'll compute the prediction histogram by simulating the test's prediction method.
    pred_intentions = []
    for _ in true_intentions:
        pred_dist = random_baseline.predict_intention()
        pred_intent = max(pred_dist, key=pred_dist.get)  # argmax with tie-breaking by first max
        pred_intentions.append(pred_intent)
    
    # Compute prediction histogram
    pred_counts = Counter(pred_intentions)
    pred_hist = {intent: count / total_labels for intent, count in pred_counts.items()}
    
    # Check if Random baseline is prior-argmax: 
    #   For each step, the predicted intention should be the argmax of the prior (label histogram).
    #   Since the prior is fixed, the prior-argmax is the same for all steps.
    #   We'll compute the prior-argmax from the label histogram.
    prior_argmax = max(label_hist, key=label_hist.get)
    # Check if all predictions equal the prior-argmax
    all_prior_argmax = all(p == prior_argmax for p in pred_intentions)
    
    # Output
    print("=== Random Baseline Crosswalk ===")
    print(f"Prediction rule: returns uniform distribution over intentions {intentions}")
    print(f"Distribution: {pred_dist}")
    print(f"Seed: 42 (used in RandomBaseline constructor)")
    print(f"Label histogram (true intentions): {label_hist}")
    print(f"Prediction histogram: {pred_hist}")
    print(f"Prior-argmax (from label histogram): {prior_argmax}")
    print(f"All predictions equal prior-argmax? {all_prior_argmax}")
    if all_prior_argmax:
        print("Conclusion: Random baseline behaves as prior-argmax at artifact level.")
    else:
        print("Conclusion: Random baseline does not behave as prior-argmax.")
    

if __name__ == "__main__":
    main()
