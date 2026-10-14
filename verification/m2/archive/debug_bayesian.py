#!/usr/bin/env python3
"""Debug script to see what the Bayesian baseline is doing."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from src.agents.intentions import ALL_INTENTIONS
from src.evaluation.baselines import RandomBaseline, DistanceHeuristicBaseline, BayesianBeliefBaseline
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
    print(f"Generating {num_episodes} episodes for baseline evaluation...")
    
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
    
    print(f"Generated {len(episodes)} episodes")
    return episodes


def debug_bayesian(episodes: List[Dict[str, Any]]):
    """Debug what the Bayesian baseline is doing."""
    print("=== DEBUGGING BAYESIAN BASELINE ===")
    
    # Extract targets and intentions for baseline initialization
    config = make_basic_config()
    targets = [list(t) for t in config.targets]  # Convert tuples to lists
    intentions = list(ALL_INTENTIONS)
    
    print(f"Targets: {targets}")
    print(f"Intentions: {intentions}")
    
    # Initialize Bayesian baseline
    bayesian_baseline = BayesianBeliefBaseline(targets=targets, intentions=intentions, persist=0.85, beta=1.0)
    
    print("\nProcessing first episode...")
    for episode_idx, episode in enumerate(episodes[:1]):  # Just first episode for debugging
        print(f"\n--- Episode {episode_idx} ---")
        
        # Reset Bayesian baseline for each episode
        bayesian_baseline.reset()
        
        # Get the steps for the focal agent
        focal_agent_id = episode["focal_agent_id"]
        steps = [s for s in episode["steps"] if s["agent_id"] == focal_agent_id]
        
        if not steps:
            continue
            
        print(f"Number of steps: {len(steps)}")
        
        # Reset Bayesian baseline at start of episode
        bayesian_baseline.reset()
        
        prev_pos = None
        for step_idx, step in enumerate(steps[:10]):  # Just first 10 steps for debugging
            true_intention = step["intention"]
            current_pos = tuple(step["position"])
            realized_action = step.get("realized_action", "WAIT")  # Default to WAIT if not present
            
            print(f"\nStep {step_idx}:")
            print(f"  Position: {current_pos}")
            print(f"  True intention: {true_intention}")
            print(f"  Realized action: {realized_action}")
            
            # Show belief state before update
            if step_idx > 0:  # Not the first step
                print(f"  Belief before update: {dict(list(bayesian_baseline.belief.items())[:5])}...")  # Show first 5 items
            
            # For Bayesian baseline: update belief state
            # Skip update on first step since we don't have previous position
            if prev_pos is not None:
                bayesian_baseline.update(prev_pos, current_pos, realized_action)
                print(f"  Belief after update: {dict(list(bayesian_baseline.belief.items())[:5])}...")
            else:
                print(f"  Skipping update (first step)")
            
            # Get intention distribution by marginalizing over goals
            bayesian_intent_dist = {}
            for intention in intentions:
                bayesian_intent_dist[intention] = 0.0
            for (g, i), prob in bayesian_baseline.belief.items():
                if i in bayesian_intent_dist:
                    bayesian_intent_dist[i] += prob
            # Normalize
            total = sum(bayesian_intent_dist.values())
            if total > 0:
                bayesian_intent_dist = {i: p / total for i, p in bayesian_intent_dist.items()}
            
            print(f"  Intention distribution: {bayesian_intent_dist}")
            
            # Get most likely intention
            if bayesian_intent_dist:
                bayesian_pred_intent = max(bayesian_intent_dist, key=bayesian_intent_dist.get)
                print(f"  Predicted intention: {bayesian_pred_intent}")
                print(f"  Correct? {bayesian_pred_intent == true_intention}")
            
            # Update previous position for next iteration
            prev_pos = current_pos
            
            # Break after a few steps
            if step_idx >= 9:
                break


if __name__ == "__main__":
    config = make_basic_config()
    episodes = generate_episodes_for_baselines(config, num_episodes=2, seed=42)
    debug_bayesian(episodes)