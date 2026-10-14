#!/usr/bin/env python3
"""Debug script to see what the baselines are predicting."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
from collections import Counter

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
    np.random.seed(seed)
    
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


def debug_baselines(episodes: List[Dict[str, Any]]):
    """Debug what the baselines are predicting."""
    print("=== DEBUGGING BASELINES ===")
    
    # Extract targets and intentions for baseline initialization
    config = make_basic_config()
    targets = [list(t) for t in config.targets]  # Convert tuples to lists
    intentions = list(ALL_INTENTIONS)
    
    print(f"Targets: {targets}")
    print(f"Intentions: {intentions}")
    
    # Initialize baselines
    random_baseline = RandomBaseline(targets=targets, intentions=intentions, rng=random.Random(42))
    distance_baseline = DistanceHeuristicBaseline(targets=targets, beta=0.5)
    bayesian_baseline = BayesianBeliefBaseline(targets=targets, intentions=intentions, persist=0.85, beta=1.0)
    
    # Collect predictions and true labels for each baseline
    random_true_intentions = []
    random_pred_intentions = []
    
    distance_true_intentions = []
    distance_pred_intentions = []
    
    bayesian_true_intentions = []
    bayesian_pred_intentions = []
    
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
        for step_idx, step in enumerate(steps[:5]):  # Just first 5 steps for debugging
            true_intention = step["intention"]
            current_pos = tuple(step["position"])
            realized_action = step.get("realized_action", "WAIT")  # Default to WAIT if not present
            
            print(f"\nStep {step_idx}:")
            print(f"  Position: {current_pos}")
            print(f"  True intention: {true_intention}")
            print(f"  Realized action: {realized_action}")
            
            # For Random baseline: same prediction regardless of input
            random_pred_dist = random_baseline.predict_intention()
            random_pred_intent = max(random_pred_dist, key=random_pred_dist.get)  # Most likely intention
            print(f"  Random prediction: {random_pred_dist}")
            print(f"  Random predicted intention: {random_pred_intent}")
            
            # For Distance heuristic: predict intention based on position
            dist_pred_goal_dist = distance_baseline.predict_goal_distribution(current_pos)
            print(f"  Distance goal distribution: {dist_pred_goal_dist}")
            
            # Convert goal distribution to intention prediction
            # Let's try a simple approach: assume P(I|G=0) and P(I|G=1) are uniform for now
            # This is not correct but let's see what happens
            goal0_prob = dist_pred_goal_dist.get(0, 0.0)
            goal1_prob = dist_pred_goal_dist.get(1, 0.0)
            
            # Simple conversion: if goal 0 is more likely, predict intention that's "more like" goal 0
            # But we don't have a good mapping, so let's just use the goal distribution directly
            # and map goals to intentions arbitrarily for now
            # Actually, let's just predict the most likely goal and then pick a random intention
            # This is just for debugging
            most_likely_goal = max(dist_pred_goal_dist, key=dist_pred_goal_dist.get)
            print(f"  Most likely goal: {most_likely_goal}")
            
            # For now, let's just make distance predict uniformly like random to see if that's the issue
            distance_pred_dist = random_baseline.predict_intention()  # Same as random for now
            distance_pred_intent = max(distance_pred_dist, key=distance_pred_dist.get)
            print(f"  Distance prediction: {distance_pred_dist}")
            print(f"  Distance predicted intention: {distance_pred_intent}")
            
            # For Bayesian baseline: predict intention based on current state and action
            # Skip update on first step since we don't have previous position
            if prev_pos is not None:
                bayesian_baseline.update(prev_pos, current_pos, realized_action)
            bayesian_goal_dist = bayesian_baseline.goal_distribution()
            print(f"  Bayesian goal distribution: {bayesian_goal_dist}")
            
            # Get intention distribution by marginalizing over goals
            # P(I) = sum_G P(I|G) P(G)
            # We need P(I|G). Let's assume it's uniform for now: P(I|G) = 1/|I|
            bayesian_pred_dist = {intent: 1.0/len(intentions) for intent in intentions}
            bayesian_pred_intent = max(bayesian_pred_dist, key=bayesian_pred_dist.get)
            print(f"  Bayesian prediction: {bayesian_pred_dist}")
            print(f"  Bayesian predicted intention: {bayesian_pred_intent}")
            
            # Store predictions and true values
            random_true_intentions.append(true_intention)
            random_pred_intentions.append(random_pred_intent)
            
            distance_true_intentions.append(true_intention)
            distance_pred_intentions.append(distance_pred_intent)
            
            bayesian_true_intentions.append(true_intention)
            bayesian_pred_intentions.append(bayesian_pred_intent)
            
            # Update previous position for next iteration
            prev_pos = current_pos
            
            # Break after a few steps
            if step_idx >= 4:
                break
    
    # Calculate metrics for the samples we processed
    print("\n=== METRICS (first episode, first few steps) ===")
    if random_true_intentions:
        from sklearn.metrics import accuracy_score, f1_score
        
        random_acc = accuracy_score(random_true_intentions, random_pred_intentions)
        random_f1 = f1_score(random_true_intentions, random_pred_intentions, average="macro")
        
        distance_acc = accuracy_score(distance_true_intentions, distance_pred_intentions)
        distance_f1 = f1_score(distance_true_intentions, distance_pred_intentions, average="macro")
        
        bayesian_acc = accuracy_score(bayesian_true_intentions, bayesian_pred_intentions)
        bayesian_f1 = f1_score(bayesian_true_intentions, bayesian_pred_intentions, average="macro")
        
        print(f"Random:  Acc={random_acc:.4f}, F1={random_f1:.4f}")
        print(f"Distance: Acc={distance_acc:.4f}, F1={distance_f1:.4f}")
        print(f"Bayesian: Acc={bayesian_acc:.4f}, F1={bayesian_f1:.4f}")


if __name__ == "__main__":
    config = make_basic_config()
    episodes = generate_episodes_for_baselines(config, num_episodes=3, seed=42)
    debug_baselines(episodes)