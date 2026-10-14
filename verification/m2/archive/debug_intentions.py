#!/usr/bin/env python3
"""Debug script to see intention distribution."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
from collections import Counter

from src.agents.intentions import ALL_INTENTIONS
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


def analyze_intentions(episodes: List[Dict[str, Any]]):
    """Analyze the intention distribution in the episodes."""
    all_intentions = []
    
    for episode in episodes:
        focal_agent_id = episode["focal_agent_id"]
        steps = [s for s in episode["steps"] if s["agent_id"] == focal_agent_id]
        
        for step in steps:
            all_intentions.append(step["intention"])
    
    counter = Counter(all_intentions)
    total = len(all_intentions)
    
    print(f"Total intention samples: {total}")
    print("Intention distribution:")
    for intention in ALL_INTENTIONS:
        count = counter.get(intention, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {intention}: {count} ({percentage:.1f}%)")
    
    return counter


if __name__ == "__main__":
    config = make_basic_config()
    episodes = generate_episodes_for_baselines(config, num_episodes=5, seed=42)
    analyze_intentions(episodes)