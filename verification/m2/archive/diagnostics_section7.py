#!/usr/bin/env python3
"""Diagnostic script for Section 7 baselines."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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


def run_diagnostics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run diagnostics and return detailed metrics."""
    print("Running diagnostics...")
    
    config = make_basic_config()
    targets = [list(t) for t in config.targets]
    intentions = list(ALL_INTENTIONS)
    
    random_baseline = RandomBaseline(targets=targets, intentions=intentions, rng=random.Random(42))
    distance_baseline = DistanceHeuristicBaseline(targets=targets, beta=0.5)
    bayesian_baseline = BayesianBeliefBaseline(targets=targets, intentions=intentions, persist=0.85, beta=1.0)
    
    # Storage for detailed predictions
    random_true = []
    random_pred = []
    distance_true = []
    distance_pred = []
    bayesian_true = []
    bayesian_pred = []
    bayesian_probs = []  # Store full probability distributions for NLL
    
    # For distance score variance
    distance_goal_scores_all = []  # List of dicts per step: {goal_idx: score}
    
    # For temporal alignment check: we'll also store what we predict for intention vs what we should
    
    print("Processing episodes...")
    for episode_idx, episode in enumerate(episodes):
        bayesian_baseline.reset()
        
        focal_agent_id = episode["focal_agent_id"]
        steps = [s for s in episode["steps"] if s["agent_id"] == focal_agent_id]
        
        if not steps:
            continue
        
        prev_pos = None
        for step_idx, step in enumerate(steps):
            true_intention = step["intention"]
            current_pos = tuple(step["position"])
            realized_action = step.get("realized_action", "WAIT")
            
            # Random baseline
            random_pred_dist = random_baseline.predict_intention()
            random_pred_intent = max(random_pred_dist, key=random_pred_dist.get)
            
            # Distance baseline: get goal distribution, then we need to map to intention
            # We'll use the same mapping as before: fixed intention distribution based on behavior weights
            dist_pred_goal_dist = distance_baseline.predict_goal_distribution(current_pos)
            # Store goal scores for variance check
            distance_goal_scores_all.append(dist_pred_goal_dist.copy())
            # Convert to intention prediction: we use the fixed intention distribution
            dist_pred_intent_dist = {
                "DIRECT": 0.6,
                "DETOUR_LEFT": 0.15,
                "DETOUR_RIGHT": 0.15,
                "WAIT": 0.1
            }
            distance_pred_intent = max(dist_pred_intent_dist, key=dist_pred_intent_dist.get)
            
            # Bayesian baseline: update belief if not first step
            if prev_pos is not None:
                bayesian_baseline.update(prev_pos, current_pos, realized_action)
            # Get intention distribution by marginalizing over goals
            bayesian_intent_dist = {}
            for intention in intentions:
                bayesian_intent_dist[intention] = 0.0
            for (g, i), prob in bayesian_baseline.belief.items():
                if i in bayesian_intent_dist:
                    bayesian_intent_dist[i] += prob
            total = sum(bayesian_intent_dist.values())
            if total > 0:
                bayesian_intent_dist = {i: p / total for i, p in bayesian_intent_dist.items()}
            # Sample? No, we'll use the distribution for prediction and for NLL
            bayesian_pred_intent = max(bayesian_intent_dist, key=bayesian_intent_dist.get)
            bayesian_probs.append(bayesian_intent_dist.copy())
            
            # Store
            random_true.append(true_intention)
            random_pred.append(random_pred_intent)
            
            distance_true.append(true_intention)
            distance_pred.append(distance_pred_intent)
            
            bayesian_true.append(true_intention)
            bayesian_pred.append(bayesian_pred_intent)
            
            prev_pos = current_pos
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score
    
    random_acc = accuracy_score(random_true, random_pred)
    random_f1 = f1_score(random_true, random_pred, average="macro")
    
    distance_acc = accuracy_score(distance_true, distance_pred)
    distance_f1 = f1_score(distance_true, distance_pred, average="macro")
    
    bayesian_acc = accuracy_score(bayesian_true, bayesian_pred)
    bayesian_f1 = f1_score(bayesian_true, bayesian_pred, average="macro")
    
    # Confusion matrix for Bayesian
    cm = confusion_matrix(bayesian_true, bayesian_pred, labels=intentions)
    conf_matrix = {}
    for i, true_intent in enumerate(intentions):
        conf_matrix[true_intent] = {}
        for j, pred_intent in enumerate(intentions):
            conf_matrix[true_intent][pred_intent] = int(cm[i][j])
    
    # Distance score variance: compute variance of scores for each goal across steps
    # We have a list of dicts per step. Convert to array per goal.
    n_goals = len(targets)
    goal_scores = [[] for _ in range(n_goals)]
    for step_scores in distance_goal_scores_all:
        for g in range(n_goals):
            goal_scores[g].append(step_scores.get(g, 0.0))
    
    goal_score_variance = [np.var(scores) if scores else 0.0 for scores in goal_scores]
    goal_score_mean = [np.mean(scores) if scores else 0.0 for scores in goal_scores]
    
    # Check if Random and Distance predictions are identical
    pred_identical = all(rp == dp for rp, dp in zip(random_pred, distance_pred))
    
    # Prefix results: compute cumulative accuracy at 25%, 50%, 100% of data
    n_total = len(bayesian_true)
    indices_25 = int(0.25 * n_total)
    indices_50 = int(0.50 * n_total)
    indices_100 = n_total
    
    def cumulative_metrics(true_list, pred_list, end_idx):
        acc = accuracy_score(true_list[:end_idx], pred_list[:end_idx]) if end_idx > 0 else 0.0
        f1 = f1_score(true_list[:end_idx], pred_list[:end_idx], average="macro") if end_idx > 0 else 0.0
        return acc, f1
    
    bayesian_acc_25, bayesian_f1_25 = cumulative_metrics(bayesian_true, bayesian_pred, indices_25)
    bayesian_acc_50, bayesian_f1_50 = cumulative_metrics(bayesian_true, bayesian_pred, indices_50)
    bayesian_acc_100, bayesian_f1_100 = cumulative_metrics(bayesian_true, bayesian_pred, indices_100)
    
    # NLL vs prior
    # Compute empirical prior from training? We'll use the training set intention frequencies.
    # For simplicity, we'll compute the prior from the entire dataset (not ideal but okay for diagnostic)
    # Better: we should compute prior from the training set only, but we don't have a split here.
    # We'll compute prior from the same data (note: this is optimistic for the prior).
    from collections import Counter
    intent_counts = Counter(bayesian_true)
    prior_probs = {intent: count / n_total for intent, count in intent_counts.items()}
    # Ensure all intentions are present
    for intent in intentions:
        if intent not in prior_probs:
            prior_probs[intent] = 0.0
    # Normalize
    total_prior = sum(prior_probs.values())
    if total_prior > 0:
        prior_probs = {intent: prob / total_prior for intent, prob in prior_probs.items()}
    
    # Compute NLL for Bayesian predictions
    nll_bayesian = 0.0
    for true_intent, prob_dist in zip(bayesian_true, bayesian_probs):
        prob = prob_dist.get(true_intent, 1e-12)  # avoid log(0)
        nll_bayesian += -np.log(prob)
    nll_bayesian /= n_total
    
    # Compute NLL for prior
    nll_prior = 0.0
    for true_intent in bayesian_true:
        prob = prior_probs.get(true_intent, 1e-12)
        nll_prior += -np.log(prob)
    nll_prior /= n_total
    
    # Temporal alignment: we already scored intention at t against intention at t (no shift).
    # To check if we should be scoring against action or something else, we can also compute
    # accuracy of predicting the realized action? Not needed for now.
    
    # Recursive Bayesian state: we already used the belief update that includes I_{t-1} via persist.
    # We can verify that the belief is not uniform by checking entropy.
    belief_entropies = []
    for prob_dist in bayesian_probs:
        # entropy = -sum p log p
        ent = -sum(p * np.log(p + 1e-12) for p in prob_dist.values())
        belief_entropies.append(ent)
    avg_belief_entropy = np.mean(belief_entropies) if belief_entropies else 0.0
    
    return {
        "Random": {"accuracy": random_acc, "macro_f1": random_f1},
        "Distance": {"accuracy": distance_acc, "macro_f1": distance_f1},
        "Bayesian": {
            "accuracy": bayesian_acc,
            "macro_f1": bayesian_f1,
            "confusion_matrix": conf_matrix,
            "nll": nll_bayesian,
            "nll_prior": nll_prior,
            "belief_entropy": avg_belief_entropy,
            "prefix_25": {"accuracy": bayesian_acc_25, "macro_f1": bayesian_f1_25},
            "prefix_50": {"accuracy": bayesian_acc_50, "macro_f1": bayesian_f1_50},
            "prefix_100": {"accuracy": bayesian_acc_100, "macro_f1": bayesian_f1_100},
        },
        "diagnostics": {
            "predictions_identical_random_distance": pred_identical,
            "distance_goal_score_mean": goal_score_mean,
            "distance_goal_score_variance": goal_score_variance,
        }
    }


def main():
    config = make_basic_config()
    episodes = generate_episodes_for_baselines(config, num_episodes=15, seed=42)
    results = run_diagnostics(episodes)
    
    # Print summary
    print("\n=== DIAGNOSTIC RESULTS ===")
    print(f"Random: Acc={results['Random']['accuracy']:.4f}, F1={results['Random']['macro_f1']:.4f}")
    print(f"Distance: Acc={results['Distance']['accuracy']:.4f}, F1={results['Distance']['macro_f1']:.4f}")
    print(f"Bayesian: Acc={results['Bayesian']['accuracy']:.4f}, F1={results['Bayesian']['macro_f1']:.4f}")
    print(f"Bayesian NLL: {results['Bayesian']['nll']:.4f}, Prior NLL: {results['Bayesian']['nll_prior']:.4f}")
    print(f"Predictions identical (Random vs Distance): {results['diagnostics']['predictions_identical_random_distance']}")
    print(f"Distance goal score means: {results['diagnostics']['distance_goal_score_mean']}")
    print(f"Distance goal score variances: {results['diagnostics']['distance_goal_score_variance']}")
    print(f"Bayesian belief entropy (avg): {results['Bayesian']['belief_entropy']:.4f}")
    print("Prefix results (Bayesian):")
    print(f"  25%: Acc={results['Bayesian']['prefix_25']['accuracy']:.4f}, F1={results['Bayesian']['prefix_25']['macro_f1']:.4f}")
    print(f"  50%: Acc={results['Bayesian']['prefix_50']['accuracy']:.4f}, F1={results['Bayesian']['prefix_50']['macro_f1']:.4f}")
    print(f"  100%: Acc={results['Bayesian']['prefix_100']['accuracy']:.4f}, F1={results['Bayesian']['prefix_100']['macro_f1']:.4f}")
    
    # Also output as JSON for easy parsing
    import json
    print("\nJSON output:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()