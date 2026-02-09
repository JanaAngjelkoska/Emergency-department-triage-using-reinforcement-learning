import numpy as np
from environment.EDTriageEnv import EDTriageEnv


def evaluate_agent(agent, env: EDTriageEnv, num_episodes: int = 100, epsilon: float = 0.0) -> dict:
    all_rewards = []
    all_metrics = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(state, epsilon)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        all_rewards.append(episode_reward)
        all_metrics.append(env.get_metrics())

    avg_reward = np.mean(all_rewards)
    avg_waiting = np.mean([m["average_waiting_time"] for m in all_metrics])
    avg_weighted_waiting = np.mean([m["weighted_waiting_time"] for m in all_metrics])
    avg_utilization = np.mean([m["system_utilization"] for m in all_metrics])
    avg_served = np.mean([m["total_served"] for m in all_metrics])

    return {
        "avg_episode_reward": float(avg_reward),
        "average_waiting_time": float(avg_waiting),
        "weighted_waiting_time": float(avg_weighted_waiting),
        "system_utilization": float(avg_utilization),
        "avg_patients_served": float(avg_served)
    }


def evaluate_baseline(policy, env: EDTriageEnv, num_episodes: int = 100) -> dict:
    all_rewards = []
    all_metrics = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = policy.select_action(env)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        all_rewards.append(episode_reward)
        all_metrics.append(env.get_metrics())

    avg_reward = np.mean(all_rewards)
    avg_waiting = np.mean([m["average_waiting_time"] for m in all_metrics])
    avg_weighted_waiting = np.mean([m["weighted_waiting_time"] for m in all_metrics])
    avg_utilization = np.mean([m["system_utilization"] for m in all_metrics])
    avg_served = np.mean([m["total_served"] for m in all_metrics])

    return {
        "avg_episode_reward": float(avg_reward),
        "average_waiting_time": float(avg_waiting),
        "weighted_waiting_time": float(avg_weighted_waiting),
        "system_utilization": float(avg_utilization),
        "avg_patients_served": float(avg_served)
    }
