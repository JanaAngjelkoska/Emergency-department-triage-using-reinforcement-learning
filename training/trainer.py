import numpy as np
from model.DQN import DQN
from model.DDQN import DDQN
from environment.EDTriageEnv import EDTriageEnv
from training.evaluator import evaluate_agent

def train_dqn(agent: DQN,
              env: EDTriageEnv,
              num_episodes: int = 1000,
              epsilon_start: float = 1.0,
              epsilon_end: float = 0.05,
              epsilon_decay: float = 0.995,
              target_update_freq: int = 50,
              eval_freq: int = 100,
              eval_episodes: int = 50,
              model_name: str = "dqn") -> tuple[list[float], list[dict], int]:

    epsilon = epsilon_start
    episode_rewards: list[float] = []
    eval_results: list[dict] = []

    best_eval_reward = -np.inf
    best_episode = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(state, epsilon)
            next_state, reward, done, info = env.step(action)

            agent.update_memory(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % target_update_freq == 0:
            agent.update_target_model()

        if (episode + 1) % eval_freq == 0:
            eval_env = EDTriageEnv(
                num_priority_classes=env.K,
                num_servers=env.c,
                arrival_rates=env.arrival_rates,
                service_rates=env.service_rates,
                acuity_weights=env.acuity_weights,
                max_patients=env.max_patients,
                max_steps=env.max_steps
            )
            metrics = evaluate_agent(agent, eval_env, num_episodes=eval_episodes, epsilon=0.0)
            eval_results.append(metrics)

            print(f"[DQN] Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {np.mean(episode_rewards[-eval_freq:]):.2f} | "
                  f"Eval Reward: {metrics['avg_episode_reward']:.2f} | "
                  f"Avg Wait: {metrics['average_waiting_time']:.4f} | "
                  f"Weighted Wait: {metrics['weighted_waiting_time']:.4f} | "
                  f"Epsilon: {epsilon:.4f}")

            if metrics['avg_episode_reward'] > best_eval_reward:
                best_eval_reward = metrics['avg_episode_reward']
                best_episode = episode + 1
                agent.save(model_name, best_episode)
                print(f"  -> New best model saved at episode {best_episode} with eval reward {best_eval_reward:.2f}")

    return episode_rewards, eval_results, best_episode


def train_ddqn(agent: DDQN,
               env: EDTriageEnv,
               num_episodes: int = 1000,
               epsilon_start: float = 1.0,
               epsilon_end: float = 0.05,
               epsilon_decay: float = 0.995,
               target_update_freq: int = 50,
               eval_freq: int = 100,
               eval_episodes: int = 50,
               model_name: str = "ddqn") -> tuple[list[float], list[dict], int]:

    epsilon = epsilon_start
    episode_rewards: list[float] = []
    eval_results: list[dict] = []

    best_eval_reward = -np.inf
    best_episode = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(state, epsilon)
            next_state, reward, done, info = env.step(action)

            agent.update_memory(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode + 1) % target_update_freq == 0:
            agent.update_target_model()

        if (episode + 1) % eval_freq == 0:
            eval_env = EDTriageEnv(
                num_priority_classes=env.K,
                num_servers=env.c,
                arrival_rates=env.arrival_rates,
                service_rates=env.service_rates,
                acuity_weights=env.acuity_weights,
                max_patients=env.max_patients,
                max_steps=env.max_steps
            )
            metrics = evaluate_agent(agent, eval_env, num_episodes=eval_episodes, epsilon=0.0)
            eval_results.append(metrics)

            print(f"[DDQN] Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {np.mean(episode_rewards[-eval_freq:]):.2f} | "
                  f"Eval Reward: {metrics['avg_episode_reward']:.2f} | "
                  f"Avg Wait: {metrics['average_waiting_time']:.4f} | "
                  f"Weighted Wait: {metrics['weighted_waiting_time']:.4f} | "
                  f"Epsilon: {epsilon:.4f}")

            if metrics['avg_episode_reward'] > best_eval_reward:
                best_eval_reward = metrics['avg_episode_reward']
                best_episode = episode + 1
                agent.save(model_name, best_episode)
                print(f"  -> New best model saved at episode {best_episode} with eval reward {best_eval_reward:.2f}")

    return episode_rewards, eval_results, best_episode
