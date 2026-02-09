import numpy as np
import torch
from model.QNetwork import QNetwork
from model.DQN import DQN
from model.DDQN import DDQN
from environment.EDTriageEnv import EDTriageEnv
from baselines.policies import FCFSPolicy, FixedPriorityPolicy
from training.trainer import train_dqn, train_ddqn
from training.evaluator import evaluate_agent, evaluate_baseline

NUM_PRIORITY_CLASSES = 3
NUM_SERVERS = 3
ARRIVAL_RATES = [0.08, 0.12, 0.05] #, 0.07, 0.04]
SERVICE_RATES =  [0.15, 0.10, 0.08] #, 0.06, 0.05]
ACUITY_WEIGHTS = [3.0, 2.0, 1.0]
MAX_PATIENTS = 20
MAX_STEPS = 500

HIDDEN_DIM = 128
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 50000

NUM_EPISODES = 1000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.990
TARGET_UPDATE_FREQ = 100

EVAL_FREQ = 100
EVAL_EPISODES = 200
FINAL_EVAL_EPISODES = 500


def make_env() -> EDTriageEnv:
    return EDTriageEnv(
        num_priority_classes=NUM_PRIORITY_CLASSES,
        num_servers=NUM_SERVERS,
        arrival_rates=ARRIVAL_RATES,
        service_rates=SERVICE_RATES,
        acuity_weights=ACUITY_WEIGHTS,
        max_patients=MAX_PATIENTS,
        max_steps=MAX_STEPS
    )


def make_dqn_agent(state_dim: int, num_actions: int) -> DQN:
    model = QNetwork(state_dim, num_actions, HIDDEN_DIM)
    target_model = QNetwork(state_dim, num_actions, HIDDEN_DIM)
    return DQN(
        state_space_shape=state_dim,
        num_actions=num_actions,
        model=model,
        target_model=target_model,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE
    )


def make_ddqn_agent(state_dim: int, num_actions: int) -> DDQN:
    model = QNetwork(state_dim, num_actions, HIDDEN_DIM)
    target_model = QNetwork(state_dim, num_actions, HIDDEN_DIM)
    return DDQN(
        state_space_shape=state_dim,
        num_actions=num_actions,
        model=model,
        target_model=target_model,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE
    )


def print_results_table(results: dict[str, dict]) -> None:
    header = f"{'Policy':<20} {'Avg Reward':>12} {'Avg Wait':>12} {'Wtd Wait':>12} {'Utilization':>12} {'Avg Served':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for name, metrics in results.items():
        print(f"{name:<20} "
              f"{metrics['avg_episode_reward']:>12.2f} "
              f"{metrics['average_waiting_time']:>12.4f} "
              f"{metrics['weighted_waiting_time']:>12.4f} "
              f"{metrics['system_utilization']:>12.4f} "
              f"{metrics['avg_patients_served']:>12.2f}")

    print("=" * len(header) + "\n")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    env = make_env()
    state_dim = env.state_dim
    num_actions = env.num_actions

    print("=" * 60)
    print(" Training DQN ")
    print("=" * 60)
    dqn_agent = make_dqn_agent(state_dim, num_actions)
    dqn_rewards, dqn_evals, dqn_best_episode = train_dqn(
        agent=dqn_agent,
        env=env,
        num_episodes=NUM_EPISODES,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE_FREQ,
        eval_freq=EVAL_FREQ,
        eval_episodes=EVAL_EPISODES,
        model_name="dqn"
    )

    print("\n" + "=" * 60)
    print(" Training DDQN ")
    print("=" * 60)
    ddqn_agent = make_ddqn_agent(state_dim, num_actions)
    ddqn_rewards, ddqn_evals, ddqn_best_episode = train_ddqn(
        agent=ddqn_agent,
        env=env,
        num_episodes=NUM_EPISODES,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE_FREQ,
        eval_freq=EVAL_FREQ,
        eval_episodes=EVAL_EPISODES,
        model_name="ddqn"
    )

    print("\n" + "=" * 60)
    print(" Final Evaluation (best models) ")
    print("=" * 60)

    eval_env = make_env()

    dqn_agent.load("dqn", dqn_best_episode)
    ddqn_agent.load("ddqn", ddqn_best_episode)

    dqn_metrics = evaluate_agent(dqn_agent, eval_env, num_episodes=FINAL_EVAL_EPISODES, epsilon=0.0)
    ddqn_metrics = evaluate_agent(ddqn_agent, eval_env, num_episodes=FINAL_EVAL_EPISODES, epsilon=0.0)

    fcfs_policy = FCFSPolicy(NUM_PRIORITY_CLASSES)
    fcfs_metrics = evaluate_baseline(fcfs_policy, eval_env, num_episodes=FINAL_EVAL_EPISODES)

    fixed_priority_policy = FixedPriorityPolicy(NUM_PRIORITY_CLASSES, ACUITY_WEIGHTS)
    fixed_priority_metrics = evaluate_baseline(fixed_priority_policy, eval_env, num_episodes=FINAL_EVAL_EPISODES)

    results = {
        "DQN": dqn_metrics,
        "DDQN": ddqn_metrics,
        "FCFS": fcfs_metrics,
        "Fixed Priority": fixed_priority_metrics
    }

    print_results_table(results)