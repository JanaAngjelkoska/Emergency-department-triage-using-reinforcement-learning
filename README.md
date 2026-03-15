# Emergency Department Triage Optimization using Deep Reinforcement Learning

> Intelligent patient prioritization in emergency departments using Deep Q-Networks and queueing theory

## Overview

This project addresses the critical problem of **emergency department (ED) overcrowding** by developing adaptive triage policies using deep reinforcement learning. Patient arrivals are modeled as a Poisson process with exponentially distributed service times, capturing the stochastic nature of ED operations. The triage problem is formulated as a Markov decision process (MDP) derived from an M/M/c/K queue, and DQN and DDQN agents are trained to learn dynamic prioritization strategies that minimize patient waiting times while maintaining high resource utilization.

### Key Features

- **Markovian Queueing Model**: Mathematically rigorous M/M/c/K queue representation of ED operations
- **Deep Reinforcement Learning**: DQN and DDQN agents that learn adaptive triage policies
- **Multi-Objective Optimization**: Balances waiting time, patient acuity, resource utilization, and patient loss prevention
- **Empirical Parameterization**: Arrival and service rates fitted per ESI class from MIMIC-IV-ED
- **Comprehensive Evaluation**: Benchmarked against traditional heuristics (FCFS, fixed-priority)

---

## Problem Statement

Emergency departments face three critical challenges:

1. **Unpredictable Patient Arrivals**: Stochastic demand patterns following a Poisson process
2. **Resource Constraints**: Limited physicians, treatment rooms, and equipment
3. **Patient Heterogeneity**: Varying acuity levels requiring intelligent prioritization

Traditional FCFS policies fail to account for patient urgency, while fixed-priority rules cannot adapt to changing system conditions. This project leverages reinforcement learning to develop **state-dependent, adaptive triage policies** that are evaluated against classical approaches.

---

## Methodology

### 1. Queueing System Model

The ED is modeled as an **M/M/c/K queueing system**:

- **First M**: Poisson arrival process
- **Second M**: Exponential service times
- **c**: Number of parallel servers (physicians)
- **K**: Maximum system capacity (including patients in service)

Patients are grouped into priority classes by acuity and served under a **non-preemptive priority rule** — once treatment begins it cannot be interrupted, preserving the Markovian structure.

### 2. Dataset and Parameters

System parameters are estimated from **MIMIC-IV-ED (v2.0)**, an emergency department dataset from PhysioNet containing de-identified clinical records with arrival/discharge timestamps and ESI triage scores. Acuity is encoded via the **Emergency Severity Index (ESI)** scale (1 = most critical, 5 = least urgent).

Per-class arrival rates λᵢ and service rates μᵢ are fitted directly from the data:

| ESI Class | Description  | λᵢ (pat/hr) | μᵢ (pat/hr) | ρᵢ = λᵢ/μᵢ |
|-----------|-------------|-------------|-------------|------------|
| 1         | Immediate   | 0.0270      | 0.1642      | 0.1647     |
| 2         | Emergent    | 0.1556      | 0.1164      | 1.3366     |
| 3         | Urgent      | 0.2516      | 0.1438      | 1.7494     |
| 4         | Less Urgent | 0.0319      | 0.2864      | 0.1114     |
| 5         | Non-Urgent  | 0.0013      | 0.4055      | 0.0032     |

ESI classes 2 and 3 exhibit ρᵢ > 1, motivating the multi-server formulation. The minimum number of servers is estimated as:
```
c_min = ⌈Σ λᵢ/μᵢ⌉ = 4
```

This yields an implied system utilization of ~0.84, consistent with empirically observed ED physician utilization ranges.

### 3. MDP Formulation

#### State Space
```
s = (n₁, n₂, ..., nₖ, o₁, o₂, ..., oᴄ)
```

Each state encodes queue lengths per ESI class and server occupancy. For each class, the state additionally includes the maximum and mean waiting time of queued patients, giving `state_dim = K×3 + c`.

#### Action Space

Actions correspond to priority orderings over the K classes — all K! permutations are valid actions. The agent selects an ordering and servers are filled greedily according to it:
```
|A| = K! = 120  (for K=5)
```

#### Reward Function
```
r(s, a) = r_wait + r_idle + r_loss + r_penalty
```

| Component | Formula | Weight |
|-----------|---------|--------|
| Waiting time penalty | `-Σ wᵢ × Wᵢ` for assigned patients | — |
| Queue stagnation penalty | `-0.5 × Σ wᵢ × W̄ᵢ` when no assignment | 0.5 |
| Patient loss penalty | `-5.0 × Lₜ` | 5.0 |
| Server idleness penalty | `-0.1 × n_idle` | 0.1 |

### 4. Algorithms

#### Deep Q-Network (DQN)

Approximates Q*(s, a) by minimizing the TD error with a target network θ⁻:
```
y = r + γ × max Q(s', a'; θ⁻)
L(θ) = E[(y - Q(s, a; θ))²]
```

#### Double DQN (DDQN)

Decouples action selection from value estimation to reduce overestimation bias:
```
y = r + γ × Q(s', argmax Q(s', a'; θ), θ⁻)
```

#### Network Architecture
```
Input  →  Linear(state_dim, 256)  →  ReLU
       →  Linear(256, 256)        →  ReLU
       →  Linear(256, 128)        →  ReLU  →  Dropout(0.1)
       →  Linear(128, num_actions)
```

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.0005 |
| Discount factor γ | 0.99 |
| Batch size | 128 |
| Replay buffer | 100,000 |
| ε start / end | 1.0 / 0.01 |
| ε decay | 0.995 |
| Soft update τ | 0.005 |
| Max steps per episode | 1,000 |
| Training episodes | 2,000 |

### 5. Evaluation Metrics

All metrics are computed per episode and averaged across evaluation episodes:

| Metric | Description |
|--------|-------------|
| **Average episode reward** | Mean cumulative return $\bar{G} = \frac{1}{M}\sum_m G^{(m)}$ |
| **Average waiting time** | Mean waiting time across all served patients |
| **Weighted waiting time** | Acuity-weighted mean waiting time |
| **Per-class waiting time** | Mean waiting time per ESI class |
| **System utilization** | ρ = Σλᵢ / (μ̄ · c) |
| **Patients served** | Total patients completing service per episode |
| **Lost patients** | Arrivals rejected due to system at capacity K |
| **Patients remaining in queue** | Queue occupancy at episode end |
| **Patients in service** | Server occupancy at episode end |

### 6. Baselines

- **FCFS**: Patients served in arrival order, no acuity consideration
- **Fixed Priority**: Static ordering by descending acuity weight (ESI 1 → 5)

---

*Last Updated: March 2026*
