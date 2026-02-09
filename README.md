# Emergency Department Triage Optimization using Deep Reinforcement Learning

> Intelligent patient prioritization in emergency departments using Deep Q-Networks and queueing theory

## Overview

This project addresses the critical problem of **emergency department (ED) overcrowding** by developing adaptive triage policies using deep reinforcement learning. By modeling patient arrivals as a Poisson process with exponentially distributed service times, we capture the stochastic nature of ED operations and learn dynamic prioritization strategies that minimize patient waiting times while maintaining high resource utilization.

### Key Features

- **Markovian Queueing Model**: Mathematically rigorous M/M/c/K queue representation of ED operations
- **Deep Reinforcement Learning**: DQN and DDQN agents that learn adaptive triage policies
- **Multi-Objective Optimization**: Balances waiting time, patient acuity, resource utilization, and patient loss prevention
- **Realistic Simulation**: Captures stochastic arrivals, variable service times, and finite system capacity
- **Comprehensive Evaluation**: Benchmarked against traditional heuristics (FCFS, fixed-priority)

---

## Problem Statement

Emergency departments face three critical challenges:

1. **Unpredictable Patient Arrivals**: Stochastic demand patterns that vary by time of day, season, and external factors
2. **Resource Constraints**: Limited physicians, treatment rooms, and equipment
3. **Patient Heterogeneity**: Varying acuity levels requiring intelligent prioritization

Traditional first-come, first-served (FCFS) policies fail to account for patient urgency, while fixed-priority rules cannot adapt to changing system conditions. This project leverages reinforcement learning to develop **state-dependent, adaptive triage policies** that outperform classical approaches.

---

## Methodology

### 1. Queueing System Model

The emergency department is modeled as an **M/M/c/K queueing system**:

- **M/M/c/K** notation:
  - First M: Poisson arrival process (memoryless arrivals)
  - Second M: Exponential service times (memoryless service)
  - c: Number of parallel servers (physicians/treatment units)
  - K: Maximum system capacity (including patients in service)

#### Mathematical Foundation

**Poisson Arrivals**: Patient arrivals follow a Poisson process with rate λ:

```
P(N(t) = k) = (λt)^k × e^(-λt) / k!
```

**Exponential Service Times**: Treatment durations are exponentially distributed with rate μ:

```
f(t) = μ × e^(-μt),  t ≥ 0
```

**Key Property**: The memoryless nature of both distributions enables modeling as a **Markov process**, where future system evolution depends only on the current state.

#### System Characteristics

- **Multiple Priority Classes**: Patients grouped by acuity level
- **Finite Capacity**: System can hold at most K patients (arrivals when full are lost)
- **Non-Preemptive Service**: Once treatment begins, it cannot be interrupted
- **Birth-Death Process**: System evolves on finite state space {0, 1, 2, ..., K}

### 2. Markov Decision Process (MDP) Formulation

The triage optimization problem is formalized as an MDP: **( S, A, P, R, γ )**

#### State Space (S)

Each state represents the complete ED configuration:

```
s = (n₁, n₂, ..., nₖ, o₁, o₂, ..., oᴄ)
```

- **nᵢ**: Number of patients waiting in priority class i
- **oⱼ**: Occupancy status of server j (0 = idle, 1 = busy)
- **Constraint**: Total patients ≤ K (system capacity)

#### Action Space (A)

Actions specify which priority class to serve next when servers become available:

```
A(s) = {i : nᵢ > 0, ∃j such that oⱼ = 0}
```

When multiple servers are idle, actions assign waiting patients according to learned priorities.

#### Transition Dynamics (P)

State transitions follow birth-death dynamics:

- **Arrivals**: Class i arrivals occur at rate λᵢ (if capacity permits)
- **Service Completions**: Class i service completes at rate μᵢ

#### Reward Function (R)

Multi-component reward balancing operational objectives:

```
r(s, a) = r_wait + r_idle + r_loss + r_penalty
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Waiting Time Penalty** | `-Σ wᵢ × waiting_time` | Minimize acuity-weighted delays |
| **Queue Stagnation** | `-0.5 × avg_waiting_time` | Prevent prolonged queue buildup |
| **Patient Loss** | `-5.0 × patients_lost` | Strongly discourage capacity saturation |
| **Server Idleness** | `-0.1 × idle_servers` | Encourage resource utilization |

**Penalty Hierarchy**: Patient loss (5.0) >> Queue stagnation (0.5) >> Idleness (0.1)

#### Objective

Find optimal policy π* that maximizes expected cumulative discounted reward:

```
π* = argmax E[Σ γᵗ × r(sₜ, aₜ)]
```

### 3. Deep Reinforcement Learning Algorithms

#### Deep Q-Network (DQN)

Approximates optimal action-value function Q*(s, a) using a neural network Q(s, a; θ).

**Loss Function**:
```
L(θ) = E[(y - Q(s, a; θ))²]

where y = r + γ × max Q(s', a'; θ⁻)
              a'
```

**Key Techniques**:
- **Target Network**: Separate network with frozen parameters θ⁻ updated every τ steps
- **Experience Replay**: Store transitions in buffer D, sample mini-batches to break correlations
- **ε-Greedy Exploration**: Balance exploration (random action) vs exploitation (greedy action)

#### Double Deep Q-Network (DDQN)

Addresses DQN's overestimation bias by decoupling action selection from value estimation:

```
y_DDQN = r + γ × Q(s', argmax Q(s', a'; θ), θ⁻)
                        a'
```

- **Online network** (θ) selects best action
- **Target network** (θ⁻) evaluates action value
- Reduces overestimation, improves stability under high load

#### Network Architecture

```
Input Layer:  State vector s (queue lengths + server occupancy)
      ↓
Hidden Layers: Fully connected with ReLU activation
      ↓
Output Layer:  Q-values for all valid actions
```

Optional: Batch normalization and dropout for improved generalization.

### 4. Evaluation Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Average Waiting Time** | `W̄ = (1/N) Σ Wᵢ` | Mean time patients spend waiting |
| **Weighted Waiting Time** | `W̄_weighted = (1/N) Σ wᵢWᵢ` | Acuity-adjusted waiting time |
| **System Utilization** | `ρ = λ/(μc)` | Fraction of time servers are busy |
| **Patient Loss Rate** | `Loss % = (lost/arrivals) × 100` | Percentage of patients turned away |

**Baseline Comparisons**:
- First-Come, First-Served (FCFS)
- Fixed-Priority Rules


---

*Last Updated: February 2026*
