import numpy as np
from itertools import permutations
from typing import Optional


class EDTriageEnv:
    def __init__(self,
                 num_priority_classes: int = 3,
                 num_servers: int = 3,
                 arrival_rates: Optional[list[float]] = None,
                 service_rates: Optional[list[float]] = None,
                 acuity_weights: Optional[list[float]] = None,
                 max_patients: int = 30,
                 max_steps: int = 500):

        self.K = num_priority_classes
        self.c = num_servers
        self.max_patients = max_patients
        self.max_steps = max_steps

        self.arrival_rates = arrival_rates
        self.service_rates = service_rates
        self.acuity_weights = acuity_weights

        assert len(self.arrival_rates) == self.K
        assert len(self.service_rates) == self.K
        assert len(self.acuity_weights) == self.K

        self.priority_orderings = list(permutations(range(self.K)))
        self.num_actions = len(self.priority_orderings)
        self.state_dim = self.K * 3 + self.c

        self.queues: list[list[float]] = [[] for _ in range(self.K)]
        self.server_occupancy = np.zeros(self.c, dtype=np.int32)
        self.server_serving_class = np.full(self.c, -1, dtype=np.int32)
        self.step_count = 0
        self.current_time = 0.0
        self.served_records: list[tuple[int, float]] = []
        self.lost_patients = 0
        self._step_assignments: list[tuple[int, float]] = []
        self._wait_norm = 100.0
        self._qlen_norm = 20.0

    def reset(self) -> np.ndarray:
        self.queues = [[] for _ in range(self.K)]
        self.server_occupancy = np.zeros(self.c, dtype=np.int32)
        self.server_serving_class = np.full(self.c, -1, dtype=np.int32)
        self.step_count = 0
        self.current_time = 0.0
        self.served_records = []
        self.lost_patients = 0
        self._step_assignments = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        features = []
        for i in range(self.K):
            q_len = len(self.queues[i])
            if q_len == 0:
                max_wait = 0.0
                mean_wait = 0.0
            else:
                waits = [self.current_time - t for t in self.queues[i]]
                max_wait = float(max(waits))
                mean_wait = float(np.mean(waits))

            features.extend([
                q_len / self._qlen_norm,
                max_wait / self._wait_norm,
                mean_wait / self._wait_norm,
            ])

        features.extend(self.server_occupancy.astype(np.float32).tolist())
        return np.array(features, dtype=np.float32)

    def _total_patients(self) -> int:
        return sum(len(q) for q in self.queues) + int(np.sum(self.server_occupancy))

    def _compute_reward(self) -> float:
        reward = 0.0
        for cls, wait in self._step_assignments:
            reward -= self.acuity_weights[cls] * wait

        if len(self._step_assignments) == 0:
            for i in range(self.K):
                if len(self.queues[i]) > 0:
                    avg_wait = np.mean([self.current_time - t for t in self.queues[i]])
                    reward -= 0.5 * self.acuity_weights[i] * avg_wait

        reward -= 5.0 * self.lost_patients

        num_idle = np.sum(self.server_occupancy == 0)
        reward -= 0.1 * num_idle

        return reward

    def _simulate_arrivals(self) -> None:
        for i in range(self.K):
            num_arrivals = np.random.poisson(self.arrival_rates[i])
            for _ in range(num_arrivals):
                if self._total_patients() < self.max_patients:
                    self.queues[i].append(self.current_time)
                else:
                    self.lost_patients += 1

    def _simulate_service_completions(self) -> None:
        for j in range(self.c):
            if self.server_occupancy[j] == 1:
                serving_class = self.server_serving_class[j]
                if np.random.rand() < self.service_rates[serving_class]:
                    self.server_occupancy[j] = 0
                    self.server_serving_class[j] = -1

    def _assign_patient_to_server(self, priority_class: int) -> bool:
        if len(self.queues[priority_class]) == 0:
            return False

        for j in range(self.c):
            if self.server_occupancy[j] == 0:
                self.server_occupancy[j] = 1
                self.server_serving_class[j] = priority_class
                arrival_time = self.queues[priority_class].pop(0)
                wait_time = self.current_time - arrival_time
                self.served_records.append((priority_class, wait_time))
                self._step_assignments.append((priority_class, wait_time))
                return True

        return False

    def _fill_servers_by_ordering(self, ordering: tuple) -> None:
        while np.any(self.server_occupancy == 0):
            assigned_any = False
            for cls in ordering:
                if len(self.queues[cls]) > 0 and np.any(self.server_occupancy == 0):
                    if self._assign_patient_to_server(cls):
                        assigned_any = True
            if not assigned_any:
                break

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.current_time += 1.0
        self.step_count += 1
        self._step_assignments = []

        self._simulate_arrivals()
        self._simulate_service_completions()

        action = int(action) % self.num_actions
        ordering = self.priority_orderings[action]
        self._fill_servers_by_ordering(ordering)

        reward = self._compute_reward()
        state = self._get_state()
        done = (self.step_count >= self.max_steps or
                self._total_patients() >= self.max_patients)

        info = {
            "queue_lengths": [len(q) for q in self.queues],
            "server_occupancy": self.server_occupancy.copy(),
            "total_patients_in_system": self._total_patients(),
            "utilization": float(np.sum(self.server_occupancy)) / self.c,
            "lost_patients": self.lost_patients,
        }

        return state, reward, done, info

    def get_metrics(self) -> dict:
        if len(self.served_records) == 0:
            avg_wait = 0.0
            weighted_wait = 0.0
            per_class_avg_wait = {i: 0.0 for i in range(self.K)}
        else:
            avg_wait = float(np.mean([w for _, w in self.served_records]))
            weighted_wait = float(
                sum(self.acuity_weights[cls] * w for cls, w in self.served_records) /
                sum(self.acuity_weights[cls] for cls, _ in self.served_records)
            )

            per_class_avg_wait = {}
            for i in range(self.K):
                class_waits = [w for cls, w in self.served_records if cls == i]
                per_class_avg_wait[i] = float(np.mean(class_waits)) if class_waits else 0.0

        total_lambda = sum(self.arrival_rates)
        avg_mu = sum(self.service_rates) / self.K
        utilization = total_lambda / (avg_mu * self.c)

        return {
            "average_waiting_time": avg_wait,
            "weighted_waiting_time": weighted_wait,
            "per_class_avg_wait": per_class_avg_wait,
            "system_utilization": utilization,
            "total_served": len(self.served_records),
            "lost_patients": self.lost_patients,
            "patients_remaining_in_queue": sum(len(q) for q in self.queues),
            "patients_in_service": int(np.sum(self.server_occupancy)),
        }
