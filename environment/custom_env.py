import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TrafficSignalEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.directions = ["N", "S", "E", "W"]
        self.arrival_prob = 0.4
        self.queues = None
        self.current_step = 0

        self.current_phase = 0
        self.time_in_phase = 0

        self.phase_directions = {
            0: ["N", "S"],
            1: ["E", "W"],
            2: ["N", "S"],
            3: ["E", "W"],
            4: [],
        }

        self.clearance_range = {
            0: (1, 2),
            1: (1, 2),
            2: (0, 1),
            3: (0, 1),
            4: (0, 0),
        }

        self.wait_weight = 0.1
        self.switch_penalty = 20.0
        self.all_red_penalty = 25.0
        self.clear_bonus = 2.0

        self.gridlock_threshold = 20
        self.max_steps = 500

    def reset(self, seed=None, options=None, randomize_start=False):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_phase = 0
        self.time_in_phase = 0

        if randomize_start:
            self.queues = {
                d: [self.current_step] * int(self.np_random.integers(0, 9))
                for d in self.directions
            }
        else:
            self.queues = {d: [] for d in self.directions}

        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        self._generate_arrivals()

        prev_action = self.current_phase
        prev_queue_lengths = {d: len(self.queues[d]) for d in self.directions}

        self._update_phase(action)
        self._clear_vehicles(action)

        reward = self._calculate_reward(action, prev_action, prev_queue_lengths)

        observation = self._get_observation()
        terminated = self._check_gridlock()
        truncated = self.current_step >= self.max_steps

        info = {"gridlock": terminated}

        return observation, reward, terminated, truncated, info

    def _generate_arrivals(self):
        for d in self.directions:
            if self.np_random.random() < self.arrival_prob:
                self.queues[d].append(self.current_step)

    def _update_phase(self, action):
        if action == self.current_phase:
            self.time_in_phase += 1
        else:
            self.current_phase = action
            self.time_in_phase = 0

    def _clear_vehicles(self, action):
        served_directions = self.phase_directions[action]
        low, high = self.clearance_range[action]

        for d in served_directions:
            clear_count = self.np_random.integers(low, high + 1)
            self.queues[d] = self.queues[d][clear_count:]

    def _average_wait(self, direction):
        queue = self.queues[direction]
        if not queue:
            return 0.0
        wait_times = [self.current_step - arrival_step for arrival_step in queue]
        return float(np.mean(wait_times))

    def _calculate_reward(self, action, prev_action, prev_queue_lengths):
        total_queue = sum(len(self.queues[d]) for d in self.directions)
        total_wait = sum(self._average_wait(d) * len(self.queues[d]) for d in self.directions)

        reward = -total_queue - (self.wait_weight * total_wait)

        if action != prev_action:
            reward -= self.switch_penalty
            if action == 4:
                reward -= self.all_red_penalty

        for d in self.directions:
            if prev_queue_lengths[d] > 0 and len(self.queues[d]) == 0:
                reward += self.clear_bonus

        return reward

    def _check_gridlock(self):
        return any(len(self.queues[d]) > self.gridlock_threshold for d in self.directions)

    def _get_observation(self):
        queue_lengths = [len(self.queues[d]) for d in self.directions]
        avg_waits = [self._average_wait(d) for d in self.directions]
        phase_info = [float(self.current_phase), float(self.time_in_phase)]
        return np.array(queue_lengths + phase_info + avg_waits, dtype=np.float32)

    def render(self):
        pass

    def close(self):
        pass