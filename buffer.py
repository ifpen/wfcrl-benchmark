

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3.common.buffers import RolloutBuffer
import torch

class DelayedRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        records_shape: Tuple,
        delay: Union[Iterable, Callable],
        delayed_compute: Callable,
        device: Union[torch.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        temp_buffer_ratio: int = int(1e4)
    ):
        self._final_buffer = RolloutBuffer(
            buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs
        )
        super(DelayedRolloutBuffer, self).__init__(
            temp_buffer_ratio*buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs
        )
        super(DelayedRolloutBuffer, self).reset()

        if isinstance(delay, Iterable):
            self._get_agent_delays = lambda x: delay
        else:
            self._get_agent_delays = delay
        self._compute_fn = delayed_compute
        self._timestamp = -1
        self.buffer_size = temp_buffer_ratio * buffer_size
        self._last_processed = None
        self._records_to_process = False

        self.delay_passed = np.zeros((self.buffer_size, self.n_envs), dtype=np.bool8)
        self.records = np.zeros((self.buffer_size, self.n_envs) + records_shape, dtype=np.float32)
        self.timemarks = np.inf * np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
        self.delays = np.ones((self.buffer_size, self.n_envs), dtype=np.bool8) * np.inf


    @property
    def _delay(self):
        return np.max(self._get_agent_delays())

    def _roll_arrays(self):
        self.observations = np.roll(self.observations, -1, axis=0)
        self.actions = np.roll(self.actions, -1, axis=0)
        self.records = np.roll(self.records,  -1, axis=0)
        self.episode_starts = np.roll(self.episode_starts,  -1, axis=0)
        self.values = np.roll(self.values, -1, axis=0)
        self.log_probs = np.roll(self.log_probs,  -1, axis=0)
        self.timemarks = np.roll(self.timemarks,  -1, axis=0)
        self.delays = np.roll(self.delays,  -1, axis=0)
        self.delay_passed = np.roll(self.delay_passed,  -1, axis=0)
        self.rewards = np.roll(self.rewards,  -1, axis=0)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        record: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        delay: np.ndarray = None,
        timestamp: np.ndarray = None
    ) -> None:

        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        if delay is None:
            delay = np.max(self._get_agent_delays(*obs.squeeze()[-2:])) # obs[-2:] = dir, speed !
        if timestamp is None:
            timestamp = self._timestamp
            self._timestamp += 1
        else:
            assert timestamp > self._timestamp
            self._timestamp = timestamp

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.records[self.pos] = np.array(record).copy()
        self.timemarks[self.pos] = np.array([timestamp]).copy()
        self.delays[self.pos] = np.array([delay]).copy()
        self.delay_passed[self.pos] = np.array([False]).copy()
        
        if self.pos == self.buffer_size-1:
           self._roll_arrays()
        else:
            self.pos += 1

    def compute_rewards(self):
        # Check is some rewards need to be computed
        ready = (self.timemarks + self.delays) < self._timestamp
        ready = ready & (~self.delay_passed)
        indices = np.where(ready)[0]
        
        if len(indices) > 0:
            idx = indices[0]
            self.delay_passed[idx] = True
            prod2 = 0
            delays = self._get_agent_delays(*self.observations[idx].squeeze()[-2:])
            records1 = self.records[max(0, idx-1)]
            records2 = self.records[idx-1+delays.astype(int), :, np.arange(delays.size)].squeeze()
            reward = self._compute_fn((records1, records2))
            self._final_buffer.add(
                self.observations[idx].copy(),
                self.actions[idx].copy(),
                np.array(reward, dtype=np.float32),
                self.episode_starts[idx].copy(),
                torch.as_tensor(self.values[idx]),
                torch.as_tensor(self.log_probs[idx])
            )
            self.rewards[idx] = reward
            self._last_processed = self.timemarks[idx]
            self.full = self._final_buffer.full
            self._records_to_process = True
        return self._records_to_process
    
    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        # Check is there are some transitions ready to be processed
        if self._records_to_process:
            idx = int(np.where(self.timemarks.flatten() == self._last_processed)[0])
            last_values = torch.as_tensor(self.values[idx+1])
            dones = self.episode_starts[idx+1]
            # num_ready_to_process = np.sum(self.delay_passed)
            # self.buffer_size = num_ready_to_process
            # last_values = torch.as_tensor(self.values[num_ready_to_process])
            self._final_buffer.compute_returns_and_advantage(last_values, dones)
            self.advantages[idx] = self._final_buffer.advantages[0]
            self._records_to_process = False

    def reset(self) -> None:        
        self._final_buffer.reset()
        self.full = False
    
    def get(self, batch_size: Optional[int] = None):
        return self._final_buffer.get(batch_size)

    def save_buffer(self, path):
        data = {}
        for attr, obj in self.__dict__.items():
            if isinstance(obj, np.ndarray) and obj.shape[0] == self.buffer_size:
                obj = obj.squeeze()
                if obj.ndim > 2:
                    continue
                if obj.ndim > 1:
                    data.update({
                        f"{attr}_{j}": obj[:,j]
                        for j in range(obj.shape[1])
                    })
                else:
                    data.update({
                        attr: obj
                    })
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)