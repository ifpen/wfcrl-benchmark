from collections import OrderedDict
import itertools
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter

class VectorExtractor(nn.Module):
    """dict-vector mapping for vectors OF DIMENSION 1"""
    def __init__(self, space, filter_out=["pitch", "torque"]):
        super().__init__()
        lows = []
        highs = []
        size = 0
        indice = 0
        self.keys = OrderedDict()
        for item, item_space in space.items():
            if item not in filter_out:
                lows.append(item_space.low)
                highs.append(item_space.high)
                self.keys[item] = [indice, indice + item_space.shape[0]]
                size += item_space.shape[0]
            indice += item_space.shape[0]

        self.space = gym.spaces.Box(
            low=np.concatenate(lows),
            high=np.concatenate(highs),
            shape=(size,)
        )

    def forward(self, dic):
        return np.concatenate([np.atleast_2d(dic[key]) for key in self.keys],1).squeeze()
    
    def make_dict(self, vector):
        return {
            key: vector[idx1:idx2]
            for key, (idx1, idx2) in self.keys.items()
        }

class DfacSPaceExtractor(nn.Module):
    def __init__(self, local_observation_space, global_observation_space):
        super().__init__()
        yaw_space = local_observation_space["yaw"]
        wind_space = global_observation_space["freewind_measurements"]
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([yaw_space.low, wind_space.low]),
            high=np.concatenate([yaw_space.high, wind_space.high]),
            shape=(yaw_space.shape[0] + wind_space.shape[0],)
        )

    def forward(self, local_obs, global_obs):
        return np.concatenate([[local_obs["yaw"]], global_obs["freewind_measurements"].flatten()],0)

class FourierExtractor(nn.Module):
    """
    Feature extracting fourier features from an observation.
    Used for continous observation spaces.
    :param observation_space:
    :param order of torch. Fourier extracts:
    :param learnable: True if generated matrix is also to be trained (NOT witorch.hyper network)
    :param max_dim: maximum dimension of torch. generated matrix. random generation will be used
        if max_dim is inferior to (order+1)^(dim of observation space)
    :param seed: seed used to generate random matrix
    :param hyper: True if a hyper network will be used to generate torch. matrix
    :param hyper_train: True if torch. hyper network is to be trained
    """

    def __init__(
            self, 
            observation_space: gym.Space, 
            order: int, learnable = False,
            max_dim: int = None, 
            seed: int = None, 
            hyper: bool = False, 
            hyper_train: bool = False,
            hyper_network_class: torch.nn.Module = None, 
            fourier_hyper_arch: Iterable[int] = None
        ):
        assert (isinstance(observation_space, gym.spaces.Box), 
            "Use Fourier extractor with continuous observation spaces")
        super().__init__()
        
        self._observation_space = observation_space
        self._learnable = learnable
        self.observation_dim = observation_space.shape[0]
        features_dim = (order + 1) ** self.observation_dim
        random = False
        if (max_dim is not None) and (features_dim > max_dim):
            features_dim = max_dim
            random = True
        self._features_dim = features_dim

        self.order = order
        self.rng = None
        self.ub = observation_space.high
        self.lb = observation_space.low
        self.combin = None
        self.hyper = hyper
        self.hyper_train = hyper_train

        if not hyper:
            if random:
                self.rng = np.random.default_rng(seed)
                combin = self.rng.integers(low=0, high=self.order+1, size=(max_dim, self.observation_dim))
            else:
                combin = list((itertools.product(*[np.arange(self.order+1) for _ in range(self.observation_dim)])))
            
            self.combin = torch.as_tensor(combin).T
            self.combin = Parameter(data=self.combin.to(float), requires_grad=self._learnable)
        elif not hyper_train:
            self.fmat_network = hyper_network_class(
                input_dim = observation_space.shape[0] - 1,
                order = order,
                output_dims= (max_dim, observation_space.shape[0]),
                architecture=fourier_hyper_arch if fourier_hyper_arch else [],
                input_lb=observation_space.low[1:],
                input_ub=observation_space.high[1:], 
            )
            self.fmat_network.train(False)
            for p in self.fmat_network.parameters():
                p.requires_grad = False

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if (self.hyper and not self.hyper_train):
            self.combin = self.fmat_network(observations[:, 1:])
        clipped_observations = torch.clip(observations, torch.as_tensor(self.lb[None,:]), torch.as_tensor(self.ub[None,:]))
        observations = (clipped_observations - self.lb[None,:])/(self.ub[None,:] - self.lb[None,:])
        combin = self.combin.to(observations.dtype).to(observations.device)
        x = (
            torch.bmm(observations[:, None, :], combin.transpose(1,2))
            if self.hyper
            else observations@combin
        )
        features = torch.cos(np.pi * x.squeeze())
        return features

    def set_combin_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        self.combin = matrix