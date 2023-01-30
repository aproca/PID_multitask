import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from stable_baselines3.common.policies import ActorCriticPolicy
import gym
from typing import Callable, Dict, List, Optional, Type, Union

class ActorCriticNetwork(nn.Module):
    """
    Policy network for PPO models. Stores activations for easy extraction.
    """
    def __init__(self, obs_shape, actor_layer_sizes, critic_layer_sizes, action_num = 9):
        super(ActorCriticNetwork, self).__init__()
        self.obs_shape = obs_shape
        self.device = torch.device('cpu')
        self.fc_a = nn.ModuleList()
        self.fc_c = nn.ModuleList()
        actor_layer_sizes.insert(0, self.obs_shape)
        actor_layer_sizes.append(action_num)
        critic_layer_sizes.insert(0, self.obs_shape)
        critic_layer_sizes.append(1)
        for i in range(len(actor_layer_sizes)-1):
            self.fc_a.append(nn.Linear(actor_layer_sizes[i], actor_layer_sizes[i+1], bias = True))
        for i in range(len(critic_layer_sizes)-1):
            self.fc_c.append(nn.Linear(critic_layer_sizes[i], critic_layer_sizes[i+1], bias = True))
        self.activations = []
        self.latent_dim_pi = actor_layer_sizes[-1]
        self.latent_dim_vf = 1

    def forward(self, x):
        self.activations = []
        x_a = torch.tensor(x).to(self.device)
        x_c = torch.tensor(x).to(self.device)
        for l in self.fc_a:
            x_a = F.relu(l(x_a))
            self.activations.append(x_a.detach().cpu().numpy())
        for l in self.fc_c:
            x_c = F.relu(l(x_c))
        return x_a, x_c

    def forward_actor(self, x):
        self.activations = []
        x_a = torch.tensor(x).to(self.device)
        for l in self.fc_a:
            x_a = F.relu(l(x_a))
            self.activations.append(x_a.detach().cpu().numpy())
        return x_a

    def forward_critic(self, x):
        x_c = torch.tensor(x).to(self.device)
        for l in self.fc_c:
            x_c = F.relu(l(x_c))
        return x_c


class NetActorCriticPolicy(ActorCriticPolicy):
    """
    Modified from stable baselines 3 source code. 
    Original: https://github.com/DLR-RM/stable-baselines3/blob/503425932f5dc59880f854c4f0db3255a3aa8c1e/stable_baselines3/common/policies.py#L344
    StablesBaselines-compatible policy object for A2C, PPO.
    """
    def __init__(self, 
        observation_space: gym.spaces.Space, 
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn : Type[nn.Module] = nn.ReLU, 
        *args,
        **kwargs,
        ):

        super(NetActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ActorCriticNetwork(self.features_dim, copy.deepcopy(self.net_arch['pi']), copy.deepcopy(self.net_arch['vf']), self.action_space.n)
        self.device1 = torch.device('cpu') 
        self.mlp_extractor.to(self.device1)