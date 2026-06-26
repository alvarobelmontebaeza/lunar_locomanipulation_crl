import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from lunar_locomanipulation_crl.envs import ConstrainedRlEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

class RslRlCatVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL VecEnv wrapper for CaT environments.
    
    This wrapper adapts Constrained RL environments to be compatible with RSL-RL's VecEnv interface. This is done
    by not combining terminated and truncated flags into a single done signal, but instead returning them separately."""

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, cat_dones, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        # cat_dones is a float tensor in [0, 1]. Treat values >= 1.0 as True
        # and perform elementwise logical OR with the `truncated` boolean tensor.
        cat_dones_bool = cat_dones >= 1.0
        truncated_bool = truncated.bool()
        true_dones = torch.logical_or(cat_dones_bool, truncated_bool).to(dtype=torch.long)
        # move time out information and terminated to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
            extras["terminated"] = true_dones
        # return the step information
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, cat_dones, extras
