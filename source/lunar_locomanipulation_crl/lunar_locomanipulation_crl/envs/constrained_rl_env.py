# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import gymnasium as gym
import inspect
import math
import numpy as np
import torch
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import MISSING
from typing import Any, ClassVar

import isaacsim.core.utils.torch as torch_utils
import omni.kit.app
import omni.log
import omni.physx
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.version import get_version

from isaaclab.managers import EventManager, CurriculumManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import attach_stage_to_usd_context, use_stage
from isaaclab.utils.noise import NoiseModel
from isaaclab.utils.timer import Timer

from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from .constrained_rl_env_cfg import ConstrainedRlEnvCfg
from isaaclab.envs.ui import ViewportCameraController

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

class ConstrainedRlEnv(DirectRLEnv):
    cfg: ConstrainedRlEnvCfg

    def __init__(self, cfg: ConstrainedRlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Load ConstraintManager here if present in cfg
        if self.cfg.constraint_manager is not None:
            self.constraint_manager = self.cfg.constraint_manager
            print("[INFO]: Constraint Manager: ", self.constraint_manager)
        
        # Load curriculum manager if present in cfg
        if self.cfg.curriculum is not None:
            self.curriculum_manager = CurriculumManager(self.cfg.curriculum, env=self)
            print("[INFO]: Curriculum Manager: ", self.curriculum_manager)
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        # Compute CaT constraints
        if hasattr(self.cfg, "constraints") and self.cfg.constraints is not None:
            cstr_prob = torch.clip(self.constraint_manager.compute(), min=0.0, max=1.0)
            # -- Constrained reward computation
            raw_rewards = self._get_rewards()
            self.reward_buf = torch.clip(
                raw_rewards * (1.0 - cstr_prob),
                min=0.0,
                max=None,
            )
            dones = cstr_prob.clone()
        else:
            self.reward_buf = self._get_rewards()
            dones = torch.zeros(self.num_envs, device=self.device)        

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            dones[reset_env_ids] = 1.0 # Reset env if done by timeout or termination, even if no hard constrained is violated
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        # Instead of only termination resets, we return a combination of termination and constraint-based resets
        return self.obs_buf, self.reward_buf, dones, self.reset_time_outs, self.extras
    
    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        if hasattr(self.cfg, "curriculum") and self.cfg.curriculum is not None:
            self.curriculum_manager.compute(env_ids)

        self.scene.reset(env_ids)

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # reset noise models
        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0    