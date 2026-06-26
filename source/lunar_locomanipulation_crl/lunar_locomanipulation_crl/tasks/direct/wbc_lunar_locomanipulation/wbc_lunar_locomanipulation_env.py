# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import gym

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, RayCaster
from lunar_locomanipulation_crl.envs import ConstrainedRlEnv
from lunar_locomanipulation_crl.modules import ConstraintManager
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import compute_pose_error, subtract_frame_transforms, combine_frame_transforms, quat_error_magnitude, quat_from_euler_xyz, quat_unique, euler_xyz_from_quat
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

from .wbc_lunar_locomanipulation_env_cfg import WBCLunarLocomanipulationEnvCfg


class WBCLunarLocomanipulationEnv(ConstrainedRlEnv):
    cfg: WBCLunarLocomanipulationEnvCfg

    def __init__(self, cfg: WBCLunarLocomanipulationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        
        # Get gravity vector for convenience
        self._gravity_w = torch.tensor([0.0, 0.0, self.cfg.sim.gravity[2]], device=self.device)

        # Up direction (unit)
        self._up_w = torch.nn.functional.normalize(-self._gravity_w, dim=0)

        # Build a fixed orthonormal basis {u, v, up} for projecting to the ground plane
        _tmp = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        if torch.allclose(torch.abs(self._up_w), _tmp, atol=1e-3):
            _tmp = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self._u_w = torch.nn.functional.normalize(torch.cross(self._up_w, _tmp), dim=0)  # (3,)
        self._v_w = torch.nn.functional.normalize(torch.cross(self._up_w, self._u_w), dim=0)   # (3,)   

        # Initialize stability metrics value
        self._last_giim = torch.zeros(self.num_envs, device=self.device)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # Target end-effector pose (position + orientation) in quaternion WXYZ format
        self._target_ee_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self._target_ee_pose_w = torch.zeros(self.num_envs, 7, device=self.device)

        # Target sampling ranges
        self._target_pos_x_range = torch.tensor(self.cfg.target_pos_x_range, device=self.device)
        self._target_pos_y_range = torch.tensor(self.cfg.target_pos_y_range, device=self.device)
        self._target_pos_z_range = torch.tensor(self.cfg.target_pos_z_range, device=self.device)

        # Action related params
        self._leg_action_scale = torch.tensor(self.cfg.leg_action_scale, device=self.device)
        self._arm_action_scale = torch.tensor(self.cfg.arm_action_scale, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "pose_tracking",
                "stability",
                "leg_low_power",
                "arm_low_power",
                "leg_action_rate",
                "arm_action_rate",
            ]
        }

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies(".*base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies([".*base", ".*Head.*", ".*hip", ".*thigh", ".*calf", ".*wx250s.*"])
        self._ee_id, _ = self._robot.find_bodies(".*wx250s_ee_gripper_link")

        self._leg_joint_ids, _ = self._robot.find_joints([".*hip_joint", ".*thigh_joint", ".*calf_joint"])
        self._arm_joint_ids, _ = self._robot.find_joints([".*waist", ".*shoulder", ".*elbow", ".*forearm_roll", ".*wrist_angle", ".*wrist_rotate"])

        # Initialize constraint manager
        self.constraint_manager = ConstraintManager(self.cfg.constraints, self)

        # Add handle for debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        # Add robot and sensors to scene
        # Instanciate assets
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if hasattr(self.cfg, "height_scanner") and self.cfg.height_scanner is not None:
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        # Add terrain to scene
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Process the raw policy actions before applying them to the robot.
        """
        self._actions = actions.clone()
        # Process leg actions
        self._leg_actions = self._actions[:, :12] * self._leg_action_scale + self._robot.data.default_joint_pos[:, self._leg_joint_ids]
        # Process arm actions
        self._arm_actions = self._actions[:, 12:] * self._arm_action_scale + self._robot.data.default_joint_pos[:, self._arm_joint_ids]
        # Combine processed actions
        self._processed_actions = torch.cat([self._leg_actions, self._arm_actions], dim=1)


        # Update command in base frame at every step.
        # This recomputes the target pose in base frame based on the current robot base pose.
        self._update_command()


    def _apply_action(self):
        self._robot.set_joint_position_target(self._leg_actions, joint_ids=self._leg_joint_ids)
        self._robot.set_joint_position_target(self._arm_actions, joint_ids=self._arm_joint_ids)
    
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if self.cfg.height_scanner is not None:
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)

        # Extract observation ground truth data
        projected_gravity = self._robot.data.projected_gravity_b
        base_lin_vel = self._robot.data.root_lin_vel_b
        base_ang_vel = self._robot.data.root_ang_vel_b
        leg_joint_pos = self._robot.data.joint_pos[:, self._leg_joint_ids]
        arm_joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        leg_joint_vel = self._robot.data.joint_vel[:, self._leg_joint_ids]
        arm_joint_vel = self._robot.data.joint_vel[:, self._arm_joint_ids]
        feet_contacts = self._get_feet_contact_states()
        actions = self._actions
        target_pose = self._target_ee_pose

        # TODO: Nosify observations if needed

        # Construct observation tensor
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    projected_gravity, # 3
                    base_lin_vel, # 3
                    base_ang_vel, # 3
                    leg_joint_pos, # 12
                    arm_joint_pos, # 6
                    leg_joint_vel, # 12
                    arm_joint_vel, # 6
                    feet_contacts, # 4
                    actions, # 18
                    target_pose, # 7
                    height_data, # terrain height data 187 (optional)
                    # Total observation size: 74 (if without terrain height data), 261 (if with terrain height data)
                )
                if tensor is not None
            ],
            dim=-1,
        )

        # TODO: Think about asymmetric actor-critic observations if needed

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Task rewards
        pose_tracking = self._compute_pose_tracking_reward()

        # Stability reward
        stability_rew, giim = self._compute_stability_reward()
        self._last_giim = giim

        # Low power consumption rewards
        leg_low_power = self._compute_low_power_reward(self._leg_joint_ids, self.cfg.max_leg_power)
        arm_low_power = self._compute_low_power_reward(self._arm_joint_ids, self.cfg.max_arm_power)

        # Action rate regularization
        leg_action_rate = torch.abs(self._actions[:, :12] - self._previous_actions[:, :12])
        arm_action_rate = torch.abs(self._actions[:, 12:] - self._previous_actions[:, 12:])
        leg_action_rate_rew = torch.exp(-leg_action_rate / 0.25).mean(dim=1)
        arm_action_rate_rew = torch.exp(-arm_action_rate / 0.25).mean(dim=1)

        # Compute weighted rewards
        rewards = {
            "pose_tracking": pose_tracking * self.cfg.pose_tracking_reward_scale * self.step_dt,
            "stability": stability_rew * self.cfg.stability_reward_scale * self.step_dt,
            "leg_low_power": leg_low_power * self.cfg.leg_low_power_reward_scale * self.step_dt,
            "arm_low_power": arm_low_power * self.cfg.arm_low_power_reward_scale * self.step_dt,
            "leg_action_rate": leg_action_rate_rew * self.cfg.leg_action_rate_reward_scale * self.step_dt,
            "arm_action_rate": arm_action_rate_rew * self.cfg.arm_action_rate_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # -- Logging before reset
        # Log rewards
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        # Log constraints
        if hasattr(self.cfg, "constraints"):
            info = self.constraint_manager.reset(env_ids)
            self.extras["log"].update(info)
        # Log termination reasons
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
        # Log curriculum parameters
        if hasattr(self.cfg, "curriculum"):
            curriculum_info = self.curriculum_manager.reset(env_ids)
            self.extras["log"].update(curriculum_info)
        # Log metrics
        pos_error, rot_error = compute_pose_error(
            self._robot.data.body_com_pose_w[env_ids, self._ee_id, :3].view(-1, 3),
            self._robot.data.body_com_pose_w[env_ids, self._ee_id, 3:7].view(-1, 4),
            self._target_ee_pose_w[env_ids, :3],
            self._target_ee_pose_w[env_ids, 3:],
        )
        extras = dict()
        extras["Metrics/final_ee_pos_error"] = torch.norm(pos_error, dim=-1)
        extras["Metrics/final_ee_rot_error"] = torch.norm(rot_error, dim=-1)
        self.extras["log"].update(extras)


        # -- Start reset procedure
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        # if len(env_ids) == self.num_envs:
        #     # Spread out the resets to avoid spikes in training when many environments reset at a similar time
        #     self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))        

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._last_giim[env_ids] = 0.0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids, :3]
        # Add randomization to the robot initial XY position
        # default_root_state[:, :2] += torch.rand_like(default_root_state[:, :2], device=self.device) * 2.5
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sample new commands
        self._target_ee_pose[env_ids] = torch.zeros_like(self._target_ee_pose[env_ids])
        self._target_ee_pose_w[env_ids] = torch.zeros_like(self._target_ee_pose_w[env_ids])
        self._resample_ee_command(env_ids)   

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_marker") and not hasattr(self, "ee_marker"):
                frame_marker_cfg = FRAME_MARKER_CFG.copy()
                frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
                self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
            # set their visibility to true
            self.ee_marker.set_visibility(True)
            self.goal_marker.set_visibility(True)
        else:
            if hasattr(self, "goal_marker"):
                self.goal_marker.set_visibility(False)
            if hasattr(self, "ee_marker"):
                self.ee_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        curr_ee_pos_w = self._robot.data.body_com_pose_w[:, self._ee_id, :3].view(-1, 3)
        curr_ee_quat_w = self._robot.data.body_com_pose_w[:, self._ee_id, 3:7].view(-1, 4)
        self.ee_marker.visualize(curr_ee_pos_w, curr_ee_quat_w)
        self.goal_marker.visualize(self._target_ee_pose_w[:, :3], self._target_ee_pose_w[:, 3:7])

    """ 
    Utility functions 
    -----------------------------
    The methods below are utility functions to compute different components of the environment.
    -----------------------------
    """

    # -- Command related functions
    def _resample_ee_command(self, env_ids: torch.Tensor | None = None, make_quat_unique: bool = False):
        """Resample a new end-effector target pose command for all environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Tensor for sampling
        r = torch.empty(len(env_ids), device=self.device)    
        # Sample target position
        self._target_ee_pose[env_ids, 0] = r.uniform_(self._target_pos_x_range[0], self._target_pos_x_range[1])
        self._target_ee_pose[env_ids, 1] = r.uniform_(self._target_pos_y_range[0], self._target_pos_y_range[1])
        self._target_ee_pose[env_ids, 2] = r.uniform_(self._target_pos_z_range[0], self._target_pos_z_range[1])
        # Sample target orientation in Euler angles and convert to quaternions
        euler_angles = torch.zeros_like(self._target_ee_pose[env_ids, :3])
        euler_angles[:, 0] = r.uniform_(*self.cfg.target_rot_roll_range)
        euler_angles[:, 1] = r.uniform_(*self.cfg.target_rot_pitch_range)
        euler_angles[:, 2] = r.uniform_(*self.cfg.target_rot_yaw_range)
        target_quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        if make_quat_unique:
            self._target_ee_pose[env_ids, 3:] = quat_unique(target_quat)
        else:
            self._target_ee_pose[env_ids, 3:] = target_quat

        # Base offset pose - follow offset computation as in https://arxiv.org/pdf/2210.10044
        pos_offset = self._robot.data.root_state_w[env_ids, :3].clone()
        # pos_offset[:, 2] = 0.5  # fixed height offset
        rot_offset = self._robot.data.root_pose_w[env_ids, 3:7].clone()
        euler_x, euler_y, euler_z = euler_xyz_from_quat(rot_offset)
        rot_offset = quat_from_euler_xyz(torch.zeros_like(euler_x), torch.zeros_like(euler_y), euler_z) #roll/pitch independent offset

        # Transform target pose to world frame
        self._target_ee_pose_w[env_ids, :3], self._target_ee_pose_w[env_ids, 3:] = combine_frame_transforms(
            pos_offset,
            rot_offset,
            self._target_ee_pose[env_ids, :3],
            self._target_ee_pose[env_ids, 3:],
        )
    
    def _update_command(self):
        # Update ee command in base frame
        self._target_ee_pose[:, :3], self._target_ee_pose[:, 3:] = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._target_ee_pose_w[:, :3],
            self._target_ee_pose_w[:, 3:]
        )

    # -- Constraint functions
    def _get_feet_contact_states(self) -> torch.Tensor:
        net_contact_forces = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids]
        feet_force = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0]
        return (feet_force > 1.0).float()

    # -- Reward functions
    def _get_sigmas(self, epsilon_pos, epsilon_orn):
        """
        Function to get sigma values for position and orientation based on errors for locomanipulation.
        Tailored for base distances 0-1 m and arm reach ~0.5 m.
        
        Parameters:
        - epsilon_pos (torch.Tensor): Tensor of position errors (meters), shape (N,), range 0-1 m.
        - epsilon_orn (torch.Tensor): Tensor of orientation errors (radians), shape (N,), range 0-1.57 rad.
        
        Returns:
        - sigma_pos (torch.Tensor): Tensor of sigma values for position (meters), shape (N,).
        - sigma_orn (torch.Tensor): Tensor of sigma values for orientation (radians), shape (N,).
        """
        # Define thresholds and corresponding sigma values for position (meters)
        pos_thresholds = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
        sigma_pos_values = [0.5, 0.4, 0.25, 0.15, 0.05, 0.025]

        # Define thresholds and corresponding sigma values for orientation (radians)
        orn_thresholds = [1.57, 1.0, 0.5, 0.25, 0.1]
        sigma_orn_values = [0.5, 0.4, 0.25, 0.15, 0.05]

        # Initialize tensors with the largest sigma (for errors >= max threshold)
        sigma_pos = torch.full_like(epsilon_pos, sigma_pos_values[0])
        sigma_orn = torch.full_like(epsilon_orn, sigma_orn_values[0])

        # Assign sigma values for position errors based on thresholds
        for threshold, sigma in zip(pos_thresholds, sigma_pos_values):
            mask = epsilon_pos < threshold
            sigma_pos[mask] = sigma

        # Assign sigma values for orientation errors based on thresholds
        for threshold, sigma in zip(orn_thresholds, sigma_orn_values):
            mask = epsilon_orn < threshold
            sigma_orn[mask] = sigma
        
        return sigma_pos, sigma_orn
    
    def _compute_pose_tracking_reward(self) -> torch.Tensor:
        # Extract asset data
        data = self._robot.data
        # Obtain desired and current poses
        des_pos_b, des_quat_b = self._target_ee_pose[:, :3], self._target_ee_pose[:, 3:]
        curr_pos_w = data.body_com_pose_w[:, self._ee_id, :3].view(-1, 3)
        curr_quat_w = data.body_com_pose_w[:, self._ee_id, 3:7].view(-1, 4)
        curr_base_pos_w, curr_base_quat_w = data.root_pose_w[:, :3], data.root_pose_w[:, 3:7]
        # Convert to base frame
        curr_pos_b, curr_quat_b = subtract_frame_transforms(
            curr_base_pos_w,
            curr_base_quat_w,
            curr_pos_w,
            curr_quat_w,
        )

        # Compute position and orientation errors
        pos_error = torch.sum(torch.square(des_pos_b - curr_pos_b), dim=1)
        quat_error = quat_error_magnitude(des_quat_b, curr_quat_b)

        # TODO: Implement adaptive sigmas
        sigma_pos, sigma_rot = self._get_sigmas(pos_error, quat_error)

        pos_rew = torch.exp(-pos_error / sigma_pos)
        rot_rew = torch.exp(-quat_error / sigma_rot)

        # Manipulation reward
        pose_rew = pos_rew * rot_rew

        # Compute locomotion reward to encourage moving towards the desired position before using manipulation
        radius = self.cfg.pose_tracking_radius
        base_dist = torch.sum(torch.square(des_pos_b[:, :2]), dim=1)
        rew_base_dist = torch.exp(-(base_dist - radius) / 0.5).clamp(0.0, 1.0)

        # Compute gating function to encourage switching to manipulation reward when close enough
        gating_k = 10.0
        mu = l = radius * 2.0
        gate = torch.sigmoid(gating_k * (base_dist - mu)/l)
        gate = torch.clamp(gate, 0.0, 1.0)

        # Combined locomanipulation reward
        return pose_rew * rew_base_dist
    
    def _compute_low_power_reward(self, joint_ids, max_power) -> torch.Tensor:
        joint_torques = self._robot.data.applied_torque[:, joint_ids]
        joint_velocities = self._robot.data.joint_vel[:, joint_ids]
        power = torch.sum(torch.abs(joint_torques * joint_velocities), dim=1)

        # Exponential decay reward. Scale total power by max power for normalization
        scale = max_power * 0.5
        return torch.exp(-power / scale)
    
    def _compute_stability_reward(self):
        """
        Returns:
        stability_rew: (N,)  mean over faces of [acos(n·agi/|n||agi|) - pi/2]  (0 if <3 contacts; we set -0.1 below)
        giim_min:      (N,)  diagnostic min over faces of the same term
        Assumes:
        - Candidate ground contacts are feet: self._feet_ids (F,)
        - Contact states per foot available via self._get_feet_contact_states() -> (N, F) boolean/0-1
        - World foot COM positions: self._robot.data.body_com_pose_w[:, self._feet_ids, :3] -> (N, F, 3)
        - Root world pos/vel: self._robot.data.root_pos_w (N,3), root_lin_vel_w (N,3)
        """
        device = self.device
        eps = 1e-6

        # ----------------------------
        # 1) GIA (world, batched)
        # ----------------------------
        N = self.num_envs

        com_acc = self._robot.data.body_com_acc_w[:, self._base_id, :3].view(-1, 3)          # (N,3)
        a_gi = torch.nan_to_num(self._gravity_w - com_acc)                               # (N,3)

        # ----------------------------
        # 2) Gather contact candidates
        # ----------------------------
        feet_pos_w = self._robot.data.body_com_pose_w[:, self._feet_ids, :3]  # (N,F,3)
        contact_mask = self._get_feet_contact_states().bool()                 # (N,F)

        # Center at CoM/root (proxy for CoM is fine for RL)
        com_w = self._robot.data.root_pos_w                                   # (N,3)
        feet_from_com = feet_pos_w - com_w.unsqueeze(1)                                 # (N,F,3)

        # ----------------------------
        # 3) Order contacts by azimuth in ground plane (per-env), fully batched
        # ----------------------------
        # Project to plane basis
        # x = feet_from_com·u, y = feet_from_com·v
        x = (feet_from_com * self._u_w.view(1,1,3)).sum(-1)                             # (N,F)
        y = (feet_from_com * self._v_w.view(1,1,3)).sum(-1)                             # (N,F)
        angles = torch.atan2(y, x)                                            # (N,F)

        # Put invalid/non-contact points at +inf so they sort to the end
        angles_masked = angles.masked_fill(~contact_mask, float('inf'))       # (N,F)

        # Sort by angle; valids come first in each row
        _, sort_idx = torch.sort(angles_masked, dim=1, stable=True)
        feet_from_com_sorted = torch.gather(feet_from_com, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))     # (N,F,3)
        mask_sorted = torch.gather(contact_mask, 1, sort_idx)                            # (N,F)

        # Number of valid contacts per env
        K = mask_sorted.sum(dim=1)                                             # (N,)

        # ----------------------------
        # 4) Build polygon faces (CoM, p_i, p_{i+1}) including wrap i=K-1 -> 0
        # ----------------------------
        # Sequential edges among consecutive valid points (positions 0..K-1)
        # Non-wrapping pairs (j, j+1) for j=0..F-2 where both are valid:
        mask_seq = mask_sorted[:, :-1] & mask_sorted[:, 1:]                   # (N, F-1)
        v1_seq = feet_from_com_sorted[:, :-1, :]                                        # (N, F-1, 3)
        v2_seq = feet_from_com_sorted[:, 1:,  :]                                        # (N, F-1, 3)

        # Wrap pair (last valid -> first valid). Build a single pair per env.
        # Indices: last = K-1 (clamped); first = 0
        last_idx = torch.clamp(K - 1, min=0)                                   # (N,)
        batch_idx = torch.arange(N, device=device)

        v1_wrap = feet_from_com_sorted[batch_idx, last_idx.clamp_max(feet_from_com_sorted.size(1)-1)]        # (N,3)
        v2_wrap = feet_from_com_sorted[batch_idx, torch.zeros_like(last_idx)]                        # (N,3)

        # Valid wrap edge only if at least 2 contacts (and first is valid, which it is if K>=1)
        mask_wrap = (K >= 2).unsqueeze(-1)                                     # (N,1) for broadcasting

        # Stack edges to shape (N, F, 3)
        v1 = torch.cat([v1_seq, v1_wrap.unsqueeze(1)], dim=1)                  # (N, F, 3)  (last column is wrap)
        v2 = torch.cat([v2_seq, v2_wrap.unsqueeze(1)], dim=1)                  # (N, F, 3)
        edge_mask = torch.cat([mask_seq, mask_wrap], dim=1)                    # (N, F)

        # ----------------------------
        # 5) Face normals and angles to GIA (all batched)
        # ----------------------------
        n = torch.cross(v1, v2, dim=-1)                                        # (N, F, 3)

        # Drop degenerate faces (zero normal)
        n_norm = n.norm(dim=-1)                                               # (N, F)
        degenerate_faces = n_norm < eps                                       # (N, F)

        # Orient normals so they point toward +up
        dir_dot = (n * self._up_w.view(1,1,3)).sum(-1)                      # (N,F)
        flip = (dir_dot < 0.0).unsqueeze(-1)                                   # (N,F,1)
        n = torch.where(flip, -n, n)                                          # (N,F,3)
        n_norm = n.norm(dim=-1).clamp_min(eps)                                # (N,F) 

        # Cosine of angle between normal and GIA
        a_norm = a_gi.norm(dim=-1, keepdim=True).clamp_min(eps)                # (N,1)
        cosang = ((n * a_gi.unsqueeze(1)).sum(-1) / (n_norm * a_norm))         # (N,F)
        cosang = cosang.clamp(-1.0, 1.0)
        theta = torch.acos(cosang)                                             # (N,F)

        # θ - π/2 per face
        face_val = theta - (math.pi * 0.5)
        valid_face_mask = edge_mask & ~degenerate_faces

        # Mask out invalid edges
        face_val = torch.where(valid_face_mask, face_val, torch.zeros_like(face_val))  # (N,F)

        # Sum over valid faces and divide by their count; set fallback if <3 contacts
        sum_faces = face_val.sum(dim=1)                                        # (N,)
        num_faces = valid_face_mask.sum(dim=1).to(face_val.dtype).clamp_min(1)                       # (N,)
        stability_mean = sum_faces / num_faces                                   # (N,)

        # Scale to [0,1], with 1 being most stable (all faces horizontal)
        stability_unit = (stability_mean + (math.pi * 0.5)) / math.pi  # (N,)
        stability_unit = stability_unit.clamp(0.0, 1.0)

        # Diagnostic: min over valid faces; use +inf for invalid edges to do masked min
        big = torch.full_like(face_val, float('inf'))
        face_for_min = torch.where(valid_face_mask, face_val, big)
        giim_min = face_for_min.min(dim=1).values                              # (N,)

        # Apply fallback when not enough contacts (<3)
        valid_poly = (K >= 3).to(stability_unit.dtype)                                          # (N,)
        stability_mean = stability_unit * valid_poly
        giim_min       = giim_min * valid_poly + (0.0) * (1 - valid_poly)

        return stability_mean, giim_min