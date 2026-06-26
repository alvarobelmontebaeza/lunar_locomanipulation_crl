
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from lunar_locomanipulation_crl.envs import ConstrainedRlEnv


def joint_position(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = torch.abs(data.joint_pos[:, asset_cfg.joint_ids]) - limit
    return cstr


def joint_position_when_moving_forward(
    env: ConstrainedRlEnv,
    limit: float,
    velocity_deadzone: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = (
        torch.abs(data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids])
        - limit
    )
    cstr *= (
        (
            torch.abs(env.command_manager.get_command("base_velocity")[:, 1])
            < velocity_deadzone
        )
        .float()
        .unsqueeze(1)
    )
    return cstr


def joint_torque(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = torch.abs(data.applied_torque[:, asset_cfg.joint_ids]) - limit
    return cstr


def joint_velocity(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return torch.abs(data.joint_vel[:, asset_cfg.joint_ids]) - limit


def joint_acceleration(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return torch.abs(data.joint_acc[:, asset_cfg.joint_ids]) - limit


def upsidedown(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return data.projected_gravity_b[:, 2] > limit


def contact(
    env: ConstrainedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.any(
        torch.max(
            torch.norm(net_contact_forces[:, :, asset_cfg.body_ids], dim=-1),
            dim=1,
        )[0]
        > 1.0,
        dim=1,
    )


def base_orientation(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return torch.norm(data.projected_gravity_b[:, :2], dim=1) - limit


def air_time(
    env: ConstrainedRlEnv,
    limit: float,
    velocity_deadzone: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, asset_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, asset_cfg.body_ids]
    # Like in CaT
    command_more_than_limit = (
        (
            torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
            > velocity_deadzone
        )
        .float()
        .unsqueeze(1)
    )
    cstr = (limit - last_air_time) * touchdown.float() * command_more_than_limit
    return cstr


def n_foot_contact(
    env: ConstrainedRlEnv,
    number_of_desired_feet: int,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_cstr = torch.abs(
        (
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1
                ),
                dim=1,
            )[0]
            > 1.0
        ).sum(1)
        - number_of_desired_feet
    )

    return contact_cstr



def joint_range(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return (
        torch.abs(data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids])
        - limit
    )


def action_rate(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return (
        torch.abs(
            env.action_manager._action[:, asset_cfg.joint_ids]
            - env.action_manager._prev_action[:, asset_cfg.joint_ids]
        )
        / env.step_dt
        - limit
    )


def foot_contact_force(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_forces = (
        torch.max(torch.norm(net_contact_forces[:, :, asset_cfg.body_ids], dim=-1), dim=1)[0]
    )
    return contact_forces - limit


def min_base_height(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return limit - robot.data.root_pos_w[:, 2]

def max_base_height(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2] - limit


def no_move(
    env: ConstrainedRlEnv,
    velocity_deadzone: float,
    joint_vel_limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr_nomove = (torch.abs(data.joint_vel[:, asset_cfg.joint_ids]) - joint_vel_limit) * (
        torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
        < velocity_deadzone
    ).float().unsqueeze(1)
    return cstr_nomove

def body_velocity(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return torch.norm(data.root_lin_vel_w, dim=1) - limit

def foot_slippage(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    # Foot forces
    contact_sensor = env.scene[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    foot_force = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]

    # Foot velocities
    data = env.scene[asset_cfg.name].data
    foot_vel_xy = torch.norm(data.body_vel_w[:, asset_cfg.body_ids, :2], dim=-1)

    # Consider slippage if foot is in contact with the ground and moving
    foot_slip = foot_vel_xy * (foot_force > 1.0).float()

    return foot_slip - limit

def foot_stumble(
    env: ConstrainedRlEnv,
    sensor_cfg: SceneEntityCfg,
    coefficient: float = 4.0,
) -> torch.Tensor:
    # Foot forces
    contact_sensor = env.scene[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    foot_force_xy = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :2], dim=-1), dim=1)[0]
    foot_force_z = torch.max(torch.abs(net_contact_forces[:, :, sensor_cfg.body_ids, 2]), dim=1)[0]

    # Check if foot is stumbling: high lateral forces but low vertical forces
    cstr_foot_stumble = foot_force_xy - coefficient * foot_force_z

    return cstr_foot_stumble

def feet_force_std(
    env: ConstrainedRlEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    foot_force = torch.max(torch.norm(net_contact_forces[:, :, asset_cfg.body_ids], dim=-1), dim=1)[0]
    force_std = torch.std(foot_force, dim=1)
    return force_std - limit