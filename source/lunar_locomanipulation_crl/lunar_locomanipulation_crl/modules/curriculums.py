"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from lunar_locomanipulation_crl.envs import ConstrainedRlEnv


def modify_constraint_p(
    env: ConstrainedRlEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
    init_max_p: float,
):
    progress = min(env.common_step_counter / num_steps, 1.0)

    # Linearly interpolate the expected time for episode end: soft_p is the maximum
    # termination probability so it is an image of the expected time of death.
    T_start = 20
    T_end = 1 / init_max_p
    init_max_p = 1 / (T_start + progress * (T_end - T_start))

    # obtain term settings
    term_cfg = env.constraint_manager.get_term_cfg(term_name)
    # update term settings
    term_cfg.max_p = init_max_p
    env.constraint_manager.set_term_cfg(term_name, term_cfg)

    return init_max_p

def modify_terrain_difficulty(
    env: ConstrainedRlEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), error_threshold: float = 0.15
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = env._target_ee_pose  # type: ignore
    # compute the distance to the target
    curr_pos_w = asset.data.body_com_pose_w[env_ids, env._ee_id, :3].view(-1, 3)   
    distance = torch.norm(env._target_ee_pose_w[env_ids, :3] - curr_pos_w, dim=1)
    # robots have accuracy better than the error threshold go to harder terrains
    move_up = distance < error_threshold
    # robots that are far from the target go to easier terrains
    move_down = distance > (2.0 * error_threshold)
    #move_down *= ~move_up
    # update terrain levels
    env._terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(env._terrain.terrain_levels.float())

def modify_target_max_range(
    env: ConstrainedRlEnv,
    env_ids: Sequence[int],
    axis: str,
    initial_range: list[float],
    final_range: list[float],
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies the maximum target range for the end-effector position command.

    The maximum target range is linearly increased from ``initial_range`` to ``final_range``
    over ``num_steps`` environment steps.

    Returns:
        The current maximum target range.
    """
    initial_range = torch.tensor(initial_range, device=env.device)
    final_range = torch.tensor(final_range, device=env.device)

    progress = min(env.common_step_counter / num_steps, 1.0)
    current_range = initial_range + progress * (final_range - initial_range)
    if axis == "x":
        env._target_pos_x_range = current_range  # type: ignore
    elif axis == "y":
        env._target_pos_y_range = current_range  # type: ignore
    elif axis == "z":
        env._target_pos_z_range = current_range  # type: ignore

    return current_range[1]