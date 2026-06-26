# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from lunar_locomanipulation_crl.envs import ConstrainedRlEnv, ConstrainedRlEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import VisualizationMarkersCfg, FRAME_MARKER_CFG

# For constraints
from lunar_locomanipulation_crl.modules.constraint_term_cfg import ConstraintTermCfg as CstrTerm
from lunar_locomanipulation_crl.modules import constraints as cstr_funcs
from lunar_locomanipulation_crl.modules import curriculums as curr_funcs

##
# Pre-defined configs
##
from lunar_locomanipulation_crl.assets.widow_go2 import WIDOWGO2_CFG  # isort: skip
from lunar_locomanipulation_crl.utils.terrains import LUNAR_TERRAIN_CFG, JSC1A_MATERIAL_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range": (0.3, 1.25),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.0, 5.0),
            "operation": "add",
        },
    )

@configclass
class ConstraintsCfg:
    """
    Configuration class for constraints specific to the WBC Lunar Locomanipulation task.

    This config class defines the constraints for the CaT algorithm in the WBC Lunar Locomanipulation environment.
    
    """
    # --- SOFT CONSTRAINTS ---
    # Legs
    leg_ht_joint_torque: CstrTerm = CstrTerm(
        func=cstr_funcs.joint_torque,
        max_p=0.25,
        params={
            "limit": 23.5,  # Nm
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*hip_joint", ".*thigh_joint"]),
        },
    )
    leg_knee_joint_torque: CstrTerm = CstrTerm(
        func=cstr_funcs.joint_torque,
        max_p=0.25,
        params={
            "limit": 35.0,  # Nm
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*calf_joint"]),
        },
    )
    leg_joint_velocity: CstrTerm = CstrTerm(
        func=cstr_funcs.joint_velocity,
        max_p=0.25,
        params={
            "limit": 30.0,  # rad/s
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"]),
        },
    )
    
    # --- HARD CONSTRAINTS ---
    collision = CstrTerm(
        func=cstr_funcs.contact,
        max_p=1.0,
        params={
            "asset_cfg": SceneEntityCfg(name="contact_sensor", body_names=[".*base", ".*Head.*", ".*hip", ".*wx250s.*"]),
        },
    )
    # foot_contact_force: CstrTerm = CstrTerm(
    #     func=cstr_funcs.foot_contact_force,
    #     max_p=1.0,
    #     params={
    #         "limit": 150.0,  # Newtons
    #         "asset_cfg": SceneEntityCfg(name="contact_sensor", body_names=[".*foot"]),
    #     },
    # )
    # upsidedown = CstrTerm(
    #     func=cstr_funcs.upsidedown,
    #     max_p=1.0,
    #     params={
    #         "limit": -0.9,  # z-component of projected gravity in body frame (~cos(35°))
    #         "asset_cfg": SceneEntityCfg(name="robot"),
    #     },
    # )

    # -- STYLE CONSTRAINTS --
    base_velocity: CstrTerm = CstrTerm(
        func=cstr_funcs.body_velocity,
        max_p=0.25,
        params={
            "limit": 0.25,  # m/s
            "asset_cfg": SceneEntityCfg(name="robot"),
        },
    )
    
    leg_joint_deviation: CstrTerm = CstrTerm(
        func=cstr_funcs.joint_range,
        max_p=0.25,
        params={
            "limit": 0.4,  # rad
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"]),
        },
    )

    foot_stumble: CstrTerm = CstrTerm(
        func=cstr_funcs.foot_stumble,
        max_p=0.25,
        params={
            "sensor_cfg": SceneEntityCfg(name="contact_sensor", body_names=[".*foot"]),
        },
    )

    foot_slippage: CstrTerm = CstrTerm(
        func=cstr_funcs.foot_slippage,
        max_p=0.25,
        params={
            "limit": 0.2,  # m/s
            "asset_cfg": SceneEntityCfg(name="robot", body_names=[".*foot"]),
            "sensor_cfg": SceneEntityCfg(name="contact_sensor", body_names=[".*foot"]),
        },
    )

    # n_foot_contacts: CstrTerm = CstrTerm(
    #     func=cstr_funcs.n_foot_contact,
    #     max_p=0.25,
    #     params={
    #         "number_of_desired_feet": 3,  # minimum number of feet in contact
    #         "sensor_cfg": SceneEntityCfg(name="contact_sensor", body_names=[".*foot"]),
    #     },
    # )

    # feet_force_std: CstrTerm = CstrTerm(
    #     func=cstr_funcs.feet_force_std,
    #     max_p=0.25,
    #     params={
    #         "limit": 70.0,  # m/s
    #         "asset_cfg": SceneEntityCfg(name="contact_sensor", body_names=[".*foot"]),
    #     },
    # )

MAX_CURRICULUM_ITERATIONS = 1000
@configclass
class CurriculumCfg:
    """Configuration class for curriculum specific to the WBC Lunar Locomanipulation task."""
    leg_ht_joint_torque_p: CurrTerm = CurrTerm(
        func=curr_funcs.modify_constraint_p,
        params={
            "term_name": "leg_ht_joint_torque",
            "num_steps": MAX_CURRICULUM_ITERATIONS * 24,
            "init_max_p": 0.25,
        },
    )
    leg_knee_joint_torque_p: CurrTerm = CurrTerm(
        func=curr_funcs.modify_constraint_p,
        params={
            "term_name": "leg_knee_joint_torque",
            "num_steps": MAX_CURRICULUM_ITERATIONS * 24,
            "init_max_p": 0.25,
        },
    )
    leg_joint_velocity_p: CurrTerm = CurrTerm(
        func=curr_funcs.modify_constraint_p,
        params={
            "term_name": "leg_joint_velocity",
            "num_steps": MAX_CURRICULUM_ITERATIONS * 24,
            "init_max_p": 0.25,
        },
    )
    leg_base_velocity_p: CurrTerm = CurrTerm(
        func=curr_funcs.modify_constraint_p,
        params={
            "term_name": "base_velocity",
            "num_steps": MAX_CURRICULUM_ITERATIONS * 24,
            "init_max_p": 0.25,
        },
    )
    leg_joint_deviation_p: CurrTerm = CurrTerm(
        func=curr_funcs.modify_constraint_p,
        params={
            "term_name": "leg_joint_deviation",
            "num_steps": MAX_CURRICULUM_ITERATIONS * 24,
            "init_max_p": 0.25,
        },
    )
    foot_stumble_p: CurrTerm = CurrTerm(
        func=curr_funcs.modify_constraint_p,
        params={
            "term_name": "foot_stumble",
            "num_steps": MAX_CURRICULUM_ITERATIONS * 24,
            "init_max_p": 0.25,
        },
    )
    foot_slippage_p: CurrTerm = CurrTerm(
        func=curr_funcs.modify_constraint_p,
        params={
            "term_name": "foot_slippage",
            "num_steps": MAX_CURRICULUM_ITERATIONS * 24,
            "init_max_p": 0.25,
        },
    )
    # feet_force_std_p: CurrTerm = CurrTerm(
    #     func=curr_funcs.modify_constraint_p,
    #     params={
    #         "term_name": "feet_force_std",
    #         "num_steps": MAX_CURRICULUM_ITERATIONS * 24,
    #         "init_max_p": 0.25,
    #     },
    # )

    """ Terrain curriculum configuration. """
    terrain_difficulty: CurrTerm = CurrTerm(
        func=curr_funcs.modify_terrain_difficulty,
        params={
            "error_threshold": 0.15,
        },
    )
    """ Target pose sampling ranges. """
    target_pos_x_range: CurrTerm = CurrTerm(
        func=curr_funcs.modify_target_max_range,
        params={
            "axis": "x",
            "initial_range":[-0.3, 0.3],
            "final_range": [-0.3, 2.5],
            "num_steps": MAX_CURRICULUM_ITERATIONS * 2 * 24,
        },
    )
    target_pos_y_range: CurrTerm = CurrTerm(
        func=curr_funcs.modify_target_max_range,
        params={
            "axis": "y",
            "initial_range": [-0.3, 0.3],
            "final_range": [-1.0, 1.0],
            "num_steps": MAX_CURRICULUM_ITERATIONS * 2 * 24,
        },
    )
    target_pos_z_range: CurrTerm = CurrTerm(
        func=curr_funcs.modify_target_max_range,
        params={
            "axis": "z",
            "initial_range": [-0.1, 0.1],
            "final_range": [-0.35, 0.2],
            "num_steps": MAX_CURRICULUM_ITERATIONS * 2 * 24,
        },
    )

@configclass
class WBCLunarLocomanipulationEnvCfg(ConstrainedRlEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 4
    action_space = 18
    observation_space = 74  # without height scanner
    state_space = 0
    debug_vis = True

    # simulation
    # Increase gpu_max_rigid_patch_count for large terrains
    physx_cfg: sim_utils.PhysxCfg = sim_utils.PhysxCfg(
        gpu_max_rigid_patch_count=180_000, #262_144,
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        gravity=(0.0, 0.0, -9.81 / 6.0), # Lunar gravity
        physx=physx_cfg
    )
    

    # Add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Terrain configuration
    rough_terrain: bool = True
    if rough_terrain:
        observation_space += 187  # with height scanner
        terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=LUNAR_TERRAIN_CFG,
                max_init_terrain_level=1,
                collision_group=-1,
                physics_material=JSC1A_MATERIAL_CFG,
                # visual_material=sim_utils.MdlFileCfg(
                #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                #     project_uvw=True,
                # ),
                debug_vis=False,
        )
    else:
        height_scanner = None
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = WIDOWGO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # -- Constraints --
    constraints: ConstraintsCfg = ConstraintsCfg()

    # -- Action parameters -- 
    leg_action_scale = 0.25
    arm_action_scale = 0.5
    action_scale = [leg_action_scale] * 12 + [arm_action_scale] * 6

    # Curriculum
    curriculum: CurriculumCfg | None = CurriculumCfg()

    # -- Command parameters --
    enable_command_curriculum: bool = False
    if enable_command_curriculum:
        # EE target position ranges
        target_pos_x_range = ([-0.3, 0.3], [-0.3, 2.5]) # Min range & Max range
        target_pos_y_range = ([-0.3, 0.3], [-1.0, 1.0])
        target_pos_z_range = ([-0.1, 0.1], [-0.35, 0.2])

        curriculum.target_pos_x_range.params["initial_range"] = target_pos_x_range[0]
        curriculum.target_pos_x_range.params["final_range"] = target_pos_x_range[1]
        curriculum.target_pos_y_range.params["initial_range"] = target_pos_y_range[0]
        curriculum.target_pos_y_range.params["final_range"] = target_pos_y_range[1]
        curriculum.target_pos_z_range.params["initial_range"] = target_pos_z_range[0]
        curriculum.target_pos_z_range.params["final_range"] = target_pos_z_range[1]
    else:
        # EE target position ranges
        target_pos_x_range = [-0.3, 2.5]
        target_pos_y_range = [-1.0, 1.0]
        target_pos_z_range = [-0.35, 0.2]
        curriculum.target_pos_x_range = None
        curriculum.target_pos_y_range = None
        curriculum.target_pos_z_range = None


    # EE target orientation ranges (in radians)
    target_rot_roll_range = [0.0, 0.0]
    target_rot_pitch_range = [-math.pi * 0.25, math.pi * 0.5]  
    target_rot_yaw_range = [-math.pi * 0.5, math.pi * 0.5]

    if not rough_terrain:
        curriculum.terrain_difficulty = None  # No terrain curriculum for flat terrain

    # -- Reward parameters --
    pose_tracking_reward_scale = 5.0
    # Radius to enable manipulation pose tracking reward and saturate locomotion reward
    pose_tracking_radius = 0.3 # in meters

    # Stability reward scale
    stability_reward_scale = 0.4

    # Low power consumption reward scales
    leg_low_power_reward_scale = 0.5
    max_leg_power = 900.0 # Max value expected for leg power consumption. This is used to normalize the power consumption reward
    arm_low_power_reward_scale = 0.2
    max_arm_power = 40.0 # Max value expected for arm power consumption. This is used to normalize the power consumption reward
    leg_action_rate_reward_scale = 0.2
    arm_action_rate_reward_scale = 0.25