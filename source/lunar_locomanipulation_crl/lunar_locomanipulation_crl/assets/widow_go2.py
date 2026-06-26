
import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


WIDOWGO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/alvaro/lunar_locomanipulation_crl/source/lunar_locomanipulation_crl/lunar_locomanipulation_crl/assets/data/WidowGo2_simpleColliders.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            ".*widow_left_finger": 0.015,
            ".*widow_right_finger": -0.015,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit_sim={
                ".*_hip_joint": 35.278,
                ".*_thigh_joint": 35.278,
                ".*_calf_joint": 44.4
            },
            velocity_limit_sim=30.0,
            stiffness=40.0,
            damping=1.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*widow_waist", ".*widow_shoulder", ".*widow_elbow", ".*widow_forearm_roll", ".*widow_wrist_angle", ".*widow_wrist_rotate"],
            effort_limit_sim={
                ".*widow_waist": 10.0,
                ".*widow_shoulder": 20.0,
                ".*widow_elbow": 15.0,
                ".*widow_forearm_roll": 2.0,
                ".*widow_wrist_angle": 5.0,
                ".*widow_wrist_rotate": 1.0,
            },
            velocity_limit_sim=3.14,
            stiffness=20.0,
            damping=0.5,
        ),
    },
)
"""Configuration of WidowX wx250s arm + Unitree Go2 using DelayedPD and Implicit actuator models."""
