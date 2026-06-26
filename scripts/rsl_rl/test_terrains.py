# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of steps per episode.")
# -- Policy evaluation specific arguments
# Terrain type
parser.add_argument("--terrain", type=str, default=None, help="Name of the terrain to use for testing.")
parser.add_argument("--difficulty", type=int, default=None, help="Difficulty level of the terrain.")
# Soil types
parser.add_argument("--soil", type=str, default="nominal", help="Type of soil to use for testing. (loose, nominal, dense)")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from isaaclab.envs import (
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, export_policy_as_jit, export_policy_as_onnx
from lunar_locomanipulation_crl.algorithms import RslRlCatVecEnvWrapper, CaTOnPolicyRunner

# Math utils
from isaaclab.utils.math import quat_error_magnitude

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import lunar_locomanipulation_crl.tasks  # noqa: F401

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg , agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # --------  TERRAIN MODIFICATION FOR TESTING  --------
    env_cfg.enable_command_curriculum = False  # disable command curriculum for testing
    env_cfg.curriculum.terrain_difficulty = None
    env_cfg.terrain.max_init_terrain_level = 4

    # Set the terrain and difficulty if specified
    if args_cli.terrain is not None:
        desired_terrain = args_cli.terrain
        env_cfg.terrain.terrain_generator.curriculum = True
        env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.terrain_generator.num_rows - 1
        # Iterate through all sub terrains and set proportions
        for sub_terrain_name in env_cfg.terrain.terrain_generator.sub_terrains:
            if sub_terrain_name == desired_terrain:
                env_cfg.terrain.terrain_generator.sub_terrains[sub_terrain_name].proportion = 1.0
            else:
                env_cfg.terrain.terrain_generator.sub_terrains[sub_terrain_name].proportion = 0.0
        
    # Set difficulty level
    if args_cli.difficulty is not None:
        difficulty_level = args_cli.difficulty
        num_difficulties = env_cfg.terrain.terrain_generator.num_rows
        difficulty_level = min(difficulty_level / num_difficulties, 1)
        env_cfg.terrain.terrain_generator.difficulty_range = (difficulty_level, difficulty_level)

    # --------- SET FRICTION COEFFICIENTS FOR TESTING ---------
    # Values obtained according to the interaction of JSC-1a lunar soil simulant with typical robot foot materials (rubber)
    env_cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
    if args_cli.soil == "loose":
        env_cfg.terrain.physics_material.static_friction = 0.8
        env_cfg.terrain.physics_material.dynamic_friction = 0.6
        env_cfg.events.physics_material.compliant_contact_stiffness = 1e5
        env_cfg.events.physics_material.compliant_contact_damping = 2e4
    elif args_cli.soil == "dense":
        env_cfg.terrain.physics_material.static_friction = 1.35
        env_cfg.terrain.physics_material.dynamic_friction = 1.0
        env_cfg.events.physics_material.compliant_contact_stiffness = 1e6
        env_cfg.events.physics_material.compliant_contact_damping = 1e5
    # ----------------------------------------------------

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlCatVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    # create runner from rsl-rl
    if agent_cfg.class_name == "CaTOnPolicyRunner":
        runner = CaTOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # Get max steps info
    if args_cli.max_steps is not None:
        max_episode_steps = args_cli.max_steps
        print(f"[INFO] Overriding max episode steps to: {max_episode_steps}")
    else: 
        max_episode_steps = env_cfg.episode_length_s // dt
        print(f"[INFO] Max episode steps: {max_episode_steps}")    

    # reset environment
    obs = env.get_observations()
    timestep = 0
    constraint_violations = {}
    giim_buffer = []
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            raw_constraints = env.unwrapped.constraint_manager.cat.raw_constraints
            giim_buffer.append(env.unwrapped._last_giim.detach().cpu().numpy())
            # accumulate constraint violations
            for name in env.unwrapped.constraint_manager.cat.get_names():
                if name not in constraint_violations:
                    constraint_violations[name] = torch.zeros_like(raw_constraints[name])
                else:
                    constraint_violations[name] += (env.unwrapped.constraint_manager.cat.raw_constraints[name] > 0).float()
            timestep += 1

        if timestep == max_episode_steps - 5 and not args_cli.video:
            # Compute error metrics at the end of the episode
            curr_ee_pos_w = env.unwrapped._robot.data.body_com_pose_w[:, env.unwrapped._ee_id, :3].view(-1, 3)
            ee_pos_error = torch.norm(curr_ee_pos_w - env.unwrapped._target_ee_pose_w[:, :3], dim=-1)
            mean_pos_error = torch.mean(ee_pos_error)
            std_pos_error = torch.std(ee_pos_error)
            median_pos_error = torch.median(ee_pos_error)
            p90_pos_error = torch.quantile(ee_pos_error, 0.9)
            p95_pos_error = torch.quantile(ee_pos_error, 0.95)
            p99_pos_error = torch.quantile(ee_pos_error, 0.99)

            plot_error_KDE(ee_pos_error * 100.0, title="EE Position Error Distribution", unit="cm")

            curr_ee_quat_w = env.unwrapped._robot.data.body_com_pose_w[:, env.unwrapped._ee_id, 3:7].view(-1, 4)
            ee_quat_error = quat_error_magnitude(curr_ee_quat_w, env.unwrapped._target_ee_pose_w[:, 3:7])
            mean_quat_error = torch.mean(ee_quat_error) * (180.0 / 3.14159265)  # convert to degrees
            std_quat_error = torch.std(ee_quat_error) * (180.0 / 3.14159265)  # convert to degrees
            median_quat_error = torch.median(ee_quat_error) * (180.0 / 3.14159265)
            p90_quat_error = torch.quantile(ee_quat_error, 0.9) * (180.0 / 3.14159265)
            p95_quat_error = torch.quantile(ee_quat_error, 0.95) * (180.0 / 3.14159265)
            p99_quat_error = torch.quantile(ee_quat_error, 0.99) * (180.0 / 3.14159265)

            plot_error_KDE(ee_quat_error * (180.0 / 3.14159265), title="EE Rotation Error Distribution", unit="deg.")

            print(f"[INFO] Episode ended. Mean EE Position Error: {mean_pos_error:.4f} m, Std EE Position Error: {std_pos_error:.4f} m")
            print(f"[INFO] Median EE Position Error: {median_pos_error:.4f} m, 90th Percentile: {p90_pos_error:.4f} m, 95th Percentile: {p95_pos_error:.4f} m, 99th Percentile: {p99_pos_error:.4f} m")
            print(f"[INFO] Mean EE Quaternion Error: {mean_quat_error:.4f} deg., Std EE Quaternion Error: {std_quat_error:.4f} deg.")
            print(f"[INFO] Median EE Quaternion Error: {median_quat_error:.4f} deg., 90th Percentile: {p90_quat_error:.4f} deg., 95th Percentile: {p95_quat_error:.4f} deg., 99th Percentile: {p99_quat_error:.4f} deg.")

            # Plot GIIM evolution
            plot_giim_evolution(giim_buffer)

            # -- Compute constraint violation statistics --
            print(f"[INFO] Constraint Violations over the Episode:")
            for name in env.unwrapped.constraint_manager.cat.get_names():
                total_constraint_violations = constraint_violations[name].mean(dim=1)
                mean_constraint_violations_per_env = torch.mean(total_constraint_violations)
                mean_constraint_violations_per_env_per_step = mean_constraint_violations_per_env / max_episode_steps
                print(f"        - {name}: {total_constraint_violations} violations over the episode.")
                print(f"          Mean Violations per Env: {mean_constraint_violations_per_env.cpu().numpy()}, Mean Violations per Env per Step: {mean_constraint_violations_per_env_per_step.cpu().numpy()}")
                print(f"          Percentage of time steps with violations per env: {mean_constraint_violations_per_env_per_step.cpu().numpy() * 100.0}%")


            break
        elif args_cli.video and timestep >= args_cli.video_length:
            # stop after recording the specified video length
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


def plot_error_KDE(error_data, title="Error KDE Plot", unit: str=""):
    """Plot the KDE of the error data using matplotlib."""
    data = error_data.cpu().numpy()
    sns.kdeplot(data, fill=True)
    plt.title(title)
    plt.xlabel("Error (" + unit + ")")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()
    plt.close()

def plot_giim_evolution(giim_buffer):
    import numpy as np
    """Plot the evolution of GIIM over time."""
    giim_array = np.array(giim_buffer) * 180.0 / 3.14159265  # Convert to degrees
    mean_giim = np.mean(giim_array, axis=1)

    plt.plot(mean_giim)
    plt.title("GIIM Evolution Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Mean GIIM")
    plt.grid(True)
    plt.show()
    plt.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
