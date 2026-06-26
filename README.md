# lunar_locomanipulation_crl

## Overview

This repository implements constrained reinforcement learning for **whole-body loco-manipulation on lunar terrain**, built on top of [Isaac Lab](https://isaac-sim.github.io/IsaacLab). A WidowGo2 robot (Unitree Go2 quadruped with a WidowX 6-DOF arm) learns to traverse procedurally generated rough lunar terrain while controlling its arm, under lunar gravity.

Constraints are enforced via the **CaT (Constraints as Terminations)** algorithm, which converts constraint violations into probabilistic episode terminations. This allows the policy to learn safe behaviors without negative rewards, keeping the reward function fully positive.

This is a focused extract of the broader `constrained_rl_isaaclab` project, containing only the `wbc_lunar_locomanipulation` task and the infrastructure required to run it.

**Keywords:** constrained reinforcement learning, whole-body control, loco-manipulation, lunar terrain, space robotics, CaT, Isaac Lab

---

## Task

### `WBC-Lunar-Locomanipulation-Direct-v0` — Whole-Body Control on Lunar Terrain

A WidowGo2 robot performs whole-body loco-manipulation on procedurally generated rough lunar terrain. The policy jointly controls all legs and the arm while satisfying constraints on joint loads, foot contacts, and base velocity under lunar gravity (g/6).

- **Robot:** WidowGo2 — Unitree Go2 quadruped with a WidowX 6-DOF arm
- **Action space (18D):** 12 leg joint positions + 6 arm joint positions
- **Observation space (261D):** base state, joint state, EE pose error, height scanner (187D) for terrain perception
- **Gravity:** lunar (≈ 1.635 m/s², 1/6 of Earth's)
- **Terrain:** procedurally generated rough lunar surface with curriculum-driven difficulty
- **Episode length:** 10 s (500 policy steps at 50 Hz)
- **Curriculum:** constraint probability annealing (over 1000 iterations), terrain difficulty progression, and EE target range expansion

Active constraints:

| Constraint | Limit | max_p |
|---|---|---|
| Hip / thigh joint torque | 23.5 Nm | 0.6 |
| Knee joint torque | 35 Nm | 0.6 |
| Joint velocity | 30 rad/s | 0.6 |
| Base linear velocity | 0.25 m/s | 0.25 |
| Joint deviation from default | (per-joint) | 0.25 |
| Foot stumble | — | 0.6 |
| Foot slippage | — | 0.6 |
| Body collision (hard) | — | 1.0 |

---

## CaT Algorithm

All constraints are handled as probabilistic terminations rather than reward penalties:

```text
reward = clip(raw_reward × (1 − cstr_prob), min=0)
dones  = cstr_prob
```

The termination probability for each constraint is computed by normalizing the violation magnitude against a running Polyak-smoothed maximum, then scaling to `[0, max_p]`. The per-environment probability is the max across all active constraints.

---

## Repository Structure

```text
lunar_locomanipulation_crl/
├── scripts/
│   ├── list_envs.py
│   ├── random_agent.py
│   ├── zero_agent.py
│   └── rsl_rl/
│       ├── train_cat.py               # Train with CaT (uses CaTOnPolicyRunner)
│       ├── play_cat.py                # Visualize a trained policy
│       ├── test_terrains.py           # Terrain traversal evaluation
│       ├── plot_terrain_experiments.py
│       └── plot_soil_experiments.py
└── source/lunar_locomanipulation_crl/lunar_locomanipulation_crl/
    ├── algorithms/
    │   ├── cat_on_policy_runner.py   # CaT training runner (extends RSL-RL)
    │   └── rsl_rl_cat_vecenv_wrapper.py
    ├── assets/
    │   ├── widow_go2.py               # WIDOWGO2_CFG articulation config
    │   └── data/                      # WidowGo2.usd, WidowGo2_simpleColliders.usd
    ├── envs/
    │   ├── constrained_rl_env.py     # Base env: CaT step loop
    │   └── constrained_rl_env_cfg.py
    ├── modules/
    │   ├── constraint_manager.py     # CaT class + ConstraintManager
    │   ├── constraint_term_cfg.py    # ConstraintTermCfg dataclass
    │   ├── constraints.py            # Shared constraint functions (torque/vel/foot/collision)
    │   └── curriculums.py            # Curriculum functions
    ├── utils/
    │   └── terrains.py                # LUNAR_TERRAIN_CFG, JSC1A_MATERIAL_CFG
    └── tasks/direct/
        └── wbc_lunar_locomanipulation/
            ├── wbc_lunar_locomanipulation_env.py
            ├── wbc_lunar_locomanipulation_env_cfg.py
            └── agents/
                ├── rsl_rl_ppo_cfg.py
                └── skrl_ppo_cfg.yaml
```

---

## Installation

1. Install Isaac Lab following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Conda or uv is recommended.

2. Install the package in editable mode:

    ```bash
    python -m pip install -e source/lunar_locomanipulation_crl
    ```

3. Verify the installation by listing available environments:

    ```bash
    python scripts/list_envs.py
    ```

**Note:** the USD asset path in `assets/widow_go2.py` is currently hardcoded to `/home/alvaro/lunar_locomanipulation_crl/...`. Update it if you move this repository elsewhere.

---

## Usage

### Training

```bash
python scripts/rsl_rl/train_cat.py --task=WBC-Lunar-Locomanipulation-Direct-v0 --num_envs=4096
```

### Playing a trained policy

```bash
python scripts/rsl_rl/play_cat.py --task=WBC-Lunar-Locomanipulation-Direct-v0 --num_envs=32 --load_run=<run_name>
```

### Evaluating on terrain

```bash
python scripts/rsl_rl/test_terrains.py \
    --task=WBC-Lunar-Locomanipulation-Direct-v0 \
    --num_envs=512 \
    --load_run=<run_name>
```

### Zero / random agents

```bash
python scripts/zero_agent.py   --task=WBC-Lunar-Locomanipulation-Direct-v0
python scripts/random_agent.py --task=WBC-Lunar-Locomanipulation-Direct-v0
```

---

## IDE Setup (Optional)

Run the VSCode task to configure the Python environment:

1. Press `Ctrl+Shift+P` → `Tasks: Run Task` → `setup_python_env`
2. Enter the absolute path to your Isaac Sim installation when prompted.

This generates `.vscode/.python.env` with all Isaac Sim extension paths for Pylance indexing.

---

## Code Formatting

```bash
pip install pre-commit
pre-commit run --all-files
```
