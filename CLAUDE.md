# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unitree RL Lab provides reinforcement learning environments for Unitree robots (Go2, H1, G1-29dof, G1-23dof, B2, Go2W, H1-2) built on [IsaacLab 2.3.0](https://isaac-sim.github.io/IsaacLab) and [IsaacSim 5.1.0](https://docs.omniverse.nvidia.com/isaacsim). The codebase supports two task types: **locomotion** (velocity-tracking RL) and **mimic** (motion-tracking/dance via joint-space imitation).

## Key Directories

- `source/unitree_rl_lab/unitree_rl_lab/` - Python package
  - `assets/robots/unitree.py` - Robot articulation definitions (USD/URDF spawn configs, actuators, joint mappings for all Unitree robots)
  - `tasks/locomotion/` - Locomotion environments (velocity tracking)
    - `robots/g1/29dof/velocity_env_cfg.py` - G1-29dof locomotion env config (scene, rewards, observations, commands, terminations)
    - `robots/go2/velocity_env_cfg.py`, `robots/h1/velocity_env_cfg.py` - Robot-specific configs
    - `agents/rsl_rl_ppo_cfg.py` - Shared PPO agent config (RslRlOnPolicyRunnerCfg, 512/256/128 MLP, ELU)
      - `BasePPORunnerCfg` - default (no symmetry)
      - `BasePPORunnerWithSymmetryCfg` - with left-right symmetry augmentation + mirror loss
    - `mdp/symmetry/` - Left-right symmetry functions per robot
      - `symmetry/g1_29dof.py` - G1-29dof symmetry (swap_map + flip_set, 16 roll/yaw joints)
    - `mdp/` - MDP term functions (rewards, observations, commands, events, curriculums)
  - `tasks/mimic/` - Motion mimic environments (joint-space imitation/tracking)
    - `robots/g1_29dof/` - Dance and style-mimic tracking envs (e.g., dance_102, gangnam_style)
    - `agents/rsl_rl_ppo_cfg.py`, `mdp/` - Mimic-specific PPO config and MDP terms
  - `utils/parser_cfg.py` - `parse_env_cfg()` for loading/resolving task configs at runtime
  - `utils/export_deploy_cfg.py` - `export_deploy_cfg()` exports trained env config to YAML for deployment
  - `tasks/__init__.py` - Auto-registers all task configs via `import_packages()`
- `scripts/rsl_rl/` - Training/play scripts
  - `train.py` - Main training entry point (uses `OnPolicyRunner`, Hydra configs, logs to `logs/rsl_rl/<exp>/`)
  - `play.py` - Inference/play entry point (loads checkpoint, exports policy.pt and policy.onnx)
  - `cli_args.py` - Shared CLI argument parsing for RSL-RL options
- `scripts/mimic/` - Data conversion utilities for mimic tasks
  - `csv_to_npz.py` - Convert CSV motion capture data to NPZ format
  - `replay_npz.py` - Replay NPZ trajectory data
- `deploy/robots/<robot>/` - C++ deployment code per robot (CMakeLists.txt, main.cpp, FSM states, ONNX policies, deploy.yaml)
- `docker/docker-compose.yaml` - Docker compose for IsaacLab containerized dev

## Commands

### Installation (requires conda with IsaacLab activated)
```bash
conda activate env_isaaclab
./unitree_rl_lab.sh -i        # pip install -e, setup conda env, install git lfs
source unitree_rl_lab.sh      # re-source after install
```

### List available tasks
```bash
./unitree_rl_lab.sh -l        # list all registered Unitree tasks
```

### Train
```bash
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity   # headless training
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity-WithSymmetry   # with left-right symmetry augmentation
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --video --num_envs 8192
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --logger wandb --run_name trial1
./scripts/rsl_rl/train.py --task Unitree-G1-29dof-Velocity-WithSymmetry --headless
```

### Play / Inference
```bash
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity   # play with latest checkpoint
./scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity --checkpoint logs/.../policy.pt
./scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity --video  # record video
```

### Code quality
```bash
pre-commit run --all-files          # black + flake8 + isort + pyupgrade + codespell
pre-commit run --all-files --hook-stage manual  # pyright type checking (via pyproject.toml [tool.pyright])
```

### Docker
```bash
docker compose -f docker/docker-compose.yaml up   # launch IsaacLab container
```

## Architecture

### Task Registration
Tasks are auto-registered as Gym environments via `import_packages(__name__)` in `tasks/__init__.py`. IsaacLab's `@configclass` decorators register multiple entry points per task:
- `env_cfg_entry_point` / `play_env_cfg_entry_point` - Environment configs
- `rsl_rl_cfg_entry_point` - RL agent config (PPO runner)

Task naming convention: `Unitree-<Robot>-<TaskType>` (e.g., `Unitree-G1-29dof-Velocity`, `Unitree-G1-29dof-Mimic-Dance-102`).

### Environment Config Pattern
Each task follows IsaacLab's ManagerBasedRLEnvCfg pattern with nested config classes:
- `RobotSceneCfg` (InteractiveSceneCfg) - scene entities (robot, terrain, sensors, lights)
- `ObservationsCfg` / `ObservationsCfg.PolicyCfg` and `CriticCfg` - observation terms with noise/scaling
- `RewardsCfg` - reward terms (positive = desired, negative = penalty)
- `TerminationsCfg` - termination/done terms
- `CommandsCfg` - velocity commands (UniformLevelVelocityCommandCfg)
- `EventCfg` - randomization events (startup physics/mass randomization, reset base/joints, interval pushes)
- `CurriculumCfg` - terrain difficulty curriculum

Robot configs in `assets/robots/unitree.py` define articulation spawn paths, initial joint states, and actuator models (IdealPDActuatorCfg / ImplicitActuatorCfg) with hardware-derived stiffness/damping/armature values.

### Training Flow
1. `train.py` discovers tasks via `gym.registry`, loads env + agent configs via Hydra
2. Creates gym environment, wraps with `RslRlVecEnvWrapper`, instantiates `OnPolicyRunner`
3. Calls `runner.learn()` - logs to `logs/rsl_rl/<experiment_name>/<timestamp>/`
4. `export_deploy_cfg()` automatically exports runtime config as `deploy.yaml` in the log directory

### Deployment Pipeline
Trained models -> sim2sim (Mujoco) -> sim2real (C++):
1. `play.py` exports `policy.pt` (JIT) and `policy.onnx` to `logs/.../exported/`
2. `deploy/robots/<robot>/config/policy/<task>/` stores deploy.yaml + ONNX policy
3. C++ `robot_controller` loads ONNX via onnxruntime, reads deploy.yaml for joint mappings, action scales, observation specs
4. FSM-based state machine (FSM/ directory) handles mode switching (standby, mimic, RL)

### Key Config Files
- `pyproject.toml` - isort sections (ISAACLABPARTY, FIRSTPARTY), pyright type checking
- `.flake8` - flake8 config with google docstring convention, max-line-length 120
- `.pre-commit-config.yaml` - black (preview, 120), flake8, isort, pyupgrade, codespell
