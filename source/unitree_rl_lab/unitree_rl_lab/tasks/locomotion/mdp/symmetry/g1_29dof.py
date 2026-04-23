# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry for G1-29dof locomotion.

Applies left-right mirror symmetry to observations and actions for data augmentation.
Follows legged_lab's term-by-term iteration pattern and holosoma's config-driven approach.

Joint ordering (IsaacLab articulation):
    [ 0] left_hip_pitch_joint       [ 1] right_hip_pitch_joint
    [ 2] waist_yaw_joint            [ 3] left_hip_roll_joint    [ 4] right_hip_roll_joint
    [ 5] waist_roll_joint           [ 6] left_hip_yaw_joint     [ 7] right_hip_yaw_joint
    [ 8] waist_pitch_joint          [ 9] left_knee_joint        [10] right_knee_joint
   [11] left_shoulder_pitch_joint  [12] right_shoulder_pitch_joint
   [13] left_ankle_pitch_joint     [14] right_ankle_pitch_joint
   [15] left_shoulder_roll_joint   [16] right_shoulder_roll_joint
   [17] left_ankle_roll_joint      [18] right_ankle_roll_joint
   [19] left_shoulder_yaw_joint    [20] right_shoulder_yaw_joint
   [21] left_elbow_joint           [22] right_elbow_joint
   [23] left_wrist_roll_joint      [24] right_wrist_roll_joint
   [25] left_wrist_pitch_joint     [26] right_wrist_pitch_joint
   [27] left_wrist_yaw_joint       [28] right_wrist_yaw_joint

Flip set (16 joints: roll + yaw): {2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 23, 24, 27, 28}
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# Left-right swap map (adjacent L↔R pairs; waist joints 2,5,8 are centerline — no swap)
_SWAP_MAP: list[int] = list(range(29))
for a, b in [(0, 1), (3, 4), (6, 7), (9, 10), (11, 12), (13, 14),
             (15, 16), (17, 18), (19, 20), (21, 22), (23, 24),
             (25, 26), (27, 28)]:
    _SWAP_MAP[a] = b
    _SWAP_MAP[b] = a

# Joints whose sign must be flipped after swapping (roll and yaw axes)
_FLIP_SET: set[int] = {2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 23, 24, 27, 28}


def _build_flip_mask(n_joints: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones(n_joints, dtype=torch.float32, device=device)
    mask[list(i for i in _FLIP_SET if i < n_joints)] = -1.0
    return mask


# --- Symmetry helpers ---

# Sign flip patterns for non-joint terms
_ANG_VEL_SIGN = torch.tensor([-1.0, 1.0, -1.0])
_proj_grav_sign = torch.tensor([1.0, -1.0, 1.0])
_cmd_sign = torch.tensor([1.0, -1.0, -1.0])
_lin_vel_sign = torch.tensor([1.0, -1.0, 1.0])


def _mirror_joint_term(
    tensor: torch.Tensor,
    joint_dim_per_step: int,
    dim: int,
    history_length: int,
    start_idx: int,
    flip_mask: torch.Tensor,
    swap_idx: torch.Tensor,
    is_3d: bool,
    B: int,
) -> int:
    """Mirror a joint/last_action term (swap indices + flip sign).

    Returns the next start index after this term.
    """
    if is_3d:
        # [batch, history, per_step]: mirror per-step within each history step
        # start_idx is the per_step offset of this term
        for h in range(history_length):
            term_data = tensor[:, h, start_idx : start_idx + joint_dim_per_step]
            tensor[:, h, start_idx : start_idx + joint_dim_per_step] = (
                term_data[:, swap_idx] * flip_mask
            )
    else:
        # [batch, flat]: iterate over history, each step has joint_dim_per_step features
        for h in range(history_length):
            term_data = tensor[:, start_idx + h * joint_dim_per_step : start_idx + (h + 1) * joint_dim_per_step].view(
                B, joint_dim_per_step
            )
            tensor[:, start_idx + h * joint_dim_per_step : start_idx + (h + 1) * joint_dim_per_step] = (
                term_data[:, swap_idx] * flip_mask
            ).view(B, joint_dim_per_step)
    return start_idx + joint_dim_per_step * history_length


# --- Observation config detection ---

def _get_obs_term_structure(
    env: ManagerBasedRLEnv,
    obs_tensor: torch.Tensor,
    group_name: str,
) -> dict:
    """Detect observation term structure from the observation manager.

    Returns dict with:
      - term_dims: list of per-sample term dimensions (already includes history)
      - term_types: list of type strings ('a', 'l', 'p', 'v', 'j', or 'other')
      - term_dims_per_step: list of per-step dimensions (before history expansion)
      - history_length: int
      - is_3d: bool
    """
    is_3d = obs_tensor.ndim == 3
    obs_manager = getattr(env.unwrapped, "observation_manager", None)
    history_length = 1

    term_names = []
    term_dims = []  # per-sample dimensions
    term_dims_per_step = []  # per-step dimensions

    if obs_manager is not None:
        try:
            group_cfg = obs_manager.cfg.groups.get(group_name, None)
            if group_cfg is not None:
                history_length = getattr(group_cfg, "history_length", 1)
                term_keys = sorted(group_cfg.terms.keys())
                term_names = list(term_keys)
                for term_name in term_keys:
                    term_cfg = group_cfg.terms[term_name]
                    # Get per-step dimension by computing the term once
                    obs = obs_manager._compute_term(group_name, term_name, term_cfg)
                    # The returned obs shape is [num_envs, term_dim]
                    term_dims_per_step.append(obs.shape[1])
                    # Per-sample dimension includes history
                    if history_length > 1:
                        term_dims.append(obs.shape[1] * history_length)
                    else:
                        term_dims.append(obs.shape[1])
        except Exception:
            pass

    # Fallback if detection failed
    if not term_names:
        if is_3d:
            per_step = obs_tensor.shape[2]
            term_dims_per_step = [per_step]
            term_dims = [per_step * obs_tensor.shape[1]]
            term_names = ["unknown"]
        else:
            flat_dim = obs_tensor.shape[1]
            term_dims = [flat_dim]
            term_dims_per_step = [flat_dim]
            term_names = ["unknown"]

    # Assign types based on per-step dimension and position
    term_types = []
    first_3d_count = 0
    for name, dim_per_step in zip(term_names, term_dims_per_step):
        if dim_per_step == 3:
            if name in ("base_ang_vel", "ang_vel"):
                term_types.append("a")
            elif name in ("base_lin_vel", "lin_vel"):
                term_types.append("l")
            elif name in ("projected_gravity", "proj_grav"):
                term_types.append("p")
            elif name in ("velocity_commands", "cmd"):
                term_types.append("v")
            elif first_3d_count < 3:
                if first_3d_count == 0:
                    term_types.append("a")
                elif first_3d_count == 1:
                    term_types.append("p")
                else:
                    term_types.append("v")
            else:
                term_types.append("v")
            first_3d_count += 1
        elif dim_per_step == 29:
            term_types.append("j")
        else:
            term_types.append("other")

    return {
        "term_dims": term_dims,
        "term_types": term_types,
        "term_dims_per_step": term_dims_per_step,
        "history_length": history_length,
        "is_3d": is_3d,
    }


def _mirror_obs(
    obs_slice: torch.Tensor,
    term_info: dict,
    flip_mask: torch.Tensor,
    swap_idx: torch.Tensor,
) -> torch.Tensor:
    """Apply left-right mirror symmetry to an observation tensor.

    Iterates over each term in order (following legged_lab pattern), applying
    the appropriate symmetry transformation per term type.

    Args:
        obs_slice: Tensor to mirror (cloned from the second half of batch).
        term_info: Dict from _get_obs_term_structure().
        flip_mask: Per-joint sign flip mask.
        swap_idx: Joint swap index map.

    Returns:
        Mirrored tensor (same shape as input).
    """
    B = obs_slice.shape[0]
    is_3d = term_info["is_3d"]
    hist = term_info["history_length"]
    dims = term_info["term_dims"]
    per_step_dims = term_info["term_dims_per_step"]
    types = term_info["term_types"]

    if is_3d:
        # [batch, history, per_step] - iterate over terms using per-step offsets
        per_step_idx = 0
        for dim, dim_ps, typ in zip(dims, per_step_dims, types):
            # term order from config is significant (sorted alphabetically)
            if dim_ps == 3 and typ == "a":
                # Angular velocity: [-1, 1, -1] across all history steps
                for h in range(hist):
                    obs_slice[:, h, per_step_idx : per_step_idx + dim_ps] *= _ANG_VEL_SIGN
                per_step_idx += dim_ps
            elif dim_ps == 3 and typ == "l":
                # Linear velocity: [1, -1, 1] across all history steps
                for h in range(hist):
                    obs_slice[:, h, per_step_idx : per_step_idx + dim_ps] *= _lin_vel_sign
                per_step_idx += dim_ps
            elif dim_ps == 3 and typ == "p":
                # Projected gravity: [1, -1, 1] across all history steps
                for h in range(hist):
                    obs_slice[:, h, per_step_idx : per_step_idx + dim_ps] *= _proj_grav_sign
                per_step_idx += dim_ps
            elif dim_ps == 3 and typ == "v":
                # Velocity commands: [1, -1, -1] across all history steps
                for h in range(hist):
                    obs_slice[:, h, per_step_idx : per_step_idx + dim_ps] *= _cmd_sign
                per_step_idx += dim_ps
            elif typ == "j":
                # Joint term: swap + flip across all history steps
                per_step_idx = _mirror_joint_term(
                    obs_slice, dim_ps, dim, hist, per_step_idx, flip_mask, swap_idx, True, B,
                )
            else:
                # Unknown term: skip (per_step_idx is already per-step offset)
                per_step_idx += dim_ps
    else:
        # [batch, flat] - iterate over terms
        idx = 0
        for dim, dim_ps, typ in zip(dims, per_step_dims, types):
            # term order from config is significant (sorted alphabetically)
            if dim_ps == 3 and typ == "a":
                # Angular velocity: [-1, 1, -1] per step
                for h in range(hist):
                    obs_slice[:, idx + h * dim_ps : idx + (h + 1) * dim_ps] *= _ANG_VEL_SIGN
                idx += dim
            elif dim_ps == 3 and typ == "l":
                # Linear velocity: [1, -1, 1] per step
                for h in range(hist):
                    obs_slice[:, idx + h * dim_ps : idx + (h + 1) * dim_ps] *= _lin_vel_sign
                idx += dim
            elif dim_ps == 3 and typ == "p":
                # Projected gravity: [1, -1, 1] per step
                for h in range(hist):
                    obs_slice[:, idx + h * dim_ps : idx + (h + 1) * dim_ps] *= _proj_grav_sign
                idx += dim
            elif dim_ps == 3 and typ == "v":
                # Velocity commands: [1, -1, -1] per step
                for h in range(hist):
                    obs_slice[:, idx + h * dim_ps : idx + (h + 1) * dim_ps] *= _cmd_sign
                idx += dim
            elif typ == "j":
                # Joint term: swap + flip
                idx = _mirror_joint_term(
                    obs_slice, dim_ps, dim, hist, idx, flip_mask, swap_idx, False, B,
                )
            else:
                # Unknown term: skip
                idx += dim

    return obs_slice


# --- Public API ---

@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Augments observations and actions with left-right symmetry.

    Creates 2x batch: first half keeps original observations, second half
    has left-right joint indices swapped with sign flips on roll/yaw joints.
    Follows legged_lab's term-by-term iteration pattern for correctness.

    Args:
        env: The environment instance.
        obs: Observation TensorDict with 'policy' and 'critic' groups.
        actions: Action tensor.

    Returns:
        Tuple of (augmented_obs, augmented_actions), or (None, None) if inputs were None.
    """
    # --- Actions ---
    if actions is not None:
        batch_size = actions.shape[0]
        device = actions.device
        n_joints = actions.shape[1]
        actions_aug = torch.zeros(batch_size * 2, n_joints, device=device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _mirror_actions(actions, n_joints, device)
    else:
        actions_aug = None
        batch_size = 0
        n_joints = 29

    # --- Observations ---
    if obs is None:
        return None, actions_aug

    batch_size = obs["policy"].shape[0]
    obs_aug = obs.repeat(2)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    flip_mask = _build_flip_mask(n_joints, device)
    swap_idx = torch.tensor(_SWAP_MAP, device=device, dtype=torch.long)

    # Detect observation structure from the observation manager
    policy_info = _get_obs_term_structure(env, obs_aug["policy"], "policy")

    # Mirror only the second half (original first half stays untouched)
    mirrored_policy = _mirror_obs(
        obs_aug["policy"][batch_size : 2 * batch_size].clone(),
        policy_info, flip_mask, swap_idx,
    )
    obs_aug["policy"][batch_size : 2 * batch_size] = mirrored_policy

    # Critic group
    if "critic" in obs_aug:
        critic_info = _get_obs_term_structure(env, obs_aug["critic"], "critic")
        mirrored_critic = _mirror_obs(
            obs_aug["critic"][batch_size : 2 * batch_size].clone(),
            critic_info, flip_mask, swap_idx,
        )
        obs_aug["critic"][batch_size : 2 * batch_size] = mirrored_critic

    return obs_aug, actions_aug


def _mirror_actions(actions: torch.Tensor, n_joints: int, device: torch.device) -> torch.Tensor:
    """Mirror actions by swapping left/right joints and flipping sign on roll/yaw."""
    mirrored = torch.zeros_like(actions)
    flip_mask = _build_flip_mask(n_joints, device)
    swap_idx = torch.tensor(_SWAP_MAP, device=device, dtype=torch.long)
    mirrored[:] = actions[:, swap_idx] * flip_mask
    return mirrored
