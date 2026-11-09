"""Shared GPU task hooks: identical MDP for Isaac Lab 2 and 3 adapters."""
from __future__ import annotations
import torch
from ..robot.jetbot_cfg import JetbotAssetCfg
from .torch_mdp import build_observation, process_actions, reward_components

class NavigationTaskHooks:
    def _initialize_navigation_buffers(self):
        self._wheel_joint_ids = JetbotAssetCfg().resolve_wheel_joint_ids(self.robot.joint_names)
        self._command_velocity = torch.zeros((self.num_envs, 2), device=self.device)
        self._previous_action = torch.zeros_like(self._command_velocity)
        self._applied_action = torch.zeros_like(self._command_velocity)
        self._wheel_targets = torch.zeros_like(self._command_velocity)
        self._goals_local = torch.tensor(
            self.cfg.layout.goal, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._last_step_end_reason = torch.zeros(
            self.num_envs, dtype=torch.int8, device=self.device
        )
        self._last_obstacle_force = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_step_position = torch.zeros((self.num_envs, 2), device=self.device)
        self._last_step_yaw = torch.zeros(self.num_envs, device=self.device)
        self._last_step_velocity = torch.zeros((self.num_envs, 2), device=self.device)
        self._last_lidar_distances = torch.zeros(
            (self.num_envs, self.cfg.navigation.lidar_rays), device=self.device
        )
        self._last_lidar_sectors = torch.zeros(
            (self.num_envs, self.cfg.navigation.lidar_sectors), device=self.device
        )
        self._last_applied_action = torch.zeros_like(self._applied_action)
        self._last_wheel_targets = torch.zeros_like(self._wheel_targets)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 2):
            raise ValueError(f"actions must have shape ({self.num_envs}, 2)")
        self._previous_action.copy_(self._applied_action)
        wheels, velocity, applied = process_actions(
            actions, self._command_velocity, self.cfg.navigation
        )
        self._wheel_targets.copy_(wheels)
        self._command_velocity.copy_(velocity)
        self._applied_action.copy_(applied)


    def _get_observations(self) -> dict:
        _, _, velocity, _, sectors, distance, sine, cosine = self._navigation_state()
        observation = build_observation(
            sectors,
            distance,
            sine,
            cosine,
            velocity,
            self._applied_action,
            self.cfg.navigation,
        )
        return {"policy": observation}


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        position, yaw, velocity, distances, sectors, distance, sine, cosine = self._navigation_state()
        force, collision = self._contact_state()
        outside = (
            (position[:, 0].abs() > self.cfg.navigation.room_width / 2.0
             + self.cfg.navigation.boundary_tolerance)
            | (position[:, 1].abs() > self.cfg.navigation.room_length / 2.0
               + self.cfg.navigation.boundary_tolerance)
        )
        success = distance <= self.cfg.navigation.goal_radius
        raw_timeout = self.episode_length_buf >= self.max_episode_length
        reason = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
        reason = torch.where(raw_timeout, torch.full_like(reason, 4), reason)
        reason = torch.where(outside, torch.full_like(reason, 3), reason)
        reason = torch.where(success, torch.full_like(reason, 2), reason)
        reason = torch.where(collision, torch.full_like(reason, 1), reason)

        self._step_sine, self._step_cosine = sine, cosine
        self._step_velocity = velocity
        self._step_sectors = sectors
        self._step_distance = distance
        self._step_reason = reason
        self._last_step_end_reason.copy_(reason)
        self._last_obstacle_force.copy_(force)
        self._last_collision.copy_(collision)
        self._last_step_position.copy_(position)
        self._last_step_yaw.copy_(yaw)
        self._last_step_velocity.copy_(velocity)
        self._last_lidar_distances.copy_(distances)
        self._last_lidar_sectors.copy_(sectors)
        self._last_applied_action.copy_(self._applied_action)
        self._last_wheel_targets.copy_(self._wheel_targets)
        terminated = collision | success | outside
        time_out = raw_timeout & ~terminated
        # Preserve the successor before DirectRLEnv resets ended environments.
        self.extras["terminal_observation"] = build_observation(
            sectors, distance, self._step_sine, self._step_cosine, velocity,
            self._applied_action, self.cfg.navigation).detach().clone()
        self.extras["terminal_observation_mask"] = (terminated | time_out).clone()
        self.extras["time_outs"] = time_out.clone()
        self.extras["end_reason"] = reason.clone()
        return terminated, time_out


    def _get_rewards(self) -> torch.Tensor:
        components = reward_components(
            self._previous_distance,
            self._step_distance,
            self._previous_action,
            self._applied_action,
            self._step_sectors,
            self._step_velocity[:, 0],
            self._step_reason,
            self.cfg.navigation,
        )
        self._previous_distance.copy_(self._step_distance)
        self.extras["reward_components"] = {k: v.detach().clone() for k, v in components.items()}
        return torch.stack(tuple(components.values()), dim=0).sum(dim=0)


    def diagnostic_tensors(self) -> dict[str, torch.Tensor | dict[str, str]]:
        """Snapshot for tests/UI only; callers decide if/when to copy to CPU."""
        position, yaw, velocity, distances, sectors, distance, sine, cosine = self._navigation_state()
        force, collision = self._contact_state()
        return {
            "position_local_m": position.detach().clone(),
            "yaw_rad": yaw.detach().clone(),
            "velocity": velocity.detach().clone(),
            "lidar_distance_m": distances.detach().clone(),
            "lidar_sector_m": sectors.detach().clone(),
            "goal_distance_m": distance.detach().clone(),
            "obstacle_force_N": force.detach().clone(),
            "collision": collision.detach().clone(),
            "last_obstacle_force_N": self._last_obstacle_force.detach().clone(),
            "last_collision": self._last_collision.detach().clone(),
            "last_position_local_m": self._last_step_position.detach().clone(),
            "last_yaw_rad": self._last_step_yaw.detach().clone(),
            "last_velocity": self._last_step_velocity.detach().clone(),
            "last_lidar_distance_m": self._last_lidar_distances.detach().clone(),
            "last_lidar_sector_m": self._last_lidar_sectors.detach().clone(),
            "applied_action": self._applied_action.detach().clone(),
            "wheel_target_radps": self._wheel_targets.detach().clone(),
            "last_applied_action": self._last_applied_action.detach().clone(),
            "last_wheel_target_radps": self._last_wheel_targets.detach().clone(),
            "last_end_reason": self._last_step_end_reason.detach().clone(),
            "resolved_body_paths": dict(self._resolved_body_paths),
        }
