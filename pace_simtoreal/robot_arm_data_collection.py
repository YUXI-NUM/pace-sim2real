"""Collect dual-arm chirp data for PACE identification."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect dual-arm chirp data.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Robot-Arms-v0", help="Name of the task.")
parser.add_argument("--min_frequency", type=float, default=0.1, help="Minimum chirp frequency in Hz.")
parser.add_argument("--max_frequency", type=float, default=3.0, help="Maximum chirp frequency in Hz.")
parser.add_argument("--duration", type=float, default=20.0, help="Duration of the chirp signal in seconds.")
parser.add_argument("--amplitude_scale", type=float, default=1.0, help="Scale factor for all arm amplitudes.")
parser.add_argument("--limit_margin", type=float, default=0.05, help="Joint-limit clamp margin in radians.")
parser.add_argument(
    "--collision_threshold",
    type=float,
    default=5.0,
    help="Stop when contact force norm exceeds this threshold in N. Set <= 0 to disable.",
)
parser.add_argument(
    "--ignore_initial_seconds",
    type=float,
    default=0.25,
    help="Ignore contacts during the initial settling window.",
)
parser.add_argument("--no_stop_on_collision", action="store_true", help="Record collision info without stopping.")
parser.add_argument("--plot", action="store_true", help="Plot collected trajectories after saving.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch import pi

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root


def _find_joint_ids(articulation, joint_order, device):
    missing = [name for name in joint_order if name not in articulation.joint_names]
    if missing:
        raise ValueError(f"Task joint_order contains joints absent from the robot: {missing}")
    return torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=device)


def _update_actuator_params(articulation, joint_ids, armature, damping, friction, bias, time_lag, num_envs):
    env_ids = torch.arange(num_envs, device=joint_ids.device)
    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction

    for drive_type in articulation.actuators.keys():
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(len(articulation.joint_names), device=joint_ids.device)
            drive_indices = all_idx[drive_indices]
        drive_indices = torch.as_tensor(drive_indices, device=joint_ids.device)
        comparison_matrix = joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0)
        if not torch.all(torch.any(comparison_matrix, dim=0)):
            continue
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        articulation.actuators[drive_type].update_time_lags(time_lag)
        articulation.actuators[drive_type].update_encoder_bias(bias[:, drive_joint_idx])
        articulation.actuators[drive_type].reset(env_ids)


def _build_dual_arm_chirp(t, initial_pos, amplitude_scale, f0, f1, duration):
    phase = 2.0 * pi * (f0 * t + ((f1 - f0) / (2.0 * duration)) * t**2)
    amplitudes = torch.tensor(
        [0.30, 0.22, 0.25, 0.35, 0.30, 0.22, 0.25, 0.35],
        device=t.device,
    ) * amplitude_scale
    phase_offsets = torch.tensor(
        [0.0, 0.5 * pi, pi, 0.75 * pi, pi, 1.5 * pi, 0.0, 1.25 * pi],
        device=t.device,
    )
    return initial_pos.unsqueeze(0) + amplitudes.unsqueeze(0) * torch.sin(phase.unsqueeze(1) + phase_offsets.unsqueeze(0))


def _clamp_to_soft_limits(trajectory, articulation, joint_ids, margin):
    limits = getattr(articulation.data, "soft_joint_pos_limits", None)
    if limits is None:
        return trajectory
    joint_limits = limits[0, joint_ids, :].to(trajectory.device)
    lower = joint_limits[:, 0] + margin
    upper = joint_limits[:, 1] - margin
    valid = lower < upper
    trajectory[:, valid] = torch.max(torch.min(trajectory[:, valid], upper[valid]), lower[valid])
    return trajectory


def _get_contact_sensor(scene):
    try:
        return scene["contact_forces"]
    except Exception:
        return None


def _check_collision(contact_sensor, threshold):
    if contact_sensor is None or threshold <= 0.0:
        return False, 0.0, ""
    net_forces = getattr(contact_sensor.data, "net_forces_w", None)
    if net_forces is None:
        return False, 0.0, ""
    force_norms = torch.linalg.norm(net_forces, dim=-1)
    max_force, flat_index = torch.max(force_norms.reshape(-1), dim=0)
    max_force_value = float(max_force.item())
    if max_force_value <= threshold:
        return False, max_force_value, ""

    body_name = ""
    body_names = getattr(contact_sensor, "body_names", None)
    if body_names:
        body_index = int(flat_index.item() % force_norms.shape[-1])
        if body_index < len(body_names):
            body_name = body_names[body_index]
    return True, max_force_value, body_name


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    joint_ids = _find_joint_ids(articulation, joint_order, env.unwrapped.device)
    num_joints = len(joint_ids)
    num_envs = env.unwrapped.num_envs

    if num_envs != 1:
        print("[WARNING]: Data collection saves env 0 only. Use --num_envs 1 for hardware-like logs.")

    env.reset()

    armature = torch.full((num_envs, num_joints), 0.1, device=env.unwrapped.device)
    damping = torch.full((num_envs, num_joints), 4.5, device=env.unwrapped.device)
    friction = torch.full((num_envs, num_joints), 0.05, device=env.unwrapped.device)
    bias = torch.zeros((num_envs, num_joints), device=env.unwrapped.device)
    time_lag = torch.full((num_envs, 1), 5, dtype=torch.int, device=env.unwrapped.device)
    _update_actuator_params(articulation, joint_ids, armature, damping, friction, bias, time_lag, num_envs)

    data_dir = project_root() / "data" / env_cfg.sim2real.robot_name
    sample_rate = 1.0 / env.unwrapped.sim.get_physics_dt()
    num_steps = int(args_cli.duration * sample_rate)
    t = torch.arange(num_steps, device=env.unwrapped.device, dtype=torch.float32) / sample_rate

    initial_pos = articulation.data.default_joint_pos[0, joint_ids].to(env.unwrapped.device)
    trajectory = _build_dual_arm_chirp(
        t,
        initial_pos,
        args_cli.amplitude_scale,
        args_cli.min_frequency,
        args_cli.max_frequency,
        args_cli.duration,
    )
    trajectory = _clamp_to_soft_limits(trajectory, articulation, joint_ids, args_cli.limit_margin)

    articulation.write_joint_position_to_sim(trajectory[0, :].unsqueeze(0).repeat(num_envs, 1), joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim(torch.zeros((num_envs, num_joints), device=env.unwrapped.device), joint_ids=joint_ids)

    action_dim = env.action_space.shape[-1]
    dof_pos_buffer = torch.zeros(num_steps, num_joints, device=env.unwrapped.device)
    dof_target_pos_buffer = torch.zeros(num_steps, num_joints, device=env.unwrapped.device)
    max_contact_force_buffer = torch.zeros(num_steps, device=env.unwrapped.device)
    contact_sensor = _get_contact_sensor(env.unwrapped.scene)
    if contact_sensor is None and args_cli.collision_threshold > 0.0:
        print("[WARNING]: No contact_forces sensor found; collision stop is disabled.")

    collision_info = {
        "collided": False,
        "step": -1,
        "time": -1.0,
        "max_force": 0.0,
        "body": "",
        "threshold": args_cli.collision_threshold,
    }

    counter = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            dof_pos_buffer[counter, :] = articulation.data.joint_pos[0, joint_ids]
            actions = torch.zeros((num_envs, action_dim), device=env.unwrapped.device)
            actions[:, joint_ids] = trajectory[counter, :].unsqueeze(0).repeat(num_envs, 1)
            env.step(actions)
            dof_target_pos_buffer[counter, :] = trajectory[counter, :]

            collided, max_force, body_name = _check_collision(contact_sensor, args_cli.collision_threshold)
            max_contact_force_buffer[counter] = max_force
            check_time = float(t[counter].item())
            if collided and check_time >= args_cli.ignore_initial_seconds:
                collision_info.update(
                    {
                        "collided": True,
                        "step": counter,
                        "time": check_time,
                        "max_force": max_force,
                        "body": body_name,
                    }
                )
                print(
                    f"[WARNING]: Collision detected at {check_time:.3f}s "
                    f"(max force {max_force:.2f} N, body='{body_name}')."
                )
                counter += 1
                if not args_cli.no_stop_on_collision:
                    break
            else:
                counter += 1

            if counter % int(sample_rate) == 0:
                print(f"[INFO]: Step {counter / sample_rate:.1f} / {args_cli.duration:.1f} seconds")
            if counter >= num_steps:
                break

    env.close()

    data_dir.mkdir(parents=True, exist_ok=True)
    saved_steps = counter
    data = {
        "time": t[:saved_steps].cpu(),
        "dof_pos": dof_pos_buffer[:saved_steps].cpu(),
        "des_dof_pos": dof_target_pos_buffer[:saved_steps].cpu(),
        "joint_order": joint_order,
        "max_contact_force": max_contact_force_buffer[:saved_steps].cpu(),
        "collision_info": collision_info,
    }
    torch.save(data, data_dir / "chirp_data.pt")
    print(f"[INFO]: Saved {saved_steps} samples to {data_dir / 'chirp_data.pt'}")
    print(f"[INFO]: Collision info: {collision_info}")

    if args_cli.plot:
        import matplotlib.pyplot as plt

        for i, joint_name in enumerate(joint_order):
            plt.figure()
            plt.plot(data["time"].numpy(), data["dof_pos"][:, i].numpy(), label=f"{joint_name} pos")
            plt.plot(data["time"].numpy(), data["des_dof_pos"][:, i].numpy(), label=f"{joint_name} target", linestyle="dashed")
            plt.title(f"Joint {joint_name} Trajectory")
            plt.xlabel("Time [s]")
            plt.ylabel("Joint position [rad]")
            plt.grid()
            plt.legend()
            plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
    simulation_app.close()
