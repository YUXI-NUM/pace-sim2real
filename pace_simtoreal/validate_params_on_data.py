"""Validate identified PACE parameters on a separate real trajectory."""

import argparse
import json
import re
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate PACE parameters on a held-out trajectory.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Robot-Arms-v0", help="Name of the Isaac Lab task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--log_dir", type=str, required=True, help="Training log directory containing config.pt and mean_*.pt.")
parser.add_argument("--mean_name", type=str, default=None, help="Parameter file name, e.g. mean_199.pt or best_params.pt. Defaults to latest mean.")
parser.add_argument("--data_file", type=str, required=True, help="Held-out PT trajectory file.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory for validation plots and tensors.")
parser.add_argument("--plot_bench_module", action="store_true", help="Also simulate the single-joint bench module parameters.")
parser.add_argument(
    "--gravity_mode",
    choices=("auto", "pt", "mujoco-current", "isaaclab-current", "none"),
    default="auto",
    help=(
        "Gravity effort source: auto/pt use data['gravity_torque']; "
        "isaaclab-current uses PhysX gravity compensation; mujoco-current computes from current simulated q."
    ),
)
parser.add_argument("--gravity_xml", type=str, default=None, help="MuJoCo XML used by --gravity_mode mujoco-current.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import pace_sim2real.tasks  # noqa: F401
from mujoco_gravity_utils import DEFAULT_ROBOT_XML, MujocoGravityProvider


_MEAN_PATTERN = re.compile(r"^mean_(\d+)\.pt$")


def _latest_mean_path(log_dir: Path) -> Path:
    best = None
    for path in log_dir.glob("mean_*.pt"):
        match = _MEAN_PATTERN.match(path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if best is None or iteration > best[0]:
            best = (iteration, path)
    if best is None:
        raise FileNotFoundError(f"No mean_*.pt files found under {log_dir}")
    return best[1]


def _safe_stem(path: Path) -> str:
    parent = path.parent.name
    grandparent = path.parent.parent.name
    dataset = path.parent.parent.parent.name
    return f"{dataset}_{grandparent}_{parent}_{path.stem}"


# NOTE: placeholder values only — replace with your own bench-identified
# actuator parameters (these are not measured data).
BENCH_MODULE_PARAMS = {
    "armature": 0.3,
    "viscous_friction": 2.0,
    "friction": 2.0,
    "dynamic_friction": 2.0,
    "encoder_bias": 0.0,
    "delay": 4.0,
}


def _set_external_effort(articulation, joint_ids: torch.Tensor, effort: torch.Tensor):
    for actuator in articulation.actuators.values():
        if not hasattr(actuator, "update_external_effort"):
            continue
        drive_indices = actuator.joint_indices
        if isinstance(drive_indices, slice):
            drive_indices = torch.arange(len(articulation.joint_names), device=joint_ids.device)[drive_indices]
        else:
            drive_indices = torch.as_tensor(drive_indices, device=joint_ids.device)
        actuator_effort = torch.zeros(
            (effort.shape[0], drive_indices.numel()),
            device=effort.device,
            dtype=effort.dtype,
        )
        for local_joint_idx, joint_id in enumerate(joint_ids):
            matches = torch.nonzero(drive_indices == joint_id, as_tuple=False).flatten()
            if matches.numel() > 0:
                actuator_effort[:, matches[0]] = effort[:, local_joint_idx]
        actuator.update_external_effort(actuator_effort)


def _apply_params(
    articulation,
    joint_ids: torch.Tensor,
    params: torch.Tensor,
    initial_position: torch.Tensor,
    dynamic_friction_value: float | None = None,
):
    """Apply a single identified parameter vector to all validation environments."""
    device = joint_ids.device
    num_envs = initial_position.shape[0]
    env_ids = torch.arange(num_envs, device=device)

    num_joints = joint_ids.numel()
    per_joint_timing = params.numel() in (15 * num_joints, 16 * num_joints)
    has_gravity_scale = params.numel() == 16 * num_joints
    armature = params[:num_joints].unsqueeze(0).repeat(num_envs, 1)
    damping = params[num_joints:2 * num_joints].unsqueeze(0).repeat(num_envs, 1)
    friction = params[2 * num_joints:3 * num_joints].unsqueeze(0).repeat(num_envs, 1)
    bias = params[3 * num_joints:4 * num_joints].unsqueeze(0).repeat(num_envs, 1)
    if per_joint_timing:
        delay = params[4 * num_joints:5 * num_joints].unsqueeze(0).repeat(num_envs, 1)
        directional_friction_pos = params[5 * num_joints:6 * num_joints].unsqueeze(0).repeat(num_envs, 1)
        directional_friction_neg = params[6 * num_joints:7 * num_joints].unsqueeze(0).repeat(num_envs, 1)
        directional_damping_pos = params[7 * num_joints:8 * num_joints].unsqueeze(0).repeat(num_envs, 1)
        directional_damping_neg = params[8 * num_joints:9 * num_joints].unsqueeze(0).repeat(num_envs, 1)
        high_speed_damping_pos = params[9 * num_joints:10 * num_joints].unsqueeze(0).repeat(num_envs, 1)
        high_speed_damping_neg = params[10 * num_joints:11 * num_joints].unsqueeze(0).repeat(num_envs, 1)
        positive_position_torque_gain = params[11 * num_joints:12 * num_joints].unsqueeze(0).repeat(num_envs, 1)
        negative_position_torque_gain = params[12 * num_joints:13 * num_joints].unsqueeze(0).repeat(num_envs, 1)
        high_speed_threshold_pos = params[13 * num_joints:14 * num_joints].unsqueeze(0).repeat(num_envs, 1)
        high_speed_threshold_neg = params[14 * num_joints:15 * num_joints].unsqueeze(0).repeat(num_envs, 1)
        gravity_scale = (
            params[15 * num_joints:16 * num_joints].unsqueeze(0).repeat(num_envs, 1)
            if has_gravity_scale
            else torch.zeros_like(bias)
        )
    else:
        # Legacy layout: one shared delay and shared high-speed thresholds for all matched joints.
        delay = int(params[4 * num_joints].item())
        has_directional_friction = params.numel() >= 6 * num_joints + 1
        has_directional_damping = params.numel() >= 8 * num_joints + 1
        has_high_speed_damping = params.numel() >= 10 * num_joints + 1
        has_positive_position_torque = params.numel() >= 11 * num_joints + 1
        has_negative_position_torque = params.numel() >= 12 * num_joints + 1
        has_high_speed_threshold_pos = params.numel() >= 12 * num_joints + 2
        has_high_speed_threshold_neg = params.numel() >= 12 * num_joints + 3
        directional_friction_pos = (
            params[4 * num_joints + 1:5 * num_joints + 1].unsqueeze(0).repeat(num_envs, 1)
            if has_directional_friction
            else torch.zeros_like(bias)
        )
        directional_friction_neg = (
            params[5 * num_joints + 1:6 * num_joints + 1].unsqueeze(0).repeat(num_envs, 1)
            if has_directional_friction
            else torch.zeros_like(bias)
        )
        directional_damping_pos = (
            params[6 * num_joints + 1:7 * num_joints + 1].unsqueeze(0).repeat(num_envs, 1)
            if has_directional_damping
            else torch.zeros_like(bias)
        )
        directional_damping_neg = (
            params[7 * num_joints + 1:8 * num_joints + 1].unsqueeze(0).repeat(num_envs, 1)
            if has_directional_damping
            else torch.zeros_like(bias)
        )
        high_speed_damping_pos = (
            params[8 * num_joints + 1:9 * num_joints + 1].unsqueeze(0).repeat(num_envs, 1)
            if has_high_speed_damping
            else torch.zeros_like(bias)
        )
        high_speed_damping_neg = (
            params[9 * num_joints + 1:10 * num_joints + 1].unsqueeze(0).repeat(num_envs, 1)
            if has_high_speed_damping
            else torch.zeros_like(bias)
        )
        positive_position_torque_gain = (
            params[10 * num_joints + 1:11 * num_joints + 1].unsqueeze(0).repeat(num_envs, 1)
            if has_positive_position_torque
            else torch.zeros_like(bias)
        )
        negative_position_torque_gain = (
            params[11 * num_joints + 1:12 * num_joints + 1].unsqueeze(0).repeat(num_envs, 1)
            if has_negative_position_torque
            else torch.zeros_like(bias)
        )
        high_speed_threshold_pos = None
        high_speed_threshold_neg = None
        gravity_scale = torch.zeros_like(bias)

    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction
    if dynamic_friction_value is not None:
        dynamic_friction = torch.full_like(friction, float(dynamic_friction_value))
        articulation.write_joint_dynamic_friction_coefficient_to_sim(
            dynamic_friction,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )
        articulation.data.default_joint_dynamic_friction_coeff[:, joint_ids] = dynamic_friction
    articulation.write_joint_position_to_sim(initial_position + bias, joint_ids=joint_ids, env_ids=env_ids)
    articulation.write_joint_velocity_to_sim(torch.zeros_like(initial_position), joint_ids=joint_ids, env_ids=env_ids)

    for actuator in articulation.actuators.values():
        if not hasattr(actuator, "update_encoder_bias") or not hasattr(actuator, "update_time_lags"):
            continue
        drive_indices = actuator.joint_indices
        if isinstance(drive_indices, slice):
            drive_indices = torch.arange(len(articulation.joint_names), device=device)[drive_indices]
        else:
            drive_indices = torch.as_tensor(drive_indices, device=device)

        actuator_bias = torch.zeros((num_envs, drive_indices.numel()), device=device, dtype=bias.dtype)
        actuator_directional_friction_pos = torch.zeros_like(actuator_bias)
        actuator_directional_friction_neg = torch.zeros_like(actuator_bias)
        actuator_directional_damping_pos = torch.zeros_like(actuator_bias)
        actuator_directional_damping_neg = torch.zeros_like(actuator_bias)
        actuator_high_speed_damping_pos = torch.zeros_like(actuator_bias)
        actuator_high_speed_damping_neg = torch.zeros_like(actuator_bias)
        actuator_positive_position_torque_gain = torch.zeros_like(actuator_bias)
        actuator_negative_position_torque_gain = torch.zeros_like(actuator_bias)
        actuator_high_speed_threshold_pos = torch.zeros_like(actuator_bias)
        actuator_high_speed_threshold_neg = torch.zeros_like(actuator_bias)
        actuator_gravity_scale = torch.zeros_like(actuator_bias)
        actuator_delay = None
        matched = False
        for local_joint_idx, joint_id in enumerate(joint_ids):
            matches = torch.nonzero(drive_indices == joint_id, as_tuple=False).flatten()
            if matches.numel() == 0:
                continue
            actuator_bias[:, matches[0]] = bias[:, local_joint_idx]
            actuator_directional_friction_pos[:, matches[0]] = directional_friction_pos[:, local_joint_idx]
            actuator_directional_friction_neg[:, matches[0]] = directional_friction_neg[:, local_joint_idx]
            actuator_directional_damping_pos[:, matches[0]] = directional_damping_pos[:, local_joint_idx]
            actuator_directional_damping_neg[:, matches[0]] = directional_damping_neg[:, local_joint_idx]
            actuator_high_speed_damping_pos[:, matches[0]] = high_speed_damping_pos[:, local_joint_idx]
            actuator_high_speed_damping_neg[:, matches[0]] = high_speed_damping_neg[:, local_joint_idx]
            actuator_positive_position_torque_gain[:, matches[0]] = positive_position_torque_gain[:, local_joint_idx]
            actuator_negative_position_torque_gain[:, matches[0]] = negative_position_torque_gain[:, local_joint_idx]
            if per_joint_timing:
                actuator_high_speed_threshold_pos[:, matches[0]] = high_speed_threshold_pos[:, local_joint_idx]
                actuator_high_speed_threshold_neg[:, matches[0]] = high_speed_threshold_neg[:, local_joint_idx]
                if actuator_delay is None:
                    actuator_delay = delay[:, local_joint_idx]
            actuator_gravity_scale[:, matches[0]] = gravity_scale[:, local_joint_idx]
            matched = True
        # Non-target actuators must not inherit optimized bias, delay, directional friction, or directional damping.
        actuator.update_encoder_bias(actuator_bias)
        if hasattr(actuator, "update_directional_friction"):
            actuator.update_directional_friction(actuator_directional_friction_pos, actuator_directional_friction_neg)
        if hasattr(actuator, "update_directional_damping"):
            actuator.update_directional_damping(actuator_directional_damping_pos, actuator_directional_damping_neg)
        if hasattr(actuator, "update_high_speed_damping"):
            actuator.update_high_speed_damping(actuator_high_speed_damping_pos, actuator_high_speed_damping_neg)
        if hasattr(actuator, "update_positive_position_torque_gain"):
            actuator.update_positive_position_torque_gain(actuator_positive_position_torque_gain)
        if hasattr(actuator, "update_negative_position_torque_gain"):
            actuator.update_negative_position_torque_gain(actuator_negative_position_torque_gain)
        if hasattr(actuator, "update_gravity_scale"):
            actuator.update_gravity_scale(actuator_gravity_scale)
        if per_joint_timing or has_high_speed_threshold_pos:
            if per_joint_timing:
                threshold_pos = actuator_high_speed_threshold_pos
                threshold_neg = actuator_high_speed_threshold_neg
            else:
                threshold_pos = params[12 * num_joints + 1].reshape(1, 1).repeat(num_envs, 1)
                threshold_neg = (
                    params[12 * num_joints + 2].reshape(1, 1).repeat(num_envs, 1)
                    if has_high_speed_threshold_neg
                    else threshold_pos
                )
            if hasattr(actuator, "update_high_speed_damping_velocity_thresholds"):
                actuator.update_high_speed_damping_velocity_thresholds(threshold_pos, threshold_neg)
            elif hasattr(actuator, "update_high_speed_damping_velocity_threshold"):
                actuator.update_high_speed_damping_velocity_threshold(threshold_pos)
        if per_joint_timing:
            actuator.update_time_lags(actuator_delay.to(torch.int) if matched else 0)
        else:
            actuator.update_time_lags(delay if matched else 0)
        actuator.reset(env_ids)

    return bias


def _simulate_trajectory(
    env,
    articulation,
    joint_ids: torch.Tensor,
    params: torch.Tensor,
    target_dof_pos: torch.Tensor,
    measured_dof_pos: torch.Tensor,
    gravity_torque: torch.Tensor | None = None,
    gravity_provider: MujocoGravityProvider | None = None,
    gravity_mode: str = "none",
    dynamic_friction_value: float | None = None,
):
    env.reset()
    stand_action = articulation.data.default_joint_pos.clone()
    all_joint_ids = torch.arange(len(articulation.joint_names), device=env.unwrapped.device)
    hold_joint_ids = all_joint_ids[~torch.isin(all_joint_ids, joint_ids)]
    hold_positions = stand_action[:, hold_joint_ids].clone()
    hold_velocities = torch.zeros_like(hold_positions)
    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    encoder_bias = _apply_params(
        articulation,
        joint_ids,
        params,
        initial_dof_pos,
        dynamic_friction_value=dynamic_friction_value,
    )
    if hold_joint_ids.numel() > 0:
        articulation.write_joint_position_to_sim(hold_positions, joint_ids=hold_joint_ids)
        articulation.write_joint_velocity_to_sim(hold_velocities, joint_ids=hold_joint_ids)

    sim_true = torch.zeros((target_dof_pos.shape[0], joint_ids.numel()), device=env.unwrapped.device)
    applied_gravity_torque = None
    if gravity_torque is not None or gravity_provider is not None or gravity_mode == "isaaclab-current":
        applied_gravity_torque = torch.zeros_like(sim_true)
    for counter in range(target_dof_pos.shape[0]):
        sim_true[counter, :] = articulation.data.joint_pos[0, joint_ids]
        actions = stand_action.clone()
        actions[:, joint_ids] = target_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
        if gravity_provider is not None:
            step_gravity = gravity_provider.compute(articulation.data.joint_pos[:, joint_ids])
            if applied_gravity_torque is not None:
                applied_gravity_torque[counter, :] = step_gravity[0, :]
            _set_external_effort(articulation, joint_ids, step_gravity)
        elif gravity_mode == "isaaclab-current":
            step_gravity = articulation.root_physx_view.get_gravity_compensation_forces()[:, joint_ids]
            if applied_gravity_torque is not None:
                applied_gravity_torque[counter, :] = step_gravity[0, :]
            _set_external_effort(articulation, joint_ids, step_gravity)
        elif gravity_torque is not None:
            step_gravity = gravity_torque[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            if applied_gravity_torque is not None:
                applied_gravity_torque[counter, :] = step_gravity[0, :]
            _set_external_effort(
                articulation,
                joint_ids,
                step_gravity,
            )
        env.step(actions)
        if hold_joint_ids.numel() > 0:
            articulation.write_joint_position_to_sim(hold_positions, joint_ids=hold_joint_ids)
            articulation.write_joint_velocity_to_sim(hold_velocities, joint_ids=hold_joint_ids)

    sim_encoder = sim_true - encoder_bias[0, :].unsqueeze(0)
    return sim_true, sim_encoder, encoder_bias, applied_gravity_torque


def _compute_metrics(sim_encoder: torch.Tensor, measured_dof_pos: torch.Tensor):
    error = sim_encoder - measured_dof_pos
    return {
        "error": error,
        "rmse": torch.sqrt(torch.mean(error ** 2, dim=0)),
        "mae": torch.mean(torch.abs(error), dim=0),
        "max_abs": torch.max(torch.abs(error), dim=0).values,
    }


def main():
    log_dir = Path(args_cli.log_dir).expanduser().resolve()
    data_file = Path(args_cli.data_file).expanduser().resolve()
    params_path = log_dir / args_cli.mean_name if args_cli.mean_name else _latest_mean_path(log_dir)
    output_dir = Path(args_cli.output_dir).expanduser().resolve() if args_cli.output_dir else log_dir / "validation" / _safe_stem(data_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    articulation = env.unwrapped.scene["robot"]

    config = torch.load(log_dir / "config.pt", map_location="cpu", weights_only=False)
    joint_order = config["joint_order"]
    joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device)

    params = torch.load(params_path, map_location=env.unwrapped.device, weights_only=False).flatten()
    bench_params = torch.tensor(
        [
            BENCH_MODULE_PARAMS["armature"],
            BENCH_MODULE_PARAMS["viscous_friction"],
            BENCH_MODULE_PARAMS["friction"],
            BENCH_MODULE_PARAMS["encoder_bias"],
            BENCH_MODULE_PARAMS["delay"],
        ],
        device=env.unwrapped.device,
        dtype=params.dtype,
    )
    data = torch.load(data_file, map_location=env.unwrapped.device, weights_only=False)
    data_joint_order = data.get("joint_order")
    if data_joint_order is not None and list(data_joint_order) != list(joint_order):
        raise ValueError(
            "Data joint_order does not match log config joint_order. "
            f"data={list(data_joint_order)}, config={list(joint_order)}"
        )
    time_data = data["time"].to(env.unwrapped.device)
    target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)
    gravity_torque = data.get("gravity_torque")
    if gravity_torque is not None:
        gravity_torque = gravity_torque.to(env.unwrapped.device)
        if gravity_torque.shape != measured_dof_pos.shape:
            raise ValueError(f"gravity_torque shape {gravity_torque.shape} does not match dof_pos {measured_dof_pos.shape}")
    gravity_provider = None
    gravity_mode = args_cli.gravity_mode
    if gravity_mode == "auto":
        gravity_mode = "pt" if gravity_torque is not None else "none"
    if gravity_mode == "pt" and gravity_torque is None:
        raise ValueError("--gravity_mode pt requested but data_file does not contain gravity_torque")
    if gravity_mode == "none":
        gravity_torque = None
    elif gravity_mode == "mujoco-current":
        gravity_torque = None
        gravity_provider = MujocoGravityProvider(args_cli.gravity_xml or DEFAULT_ROBOT_XML, joint_order)
        print(f"[INFO]: Computing gravity torque online from current simulated q using {gravity_provider.xml_path}")
    elif gravity_mode == "isaaclab-current":
        gravity_torque = None
        print("[INFO]: Computing gravity torque online from IsaacLab/PhysX current articulation state")

    if measured_dof_pos.shape[1] != len(joint_order):
        raise ValueError(f"Data has {measured_dof_pos.shape[1]} joints but log config has {len(joint_order)}: {joint_order}")

    sim_dt = env.unwrapped.sim.cfg.dt
    data_dt = torch.mean(torch.diff(time_data)).item()
    if abs(data_dt - sim_dt) > 1e-4:
        raise ValueError(f"Simulation dt ({sim_dt:.6f}s) does not match trajectory dt ({data_dt:.6f}s).")

    sim_true, sim_encoder, encoder_bias, applied_gravity_torque = _simulate_trajectory(
        env,
        articulation,
        joint_ids,
        params,
        target_dof_pos,
        measured_dof_pos,
        gravity_torque=gravity_torque,
        gravity_provider=gravity_provider,
        gravity_mode=gravity_mode,
    )
    identified_metrics = _compute_metrics(sim_encoder, measured_dof_pos)
    simulations = {
        "identified": {
            "sim_true": sim_true,
            "sim_encoder": sim_encoder,
            "encoder_bias": encoder_bias,
            "params": params,
            "metrics": identified_metrics,
        }
    }

    if args_cli.plot_bench_module:
        bench_true, bench_encoder, bench_bias, bench_applied_gravity_torque = _simulate_trajectory(
            env,
            articulation,
            joint_ids,
            bench_params,
            target_dof_pos,
            measured_dof_pos,
            gravity_torque=gravity_torque,
            gravity_provider=gravity_provider,
            gravity_mode=gravity_mode,
            dynamic_friction_value=BENCH_MODULE_PARAMS["dynamic_friction"],
        )
        simulations["bench_module"] = {
            "sim_true": bench_true,
            "sim_encoder": bench_encoder,
            "encoder_bias": bench_bias,
            "params": bench_params,
            "metrics": _compute_metrics(bench_encoder, measured_dof_pos),
            "bench_module_cfg": BENCH_MODULE_PARAMS,
        }

    result = {
        "task": args_cli.task,
        "log_dir": str(log_dir),
        "params_path": str(params_path),
        "data_file": str(data_file),
        "gravity_mode": gravity_mode,
        "gravity_xml": str(gravity_provider.xml_path) if gravity_provider is not None else None,
        "joint_order": joint_order,
        "params": params.detach().cpu().tolist(),
        "bench_module_params": BENCH_MODULE_PARAMS if args_cli.plot_bench_module else None,
        "dt": data_dt,
        "metrics": {
            name: {
                "rmse": item["metrics"]["rmse"].detach().cpu().tolist(),
                "mae": item["metrics"]["mae"].detach().cpu().tolist(),
                "max_abs": item["metrics"]["max_abs"].detach().cpu().tolist(),
            }
            for name, item in simulations.items()
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(
        {
            "time": time_data.detach().cpu(),
            "simulations": {
                name: {
                    "sim_true": item["sim_true"].detach().cpu(),
                    "sim_encoder": item["sim_encoder"].detach().cpu(),
                    "encoder_bias": item["encoder_bias"][0, :].detach().cpu(),
                    "params": item["params"].detach().cpu(),
                }
                for name, item in simulations.items()
            },
            "real": measured_dof_pos.detach().cpu(),
            "target": target_dof_pos.detach().cpu(),
            "gravity_torque": (
                applied_gravity_torque.detach().cpu()
                if applied_gravity_torque is not None
                else gravity_torque.detach().cpu() if gravity_torque is not None else None
            ),
            "gravity_mode": gravity_mode,
            "joint_order": joint_order,
        },
        output_dir / "validation_trajectory.pt",
    )

    time_cpu = time_data.detach().cpu().numpy()
    for i, joint_name in enumerate(joint_order):
        plt.figure(figsize=(10, 4.8))
        plt.plot(
            time_cpu,
            simulations["identified"]["sim_encoder"][:, i].detach().cpu().numpy(),
            label="Identified Sim",
            color="tab:orange",
            linewidth=2,
        )
        if "bench_module" in simulations:
            plt.plot(
                time_cpu,
                simulations["bench_module"]["sim_encoder"][:, i].detach().cpu().numpy(),
                label="Bench Module Sim",
                color="tab:blue",
                linewidth=1.8,
                alpha=0.9,
            )
        plt.plot(time_cpu, measured_dof_pos[:, i].detach().cpu().numpy(), label="Real", color="tab:green", linestyle="--", linewidth=2)
        plt.plot(time_cpu, target_dof_pos[:, i].detach().cpu().numpy(), label="Target", color="grey", linestyle="--", alpha=0.55)
        title = f"{joint_name} validation | Identified RMSE={identified_metrics['rmse'][i].item():.5f} rad"
        if "bench_module" in simulations:
            title += f" | Bench RMSE={simulations['bench_module']['metrics']['rmse'][i].item():.5f} rad"
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{joint_name}_validation.png", dpi=160)
        plt.close()

    print(f"[INFO]: Params: {params_path}")
    print(f"[INFO]: Data: {data_file}")
    for name, item in simulations.items():
        for i, joint_name in enumerate(joint_order):
            metrics = item["metrics"]
            print(
                f"[RESULT] {name} {joint_name}: RMSE={metrics['rmse'][i].item():.6f} rad, "
                f"MAE={metrics['mae'][i].item():.6f} rad, MaxAbs={metrics['max_abs'][i].item():.6f} rad"
            )
    print(f"[INFO]: Saved validation outputs to {output_dir}")
    env.close()


if __name__ == "__main__":
    main()