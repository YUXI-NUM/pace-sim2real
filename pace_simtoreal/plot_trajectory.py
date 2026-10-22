import torch
import matplotlib.pyplot as plt
import re
from pathlib import Path

import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--folder_name", type=str, default=None, help="Name of the folder to use.")
parser.add_argument("--mean_name", type=str, default=None, help="Name of the parameters file to use.")
parser.add_argument("--robot_name", type=str, default="anymal_d_sim", help="Name of the robot.")
parser.add_argument("--plot_trajectory", action="store_true", help="Whether to plot the trajectory.")
parser.add_argument("--plot_score", action="store_true", help="Whether to plot the score over iterations.")

args = parser.parse_args()
folder_name = args.folder_name
mean_name = args.mean_name
robot_name = args.robot_name
plot_trajectory = args.plot_trajectory
plot_score = args.plot_score

current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent

# folder_name = "25_10_24_12-05-07"
log_dir = project_root / "logs" / "pace" / robot_name

if not log_dir.exists():
    raise FileNotFoundError(f"No logs for robot {robot_name} under {log_dir}")

_pattern = re.compile(r"^mean_(\d+)\.pt$")


def find_latest_params(root: Path):
    best = None  # tuple (int, Path)
    for p in root.rglob("mean_*.pt"):
        m = _pattern.match(p.name)
        if not m:
            continue
        num = int(m.group(1))
        if best is None or num > best[0]:
            best = (num, p)
    return None if best is None else best[1], best[0]


# if no folder_name given, pick the most recent run folder for the robot
if not folder_name:
    robot_dir = log_dir
    candidates = [p for p in robot_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run folders found under {robot_dir}")
    latest_folder = max(candidates, key=lambda p: p.stat().st_mtime)
    folder_name = latest_folder.name

# now point log_dir at the chosen robot/run folder
log_dir = log_dir / folder_name

if mean_name is None:
    params_path, params_num = find_latest_params(log_dir)
else:
    params_path = log_dir / mean_name
    params_num = int(_pattern.match(mean_name).group(1))
    if not params_path.exists():
        raise FileNotFoundError(f"Given params file {params_path} does not exist")
if params_path is None:
    raise FileNotFoundError(f"No mean_*.pt files found under {log_dir}")
print(f"Latest params file: {params_path}")

mean = torch.load(params_path)
config = torch.load(log_dir / "config.pt")

joint_order = config["joint_order"]
trajectories = torch.load(log_dir / "best_trajectory.pt")  # time x joints
real_trajectories = config["dof_pos"]  # time x joints
target_trajectories = config["des_dof_pos"]  # time x joints
time = config["time"]  # time

print(f"Best parameter set: {mean}")
print(f"Armature params: {mean[:len(joint_order)]}")
print(f"Viscous friction params: {mean[len(joint_order):2 * len(joint_order)]}")
print(f"Static friction params: {mean[2 * len(joint_order):3 * len(joint_order)]}")
print(f"Encoder bias params: {mean[3 * len(joint_order):4 * len(joint_order)]}")
delay_idx = 4 * len(joint_order)
print(f"Delay param: {mean[delay_idx].item()}")
has_directional_friction = mean.numel() >= 6 * len(joint_order) + 1
has_directional_damping = mean.numel() >= 8 * len(joint_order) + 1
has_high_speed_damping = mean.numel() >= 10 * len(joint_order) + 1
has_positive_position_torque = mean.numel() >= 11 * len(joint_order) + 1
has_negative_position_torque = mean.numel() >= 12 * len(joint_order) + 1
has_high_speed_threshold_pos = mean.numel() >= 12 * len(joint_order) + 2
has_high_speed_threshold_neg = mean.numel() >= 12 * len(joint_order) + 3
if has_directional_friction:
    print(f"Directional positive-velocity friction params: {mean[delay_idx + 1:5 * len(joint_order) + 1]}")
    print(f"Directional negative-velocity friction params: {mean[5 * len(joint_order) + 1:6 * len(joint_order) + 1]}")
if has_directional_damping:
    print(f"Directional positive-velocity damping params: {mean[6 * len(joint_order) + 1:7 * len(joint_order) + 1]}")
    print(f"Directional negative-velocity damping params: {mean[7 * len(joint_order) + 1:8 * len(joint_order) + 1]}")
if has_high_speed_damping:
    print(f"High-speed positive-velocity damping params: {mean[8 * len(joint_order) + 1:9 * len(joint_order) + 1]}")
    print(f"High-speed negative-velocity damping params: {mean[9 * len(joint_order) + 1:10 * len(joint_order) + 1]}")
if has_positive_position_torque:
    print(f"Positive-position torque gain params: {mean[10 * len(joint_order) + 1:11 * len(joint_order) + 1]}")
if has_negative_position_torque:
    print(f"Negative-position torque gain params: {mean[11 * len(joint_order) + 1:12 * len(joint_order) + 1]}")
if has_high_speed_threshold_pos:
    print(f"High-speed damping positive-velocity threshold param: {mean[12 * len(joint_order) + 1].item()}")
if has_high_speed_threshold_neg:
    print(f"High-speed damping negative-velocity threshold param: {mean[12 * len(joint_order) + 2].item()}")
if not has_directional_friction and mean.numel() > delay_idx + 1:
    print(f"Extra params after delay: {mean[delay_idx + 1:]}")
encoder_bias = mean[3 * len(joint_order):4 * len(joint_order)]  # extract encoder bias
best_params_path = log_dir / "best_params.pt"
if best_params_path.exists():
    trajectory_params = torch.load(best_params_path)
    trajectory_encoder_bias = trajectory_params[3 * len(joint_order):4 * len(joint_order)]
    print(f"Cached trajectory parameter set: {trajectory_params}")
else:
    trajectory_encoder_bias = encoder_bias
    print(
        "[WARNING] best_params.pt is missing. This older log did not save the "
        "population candidate associated with best_trajectory.pt. Falling back "
        "to the mean encoder bias."
    )

if plot_score:
    try:
        progress = torch.load(log_dir / "progress.pt")
        print("Loaded optimization progress.")
    except FileNotFoundError:
        progress = None
        print("No optimization progress file found. Skipping score plot.")
        plot_score = False

if plot_score:
    plt.figure()
    plt.title("CMA-ES Score over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    data = torch.min(progress["scores_buffer"][:params_num + 1], dim=1).values.cpu().numpy()
    plt.semilogy(data)
    plt.xlim(0, params_num)
    # plt.ylim(0, None)
    plt.grid()
    plt.show()

if plot_trajectory:
    for i in range(len(joint_order)):
        plt.figure(figsize=(8, 4.5))
        plt.plot(time, trajectories[:, i].cpu().numpy() - trajectory_encoder_bias[i].item(), c="tab:orange", label="Sim", linewidth=2)  # in encoder frame
        plt.plot(time, real_trajectories[:, i].cpu().numpy(), label="Real", c="tab:green", linestyle="--", linewidth=2)
        plt.plot(time, target_trajectories[:, i].cpu().numpy(), c="grey", label="Target", linestyle="--", alpha=0.5)
        plt.title(f"Joint {joint_order[i]}")  # Use joint names from config
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position [rad]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

print("Plotting complete.")
