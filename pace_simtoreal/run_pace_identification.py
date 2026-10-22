"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Robot-YX-left-shoulder-pitch-joint-v0", help="Name of the task.")
parser.add_argument("--data_file", type=str, default=None, help="Optional PT trajectory file overriding the task config.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root
from pace_sim2real import CMAESOptimizer


def _set_hold_gains(articulation, sim_joint_ids: torch.Tensor, hold_stiffness: float, hold_damping: float):
    """Set high PD gains on every joint that is NOT being identified.

    DCMotor is an explicit actuator whose gains live in actuator.stiffness / actuator.damping
    (shape: num_envs × num_joints_in_actuator).  Writing to those tensors directly takes effect
    on the next control step and persists across episode resets.
    """
    for actuator in articulation.actuators.values():
        j_ids = actuator.joint_indices
        if isinstance(j_ids, slice):
            j_ids = torch.arange(articulation.num_joints, device=articulation.device)[j_ids]
        # local column indices within this actuator that are NOT the identified joint
        is_hold = ~torch.isin(j_ids, sim_joint_ids)
        hold_local = torch.where(is_hold)[0]
        if hold_local.numel() > 0:
            actuator.stiffness[:, hold_local] = hold_stiffness
            actuator.damping[:, hold_local] = hold_damping


def main():
    """Zero actions agent with Isaac Lab environment."""
    if args_cli.num_envs < 2:
        raise ValueError("CMA-ES requires --num_envs >= 2 because num_envs is used as the population size.")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Create optimization
    bounds_params = env_cfg.sim2real.bounds_params.to(env.unwrapped.device)
    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    sim_joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device)

    if args_cli.data_file is not None:
        data_file = Path(args_cli.data_file).expanduser()
    else:
        configured_data = Path(env_cfg.sim2real.data_dir)
        data_file = configured_data if configured_data.is_absolute() else project_root() / "data" / configured_data
    log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name

    # data = torch.load(data_file)
    data = torch.load(data_file, weights_only=False)
    time_data = data["time"].to(env.unwrapped.device)
    target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)

    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

    time_steps = time_data.shape[0]
    sim_dt = env.unwrapped.sim.cfg.dt
    data_dt = torch.mean(torch.diff(time_data)).item()
    if abs(data_dt - sim_dt) > 1e-4:
        raise ValueError(f"Simulation dt ({sim_dt:.6f}s) does not match trajectory dt ({data_dt:.6f}s).")
    print(f"[INFO]: Trajectory dt={data_dt:.6f}s ({1.0 / data_dt:.1f}Hz), sim dt={sim_dt:.6f}s")

    opt = CMAESOptimizer(
        bounds=bounds_params,
        population_size=env.unwrapped.num_envs,
        log_dir=log_dir,
        joint_order=joint_order,
        max_iteration=env_cfg.sim2real.cmaes.max_iteration,
        data=data,
        device=env.unwrapped.device,
        epsilon=env_cfg.sim2real.cmaes.epsilon,
        sigma=env_cfg.sim2real.cmaes.sigma,
        save_interval=env_cfg.sim2real.cmaes.save_interval,
        save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process,
    )

    # Build A-pose action tensor for all joints (non-identified joints hold A-pose)
    a_pose_actions = torch.zeros(env.unwrapped.num_envs, len(articulation.joint_names), device=env.unwrapped.device)
    for joint_name, angle in env_cfg.sim2real.a_pose_angles.items():
        if joint_name in articulation.joint_names:
            idx = articulation.joint_names.index(joint_name)
            a_pose_actions[:, idx] = angle

    # Pre-compute hold joint IDs and their A-pose positions/velocities
    all_joint_ids_t = torch.arange(len(articulation.joint_names), device=env.unwrapped.device)
    hold_joint_ids = all_joint_ids_t[~torch.isin(all_joint_ids_t, sim_joint_ids)]
    hold_positions = a_pose_actions[:, hold_joint_ids]  # (num_envs, num_hold_joints)
    hold_zeros = torch.zeros_like(hold_positions)

    env.reset()
    opt.update_simulator(articulation, sim_joint_ids, initial_dof_pos)

    counter = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            opt.tell(env.unwrapped.scene.articulations["robot"].data.joint_pos[:, sim_joint_ids], measured_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1))
            # non-identified joints hold A-pose; identified joint follows chirp target
            actions = a_pose_actions.clone()
            actions[:, sim_joint_ids] = target_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # apply actions
            env.step(actions)
            # Hard-lock non-identified joints: overwrite physics result with exact A-pose.
            # This bypasses effort limits entirely, matching real hardware mechanical locking.
            articulation.write_joint_position_to_sim(hold_positions, joint_ids=hold_joint_ids)
            articulation.write_joint_velocity_to_sim(hold_zeros, joint_ids=hold_joint_ids)
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter * sim_dt:.1f} / {time_data[-1]:.1f} seconds ({counter / time_steps * 100:.1f} %)")
            if counter >= time_steps:
                print("[INFO]: Reached the end of the trajectory, exiting.")
                counter = 0
                opt.evolve()
                if opt.finished():
                    break
                env.reset()
                opt.update_simulator(env.unwrapped.scene["robot"], sim_joint_ids, initial_dof_pos)
                articulation.write_joint_position_to_sim(hold_positions, joint_ids=hold_joint_ids)
                articulation.write_joint_velocity_to_sim(hold_zeros, joint_ids=hold_joint_ids)
    # close optimizer
    opt.close()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
