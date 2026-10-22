"""MuJoCo gravity torque helper for PACE arm trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    import mujoco
except ImportError:  # pragma: no cover - depends on the runtime environment
    mujoco = None


DEFAULT_ROBOT_XML = (
    Path(__file__).resolve().parents[2]
    / "source/pace_sim2real/pace_sim2real/assets/robot/mujoco/robot.xml"
)


class MujocoGravityProvider:
    """Compute generalized gravity torque for a selected joint order."""

    def __init__(self, xml_path: str | Path, joint_order: list[str]):
        if mujoco is None:
            raise ImportError("mujoco is required for real-time gravity torque computation")
        self.xml_path = Path(xml_path).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.qpos_addr, self.dof_addr = self._joint_addresses(joint_order)

    def _joint_addresses(self, joint_order: list[str]) -> tuple[list[int], list[int]]:
        qpos_addr = []
        dof_addr = []
        for joint_name in joint_order:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint {joint_name} not found in MuJoCo model {self.xml_path}")
            qpos_addr.append(int(self.model.jnt_qposadr[joint_id]))
            dof_addr.append(int(self.model.jnt_dofadr[joint_id]))
        return qpos_addr, dof_addr

    def compute(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Return gravity torque for joint_pos shaped (num_envs, num_joints) or (num_joints,)."""
        squeeze = joint_pos.ndim == 1
        if squeeze:
            joint_pos = joint_pos.unsqueeze(0)
        q_np = joint_pos.detach().cpu().numpy()
        gravity = np.zeros((q_np.shape[0], len(self.dof_addr)), dtype=np.float32)
        for env_idx in range(q_np.shape[0]):
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            self.data.qpos[3] = 1.0  # identity floating-base quaternion
            for local_idx, qpos_i in enumerate(self.qpos_addr):
                self.data.qpos[qpos_i] = q_np[env_idx, local_idx]
            mujoco.mj_forward(self.model, self.data)
            gravity[env_idx, :] = self.data.qfrc_bias[self.dof_addr]
        out = torch.as_tensor(gravity, device=joint_pos.device, dtype=joint_pos.dtype)
        return out.squeeze(0) if squeeze else out

    def compute_trajectory(self, q_traj: torch.Tensor) -> torch.Tensor:
        return self.compute(q_traj)
