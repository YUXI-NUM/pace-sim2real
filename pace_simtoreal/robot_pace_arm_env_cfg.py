from isaaclab.utils import configclass

from pace_sim2real.robots.robot import ROBOT_CYLINDER_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from pace_sim2real.assets import ASSET_DIR
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch


# NOTE: placeholder standing pose only — replace every value with your own
# robot's calibrated joint angles before using this for anything but a demo.
ROBOT_STAND_JOINT_POS = {
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_pitch_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_pitch_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": -1.3,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_pitch_joint": -0.3,
    "left_elbow_yaw_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 1.3,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": 0.3,
    "right_elbow_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
}

TARGET_JOINT_NAME = "left_elbow_yaw_joint"
# NOTE: point this at your own collected-data directory (see robot_arm_data_collection.py).
# This is relative to <project_root>/data (see run_pace_identification.py's path resolution) — do not
# prefix it with "data/" or the path will resolve to <project_root>/data/data/...
DEFAULT_DATA_ROOT = "robot/run"
DEFAULT_ARMS_COMBINED_DATA = f"{DEFAULT_DATA_ROOT}/arms_combined.pt"

JOINT_INDEX = {
    "left_hip_pitch_joint": 0,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 2,
    "left_knee_pitch_joint": 3,
    "left_ankle_pitch_joint": 4,
    "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6,
    "right_hip_roll_joint": 7,
    "right_hip_yaw_joint": 8,
    "right_knee_pitch_joint": 9,
    "right_ankle_pitch_joint": 10,
    "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12,
    "waist_pitch_joint": 13,
    "left_shoulder_pitch_joint": 14,
    "left_shoulder_roll_joint": 15,
    "left_shoulder_yaw_joint": 16,
    "left_elbow_pitch_joint": 17,
    "left_elbow_yaw_joint": 18,
    "left_wrist_yaw_joint": 19,
    "left_wrist_roll_joint": 20,
    "right_shoulder_pitch_joint": 21,
    "right_shoulder_roll_joint": 22,
    "right_shoulder_yaw_joint": 23,
    "right_elbow_pitch_joint": 24,
    "right_elbow_yaw_joint": 25,
    "right_wrist_yaw_joint": 26,
    "right_wrist_roll_joint": 27,
    "head_yaw_joint": 28,
    "head_pitch_joint": 29,
}

# NOTE: generic motor-type labels (A/B/C) — these no longer encode real
# actuator torque ratings. Replace the per-type parameters below with your
# own actuator specs.
MOTOR_TYPE_A_JOINTS = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_pitch_joint",
    "right_knee_pitch_joint",
]
MOTOR_TYPE_B_JOINTS = [
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
]
MOTOR_TYPE_C_JOINTS = [
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
]
MOTOR_SHOULDER_PITCH_ROLL_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
]
MOTOR_ARMS_UPPER_JOINTS = [
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
]
MOTOR_ARMS_LOWER_JOINTS = [
    "left_elbow_yaw_joint",
    "right_elbow_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]
MOTOR_HEAD_JOINTS = [
    "head_yaw_joint",
    "head_pitch_joint",
]

ARM_JOINT_ORDER = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    # "right_shoulder_pitch_joint",
    # "right_shoulder_roll_joint",
    # "right_shoulder_yaw_joint",
    # "right_elbow_pitch_joint",
    # "right_elbow_yaw_joint",
    # "right_wrist_yaw_joint",
    # "right_wrist_roll_joint",
]


def _without_target(joints: list[str]) -> list[str]:
    return [joint for joint in joints if joint != TARGET_JOINT_NAME]


def _without_active(joints: list[str], active_joints: list[str]) -> list[str]:
    active_set = set(active_joints)
    return [joint for joint in joints if joint not in active_set]


def _only_active(joints: list[str], active_joints: list[str]) -> list[str]:
    active_set = set(active_joints)
    return [joint for joint in joints if joint in active_set]


def _motor_cfg(joints: list[str], **kwargs) -> PaceDCMotorCfg | None:
    if not joints:
        return None
    return PaceDCMotorCfg(joint_names_expr=joints, encoder_bias=0.0, max_delay=10, **kwargs)


def _active_motor_cfgs(active_joints: list[str]) -> dict[str, PaceDCMotorCfg]:
    return {
        f"active_{joint_name}": _motor_cfg([joint_name], **_target_kwargs(joint_name))
        for joint_name in active_joints
    }


def _target_kwargs(joint_name: str) -> dict:
    # NOTE: every value below is a placeholder (not measured data).
    # Replace with your own actuator's identified parameters.
    if joint_name in MOTOR_SHOULDER_PITCH_ROLL_JOINTS:
        return dict(
            saturation_effort=45.0,
            effort_limit=45.0,
            velocity_limit=5.0,
            stiffness={".*": 35.0},
            damping={".*": 5.0},
        )
    if joint_name in MOTOR_ARMS_UPPER_JOINTS:
        return dict(
            saturation_effort=45.0,
            effort_limit=45.0,
            velocity_limit=3.0,
            stiffness={".*": 60.0},
            damping={".*": 5.0},
        )
    if joint_name in MOTOR_ARMS_LOWER_JOINTS:
        return dict(
            saturation_effort=8.0,
            effort_limit=8.0,
            velocity_limit=2.0,
            stiffness={".*": 15.0},
            damping={".*": 2.0},
        )
    if joint_name in MOTOR_TYPE_A_JOINTS:
        return dict(
            saturation_effort=250.0,
            effort_limit=250.0,
            velocity_limit=10.0,
            stiffness={".*": 120.0},
            damping={".*": 20.0},
        )
    if joint_name in MOTOR_TYPE_B_JOINTS:
        return dict(
            saturation_effort=150.0,
            effort_limit=150.0,
            velocity_limit=10.0,
            stiffness={".*": 100.0},
            damping={".*": 10.0},
        )
    if joint_name in MOTOR_TYPE_C_JOINTS:
        return dict(
            saturation_effort=70.0,
            effort_limit=70.0,
            velocity_limit=10.0,
            stiffness={".*": 50.0},
            damping={".*": 1.0},
        )
    if joint_name in MOTOR_HEAD_JOINTS:
        return dict(
            saturation_effort=10.0,
            effort_limit=10.0,
            velocity_limit=2.0,
            stiffness={".*": 5.0},
            damping={".*": 2.0},
        )
    raise ValueError(f"Unsupported TARGET_JOINT_NAME: {joint_name}")


def _armature_bounds(joint_name: str) -> tuple[float, float]:
    if joint_name in MOTOR_ARMS_LOWER_JOINTS:
        return 0.01, 0.20
    if joint_name in MOTOR_ARMS_UPPER_JOINTS:
        return 0.05, 0.80
    return 0.15, 0.80


def _is_shoulder_pitch_roll(joint_name: str) -> bool:
    return "shoulder_pitch_joint" in joint_name or "shoulder_roll_joint" in joint_name


def _is_elbow_pitch(joint_name: str) -> bool:
    return "elbow_pitch_joint" in joint_name


def _is_waist_yaw(joint_name: str) -> bool:
    return joint_name == "waist_yaw_joint"


def _make_bounds_params(joint_order: list[str]) -> torch.Tensor:
    num_joints = len(joint_order)
    bounds_params = torch.zeros((16 * num_joints, 2))
    for joint_idx, joint_name in enumerate(joint_order):
        armature_lower, armature_upper = _armature_bounds(joint_name)
        bounds_params[joint_idx, 0] = armature_lower
        bounds_params[joint_idx, 1] = armature_upper

    bounds_params[num_joints:2 * num_joints, 0] = 0.0  # damping lower bound [Nm s/rad]
    bounds_params[num_joints:2 * num_joints, 1] = 5.0  # damping upper bound [Nm s/rad]

    bounds_params[2 * num_joints:3 * num_joints, 0] = 0.0  # symmetric friction lower bound
    bounds_params[2 * num_joints:3 * num_joints, 1] = 1.0  # symmetric friction upper bound

    bounds_params[3 * num_joints:4 * num_joints, 0] = 0.0
    bounds_params[3 * num_joints:4 * num_joints, 1] = 0.0  # bias currently fixed

    bounds_params[4 * num_joints:5 * num_joints, 0] = 0
    bounds_params[4 * num_joints:5 * num_joints, 1] = 0  # per-joint delay between 0.0 - 4.0 [sim steps]

    # Position torque gains are intentionally disabled; real-time gravity should carry posture load.
    bounds_params[11 * num_joints:12 * num_joints, :] = 0.0
    bounds_params[12 * num_joints:13 * num_joints, :] = 0.0

    for joint_idx, joint_name in enumerate(joint_order):
        is_shoulder_pr = _is_shoulder_pitch_roll(joint_name)
        is_elbow_pitch = _is_elbow_pitch(joint_name)
        is_waist_yaw = _is_waist_yaw(joint_name)

        # Keep direction-asymmetric Coulomb friction for all arm joints.
        bounds_params[5 * num_joints + joint_idx, 1] = 3.0  # positive-velocity friction [Nm]
        bounds_params[6 * num_joints + joint_idx, 1] = 3.0  # negative-velocity friction [Nm]

        # Low-speed directional damping is useful for load-bearing pitch/roll joints and waist yaw.
        if is_shoulder_pr or is_elbow_pitch or is_waist_yaw:
            bounds_params[7 * num_joints + joint_idx, 1] = 0.8  # positive damping [Nm s/rad]
            bounds_params[8 * num_joints + joint_idx, 1] = 3.0  # negative damping [Nm s/rad]

        # High-speed damping and its transition are reserved for shoulder pitch/roll.
        if is_shoulder_pr:
            bounds_params[9 * num_joints + joint_idx, 1] = 8.0  # positive high-speed damping [Nm s/rad]
            bounds_params[10 * num_joints + joint_idx, 1] = 8.0  # negative high-speed damping [Nm s/rad]
            bounds_params[13 * num_joints + joint_idx, 1] = 3.0  # positive threshold [rad/s]
            bounds_params[14 * num_joints + joint_idx, 1] = 3.0  # negative threshold [rad/s]

        # Gravity scale is useful for pitch/roll load-bearing joints and elbow pitch only.
        if is_shoulder_pr:
            bounds_params[15 * num_joints + joint_idx, 1] = 1.2
        elif is_elbow_pitch:
            bounds_params[15 * num_joints + joint_idx, 1] = 0.8
    return bounds_params


TARGET_MOTOR_CFG = _motor_cfg([TARGET_JOINT_NAME], **_target_kwargs(TARGET_JOINT_NAME))
# NOTE: held-joint values below are placeholders (not measured data).
MOTOR_TYPE_A_CFG = _motor_cfg(
    _without_target(MOTOR_TYPE_A_JOINTS),
    saturation_effort=120.0,
    effort_limit=120.0,
    velocity_limit=10.0,
    stiffness={".*": 100.0},
    damping={".*": 1.0},
)
MOTOR_TYPE_B_CFG = _motor_cfg(
    _without_target(MOTOR_TYPE_B_JOINTS),
    saturation_effort=50.0,
    effort_limit=50.0,
    velocity_limit=10.0,
    stiffness={".*": 100.0},
    damping={".*": 1.0},
)
MOTOR_TYPE_C_CFG = _motor_cfg(
    _without_target(MOTOR_TYPE_C_JOINTS),
    saturation_effort=25.0,
    effort_limit=25.0,
    velocity_limit=10.0,
    stiffness={".*": 50.0},
    damping={".*": 1.0},
)
MOTOR_SHOULDER_PITCH_ROLL_CFG = _motor_cfg(
    _without_target(MOTOR_SHOULDER_PITCH_ROLL_JOINTS),
    saturation_effort=40.0,
    effort_limit=40.0,
    velocity_limit=7.0,
    stiffness={".*": 100.0},
    damping={".*": 1.0},
)
MOTOR_ARMS_UPPER_CFG = _motor_cfg(
    _without_target(MOTOR_ARMS_UPPER_JOINTS),
    **_target_kwargs("left_shoulder_yaw_joint"),
)
MOTOR_ARMS_LOWER_CFG = _motor_cfg(
    _without_target(MOTOR_ARMS_LOWER_JOINTS),
    **_target_kwargs("left_elbow_yaw_joint"),
)
MOTOR_HEAD_CFG = _motor_cfg(
    _without_target(MOTOR_HEAD_JOINTS),
    **_target_kwargs("head_yaw_joint"),
)


@configclass
class RobotDPaceCfg(PaceCfg):
    """Pace configuration for the target robot."""
    robot_name: str = f"robot_arms_real_{JOINT_INDEX[TARGET_JOINT_NAME]}"
    data_dir: str = f"{DEFAULT_DATA_ROOT}/{TARGET_JOINT_NAME}.pt"
    bounds_params: torch.Tensor = torch.zeros((15, 2))  # 1 dof nonlinear motor model with directional high-speed thresholds
    joint_order: list[str] = [TARGET_JOINT_NAME]

    def __post_init__(self):
        self.bounds_params = _make_bounds_params(self.joint_order)


@configclass
class RobotArmsCombinedDPaceCfg(RobotDPaceCfg):
    """Pace configuration for coupled dual-arm trajectories."""
    robot_name: str = "robot_arms_combined"
    data_dir: str = DEFAULT_ARMS_COMBINED_DATA
    bounds_params: torch.Tensor = torch.zeros((16 * len(ARM_JOINT_ORDER), 2))
    joint_order: list[str] = ARM_JOINT_ORDER


def _make_robot_cfg(active_joints: list[str]) -> ArticulationCfg:
    robot_cfg = ROBOT_CYLINDER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos=ROBOT_STAND_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        actuators={
            name: cfg
            for name, cfg in {
                **_active_motor_cfgs(active_joints),
                "motor_type_a": _motor_cfg(
                    _without_active(MOTOR_TYPE_A_JOINTS, active_joints),
                    saturation_effort=120.0,
                    effort_limit=120.0,
                    velocity_limit=10.0,
                    stiffness={".*": 100.0},
                    damping={".*": 1.0},
                ),
                "motor_type_b": _motor_cfg(
                    _without_active(MOTOR_TYPE_B_JOINTS, active_joints),
                    saturation_effort=50.0,
                    effort_limit=50.0,
                    velocity_limit=10.0,
                    stiffness={".*": 100.0},
                    damping={".*": 1.0},
                ),
                "motor_type_c": _motor_cfg(
                    _without_active(MOTOR_TYPE_C_JOINTS, active_joints),
                    saturation_effort=25.0,
                    effort_limit=25.0,
                    velocity_limit=10.0,
                    stiffness={".*": 50.0},
                    damping={".*": 1.0},
                ),
                "motor_shoulder_pitch_roll": _motor_cfg(
                    _without_active(MOTOR_SHOULDER_PITCH_ROLL_JOINTS, active_joints),
                    saturation_effort=40.0,
                    effort_limit=40.0,
                    velocity_limit=7.0,
                    stiffness={".*": 100.0},
                    damping={".*": 1.0},
                ),
                "motor_arms_upper": _motor_cfg(
                    _without_active(MOTOR_ARMS_UPPER_JOINTS, active_joints),
                    **_target_kwargs("left_shoulder_yaw_joint"),
                ),
                "motor_arms_lower": _motor_cfg(
                    _without_active(MOTOR_ARMS_LOWER_JOINTS, active_joints),
                    **_target_kwargs("left_elbow_yaw_joint"),
                ),
                "motor_head": _motor_cfg(
                    _without_active(MOTOR_HEAD_JOINTS, active_joints),
                    **_target_kwargs("head_yaw_joint"),
                ),
            }.items()
            if cfg is not None
        },
    )
    robot_cfg.spawn.asset_path = f"{ASSET_DIR}/robot/robot_30dof.urdf"
    robot_cfg.spawn.fix_base = True
    robot_cfg.spawn.articulation_props.enabled_self_collisions = False
    return robot_cfg


def _make_arm_robot_cfg() -> ArticulationCfg:
    return _make_robot_cfg([TARGET_JOINT_NAME])


@configclass
class RobotPaceSceneCfg(PaceSim2realSceneCfg):
    robot: ArticulationCfg = _make_arm_robot_cfg()
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )


@configclass
class RobotPaceEnvCfg(PaceSim2realEnvCfg):

    scene: RobotPaceSceneCfg = RobotPaceSceneCfg()
    sim2real: PaceCfg = RobotDPaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.005  # 200Hz simulation, matching the real robot CSV data
        self.decimation = 1  # 200Hz control
        self.episode_length_s = 30.0
        self.sim2real.cmaes.max_iteration = 200
        self.sim2real.cmaes.epsilon = 1e-4
        self.sim2real.cmaes.sigma = 0.3
        self.sim2real.cmaes.save_interval = 10
        self.sim2real.cmaes.save_optimization_process = True


@configclass
class RobotArmsCombinedSceneCfg(PaceSim2realSceneCfg):
    robot: ArticulationCfg = _make_robot_cfg(ARM_JOINT_ORDER)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )


@configclass
class RobotArmsCombinedEnvCfg(RobotPaceEnvCfg):
    scene: RobotArmsCombinedSceneCfg = RobotArmsCombinedSceneCfg()
    sim2real: PaceCfg = RobotArmsCombinedDPaceCfg()