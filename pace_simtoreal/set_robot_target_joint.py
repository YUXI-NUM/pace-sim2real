#!/usr/bin/env python3
"""Switch the one-joint PACE identification target in the env config."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = REPO_ROOT / "source/pace_sim2real/pace_sim2real/tasks/manager_based/pace/robot_pace_arm_env_cfg.py"


def _load_literal_assignment(module: ast.Module, name: str):
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise RuntimeError(f"Cannot find literal assignment for {name} in {CFG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("joint_name", help="Target joint name, e.g. left_wrist_roll_joint")
    parser.add_argument(
        "--data-root",
        help="Directory containing <joint_name>.pt. If omitted, keeps the current DEFAULT_DATA_ROOT.",
    )
    args = parser.parse_args()

    text = CFG_PATH.read_text()
    module = ast.parse(text)
    joint_index = _load_literal_assignment(module, "JOINT_INDEX")

    if args.joint_name not in joint_index:
        choices = "\n  ".join(joint_index)
        raise SystemExit(f"Unknown joint: {args.joint_name}\nAvailable joints:\n  {choices}")

    text = re.sub(
        r'^TARGET_JOINT_NAME = ".*"$',
        f'TARGET_JOINT_NAME = "{args.joint_name}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if args.data_root is not None:
        text = re.sub(
            r'^DEFAULT_DATA_ROOT = ".*"$',
            f'DEFAULT_DATA_ROOT = "{args.data_root}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )

    CFG_PATH.write_text(text)
    data_root = args.data_root or _load_literal_assignment(module, "DEFAULT_DATA_ROOT")
    print(f"Updated {CFG_PATH}")
    print(f"TARGET_JOINT_NAME = {args.joint_name}")
    print(f"robot_name = robot_arms_real_{joint_index[args.joint_name]}")
    print(f"default data_file = {data_root}/{args.joint_name}.pt")


if __name__ == "__main__":
    main()
