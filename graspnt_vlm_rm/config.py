from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def require_keys(config: dict[str, Any], dotted_keys: list[str]) -> None:
    for dotted_key in dotted_keys:
        cursor: Any = config
        for part in dotted_key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                raise KeyError(dotted_key)
            cursor = cursor[part]


def validate_runtime_config(config: dict[str, Any]) -> None:
    require_keys(
        config,
        [
            "graspnet.root",
            "graspnet.checkpoint",
            "hand_eye.rotation",
            "hand_eye.translation",
            "safety.gripper_length",
            "safety.min_grasp_z",
            "vlm.endpoint",
            "vlm.model",
        ],
    )
