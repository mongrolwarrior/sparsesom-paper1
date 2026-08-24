"""Config resolver: defaults ← experiment override → frozen hash."""

import hashlib
import json
import copy
from pathlib import Path

import yaml


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_defaults() -> dict:
    with open(CONFIGS_DIR / "defaults.yaml") as f:
        return yaml.safe_load(f)


def load_experiment(experiment_id: str) -> dict:
    for p in sorted((CONFIGS_DIR / "experiments").glob("*.yaml")):
        with open(p) as f:
            cfg = yaml.safe_load(f)
        if cfg.get("id") == experiment_id:
            return cfg
    raise FileNotFoundError(f"No experiment config with id={experiment_id}")


def resolve(experiment_id: str) -> dict:
    """Merge defaults ← experiment and add a config_hash."""
    cfg = load_defaults()
    cfg = _deep_merge(cfg, load_experiment(experiment_id))
    cfg["config_hash"] = config_hash(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    """Deterministic hash of the resolved config (excludes the hash field itself)."""
    c = {k: v for k, v in cfg.items() if k != "config_hash"}
    blob = json.dumps(c, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
