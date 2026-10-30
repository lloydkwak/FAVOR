"""Config loading: one yaml tree (config/default.yaml) plus optional override files and
dotted --set overrides. Paths inside the tree are resolved relative to sim_xarm/."""
import copy
import os

import yaml

SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(SIM_ROOT, "config", "default.yaml")


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg, key, value):
    d = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def load_config(paths=None, overrides=None):
    """paths: list of yaml files merged in order (default.yaml first if not given).
    overrides: list of 'a.b.c=value' strings (value parsed as yaml)."""
    paths = list(paths or [])
    if not paths or os.path.abspath(paths[0]) != os.path.abspath(DEFAULT_CONFIG):
        paths = [DEFAULT_CONFIG] + paths
    cfg = {}
    for p in paths:
        with open(p) as f:
            cfg = _merge(cfg, yaml.safe_load(f))
    for o in overrides or []:
        k, v = o.split("=", 1)
        set_dotted(cfg, k.strip(), yaml.safe_load(v))
    return cfg


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(SIM_ROOT, path)


def add_cli(ap):
    ap.add_argument("--config", action="append", default=[], help="extra yaml merged over default.yaml")
    ap.add_argument("--set", action="append", default=[], help="dotted override, e.g. control.mode=online")
    return ap


def from_args(args):
    return load_config(args.config, args.set)
