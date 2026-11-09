"""Re-save SB3 checkpoints whose ScaledObservations class was pickled as bytecode.

Checkpoints trained before ScaledObservations moved to module level store the
class as Python bytecode, so they only load on the Python version that saved
them (other versions segfault). This script loads each checkpoint without
unpickling that class, re-saves it so the class is stored by import path, and
checks that the re-saved model gives identical deterministic actions.

Usage: python scripts/resave_rl_checkpoints.py path/to/policy.zip [...]
"""
import base64
import importlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO, SAC

ALGORITHMS = {"ppo": PPO, "sac": SAC, "dqn": DQN}


def metadata(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))


def resave(path):
    data = metadata(path)
    kwargs = data["policy_kwargs"]
    extractor = kwargs.get("features_extractor_class", "")
    payload = base64.b64decode(kwargs.get(":serialized:", ""))
    # cloudpickle rebuilds by-value classes with _make_skeleton_class; the readable
    # label alone is not enough (resumed SAC runs show a clean label but bytecode).
    if b"_make_skeleton_class" not in payload and "<locals>" not in extractor:
        print(f"skip   {path}: class already stored by import path")
        return
    # e.g. "<class 'pendulum_lab.algorithms.dqn.train.<locals>.ScaledObservations'>"
    module_name = extractor.split("'")[1].replace(".train.<locals>", "").rsplit(".", 1)[0]
    name = module_name.rsplit(".", 1)[1]
    cls = ALGORITHMS[name]
    scaled = importlib.import_module(module_name).ScaledObservations
    readable = {k: v for k, v in kwargs.items() if not k.startswith(":") and k != "features_extractor_class"}
    replacement = {**readable, "features_extractor_class": scaled}

    old = cls.load(str(path), device="cpu", custom_objects={"policy_kwargs": replacement})
    with tempfile.TemporaryDirectory() as tmp:
        candidate = Path(tmp) / "policy.zip"
        old.save(str(candidate))
        stored = metadata(candidate)["policy_kwargs"]
        assert b"_make_skeleton_class" not in base64.b64decode(stored[":serialized:"]), "still stored by value"
        new = cls.load(str(candidate), device="cpu")   # plain load, no workaround
        rng = np.random.default_rng(0)
        obs = rng.uniform(-1, 1, size=(500, 24)).astype(np.float32)
        a_old, _ = old.predict(obs, deterministic=True)
        a_new, _ = new.predict(obs, deterministic=True)
        assert np.array_equal(a_old, a_new), "re-saved model changed its actions"
        backup = path.with_name(path.name + ".bak")
        shutil.copy2(path, backup)
        shutil.copy2(candidate, path)
    print(f"resaved {path} ({name}, net_arch={readable.get('net_arch')}); "
          f"500/500 identical actions; backup: {backup.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        resave(Path(arg))
