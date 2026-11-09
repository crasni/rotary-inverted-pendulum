"""Swing-up check in the rig-matched simulator with the pendulum's friction scaled.

Runs rig_curriculum's deterministic `hanging` collection against SimClient, with
pendulum viscous and Coulomb friction multiplied by --damping, and reports when
the pendulum first reaches the top. Used to test whether rig swing-up stalls
(v3, v5) come from the rig losing more energy per swing than the simulator.
"""
import argparse, dataclasses, glob, sys, tempfile
from pathlib import Path
import numpy as np
import pendulum_lab.rig_session as rs
import rig_curriculum as rc

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--damping", type=float, nargs="+", default=[1.0, 2.0, 3.0])
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--seconds", type=float, default=8.0)
args = parser.parse_args()

Base = rs.SimClient
for factor in args.damping:
    class Scaled(Base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            e = self.env
            e.params = dataclasses.replace(e.params, pendulum_friction=e.params.pendulum_friction * factor,
                                           pendulum_coulomb=e.params.pendulum_coulomb * factor)
            e.model.dof_damping[e._pen_dof_addr] = e.params.pendulum_friction
            e.model.dof_frictionloss[e._pen_dof_addr] = e.params.pendulum_coulomb
    rc.SimClient = Scaled
    for ck in args.checkpoints:
        out = Path(tempfile.mkdtemp()) / "b"
        ns = argparse.Namespace(checkpoint=Path(ck), output=out, stage="hanging", episodes=args.episodes,
                                episode_seconds=args.seconds, deterministic=True, seed=11, sim=True, port=None)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            rc.collect(ns)
        tops, peaks = [], []
        for f in sorted(glob.glob(str(out / "episode_*.npz"))):
            x = np.load(f, allow_pickle=True)["samples"]; t = x[:, 0] - x[0, 0]
            hang = np.degrees(np.abs(np.angle(np.exp(1j * (x[:, 2] - np.pi)))))
            tops.append(t[np.argmax(hang > 170)] if (hang > 170).any() else np.nan); peaks.append(hang[t < 4].max())
        name = Path(ck).parent.name
        print(f"damping x{factor:g}  {name:6s} top at {np.round(tops, 1)}  max height in 4 s {np.round(peaks).astype(int)}", flush=True)
