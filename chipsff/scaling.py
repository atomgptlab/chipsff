"""Single-point inference scaling benchmark (timing + peak GPU memory) for one
calculator on cubic Si supercells -- reproduces the paper's scaling table
(max atoms, memory, t @ 21,952 atoms, OOM point).

Each size is a diamond-Si cubic cell replicated s x s x s (8*s^3 atoms); we
report the median single-point energy+force wall time and the peak
torch-tracked GPU memory, escalating until the predicted memory exceeds the
cap (OOM). Run one model at a time (GPU-exclusive).
"""

import os
import gc
import json
import time


def scaling(
    calc, out_dir=".", sides=None, reps=3, mem_cap_gb=112.0, device="cuda"
):
    import numpy as np
    import torch
    from ase.build import bulk

    if sides is None:
        sides = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    def time_ef(atoms):
        atoms.calc = calc
        if hasattr(calc, "reset"):
            calc.reset()
        atoms.get_potential_energy()
        atoms.get_forces()  # warmup
        ts = []
        for _ in range(reps):
            if hasattr(calc, "reset"):
                calc.reset()
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            atoms.get_potential_energy()
            atoms.get_forces()
            if device == "cuda":
                torch.cuda.synchronize()
            ts.append(time.time() - t0)
        return float(np.median(ts))

    res = {"natoms": [], "time_s": [], "mem_gb": [], "oom_at": None}
    print(
        f"[scaling] device={device} reps={reps} cap={mem_cap_gb}GB", flush=True
    )
    for s in sides:
        a0 = bulk("Si", "diamond", a=5.43, cubic=True) * (s, s, s)
        n = len(a0)
        # predictive OOM from the GB/atom trend
        if res["mem_gb"]:
            gpa = res["mem_gb"][-1] / res["natoms"][-1]
            if gpa * n > mem_cap_gb:
                res["oom_at"] = n
                break
        try:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            dt = time_ef(a0.copy())
            gb = (
                torch.cuda.max_memory_allocated() / 1e9
                if device == "cuda"
                else 0.0
            )
            res["natoms"].append(n)
            res["time_s"].append(dt)
            res["mem_gb"].append(gb)
            print(f"  {n:>7d} atoms  {dt:6.3f}s  {gb:5.1f}GB", flush=True)
        except Exception as ex:
            res["oom_at"] = n
            print(f"  {n} OOM/err: {str(ex)[:60]}", flush=True)
            break
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    natoms = res["natoms"]
    summary = {
        "max_atoms": natoms[-1] if natoms else 0,
        "mem_at_max_gb": (
            round(res["mem_gb"][-1], 1) if res["mem_gb"] else None
        ),
        "oom_at": res["oom_at"],
        "t_at_21952_s": (
            round(res["time_s"][natoms.index(21952)], 2)
            if 21952 in natoms
            else None
        ),
    }
    res["summary"] = summary
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "scaling.json"), "w"), indent=1)
    print(f"[scaling] summary: {summary}", flush=True)
    return summary
