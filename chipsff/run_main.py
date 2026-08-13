#!/usr/bin/env python
"""run_main.py -- run the full CHIPS-FF property benchmark for ONE model.

Examples
--------
    python -m chipsff.run_main --alignn_ff            # default MATPES-r2SCAN
    python -m chipsff.run_main --alignn_ff --model_path /path/to/model_dir
    python -m chipsff.run_main --uma
    python -m chipsff.run_main --matgl
    python -m chipsff.run_main --chgnet
    python -m chipsff.run_main --mace
    python -m chipsff.run_main --alignn_ff --n 5      # quick 5-material test

What it does (one consistent protocol; the paper's Table-4 columns):
  per material : relax (FrechetCellFilter) -> lattice a,c -> E-V/bulk modulus
                 -> formation energy -> elastic C11/C44 -> surfaces (6 millers)
                 -> monovacancy;  optionally phonons and amorphous-Si.
  interfaces   : work of adhesion over the bundled Interface.csv set.
  aggregate    : MAE vs JARVIS-DFT (chempot-consistent) -> printed + CSV table.

Robustness
  * If the model's python package is missing it is pip-installed automatically.
  * Chemical potentials are made self-consistent: the stored energy_<calc>
    entries are dropped so chipsff recomputes each elemental reference with the
    selected model on the fly (it caches them back to the run's copy).
"""

import os
import sys
import json
import glob
import csv
import shutil
import argparse
import subprocess
import importlib
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

# flag -> (chipsff calculator_type, import name, pip package, settings builder)
REGISTRY = {
    "alignn_ff": (
        "alignn_ff",
        "alignn",
        "alignn",
        lambda a: {
            "alignn_ff": (
                {
                    "path": a.model_path,
                    "model_filename": "best_model.pt",
                    "device": a.device,
                    "stress_wt": 1.0,
                }
                if a.model_path
                else {"device": a.device, "stress_wt": 1.0}
            )
        },
    ),
    "uma": (
        "uma",
        "fairchem",
        "fairchem-core",
        lambda a: {
            "uma": {
                "model_name": "uma-s-1p1",
                "task_name": "omat",
                "device": a.device,
            }
        },
    ),
    "matgl": (
        "matgl",
        "matgl",
        "matgl",
        lambda a: {"matgl": {"model": "M3GNet-PES-MatPES-PBE-2025.2"}},
    ),
    "chgnet": ("chgnet", "chgnet", "chgnet", lambda a: {"chgnet": {}}),
    "mace": (
        "mace",
        "mace",
        "mace-torch",
        lambda a: {"mace": {"device": a.device}},
    ),
}


def ensure_package(import_name, pip_name):
    try:
        importlib.import_module(import_name)
        return
    except Exception:
        print(
            f"[setup] '{import_name}' not found -> pip install {pip_name}",
            flush=True,
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name]
        )
        importlib.invalidate_caches()
        importlib.import_module(import_name)


def self_consistent_chempots(calc_type, out_dir):
    """Copy the shipped element->jid map, drop energy_<calc> so chipsff
    recomputes references self-consistently for THIS model. Returns the path.
    """
    src = os.path.join(HERE, "chemical_potentials.json")
    cp = json.load(open(src))
    key = f"energy_{calc_type}"
    for el in cp:
        cp[el].pop(key, None)
    dst = os.path.join(out_dir, "chemical_potentials.json")
    json.dump(cp, open(dst, "w"), indent=2)
    return dst


# ---- exact protocol from chipsff_frechet_run.sh -----------------------------
def protocol(with_phonons):
    props = [
        "relax_structure",
        "calculate_ev_curve",
        "calculate_formation_energy",
        "calculate_elastic_tensor",
        "analyze_surfaces",
        "analyze_defects",
    ]
    if with_phonons:
        props.append("run_phonon_analysis")
    relax = {
        "filter_type": "FrechetCellFilter",
        "relaxation_settings": {
            "fmax": 0.05,
            "steps": 200,
            "constant_volume": False,
        },
    }
    surf = {
        "indices_list": [
            [1, 0, 0],
            [1, 1, 1],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [0, 1, 0],
        ],
        "layers": 4,
        "vacuum": 18,
        "filter_type": "FrechetCellFilter",
        "relaxation_settings": {
            "fmax": 0.05,
            "steps": 200,
            "constant_volume": True,
        },
    }
    dfc = {
        "generate_settings": {
            "on_conventional_cell": True,
            "enforce_c_size": 8,
            "extend": 1,
        },
        "filter_type": "FrechetCellFilter",
        "relaxation_settings": {
            "fmax": 0.05,
            "steps": 200,
            "constant_volume": True,
        },
    }
    return props, relax, surf, dfc


def run_in(wd, fn):
    here = os.getcwd()
    os.makedirs(wd, exist_ok=True)
    os.chdir(wd)
    try:
        fn()
    except Exception as exc:
        print(
            f"    FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush=True
        )
    finally:
        os.chdir(here)


def aggregate(out_dir, calc_type, interfaces_dir, iface_rows):
    import numpy as np
    from jarvis.db.figshare import data

    BY = {
        x["jid"]: x for x in data("dft_3d")
    }  # noqa: F841 (kept for parity/refs)

    def mae(x):
        return (float(np.mean(x)), len(x)) if x else (float("nan"), 0)

    la, lc, form, c11, c44, kv, vac, surf = ([] for _ in range(8))
    for f in glob.glob(f"{out_dir}/*/*_{calc_type}/*_results.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        en = r.get("energy", {})
        if en.get("final_a") and en.get("initial_a"):
            la.append(abs(en["final_a"] - en["initial_a"]))
        if en.get("final_c") and en.get("initial_c"):
            lc.append(abs(en["final_c"] - en["initial_c"]))
        fe = r.get("form_en", {})
        if (
            fe.get("form_energy") is not None
            and fe.get("form_energy_entry") is not None
        ):
            form.append(abs(fe["form_energy"] - fe["form_energy_entry"]))
        et = r.get("elastic_tensor", {})
        if et.get("c11") is not None and et.get("c11_entry"):
            c11.append(abs(et["c11"] - et["c11_entry"]))
        if et.get("c44") is not None and et.get("c44_entry"):
            c44.append(abs(et["c44"] - et["c44_entry"]))
        m = r.get("modulus", {})
        if m.get("kv") is not None and m.get("kv_entry"):
            kv.append(abs(m["kv"] - m["kv_entry"]))
        for v in r.get("vacancy_energy", []):
            p, e = v.get("vac_en"), v.get("vac_en_entry")
            if p is not None and e not in (None,) and abs(e) > 1e-6:
                vac.append(abs(p - e))
        for s in r.get("surface_energy", []):
            p, e = s.get("surf_en"), s.get("surf_en_entry")
            if p is not None and e not in (None,) and 1e-6 < abs(e) <= 5.0:
                surf.append(abs(p - e))

    wad = []
    for it in iface_rows:
        wd = os.path.join(
            interfaces_dir,
            f"JVASP-{it['film']}_JVASP-{it['subs']}_{it['fm']}_{it['sm']}",
        )
        pred = None
        for gg in glob.glob(f"{wd}/**/*.json", recursive=True):
            try:
                dd = json.load(open(gg))
            except Exception:
                continue
            if isinstance(dd, dict):
                for key in (
                    "best_z_wad",
                    "min_wad",
                    "work_of_adhesion",
                    "wad",
                ):
                    if key in dd and isinstance(dd[key], (int, float)):
                        pred = dd[key]
        if (
            pred is not None and abs(pred) <= 7.0
        ):  # physical filter (drops cell-mismatch blow-ups)
            wad.append(abs(pred - it["wad_dft"]))

    order = [
        ("a_A", la, "%.3f"),
        ("c_A", lc, "%.3f"),
        ("Ef_eV", form, "%.3f"),
        ("C11_GPa", c11, "%.1f"),
        ("C44_GPa", c44, "%.1f"),
        ("Kv_GPa", kv, "%.1f"),
        ("Vac_eV", vac, "%.3f"),
        ("Surf_Jm2", surf, "%.3f"),
        ("Wad_Jm2", wad, "%.3f"),
    ]
    return order, mae


def main():
    ap = argparse.ArgumentParser(
        description="Full CHIPS-FF benchmark for one model."
    )
    g = ap.add_mutually_exclusive_group(required=True)
    for flag in REGISTRY:
        g.add_argument(
            f"--{flag}", action="store_true", help=f"use the {flag} calculator"
        )
    ap.add_argument(
        "--model_path",
        default="",
        help="ALIGNN-FF checkpoint dir (best_model.pt+config.json)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--n",
        type=int,
        default=0,
        help="limit to first N materials (0 = all 104)",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-interfaces", action="store_true")
    ap.add_argument(
        "--phonons", action="store_true", help="also compute phonons (slow)"
    )
    # optional WBM / Matbench-Discovery task (relax on a cluster array, score)
    ap.add_argument(
        "--wbm-relax",
        dest="wbm_relax",
        action="store_true",
        help="relax a shard of WBM (uses SHARD/NSHARD env or "
        "--shard/--nshard)",
    )
    ap.add_argument(
        "--wbm-score",
        dest="wbm_score",
        action="store_true",
        help="score relaxed WBM: e_form MAE, stability F1, RMSD",
    )
    ap.add_argument(
        "--diatomics",
        action="store_true",
        help="reference-free diatomic-curve metrics (tortuosity)",
    )
    ap.add_argument(
        "--scaling",
        action="store_true",
        help="inference timing + peak memory on cubic Si supercells "
        "(t @ 21,952 atoms, max atoms, OOM)",
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=0)
    ap.add_argument(
        "--wbm_zip",
        default="",
        help="wbm-initial-atoms.extxyz.zip (else matbench_discovery)",
    )
    ap.add_argument(
        "--out", default="", help="WBM output dir (default wbm_<tag>)"
    )
    ap.add_argument(
        "--leaderboard",
        default="",
        help="jarvis_leaderboard dir; write contribution CSVs",
    )
    args = ap.parse_args()

    model = next(k for k in REGISTRY if getattr(args, k))
    calc_type, imp, pip_name, settings = REGISTRY[model]
    tag = args.tag or (
        model
        if not args.model_path
        else os.path.basename(args.model_path.rstrip("/"))
    )
    out_dir = os.path.abspath(f"chipsff_frechet_{tag}")
    iface_dir = os.path.abspath(f"iface_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(iface_dir, exist_ok=True)

    print(
        f"=== CHIPS-FF full benchmark: model={model} calc={calc_type} "
        f"tag={tag} device={args.device} ===",
        flush=True,
    )
    ensure_package(imp, pip_name)
    CSET = settings(args)

    # ---- optional WBM / Matbench-Discovery task ----
    if args.wbm_relax or args.wbm_score or args.diatomics or args.scaling:
        from chipsff.calcs import setup_calculator

        calc = setup_calculator(calc_type, CSET[calc_type])
        if args.scaling:
            from chipsff import scaling as scaling_mod

            scaling_mod.scaling(
                calc, out_dir=f"scaling_{tag}", device=args.device
            )
        if args.wbm_relax or args.wbm_score or args.diatomics:
            from chipsff import wbm

            ensure_package("matbench_discovery", "matbench-discovery")
            wout = os.path.abspath(args.out or f"wbm_{tag}")
        if args.diatomics:
            print(
                "diatomic metrics:",
                wbm.diatomics(calc, out_dir=f"diatomics_{tag}"),
                flush=True,
            )
        if args.wbm_relax:
            shard = args.shard or int(
                os.environ.get("SLURM_ARRAY_TASK_ID", "1")
            )
            nshard = args.nshard or int(os.environ.get("NSHARD", "490"))
            wbm.relax(calc, shard, nshard, wout, wbm_zip=args.wbm_zip)
        if args.wbm_score:
            metrics = wbm.score(wout, calc=calc)
            print("WBM metrics:", metrics, flush=True)
            if args.leaderboard:
                wbm.write_leaderboard(
                    wout,
                    tag,
                    args.leaderboard,
                    author_email="",
                    project_url="https://github.com/atomgptlab/alignn",
                )
        return

    # chipsff analysis backends for the property benchmark
    ensure_package("elastic", "elastic")
    if args.phonons:
        ensure_package("phonopy", "phonopy")
    if not args.skip_interfaces:
        ensure_package("intermat", "intermat")

    chempot = self_consistent_chempots(calc_type, out_dir)
    jids = json.load(open(os.path.join(HERE, "lb_jids.json")))
    if args.n:
        jids = jids[: args.n]
    iface_rows = []
    with open(os.path.join(HERE, "Interface.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            w = r["PrevData W_adhesion (Jm-2)"].strip()
            if w:
                iface_rows.append(
                    {
                        "film": r["JARVISID-Film"].strip(),
                        "subs": r["JARVISID-Subs"].strip(),
                        "fm": r["Film-miller"].strip().strip('"'),
                        "sm": r["Subs-miller"].strip().strip('"'),
                        "wad_dft": float(w),
                    }
                )

    from chipsff.general_material_analyzer import MaterialsAnalyzer

    props, relax, surf, dfc = protocol(args.phonons)

    # 1) per-material suite
    for i, jid in enumerate(jids, 1):
        wd = os.path.join(out_dir, jid)
        if glob.glob(
            f"{wd}/**/{jid}_{calc_type}_results.json", recursive=True
        ):
            print(f"[{i}/{len(jids)}] {jid} cached", flush=True)
            continue
        print(f"[{i}/{len(jids)}] {jid}", flush=True)
        run_in(
            wd,
            lambda j=jid: MaterialsAnalyzer(
                jid=j,
                calculator_type=calc_type,
                calculator_settings=CSET,
                chemical_potentials_file=chempot,
                properties_to_calculate=props,
                use_conventional_cell=True,
                bulk_relaxation_settings=relax,
                surface_settings=surf,
                defect_settings=dfc,
            ).run_all(),
        )

    # 2) interfaces -> work of adhesion
    if not args.skip_interfaces:

        def _mi(s):
            return "_".join(list(s))

        for k, it in enumerate(iface_rows, 1):
            film, subs = f"JVASP-{it['film']}", f"JVASP-{it['subs']}"
            wd = os.path.join(
                iface_dir, f"{film}_{subs}_{it['fm']}_{it['sm']}"
            )
            if os.path.exists(os.path.join(wd, ".done")):
                continue
            print(
                f"[iface {k}/{len(iface_rows)}] {film}/{subs}",
                flush=True,
            )
            run_in(
                wd,
                lambda it=it, film=film, subs=subs: MaterialsAnalyzer(
                    film_jid=film,
                    substrate_jid=subs,
                    film_index=_mi(it["fm"]),
                    substrate_index=_mi(it["sm"]),
                    calculator_type=calc_type,
                    calculator_settings=CSET,
                    chemical_potentials_file=chempot,
                    bulk_relaxation_settings=relax,
                ).analyze_interfaces(),
            )
            open(os.path.join(wd, ".done"), "w").close()

    # 3) aggregate + print/save table
    order, mae = aggregate(out_dir, calc_type, iface_dir, iface_rows)
    print("\n=== MAE vs JARVIS-DFT ===")
    hdr = ["model"] + [f"{name}(n)" for name, _, _ in order]
    print("  ".join(f"{h:>12s}" for h in hdr))
    cells = [tag]
    for name, vals, fmt in order:
        v, n = mae(vals)
        cells.append(("--" if v != v else fmt % v) + f" ({n})")
    print("  ".join(f"{c:>12s}" for c in cells))
    csv_path = os.path.abspath(f"chipsff_table_{tag}.csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(hdr) + "\n" + ",".join(cells) + "\n")
    print(f"\nsaved {csv_path}")
    if not args.phonons:
        print(
            "note: phonon MAE (omega_ph) not run; add --phonons. "
            "W_ad uses chipsff's InterMat method with |Wad|<=7 filter."
        )


if __name__ == "__main__":
    main()
