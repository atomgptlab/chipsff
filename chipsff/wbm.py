"""Optional WBM / Matbench-Discovery task for CHIPS-FF.

Cluster-friendly, sharded relaxation of the WBM initial structures with any
chipsff calculator, followed by scoring (formation-energy MAE, stability F1,
structure RMSD) and jarvis-leaderboard-compatible output.

Stages (see ``chipsff.run_main`` flags and ``submit_slurm.sh``)
  relax : relax shard SHARD/NSHARD of WBM -> <out>/shard_XXXX.jsonl (array)
  score : combine shards, compute e_form (self-consistent chempots), F1, RMSD
  leaderboard : write jarvis_leaderboard contribution CSV.zip files

Data
  WBM initial structures: pass ``--wbm_zip`` (an .extxyz zip, one member per
  structure) or install ``matbench_discovery`` and let it download them.
  Scoring/leaderboard need ``matbench_discovery`` (provides ``df_wbm`` refs).
"""

import os
import io
import json
import glob
import math
import zipfile
import warnings

warnings.filterwarnings("ignore")

# jarvis-leaderboard contribution / benchmark names
EFORM_CSV = "AI-SinglePropertyPrediction-e_form-wbm-test-mae.csv"
RMSD_CSV = "AI-AtomGen-relaxed_structure-wbm-test-rmse.csv"
FORM = "e_form_per_atom_mp2020_corrected"
HULL = "e_above_hull_mp2020_corrected_ppd_mp"

# reference-free diatomic-curve metrics (tortuosity is the paper's tau)
DIATOMIC_ELEMENTS = [
    "H",
    "Li",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Ti",
    "Fe",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "Mo",
    "Ag",
    "Sn",
    "W",
    "Pt",
    "Au",
]
DIATOMIC_METRICS = [
    "tortuosity",
    "energy_jump",
    "force_flips",
    "energy_grad_norm_max",
    "energy_diff_flips",
    "force_total_variation",
    "second_deriv_smoothness",
    "conservation_deviation",
]


def diatomics(calc, out_dir="."):
    """Reference-free diatomic-curve metrics (tortuosity is the paper's tau,
    plus energy_jump, force_flips, ...) via matbench_discovery. Sweeps each
    homonuclear dimer over 0.5-6.0 A and returns the element-averaged dict."""
    import numpy as np
    from ase import Atoms
    from matbench_discovery.metrics.diatomics import (
        DiatomicCurves,
        calc_diatomic_metrics,
    )

    dist = np.round(np.linspace(0.5, 6.0, 56), 4)
    homo = {}
    for el in DIATOMIC_ELEMENTS:
        E, F = [], []
        for d in dist:
            a = Atoms(
                el * 2,
                positions=[[0, 0, 0], [0, 0, float(d)]],
                cell=[30, 30, 30],
                pbc=False,
            )
            a.calc = calc
            try:
                E.append(float(a.get_potential_energy()))
                F.append(a.get_forces().tolist())
            except Exception:
                E.append(float("nan"))
                F.append([[0, 0, 0], [0, 0, 0]])
        homo[f"{el}{el}"] = {"energies": E, "forces": F}
    curves = DiatomicCurves.from_dict(
        {"distances": dist.tolist(), "homo-nuclear": homo}
    )
    m = calc_diatomic_metrics(None, curves)
    agg = {}
    for metric in DIATOMIC_METRICS:
        vals = [
            m[f][metric]
            for f in m
            if metric in m[f]
            and m[f][metric] is not None
            and np.isfinite(m[f][metric])
        ]
        agg[metric] = round(float(np.mean(vals)), 4) if vals else float("nan")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(
        agg, open(os.path.join(out_dir, "diatomics.json"), "w"), indent=1
    )
    return agg


# ----------------------------------------------------------------------------
# WBM initial structures
# ----------------------------------------------------------------------------
def _wbm_members(wbm_zip):
    """Yield (material_id, ase.Atoms) for every WBM initial structure."""
    from ase.io import read

    if wbm_zip and os.path.exists(wbm_zip):
        zf = zipfile.ZipFile(wbm_zip)
        for name in sorted(zf.namelist()):
            if not name.endswith(".extxyz"):
                continue
            mid = name.replace(".extxyz", "").split("/")[-1]
            yield mid, read(
                io.StringIO(zf.read(name).decode()), format="extxyz"
            )
        return
    # fall back to matbench_discovery download
    from matbench_discovery.data import ase_atoms_from_zip, DataFiles

    for at in ase_atoms_from_zip(DataFiles.wbm_initial_atoms.path):
        mid = at.info.get("material_id") or at.info.get("mat_id")
        yield mid, at


# ----------------------------------------------------------------------------
# relax one shard
# ----------------------------------------------------------------------------
def relax(calc, shard, nshard, out_dir, wbm_zip=None, fmax=0.05, steps=500):
    """Relax a contiguous shard of WBM with ``calc`` (an ASE calculator);
    write one jsonl line per structure. Resumable."""
    from ase.filters import ExpCellFilter
    from ase.optimize import FIRE

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"shard_{shard:04d}.jsonl")
    done = (
        sum(1 for _ in open(out_path))
        if os.path.exists(out_path) and os.path.getsize(out_path)
        else 0
    )

    items = list(_wbm_members(wbm_zip))
    per = math.ceil(len(items) / nshard)
    mine = items[(shard - 1) * per : shard * per]
    print(
        f"[wbm-relax] shard {shard}/{nshard}: {len(mine)} structures "
        f"({done} done)",
        flush=True,
    )

    with open(out_path, "a" if done else "w") as fout:
        for i, (mid, at) in enumerate(mine):
            if i < done:
                continue
            try:
                at.calc = calc
                FIRE(ExpCellFilter(at), logfile="/dev/null").run(
                    fmax=fmax, steps=steps
                )
                rec = {
                    "material_id": mid,
                    "energy": float(at.get_potential_energy()),
                    "cell": at.cell.array.tolist(),
                    "positions": at.get_positions().tolist(),
                    "numbers": at.get_atomic_numbers().tolist(),
                    "pbc": [bool(x) for x in at.pbc],
                }
            except Exception as exc:
                rec = {"material_id": mid, "error": repr(exc)[:200]}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(mine)}", flush=True)
    print(f"[wbm-relax] shard {shard} DONE", flush=True)


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def _combine(out_dir):
    """Merge all shard jsonl into {material_id: record} (successful only)."""
    recs = {}
    for f in sorted(glob.glob(os.path.join(out_dir, "shard_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if "energy" in r:
                recs[r["material_id"]] = r
    return recs


def _element_refs(calc, elements, out_dir):
    """Self-consistent elemental reference energies (eV/atom) for ``calc``.
    Uses the standard element reference structures from jarvis (same set as
    chipsff's chemical_potentials.json). Cached to <out>/wbm_chempots.json."""
    cache = os.path.join(out_dir, "wbm_chempots.json")
    mu = json.load(open(cache)) if os.path.exists(cache) else {}
    here = os.path.dirname(os.path.abspath(__file__))
    ref = json.load(open(os.path.join(here, "chemical_potentials.json")))
    from jarvis.db.figshare import get_jid_data
    from jarvis.core.atoms import Atoms

    for el in elements:
        if el in mu:
            continue
        jid = ref.get(el, {}).get("jid")
        if not jid:
            continue
        try:
            d = get_jid_data(jid=jid, dataset="dft_3d")
            at = Atoms.from_dict(d["atoms"]).ase_converter()
            at.calc = calc
            mu[el] = float(at.get_potential_energy()) / len(at)
        except Exception:
            pass
    json.dump(mu, open(cache, "w"))
    return mu


def score(out_dir, calc=None, do_rmsd=True):
    """Combine shards and compute WBM metrics against matbench_discovery refs.
    Returns dict with e_form_mae, F1/precision/recall (discovery), RMSD, n."""
    import numpy as np
    from ase.data import chemical_symbols
    from matbench_discovery.data import df_wbm
    from matbench_discovery.metrics.discovery import stable_metrics

    ref = df_wbm[[FORM, HULL]].dropna()

    recs = _combine(out_dir)
    # elements present across the relaxed set
    els = set()
    for r in recs.values():
        els.update(chemical_symbols[z] for z in r["numbers"])
    mu = (
        _element_refs(calc, els, out_dir)
        if calc is not None
        else json.load(open(os.path.join(out_dir, "wbm_chempots.json")))
    )

    rows, eah_true, eah_pred = [], [], []
    for mid, r in recs.items():
        if mid not in ref.index:
            continue
        comp = {}
        for z in r["numbers"]:
            comp[chemical_symbols[z]] = comp.get(chemical_symbols[z], 0) + 1
        if any(el not in mu for el in comp):
            continue
        n = len(r["numbers"])
        ef = (r["energy"] - sum(mu[el] * c for el, c in comp.items())) / n
        tgt = float(ref.loc[mid, FORM])
        rows.append((mid, tgt, ef))
        # stability: shift DFT e_above_hull by the model's formation error
        eah_true.append(float(ref.loc[mid, HULL]))
        eah_pred.append(float(ref.loc[mid, HULL]) + (ef - tgt))

    eah_true = np.array(eah_true)
    eah_pred = np.array(eah_pred)
    disc = stable_metrics(eah_true, eah_pred) if len(eah_true) else {}
    e_form_mae = (
        float(np.mean([abs(ef - t) for _, t, ef in rows]))
        if rows
        else float("nan")
    )

    out = {
        "n": len(rows),
        "e_form_mae": round(e_form_mae, 4),
        "F1": round(float(disc.get("F1", float("nan"))), 4),
        "precision": round(float(disc.get("Precision", float("nan"))), 4),
        "recall": round(float(disc.get("Recall", float("nan"))), 4),
    }
    # structure RMSD vs WBM DFT-relaxed structures (optional; needs pymatgen)
    if do_rmsd:
        out["RMSD"] = _rmsd(recs)
    # keep the per-structure predictions for leaderboard export
    json.dump(
        {"rows": rows, "metrics": out},
        open(os.path.join(out_dir, "wbm_score.json"), "w"),
    )
    return out


def _rmsd(recs):
    """Mean RMSD (Å) between relaxed and WBM DFT-relaxed structures."""
    try:
        import numpy as np
        from pymatgen.core import Structure, Lattice
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from matbench_discovery.data import df_wbm  # noqa: F401
        from matbench_discovery.data import ase_atoms_from_zip, DataFiles

        dft = {}
        for at in ase_atoms_from_zip(
            DataFiles.wbm_computed_structure_entries.path
        ):
            mid = at.info.get("material_id")
            dft[mid] = at
        sm = StructureMatcher()
        ds = []
        for mid, r in recs.items():
            if mid not in dft:
                continue
            s1 = Structure(
                Lattice(r["cell"]),
                [int(z) for z in r["numbers"]],
                r["positions"],
                coords_are_cartesian=True,
            )
            s2 = Structure(
                Lattice(dft[mid].cell.array),
                dft[mid].get_atomic_numbers(),
                dft[mid].get_positions(),
                coords_are_cartesian=True,
            )
            rms = sm.get_rms_dist(s1, s2)
            if rms:
                ds.append(rms[0])
        return round(float(np.mean(ds)), 4) if ds else float("nan")
    except Exception as exc:
        print(f"[wbm-score] RMSD skipped: {exc}")
        return float("nan")


# ----------------------------------------------------------------------------
# jarvis-leaderboard output
# ----------------------------------------------------------------------------
def write_leaderboard(
    out_dir, method, lb_dir, author_email="", project_url=""
):
    """Write jarvis_leaderboard contribution files from wbm_score.json."""
    sj = json.load(open(os.path.join(out_dir, "wbm_score.json")))
    cdir = os.path.join(lb_dir, "contributions", method)
    os.makedirs(cdir, exist_ok=True)

    lines = ["id,target,prediction"]
    for mid, tgt, pred in sj["rows"]:
        lines.append(f"{mid},{tgt:.6f},{pred:.6f}")
    _zip(os.path.join(cdir, EFORM_CSV + ".zip"), EFORM_CSV, "\n".join(lines))

    meta = {
        "model_name": method,
        "project_url": project_url,
        "author_email": author_email,
        "team_name": "AtomGPTLab",
        "metrics": sj["metrics"],
    }
    json.dump(meta, open(os.path.join(cdir, "metadata.json"), "w"), indent=2)
    print(
        f"[wbm-leaderboard] wrote {cdir}/{EFORM_CSV}.zip (n={len(lines) - 1}) "
        f"+ metadata.json  metrics={sj['metrics']}"
    )


def _zip(path, arcname, text):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(arcname, text)
