"""Beyond-CHIPS-FF application tasks, calculator-agnostic (any ASE calculator).

Each function takes an ASE calculator and returns {"rows": {...}, "mae": float,
"unit": str}. Wired into run_main as selectable tasks: sfe, neb, voltage,
diffusion. References are experimental / literature values.
"""

import warnings

import numpy as np
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import ExpCellFilter
from ase.units import J, m

warnings.filterwarnings("ignore")


# ---- stacking fault energy --------------------------------------------------
def sfe(calc):
    """Intrinsic stacking-fault energy of FCC metals via the ANNNI relation
    gamma_isf ~= 2 (E_hcp - E_fcc) / A_(111).  vs experiment (mJ/m^2)."""
    metals = [
        ("Al", 4.05, 166.0),
        ("Cu", 3.61, 45.0),
        ("Ni", 3.52, 125.0),
        ("Ag", 4.09, 16.0),
        ("Au", 4.08, 32.0),
    ]

    def epa(atoms):
        atoms.calc = calc
        FIRE(ExpCellFilter(atoms), logfile=None).run(fmax=0.02, steps=200)
        return atoms.get_potential_energy() / len(atoms), atoms

    rows, errs = {}, []
    for el, a0, exp in metals:
        try:
            efcc, fcc = epa(bulk(el, "fcc", a=a0))
            ehcp, _ = epa(
                bulk(
                    el,
                    "hcp",
                    a=a0 / np.sqrt(2),
                    c=a0 * np.sqrt(2) / np.sqrt(3) * np.sqrt(8 / 3),
                )
            )
            a = fcc.cell.cellpar()[0] * np.sqrt(2)
            area = (np.sqrt(3) / 2 * a**2) / 2.0
            g = 2.0 * (ehcp - efcc) / area / (J / m**2) * 1e3
            rows[el] = {"calc": g, "exp": exp}
            errs.append(abs(g - exp))
        except Exception as e:
            rows[el] = {"error": str(e)[:60]}
    return {
        "rows": rows,
        "mae": float(np.mean(errs)) if errs else None,
        "unit": "mJ/m^2",
    }


# ---- vacancy migration barrier (NEB) ---------------------------------------
def neb(calc, nimages=5):
    """FCC monovacancy migration barrier via ASE climbing-image NEB (eV)."""
    from ase.mep import NEB

    metals = [("Al", 4.05, 0.61), ("Cu", 3.61, 0.71), ("Ni", 3.52, 1.04)]

    def relax(atoms):
        atoms.calc = calc
        FIRE(atoms, logfile=None).run(fmax=0.03, steps=200)
        return atoms

    rows, errs = {}, []
    for el, a0, lit in metals:
        try:
            sc = bulk(el, "fcc", a=a0, cubic=True) * (3, 3, 3)
            pos = sc.get_positions()
            site = pos[0].copy()
            d = np.linalg.norm(pos - site, axis=1)
            d[0] = 1e9
            mig = int(np.argmin(d))
            ini = sc.copy()
            del ini[0]
            ini = relax(ini)
            fin = sc.copy()
            fin.positions[mig] = site
            del fin[0]
            fin = relax(fin)
            imgs = [ini] + [ini.copy() for _ in range(nimages)] + [fin]
            band = NEB(imgs, climb=True)
            band.interpolate(mic=True)
            for im in imgs:
                im.calc = calc
            FIRE(band, logfile=None).run(fmax=0.05, steps=120)
            e = [im.get_potential_energy() for im in imgs]
            em = max(e) - e[0]
            rows[el] = {"calc": float(em), "lit": lit}
            errs.append(abs(em - lit))
        except Exception as e:
            rows[el] = {"error": str(e)[:60]}
    return {
        "rows": rows,
        "mae": float(np.mean(errs)) if errs else None,
        "unit": "eV",
    }


# ---- battery cathode average voltage ---------------------------------------
def voltage(calc):
    """Cathode average voltage V = (E_deLi + n*mu_Li - E_host)/n (JARVIS
    batterymat method, single-point, no host relaxation). vs experiment (V)."""
    from jarvis.core.atoms import Atoms, get_supercell_dims
    from jarvis.analysis.thermodynamics.energetics import get_optb88vdw_energy
    from jarvis.db.figshare import get_jid_data
    from chipsff.utils import cached_data as data

    cathodes = [
        ("LiCoO2", 4.0),
        ("LiFePO4", 3.45),
        ("LiMn2O4", 4.1),
        ("LiNiO2", 3.8),
    ]

    def en(atoms):
        a = atoms.ase_converter()
        a.calc = calc
        return a.get_potential_energy()

    d3 = data("dft_3d")
    fmap = {}
    for r in d3:
        try:
            a = Atoms.from_dict(r["atoms"])
            f = a.composition.reduced_formula
            fmap.setdefault(f, a)
        except Exception:
            pass
    cp = get_optb88vdw_energy()
    liel = Atoms.from_dict(
        get_jid_data(jid=cp["Li"]["jid"], dataset="dft_3d")["atoms"]
    )
    mu = en(liel) / liel.num_atoms

    rows, errs = {}, []
    for name, exp in cathodes:
        try:
            at = fmap.get(name)
            if at is None:
                rows[name] = {"error": "not in dft_3d"}
                continue
            sc = at.make_supercell(get_supercell_dims(at, enforce_c_size=8))
            els, coords, lat = sc.elements, sc.frac_coords, sc.lattice_mat
            idx = [i for i, e in enumerate(els) if e == "Li"]
            n = len(idx)
            e_full = en(sc)
            ne = [e for j, e in enumerate(els) if j not in idx]
            nc = [coords[j] for j in range(len(els)) if j not in idx]
            deLi = Atoms(
                coords=nc, lattice_mat=lat, elements=ne, cartesian=False
            )
            v = (en(deLi) + n * mu - e_full) / n
            rows[name] = {"calc": float(v), "exp": exp}
            errs.append(abs(v - exp))
        except Exception as e:
            rows[name] = {"error": str(e)[:60]}
    return {
        "rows": rows,
        "mae": float(np.mean(errs)) if errs else None,
        "unit": "V",
    }


# ---- Li tracer diffusivity -------------------------------------------------
def diffusion(calc, temp=1000.0, nsteps=4000, dt_fs=2.0):
    """Li tracer diffusivity from the MSD slope of an NVT run on a Li-conductor
    (Li3PS4/JVASP-like); returns D in cm^2/s (no MAE reference here)."""
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.nvtberendsen import NVTBerendsen
    from jarvis.core.atoms import Atoms
    from chipsff.utils import cached_data as data

    d3 = data("dft_3d")
    at = None
    for r in d3:
        a = Atoms.from_dict(r["atoms"])
        if a.composition.reduced_formula == "Li3PS4":
            at = a
            break
    if at is None:
        return {
            "rows": {},
            "mae": None,
            "unit": "cm^2/s",
            "note": "Li3PS4 not found",
        }
    ase_at = at.make_supercell([2, 2, 2]).ase_converter()
    ase_at.calc = calc
    li = [i for i, s in enumerate(ase_at.get_chemical_symbols()) if s == "Li"]
    MaxwellBoltzmannDistribution(ase_at, temperature_K=temp)
    dyn = NVTBerendsen(ase_at, dt_fs * units.fs, temp, taut=50 * units.fs)
    p0 = ase_at.get_positions()[li].copy()
    msd = []

    def rec():
        dp = ase_at.get_positions()[li] - p0
        msd.append((dp**2).sum(axis=1).mean())

    dyn.attach(rec, interval=20)
    dyn.run(nsteps)
    msd = np.array(msd)
    t = np.arange(len(msd)) * 20 * dt_fs  # fs
    slope = np.polyfit(t[len(t) // 2 :], msd[len(t) // 2 :], 1)[0]  # A^2/fs
    D = slope / 6.0 * 1e-16 / 1e-15  # A^2/fs -> cm^2/s
    return {
        "rows": {"Li3PS4": {"D_cm2_s": float(D), "T_K": temp}},
        "mae": None,
        "unit": "cm^2/s",
    }


TASKS = {"sfe": sfe, "neb": neb, "voltage": voltage, "diffusion": diffusion}
