"""
Fast tests that need no network access or machine-learning force fields.

Structures are built locally and ASE's EMT potential stands in for the MLFF.
"""
import os

import h5py
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.constraints import ExpCellFilter
from jarvis.core.atoms import ase_to_atoms

import chipsff.general_material_analyzer as gma
import chipsff.run_chipsff as run_chipsff
from chipsff.utils import (
    absolute_error,
    collect_data,
    elastic_reference,
    get_surface_energy_entry,
    get_vacancy_energy_entry,
    match_vacancy_reference,
    reference_value,
)


@pytest.fixture
def make_analyzer(tmp_path, monkeypatch):
    """Return a factory for MaterialsAnalyzer on fcc Cu with EMT, run in tmp_path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        gma, "setup_calculator", lambda calculator_type, settings: EMT()
    )

    def make(a=3.59, jid="JVASP-TEST", cubic=False, **kwargs):
        atoms = ase_to_atoms(bulk("Cu", "fcc", a=a, cubic=cubic))
        return gma.MaterialsAnalyzer(
            jid=jid,
            calculator_type="emt",
            dataset=[{"jid": jid, "atoms": atoms.to_dict()}],
            chemical_potentials_file=str(tmp_path / "chemical_potentials.json"),
            **kwargs,
        )

    return make


def set_cu_chemical_potential(analyzer, atoms):
    """Use the EMT energy per atom of `atoms` as the Cu chemical potential."""
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = EMT()
    energy = ase_atoms.get_potential_energy() / len(ase_atoms)
    analyzer.chemical_potentials = {"Cu": {"energy_emt": energy}}


def test_capture_fire_output_returns_final_energy(make_analyzer):
    analyzer = make_analyzer(a=3.7)
    ase_atoms = analyzer.atoms.ase_converter()
    ase_atoms.calc = EMT()
    energy, nsteps = analyzer.capture_fire_output(
        ExpCellFilter(ase_atoms), fmax=0.01, steps=200
    )
    assert energy == pytest.approx(ase_atoms.get_potential_energy())
    assert nsteps < 200


def test_formation_energy_is_none_without_chemical_potential(make_analyzer):
    analyzer = make_analyzer()
    analyzer.chemical_potentials = {}
    relaxed = analyzer.relax_structure()
    analyzer.calculate_ev_curve(relaxed)
    assert analyzer.calculate_formation_energy(relaxed) is None


def test_formation_energy_without_ev_curve(make_analyzer):
    analyzer = make_analyzer()
    relaxed = analyzer.relax_structure()
    set_cu_chemical_potential(analyzer, relaxed)
    # An elemental solid relative to itself has zero formation energy
    assert analyzer.calculate_formation_energy(relaxed) == pytest.approx(
        0.0, abs=1e-6
    )


def test_vacancy_energy_does_not_depend_on_input_lattice(make_analyzer):
    # Starting 3% away from the EMT lattice constant used to add several eV
    # of strain energy, because defects were built from the input lattice.
    energies, names = [], []
    for a in (3.59, 3.70):
        analyzer = make_analyzer(a=a, jid=f"JVASP-A{a}")
        relaxed = analyzer.relax_structure()
        analyzer.calculate_ev_curve(relaxed)
        set_cu_chemical_potential(analyzer, relaxed)
        analyzer.analyze_defects()
        (vacancy,) = analyzer.job_info["all_vacancies"]
        energies.append(vacancy["vac_en"])
        names.append(vacancy["name"])
    assert energies[1] == pytest.approx(energies[0], abs=0.05)
    # Names carry the Wyckoff letter, matching vacancydb ids
    assert names == ["JVASP-A3.59_Cu_a", "JVASP-A3.7_Cu_a"]


def test_bulk_modulus_does_not_depend_on_cell_choice(make_analyzer):
    # jarvis Atoms.strain_atoms keeps Cartesian positions fixed, so the
    # strained conventional cell was distorted and its bulk modulus too high
    moduli = []
    for cubic in (False, True):
        analyzer = make_analyzer(cubic=cubic)
        relaxed = analyzer.relax_structure()
        analyzer.calculate_ev_curve(relaxed)
        moduli.append(analyzer.job_info["bulk_modulus"])
    assert moduli[1] == pytest.approx(moduli[0], rel=0.01)


def test_qha_matches_free_energy_minimum(tmp_path):
    # With E(V) = B (V - V0)^2 / (2 V0) and F(V, T) = -c T (V - V0), the
    # Gibbs energy minimum moves as V(T) = V0 + c' T V0 / B, so the thermal
    # expansion is c' / B (c' is c converted from kJ/mol to eV).
    v0, b, c = 45.0, 0.5, 1e-3
    volumes = np.linspace(40.0, 50.0, 21)
    temperatures = np.arange(0, 300, 6)
    energies = b * (volumes - v0) ** 2 / (2 * v0)
    dv = (volumes - v0)[:, None] * np.ones(len(temperatures))
    free_energies = -c * temperatures * dv  # kJ/mol, (volumes, temperatures)
    entropies = 1000 * c * dv  # J/(K mol), -dF/dT
    heat_capacities = np.zeros_like(free_energies)

    analyzer = gma.MaterialsAnalyzer.__new__(gma.MaterialsAnalyzer)
    alpha_temperatures, alpha = analyzer.perform_qha(
        volumes=volumes,
        energies=energies,
        free_energies=free_energies,
        heat_capacities=heat_capacities,
        entropies=entropies,
        temperatures=temperatures,
        output_dir=str(tmp_path),
    )
    expected = c * 0.0103643 / b
    middle = (alpha_temperatures > 50) & (alpha_temperatures < 250)
    assert np.allclose(alpha[middle], expected, rtol=0.05)


def test_thermal_expansion_keeps_fitted_equilibrium_energy(make_analyzer):
    analyzer = make_analyzer()
    ase_atoms = analyzer.atoms.ase_converter()
    ase_atoms.calc = EMT()
    input_energy = ase_atoms.get_potential_energy()  # near the EMT minimum
    analyzer.calculate_thermal_expansion(analyzer.atoms)
    # The stored value is the fitted minimum, not the most compressed point
    # (which is about 0.13 eV higher here)
    assert analyzer.job_info["equilibrium_energy"] == pytest.approx(
        input_energy, abs=0.01
    )
    alpha = np.array(analyzer.job_info["thermal_expansion"])
    temperatures = np.array(analyzer.job_info["thermal_expansion_temperatures"])
    assert len(alpha) == len(temperatures)
    assert np.all(alpha[temperatures > 100] > 0)


def test_kappa_values_are_phono3py_totals(make_analyzer):
    pytest.importorskip("phono3py")
    # A 2x2x2 supercell of the 1-atom primitive cell is too small to give
    # nonzero third-order force constants, so use the conventional cell.
    analyzer = make_analyzer(
        cubic=True, phonon3_settings={"dim": [2, 2, 2], "distance": 0.03}
    )
    analyzer.run_phonon3_analysis(analyzer.atoms)
    temperatures = analyzer.job_info["temperatures"]
    kappa_xx = analyzer.job_info["kappa_xx_values"]
    assert temperatures == list(range(100, 1001, 100))
    # Compare with the total kappa phono3py wrote to its HDF5 file
    (hdf5_file,) = [
        f for f in os.listdir(analyzer.output_dir) if f.startswith("kappa-")
    ]
    with h5py.File(os.path.join(analyzer.output_dir, hdf5_file), "r") as f:
        selected = np.isin(f["temperature"][:], temperatures)
        assert np.allclose(kappa_xx, f["kappa"][selected, 0])
    assert all(0 < k < 1e4 for k in kappa_xx)


def test_collect_data_separates_functionals_and_wyckoff_sites():
    datasets = dict(
        dft_3d=[{"jid": "JVASP-1"}],
        vacancydb=[
            {"jid": "JVASP-1", "id": "JVASP-1_Sm_a", "ef": 1.7},
            {"jid": "JVASP-1", "id": "JVASP-1_Sm_c", "ef": 1.8},
        ],
        # The first entry used to win regardless of functional
        surface_data=[
            {"name": "Surface-JVASP-1_miller_0_0_1_thickness_16_VASP_R2SCAN", "surf_en": 2.0},
            {"name": "Surface-JVASP-1_miller_0_0_1_thickness_16_VASP_PBE", "surf_en": 1.1},
            {"name": "Surface-JVASP-1_miller_0_0_1_thickness_16_VASP_PBE_noDP", "surf_en": 1.0},
            {
                "name": "Surface-JVASP-1_miller_0_0_1_x1015_thickness_16_VASP_R2SCAN_R2SCAN",
                "surf_en": 2.1,
            },
        ],
    )
    vacancies = get_vacancy_energy_entry("JVASP-1", collect_data(**datasets))
    assert {v["symbol"]: v["vac_en_entry"] for v in vacancies} == {
        "JVASP-1_Sm_a": 1.7,
        "JVASP-1_Sm_c": 1.8,
    }

    (pbe,) = get_surface_energy_entry("JVASP-1", collect_data(**datasets))
    assert pbe["name"] == "Surface-JVASP-1_miller_0_0_1"
    assert pbe["surf_en_entry"] == 1.0
    assert pbe["reference_name"].endswith("VASP_PBE_noDP")

    (r2scan,) = get_surface_energy_entry(
        "JVASP-1", collect_data(surface_functional="R2SCAN", **datasets)
    )
    assert r2scan["surf_en_entry"] == 2.0


def test_match_vacancy_reference():
    refs = [
        {"symbol": "JVASP-2_Ru_c", "vac_en_entry": 2.0},
        {"symbol": "JVASP-3_Sm_a", "vac_en_entry": 1.7},
        {"symbol": "JVASP-3_Sm_c", "vac_en_entry": 1.8},
    ]
    # One Ru site in both sets: matched although the letters differ
    assert match_vacancy_reference("JVASP-2_Ru_d", ["JVASP-2_Ru_d"], refs[:1]) == refs[0]
    # Several Sm sites: only exact letters are matched
    computed = ["JVASP-3_Sm_a", "JVASP-3_Sm_d"]
    assert match_vacancy_reference("JVASP-3_Sm_a", computed, refs[1:]) == refs[1]
    assert match_vacancy_reference("JVASP-3_Sm_d", computed, refs[1:]) is None


def test_missing_reference_values_give_nan_errors():
    assert reference_value("na") is None
    assert reference_value(None) is None
    assert reference_value("1.5") == 1.5
    assert elastic_reference("na", 3, 3) is None
    assert elastic_reference([[0]], 3, 3) is None
    assert elastic_reference(np.diag([1, 2, 3, 4, 5, 6]).tolist(), 3, 3) == 4.0
    assert np.isnan(absolute_error(None, 2.0))
    assert absolute_error(1.0, 3.5) == 2.5


def test_batch_skips_failed_materials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class FakeAnalyzer:
        def __init__(self, jid, **kwargs):
            self.jid = jid

        def run_all(self):
            if self.jid == "BAD":
                raise RuntimeError("analysis failed")
            return {"err_a": 0.1, "time": 1.0}

    monkeypatch.setattr(run_chipsff, "MaterialsAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(run_chipsff, "plot_composite_scorecard", lambda df: None)
    failures = run_chipsff.analyze_multiple_structures(
        ["GOOD", "BAD"], ["calc"], "chemical_potentials.json"
    )
    assert failures == [("BAD", "calc")]
    assert (tmp_path / "composite_error_data.csv").exists()


def test_main_runs_remaining_jobs_then_exits_with_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ran = []

    class FakeAnalyzer:
        def __init__(self, jid, calculator_type, **kwargs):
            self.jid = jid

        def run_all(self):
            ran.append(self.jid)
            if self.jid == "BAD":
                raise RuntimeError("analysis failed")
            return {"err_a": 0.1}

    monkeypatch.setattr(run_chipsff, "MaterialsAnalyzer", FakeAnalyzer)
    (tmp_path / "input.json").write_text(
        '{"jid_list": ["BAD", "GOOD"], "calculator_type": "calc"}'
    )
    monkeypatch.setattr("sys.argv", ["run_chipsff", "--input_file", "input.json"])
    with pytest.raises(SystemExit, match="1 analyses failed: BAD"):
        run_chipsff.main()
    assert ran == ["BAD", "GOOD"]
