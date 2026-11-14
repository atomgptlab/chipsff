#!/usr/bin/env python

# SlakoNet parameter sets are large (~200 MB) and slow to load, so keep one
# instance per parameter-set name for the lifetime of the process.
_SLAKONET_MODEL_CACHE = {}


def setup_calculator(calculator_type, calculator_settings):
    """
    Initializes and returns the appropriate calculator based on the calculator type and its settings.

    Args:
        calculator_type (str): The type/name of the calculator.
        calculator_settings (dict): Settings specific to the calculator.

    Returns:
        calculator: An instance of the specified calculator.
    """
    if calculator_type == "matgl":
        import matgl
        from matgl.ext.ase import PESCalculator

        model_name = calculator_settings.get(
            "model", "M3GNet-PES-MatPES-PBE-2025.2"
        )
        pot = matgl.load_model(model_name)
        return PESCalculator(pot, stress_unit="eV/A3", stress_weight=1.0)

    elif calculator_type == "matgl-direct":
        import matgl
        from matgl.ext.ase import M3GNetCalculator

        model_name = calculator_settings.get(
            "model", "M3GNet-MP-2021.2.8-DIRECT-PES"
        )
        pot = matgl.load_model(model_name)
        compute_stress = calculator_settings.get("compute_stress", True)
        stress_weight = calculator_settings.get("stress_weight", 0.01)
        return M3GNetCalculator(
            pot, compute_stress=compute_stress, stress_weight=stress_weight
        )

    elif calculator_type == "alignn_ff":
        from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

        # Honor optional settings so a custom-trained model directory can be
        # evaluated (e.g. a new default candidate) instead of the shipped
        # model. Falls back to the packaged default when unset.
        path = calculator_settings.get("path") or default_path()
        model_filename = calculator_settings.get(
            "model_filename", "best_model.pt"
        )
        ff_kwargs = {"path": path, "model_filename": model_filename}
        for key in (
            "device",
            "stress_wt",
            "force_mult_natoms",
            "force_multiplier",
        ):
            if key in calculator_settings:
                ff_kwargs[key] = calculator_settings[key]
        return AlignnAtomwiseCalculator(**ff_kwargs)
    elif calculator_type == "fairchem":
        # FairChem universal models (e.g. Meta UMA). Requires fairchem-core v2.
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

        model_name = calculator_settings.get("model_name", "uma-s-1p1")
        device = calculator_settings.get("device", "cuda")
        task_name = calculator_settings.get("task_name", "omat")
        pu = pretrained_mlip.get_predict_unit(model_name, device=device)
        return FAIRChemCalculator(pu, task_name=task_name)

    elif calculator_type == "mattersim":
        from mattersim.forcefield import MatterSimCalculator

        return MatterSimCalculator(
            load_path="MatterSim-v1.0.0-5M.pth", device="cpu"
        )

    elif calculator_type == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator()

    elif calculator_type == "mace":
        from mace.calculators import mace_mp

        return mace_mp(model="medium")

    elif calculator_type == "mace-mpa":
        from mace.calculators import mace_mp

        return mace_mp(model="medium-mpa-0")

    elif calculator_type == "mace-d3":
        from mace.calculators import mace_mp

        return mace_mp(dispersion=True)

    elif calculator_type == "mace-alexandria":
        from mace.calculators.mace import MACECalculator

        # TODO: Make an option to provide path
        model_path = calculator_settings.get(
            "model_path",
            "/users/dtw2/utils/models/alexandria_v2/mace/2D_universal_force_field_cpu.model",
        )
        device = calculator_settings.get("device", "cpu")
        return MACECalculator(model_path, device=device)

    elif calculator_type == "sevennet":
        from sevenn.sevennet_calculator import SevenNetCalculator

        checkpoint_path = calculator_settings.get(
            "checkpoint_path",
            "/users/dtw2/SevenNet/pretrained_potentials/SevenNet_0__11July2024/checkpoint_sevennet_0.pth",
        )
        device = calculator_settings.get("device", "cpu")
        return SevenNetCalculator(checkpoint_path, device=device)

    elif calculator_type == "orb-v2":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        orbff = pretrained.orb_v2()
        device = calculator_settings.get("device", "cpu")
        return ORBCalculator(orbff, device=device)

    elif calculator_type == "orb-d3-v2":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        orbff = pretrained.orb_d3_v2()
        device = calculator_settings.get("device", "cpu")
        return ORBCalculator(orbff, device=device)

    elif calculator_type == "uma":
        # fairchem v2 universal model (UMA). Needs fairchem-core>=2 + HF access.
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

        name = calculator_settings.get("model_name", "uma-s-1p1")
        task = calculator_settings.get("task_name", "omat")
        device = calculator_settings.get("device", "cpu")
        pu = pretrained_mlip.get_predict_unit(name, device=device)
        return FAIRChemCalculator(pu, task_name=task)

    elif calculator_type == "eqV2_31M_omat":
        from fairchem.core import OCPCalculator

        checkpoint_path = calculator_settings.get(
            "checkpoint_path",
            "/users/dtw2/fairchem-models/pretrained_models/eqV2_31M_omat.pt",
        )
        return OCPCalculator(checkpoint_path=checkpoint_path)

    elif calculator_type == "eqV2_86M_omat":
        from fairchem.core import OCPCalculator

        checkpoint_path = calculator_settings.get(
            "checkpoint_path",
            "/users/dtw2/fairchem-models/pretrained_models/eqV2_86M_omat.pt",
        )
        return OCPCalculator(checkpoint_path=checkpoint_path)

    elif calculator_type == "eqV2_153M_omat":
        from fairchem.core import OCPCalculator

        checkpoint_path = calculator_settings.get(
            "checkpoint_path",
            "/users/dtw2/fairchem-models/pretrained_models/eqV2_153M_omat.pt",
        )
        return OCPCalculator(checkpoint_path=checkpoint_path)

    elif calculator_type == "eqV2_31M_omat_mp_salex":
        from fairchem.core import OCPCalculator

        checkpoint_path = calculator_settings.get(
            "checkpoint_path",
            "/users/dtw2/fairchem-models/pretrained_models/eqV2_31M_omat_mp_salex.pt",
        )
        return OCPCalculator(checkpoint_path=checkpoint_path)

    elif calculator_type == "eqV2_86M_omat_mp_salex":
        from fairchem.core import OCPCalculator

        checkpoint_path = calculator_settings.get(
            "checkpoint_path",
            "/users/dtw2/fairchem-models/pretrained_models/eqV2_86M_omat_mp_salex.pt",
        )
        return OCPCalculator(checkpoint_path=checkpoint_path)

    elif calculator_type in (
        "slakonet",
        "slakonet_v0",
        "slakonet_v1",
        "slakonet_v1a",
    ):
        # SlakoNet: universal DFTB (tight-binding) parameter sets.
        # https://github.com/atomgptlab/slakonet
        import torch
        from slakonet.main import SlakoNetCalculator
        from slakonet.optim import default_model, DEFAULT_MODEL_NAME

        if calculator_type == "slakonet":
            model_name = calculator_settings.get(
                "model_name", DEFAULT_MODEL_NAME
            )
        else:
            model_name = calculator_settings.get("model_name", calculator_type)

        device = calculator_settings.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        # SlakoNet is a k-point method. One calculator instance is reused
        # across bulk cells, defect supercells and slabs, so prefer
        # `kspacing` (1/Angstrom) and let the mesh follow each cell: a
        # fixed mesh that suits a 64-atom supercell leaves spurious forces
        # well above a typical fmax=0.05 criterion on a 2-atom cell.
        kspacing = calculator_settings.get("kspacing", 0.30)
        kpoints_array = calculator_settings.get("kpoints_array", [1, 1, 1])
        if calculator_settings.get("kpoints_array") is not None and (
            "kspacing" not in calculator_settings
        ):
            kspacing = None  # explicit mesh requested, honour it

        model = _SLAKONET_MODEL_CACHE.get(model_name)
        if model is None:
            model = default_model(model_name=model_name)
            model = model.to(device).float()
            model.eval()
            _SLAKONET_MODEL_CACHE[model_name] = model

        return SlakoNetCalculator(
            model=model,
            kpoints_array=kpoints_array,
            kspacing=kspacing,
            device=device,
            compute_forces=calculator_settings.get("compute_forces", True),
        )

    elif calculator_type == "ase":
        # Generic bring-your-own ASE calculator: no per-model code needed.
        # settings["spec"] = "module.path:callable_or_Class" (imported and,
        # if callable, called with settings["kwargs"]); returns an ASE calc.
        # Lets any new force field with an ASE calculator plug in directly.
        import importlib
        import json as _json

        spec = calculator_settings.get("spec")
        if not spec:
            raise ValueError("calculator_type 'ase' needs settings['spec']")
        kwargs = calculator_settings.get("kwargs") or {}
        if isinstance(kwargs, str):
            kwargs = _json.loads(kwargs) if kwargs.strip() else {}
        mod_name, _, attr = spec.partition(":")
        if not attr:
            raise ValueError(
                "spec must be 'module.path:callable_or_Class', got " + spec
            )
        obj = getattr(importlib.import_module(mod_name), attr)
        return obj(**kwargs) if callable(obj) else obj

    else:
        raise ValueError(f"Unsupported calculator type: {calculator_type}")
