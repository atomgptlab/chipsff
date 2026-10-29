#!/usr/bin/env python
import json
import math
import numpy as np
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data

# Load each jarvis dataset once per process. The zips are large (dft_3d ~48 MB)
# and re-loading them for every material dominates a multi-material benchmark.
_DATASET_CACHE = {}


def cached_data(name):
    """Return a jarvis dataset, loading (and caching) it only once."""
    if name not in _DATASET_CACHE:
        _DATASET_CACHE[name] = data(name)
    return _DATASET_CACHE[name]


# surfacedb calculation tags accepted for each reference functional, most
# preferred first. The database mixes PBE and r2SCAN slabs whose names differ
# only after the Miller indices, so the functional has to be chosen explicitly.
SURFACE_REFERENCE_TAGS = {
    "PBE": ["VASP_PBE_noDP", "VASP_PBE", "VASP_PBE_DP"],
    "R2SCAN": ["VASP_R2SCAN", "VASP_R2SCAN_R2SCAN"],
}


def parse_surface_name(name):
    """
    Split a surfacedb name into its JID, Miller indices and calculation tag.

    "Surface-JVASP-969_miller_0_0_1_x1015_thickness_16_VASP_R2SCAN_R2SCAN"
    gives ("JVASP-969", "0_0_1", "VASP_R2SCAN_R2SCAN").
    """
    jid, rest = name.split("Surface-")[1].split("_miller_")
    miller_part, thickness_part = rest.split("_thickness_")
    miller = "_".join(miller_part.split("_")[:3])
    tag = thickness_part.split("_", 1)[1]
    return jid, miller, tag


def collect_data(
    dft_3d=None, vacancydb=None, surface_data=None, surface_functional="PBE"
):
    """
    Attach vacancy and surface reference energies to each dft_3d entry.

    Vacancy references are keyed by the vacancydb id ("JVASP-1029_Ti_a"),
    so vacancies on different Wyckoff sites stay separate. Surface references
    use only calculations done with `surface_functional` ("PBE" or "R2SCAN").
    """
    if surface_functional not in SURFACE_REFERENCE_TAGS:
        raise ValueError(
            f"Unknown surface_functional {surface_functional!r}; "
            f"choose from {list(SURFACE_REFERENCE_TAGS)}"
        )
    tags = SURFACE_REFERENCE_TAGS[surface_functional]
    if dft_3d is None:
        dft_3d = cached_data("dft_3d")
    if vacancydb is None:
        vacancydb = cached_data("vacancydb")
    if surface_data is None:
        surface_data = cached_data("surfacedb")

    vacancy_refs = {}
    for vac_entry in vacancydb:
        vacancy_refs.setdefault(vac_entry["jid"], {}).setdefault(
            vac_entry["id"], vac_entry["ef"]
        )

    # For each surface keep the calculation with the most preferred tag
    surface_refs = {}
    for surf_entry in surface_data:
        jid, miller, tag = parse_surface_name(surf_entry["name"])
        if tag not in tags:
            continue
        key = f"Surface-{jid}_miller_{miller}"
        refs = surface_refs.setdefault(jid, {})
        if key not in refs or tags.index(tag) < tags.index(refs[key]["tag"]):
            refs[key] = {
                "tag": tag,
                "surf_en": surf_entry["surf_en"],
                "name": surf_entry["name"],
            }

    aggregated_data = []
    for entry in dft_3d:
        tmp = dict(entry)
        tmp["vacancy"] = dict(vacancy_refs.get(entry["jid"], {}))
        surfaces = surface_refs.get(entry["jid"], {})
        tmp["surface"] = {k: v["surf_en"] for k, v in surfaces.items()}
        tmp["surface_source"] = {k: v["name"] for k, v in surfaces.items()}
        aggregated_data.append(tmp)

    return aggregated_data


def get_vacancy_energy_entry(jid, aggregated_data):
    """
    Retrieve the vacancy formation energy entry (vac_en_entry) for a given jid.

    Parameters:
    jid (str): The JID of the material.
    aggregated_data (list): The aggregated data containing vacancy and surface information.

    Returns:
    dict: A dictionary containing the vacancy formation energy entry and corresponding symbol.
    """
    for entry in aggregated_data:
        if entry["jid"] == jid:
            vacancy_data = entry.get("vacancy", {})
            if vacancy_data:
                return [
                    {"symbol": key, "vac_en_entry": value}
                    for key, value in vacancy_data.items()
                ]
            else:
                return f"No vacancy data found for JID {jid}"
    return f"JID {jid} not found in the data."


def match_vacancy_reference(defect_name, computed_names, vacancy_entries):
    """
    Return the reference entry for a computed vacancy named
    "<jid>_<element>_<wyckoff letter>", or None.

    An exact id match is used when available. Otherwise, if the element has
    a single site in both `computed_names` and `vacancy_entries`, those are
    matched: the same site can get a different Wyckoff letter depending on
    the cell's origin (e.g. 2c or 2d in hcp metals).
    """
    for entry in vacancy_entries:
        if entry["symbol"] == defect_name:
            return entry
    element = defect_name.split("_")[-2]
    refs = [e for e in vacancy_entries if e["symbol"].split("_")[-2] == element]
    computed = [n for n in computed_names if n.split("_")[-2] == element]
    if len(refs) == 1 and len(computed) == 1:
        return refs[0]
    return None


def get_surface_energy_entry(jid, aggregated_data):
    """
    Retrieve the surface energy entry (surf_en_entry) for a given jid.

    Parameters:
    jid (str): The JID of the material.
    aggregated_data (list): The aggregated data containing vacancy and surface information.

    Returns:
    list: A list of dictionaries containing the surface energy entry and corresponding name.
    """
    for entry in aggregated_data:
        if entry["jid"] == jid:
            surface_data = entry.get("surface", {})
            sources = entry.get("surface_source", {})
            if surface_data:
                return [
                    {
                        "name": key,
                        "surf_en_entry": value,
                        "reference_name": sources.get(key),
                    }
                    for key, value in surface_data.items()
                ]
            else:
                return f"No surface data found for JID {jid}"
    return f"JID {jid} not found in the data."


def log_job_info(message, log_file):
    """Log job information to a file and print it."""
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)


def save_dict_to_json(data_dict, filename):
    with open(filename, "w") as f:
        json.dump(data_dict, f, indent=4)


def load_dict_from_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def strain_atoms(atoms, strain):
    """
    Scale the cell of JARVIS `atoms` by (1 + strain) along each axis,
    moving the atoms with it.

    jarvis Atoms.strain_atoms keeps Cartesian coordinates fixed when the
    structure has them (as ase_to_atoms gives), which distorts every cell
    with more than one atom.
    """
    lattice_mat = np.dot(np.array(atoms.lattice_mat), (1 + strain) * np.eye(3))
    return Atoms(
        lattice_mat=lattice_mat,
        elements=atoms.elements,
        coords=atoms.frac_coords,
        cartesian=False,
    )


def reference_value(value):
    """Return a JARVIS reference value as a float, or None if it is missing ("na")."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def elastic_reference(tensor, i, j):
    """Return element (i, j) of a JARVIS elastic tensor, or None if unavailable."""
    try:
        return reference_value(tensor[i][j])
    except (TypeError, IndexError, KeyError):
        return None


def absolute_error(reference, prediction):
    """Absolute error, or NaN when either value is missing."""
    if reference is None or prediction is None:
        return float("nan")
    return abs(prediction - reference)


def get_jid_list():
    jids = [
        "JVASP-8184",
        "JVASP-10591",
        "JVASP-8118",
        "JVASP-8003",
        "JVASP-1222",
        "JVASP-106363",
        "JVASP-1109",
        "JVASP-96",
        "JVASP-20092",
        "JVASP-30",
        "JVASP-1372",
        "JVASP-23",
        "JVASP-105410",
        "JVASP-36873",
        "JVASP-113",
        "JVASP-7836",
        "JVASP-861",
        "JVASP-9117",
        "JVASP-108770",
        "JVASP-9147",
        "JVASP-1180",
        "JVASP-10703",
        "JVASP-79522",
        "JVASP-21211",
        "JVASP-1195",
        "JVASP-8082",
        "JVASP-1186",
        "JVASP-802",
        "JVASP-8559",
        "JVASP-14968",
        "JVASP-43367",
        "JVASP-22694",
        "JVASP-3510",
        "JVASP-36018",
        "JVASP-90668",
        "JVASP-110231",
        "JVASP-149916",
        "JVASP-1103",
        "JVASP-1177",
        "JVASP-1115",
        "JVASP-1112",
        "JVASP-25",
        "JVASP-10037",
        "JVASP-103127",
        "JVASP-813",
        "JVASP-1067",
        "JVASP-825",
        "JVASP-14616",
        "JVASP-111005",
        "JVASP-1002",
        "JVASP-99732",
        "JVASP-54",
        "JVASP-133719",
        "JVASP-1183",
        "JVASP-62940",
        "JVASP-14970",
        "JVASP-34674",
        "JVASP-107",
        "JVASP-58349",
        "JVASP-110",
        "JVASP-1915",
        "JVASP-816",
        "JVASP-867",
        "JVASP-34249",
        "JVASP-1216",
        "JVASP-32",
        "JVASP-1201",
        "JVASP-2376",
        "JVASP-18983",
        "JVASP-943",
        "JVASP-104764",
        "JVASP-39",
        "JVASP-10036",
        "JVASP-1312",
        "JVASP-8554",
        "JVASP-1174",
        "JVASP-8158",
        "JVASP-131",
        "JVASP-36408",
        "JVASP-85478",
        "JVASP-972",
        "JVASP-106686",
        "JVASP-1008",
        "JVASP-4282",
        "JVASP-890",
        "JVASP-1192",
        "JVASP-91",
        "JVASP-104",
        "JVASP-963",
        "JVASP-1189",
        "JVASP-149871",
        "JVASP-5224",
        "JVASP-41",
        "JVASP-1240",
        "JVASP-1408",
        "JVASP-1023",
        "JVASP-1029",
        "JVASP-149906",
        "JVASP-1327",
        "JVASP-29539",
        "JVASP-19780",
        "JVASP-85416",
        "JVASP-9166",
        "JVASP-1198",
    ]
    return jids
