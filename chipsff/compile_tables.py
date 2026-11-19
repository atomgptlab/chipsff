"""Compile the per-model CHIPS-FF outputs into the combined paper tables.

After running ``chipsff.run_main`` for several models (each writes its own
artifacts), run this in the same directory to merge them into multi-row tables:

    python -m chipsff.compile_tables

It scans the current directory for the artifacts written by ``run_main.py``:

  chipsff_table_<tag>.csv        -> Table 4 (properties): one row per model
  scaling_<tag>/scaling.json     -> Table 5 (inference scaling)
  diatomics_<tag>/diatomics.json -> tortuosity (tau) column, if present
  wbm_<tag>/wbm_score.json       -> WBM e_form MAE / F1 / RMSD, if present

and writes, for each table, ``<name>.csv``, ``<name>.md`` and ``<name>.tex``.
"""

import os
import csv
import glob
import json
import argparse


def _rows_from_property_csvs(root):
    """Return (header, rows) merged from every chipsff_table_<tag>.csv."""
    header, rows = None, []
    for fp in sorted(glob.glob(os.path.join(root, "chipsff_table_*.csv"))):
        with open(fp) as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        if len(lines) < 2:
            continue
        h = lines[0].split(",")
        header = header or h
        for data in lines[1:]:
            rows.append(dict(zip(h, data.split(","))))
    return header, rows


def _tag_from(path, prefix, suffix=""):
    base = os.path.basename(path.rstrip("/"))
    return base[len(prefix) : len(base) - len(suffix) if suffix else None]


def _tortuosity(root):
    out = {}
    for fp in glob.glob(os.path.join(root, "diatomics_*", "diatomics.json")):
        tag = os.path.basename(os.path.dirname(fp))[len("diatomics_") :]
        try:
            out[tag] = round(float(json.load(open(fp)).get("tortuosity")), 3)
        except Exception:
            pass
    return out


def _wbm(root):
    out = {}
    for fp in glob.glob(os.path.join(root, "wbm_*", "wbm_score.json")):
        tag = os.path.basename(os.path.dirname(fp))[len("wbm_") :]
        try:
            out[tag] = json.load(open(fp)).get("metrics", {})
        except Exception:
            pass
    return out


def _write(name, header, rows):
    """Write CSV + Markdown + LaTeX for a table (header + list-of-lists)."""
    with open(name + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(name + ".md", "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    with open(name + ".tex", "w") as f:
        f.write(
            "\\begin{tabular}{l" + "r" * (len(header) - 1) + "}\n\\toprule\n"
        )
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(str(x) for x in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {name}.csv / .md / .tex  ({len(rows)} rows)")


def compile_table4(root):
    """Table 4: properties (+ tortuosity / WBM columns if present)."""
    header, rows = _rows_from_property_csvs(root)
    if not rows:
        print("Table 4: no chipsff_table_*.csv found; skipping.")
        return
    tau = _tortuosity(root)
    wbm = _wbm(root)
    extra = []
    if tau:
        extra.append("tau")
    wbm_cols = []
    if wbm:
        wbm_cols = ["WBM_eform", "WBM_F1", "WBM_RMSD"]
    out_hdr = header + extra + wbm_cols
    out_rows = []
    for r in rows:
        tag = r["model"].split()[0] if "model" in r else list(r.values())[0]
        row = [r.get(h, "") for h in header]
        if tau:
            row.append(tau.get(tag, "--"))
        if wbm:
            m = wbm.get(tag, {})
            row += [
                m.get("e_form_mae", "--"),
                m.get("F1", "--"),
                m.get("RMSD", "--"),
            ]
        out_rows.append(row)
    _write(os.path.join(root, "chipsff_table4_properties"), out_hdr, out_rows)


def compile_table5(root):
    """Table 5: inference scaling (timing + memory)."""
    hdr = ["model", "max_atoms", "mem_at_max_GB", "t@21952_s", "OOM_at"]
    rows = []
    for fp in sorted(
        glob.glob(os.path.join(root, "scaling_*", "scaling.json"))
    ):
        tag = os.path.basename(os.path.dirname(fp))[len("scaling_") :]
        try:
            s = json.load(open(fp)).get("summary", {})
        except Exception:
            continue
        rows.append(
            [
                tag,
                s.get("max_atoms", "--"),
                s.get("mem_at_max_gb", "--"),
                s.get("t_at_21952_s", "--"),
                s.get("oom_at", "--"),
            ]
        )
    if not rows:
        print("Table 5: no scaling_*/scaling.json found; skipping.")
        return
    _write(os.path.join(root, "chipsff_table5_scaling"), hdr, rows)


def main():
    ap = argparse.ArgumentParser(
        description="Compile per-model CHIPS-FF outputs into Tables 4/5."
    )
    ap.add_argument(
        "--dir", default=".", help="directory with run_main outputs"
    )
    args = ap.parse_args()
    root = os.path.abspath(args.dir)
    print(f"compiling tables from {root}")
    compile_table4(root)
    compile_table5(root)


if __name__ == "__main__":
    main()
