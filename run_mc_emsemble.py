#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monte Carlo Ensemble with Latin Hypercube Sampling (LHS)
for NECO Ecosystem Model

- 25 independent parameter dimensions (kno3p2 & kno2p2 bound together)
- 1000 ensemble members
- Parallel execution using multiprocessing (16 cores)

Usage: Place this script in the same directory as NECO_ESS_perturb_splinemlz.f95
       Then run: python run_mc_ensemble.py
"""

import os
import sys
import re
import shutil
import subprocess
import time
import json
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
SOURCE_FILE = "NECO_ESS_perturb_splinemlz.f95"
COMPILER = "gfortran"
EXE_NAME = "NECO_ESS_perturb_splinemlz.exe"
RESULTS_DIR = "SplineMLD_ensemble_results"
N_ENSEMBLE = 1000
N_CORES = 16  # adjust to your CPU

# ============================================================
# PARAMETER DEFINITIONS
# Each entry: (variable_name, line_pattern, ref_value, low, high, distribution)
#   distribution: 'uniform' or 'gaussian'
#   For gaussian: low = mean, high = std
#   For uniform: sample from [low, high]
# ============================================================

# For yield parameters expressed as 1D0/XXX, we sample the denominator
# and reconstruct the expression

PARAMS = [
    # --- A. Nitrifier Growth ---
    # ynh4_bnh4: coded as "1D0/112D0", we vary the denominator (112)
    # Zakem: 112 ± 32, Gaussian
    {
        "name": "ynh4_bnh4_denom",
        "search": r"(ynh4_bnh4 = 1D0/)\d+\.?\d*D0",
        "replace_fmt": "ynh4_bnh4 = 1D0/{val:.1f}D0",
        "ref": 112.0,
        "dist": "gaussian",
        "mean": 112.0,
        "std": 32.0,
        "bounds": [50, 200],  # physical limits
    },
    {
        "name": "pn_max",
        "search": r"(pn_max = )\d+\.?\d*D0",
        "replace_fmt": "pn_max = {val:.2f}D0",
        "ref": 50.8,
        "dist": "uniform",
        "low": 25.4,   # -50%
        "high": 76.2,  # +50%
    },
    {
        "name": "kn",
        "search": r"(kn = )[\d.]+D-?\d+",
        "replace_fmt": "kn = {val}",
        "ref": 0.133e-3,
        "dist": "gaussian",
        "mean": 0.133e-3,
        "std": 0.038e-3,
        "bounds": [0.05e-3, 0.3e-3],
        "fortran_fmt": True,
    },
    # yno2_bno2: coded as "1D0/334D0"
    {
        "name": "yno2_bno2_denom",
        "search": r"(yno2_bno2 = 1D0/)\d+\.?\d*D0",
        "replace_fmt": "yno2_bno2 = 1D0/{val:.1f}D0",
        "ref": 334.0,
        "dist": "uniform",
        "low": 167.0,
        "high": 501.0,
    },
    {
        "name": "pn_max_noo",
        "search": r"(pn_max_noo = )\d+\.?\d*D0",
        "replace_fmt": "pn_max_noo = {val:.2f}D0",
        "ref": 40.3,
        "dist": "uniform",
        "low": 20.15,
        "high": 60.45,
    },
    {
        "name": "kn_noo",
        "search": r"(kn_noo = )[\d.]+D-?\d+",
        "replace_fmt": "kn_noo = {val}",
        "ref": 0.168e-3,
        "dist": "uniform",
        "low": 0.084e-3,
        "high": 0.252e-3,
        "fortran_fmt": True,
    },

    # --- B. Phytoplankton Growth ---
    {
        "name": "umaxp3",
        "search": r"(umaxp3 = )\d+\.?\d*D0",
        "replace_fmt": "umaxp3 = {val:.4f}D0",
        "ref": 0.65,
        "dist": "uniform",
        "low": 0.455,   # -30%
        "high": 0.845,  # +30%
    },
    {
        "name": "umaxp2",
        "search": r"(umaxp2 = )\d+\.?\d*D0",
        "replace_fmt": "umaxp2 = {val:.4f}D0",
        "ref": 0.65,
        "dist": "uniform",
        "low": 0.455,
        "high": 0.845,
    },
    {
        "name": "knh4p3",
        "search": r"(knh4p3 = )[\d.]+D-?\d+",
        "replace_fmt": "knh4p3 = {val}",
        "ref": 0.1e-3,
        "dist": "uniform",
        "low": 0.05e-3,
        "high": 0.15e-3,
        "fortran_fmt": True,
    },
    # kno3p2 and kno2p2 BOUND TOGETHER
    {
        "name": "kno3p2_kno2p2",
        "search": [
            r"(kno3p2 = )[\d.]+D-?\d+",
            r"(kno2p2 = )[\d.]+D-?\d+",
        ],
        "replace_fmt": [
            "kno3p2 = {val}",
            "kno2p2 = {val}",
        ],
        "ref": 0.3e-3,
        "dist": "uniform",
        "low": 0.15e-3,
        "high": 0.45e-3,
        "fortran_fmt": True,
        "bound": True,
    },

    # --- C. Heterotrophic Bacteria ---
    {
        "name": "yd_bo_pa",
        "search": r"(yd_bo_pa = )\d+\.?\d*D0",
        "replace_fmt": "yd_bo_pa = {val:.4f}D0",
        "ref": 0.20,
        "dist": "uniform",
        "low": 0.10,
        "high": 0.30,
    },
    {
        "name": "ydon_bo",
        "search": r"(ydon_bo = )\d+\.?\d*D0",
        "replace_fmt": "ydon_bo = {val:.4f}D0",
        "ref": 0.20,
        "dist": "uniform",
        "low": 0.10,
        "high": 0.30,
    },
    {
        "name": "pd_max",
        "search": r"(pd_max = )\d+\.?\d*D0",
        "replace_fmt": "pd_max = {val:.4f}D0",
        "ref": 1.0,
        "dist": "uniform",
        "low": 0.5,
        "high": 1.5,
    },
    {
        "name": "pdon_max",
        "search": r"(pdon_max = )\d+\.?\d*D0",
        "replace_fmt": "pdon_max = {val:.4f}D0",
        "ref": 1.0,
        "dist": "uniform",
        "low": 0.5,
        "high": 1.5,
    },
    {
        "name": "kd",
        "search": r"(kd = )[\d.]+D-?\d+",
        "replace_fmt": "kd = {val}",
        "ref": 0.05e-3,
        "dist": "uniform",
        "low": 0.025e-3,
        "high": 0.075e-3,
        "fortran_fmt": True,
    },
    {
        "name": "kdon",
        "search": r"(kdon = )[\d.]+D-?\d+",
        "replace_fmt": "kdon = {val}",
        "ref": 0.05e-3,
        "dist": "uniform",
        "low": 0.025e-3,
        "high": 0.075e-3,
        "fortran_fmt": True,
    },

    # --- D. Grazing & Mortality ---
    {
        "name": "gmax",
        "search": r"(gmax = )\d+\.?\d*D0",
        "replace_fmt": "gmax = {val:.4f}D0",
        "ref": 1.0,
        "dist": "uniform",
        "low": 0.5,
        "high": 1.5,
    },
    {
        "name": "kg",
        "search": r"(kg = )[\d.]+D-?\d+",
        "replace_fmt": "kg = {val}",
        "ref": 1e-3,
        "dist": "uniform",
        "low": 0.5e-3,
        "high": 1.5e-3,
        "fortran_fmt": True,
    },
    {
        "name": "gam",
        "search": r"(gam = )\d+\.?\d*D0",
        "replace_fmt": "gam = {val:.4f}D0",
        "ref": 0.5,
        "dist": "uniform",
        "low": 0.3,
        "high": 0.7,
    },
    {
        "name": "mz",
        "search": r"(mz = )[\d.]+D\d+",
        "replace_fmt": "mz = {val}",
        "ref": 700.0,
        "dist": "uniform",
        "low": 350.0,
        "high": 1050.0,
        "fortran_fmt_D3": True,
    },
    {
        "name": "mlin",
        "search": r"(mlin =  )\d+\.?\d*D-?\d+",
        "replace_fmt": "mlin =  {val}",
        "ref": 1e-2,
        "dist": "uniform",
        "low": 0.5e-2,
        "high": 1.5e-2,
        "fortran_fmt": True,
    },
    {
        "name": "mlin2",
        "search": r"(mlin2 = )\d+\.?\d*D-?\d+",
        "replace_fmt": "mlin2 = {val}",
        "ref": 1e-2,
        "dist": "uniform",
        "low": 0.5e-2,
        "high": 1.5e-2,
        "fortran_fmt": True,
    },
    {
        "name": "mlin3",
        "search": r"(mlin3 = )\d+\.?\d*D-?\d+",
        "replace_fmt": "mlin3 = {val}",
        "ref": 1e-2,
        "dist": "uniform",
        "low": 0.5e-2,
        "high": 1.5e-2,
        "fortran_fmt": True,
    },

    # --- E. Other ---
    {
        "name": "mortf",
        "search": r"(mortf = )\d+\.?\d*D0",
        "replace_fmt": "mortf = {val:.4f}D0",
        "ref": 0.0,
        "dist": "uniform",
        "low": 0.0,
        "high": 1.0,
    },
    {
        "name": "Ws",
        "search": r"(Ws = )\d+\.?\d*D0",
        "replace_fmt": "Ws = {val:.2f}D0",
        "ref": 10.0,
        "dist": "uniform",
        "low": 5.0,
        "high": 100.0,
    },
]


def to_fortran_d(val):
    """Convert a float to Fortran D-format string, e.g. 0.133E-3 -> 0.133D-3"""
    s = f"{val:.6E}"
    # Simplify: remove trailing zeros in mantissa
    parts = s.split("E")
    mantissa = parts[0].rstrip("0").rstrip(".")
    if "." not in mantissa:
        mantissa += ".0"
    exp = int(parts[1])
    return f"{mantissa}D{exp:+d}" if exp != 0 else f"{mantissa}D0"


def to_fortran_d3(val):
    """Format as X.XXD3 for values like mz = 0.7D3 (always use D3 exponent)"""
    mantissa = val / 1000.0
    return f"{mantissa:.4f}D3"


def generate_lhs_samples(n_samples, n_params, seed=42):
    """Generate Latin Hypercube Samples in [0,1]^n_params"""
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, n_params))
    for j in range(n_params):
        # Divide [0,1] into n_samples equal intervals
        intervals = np.linspace(0, 1, n_samples + 1)
        # Random point within each interval
        points = intervals[:-1] + rng.random(n_samples) * (1.0 / n_samples)
        # Random permutation
        rng.shuffle(points)
        samples[:, j] = points
    return samples


def transform_samples(lhs_unit, params):
    """Transform [0,1] LHS samples to actual parameter values"""
    n_samples = lhs_unit.shape[0]
    values = np.zeros_like(lhs_unit)

    for j, p in enumerate(params):
        u = lhs_unit[:, j]
        if p["dist"] == "uniform":
            values[:, j] = p["low"] + u * (p["high"] - p["low"])
        elif p["dist"] == "gaussian":
            from scipy.stats import norm
            raw = norm.ppf(u, loc=p["mean"], scale=p["std"])
            # Clip to physical bounds
            lo = p.get("bounds", [p["mean"] - 3*p["std"]])[0]
            hi = p.get("bounds", [0, p["mean"] + 3*p["std"]])[1]
            values[:, j] = np.clip(raw, lo, hi)

    return values


def modify_source(source_text, param_values, params):
    """Apply all parameter modifications to source text.
    Processes line-by-line to skip commented lines (starting with !)."""
    lines = source_text.split("\n")

    for j, p in enumerate(params):
        val = param_values[j]

        # Prepare search patterns and replacements
        if p.get("bound"):
            search_list = p["search"]
            repl_list = p["replace_fmt"]
        else:
            search_list = [p["search"]]
            repl_list = [p["replace_fmt"]]

        for search_pat, repl_fmt in zip(search_list, repl_list):
            # Format the value
            if p.get("fortran_fmt"):
                val_str = to_fortran_d(val)
                replacement = repl_fmt.format(val=val_str)
            elif p.get("fortran_fmt_D3"):
                val_str = to_fortran_d3(val)
                replacement = repl_fmt.format(val=val_str)
            else:
                replacement = repl_fmt.format(val=val)

            # Apply only to non-commented lines
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("!"):
                    continue  # skip comment lines
                if re.search(search_pat, line):
                    lines[i] = re.sub(search_pat, replacement, line)
                    break  # only replace first non-commented match

    return "\n".join(lines)


def run_single(args):
    """Run a single ensemble member"""
    idx, param_values, source_text, base_dir, params = args

    run_dir = os.path.join(base_dir, f"run_{idx:04d}")
    os.makedirs(run_dir, exist_ok=True)

    try:
        coef_file = "HOT_MLD_cubicspline_coefs.txt"
        if os.path.exists(coef_file):
            shutil.copy(coef_file, os.path.join(run_dir, coef_file))
        else:
            # 如果主目录下没找到，记录错误并返回
            return idx, False, f"Missing required file: {coef_file}"
        
        # Modify source
        modified = modify_source(source_text, param_values, params)

        # Write modified source
        src_path = os.path.join(run_dir, SOURCE_FILE)
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(modified)

        # Save parameter values for this run
        param_dict = {p["name"]: float(param_values[j]) for j, p in enumerate(params)}
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(param_dict, f, indent=2)

        # Compile
        exe_path = os.path.join(run_dir, EXE_NAME)
        result = subprocess.run(
            [COMPILER, "-O2", "-o", exe_path, src_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            with open(os.path.join(run_dir, "compile_error.log"), "w") as f:
                f.write(result.stderr)
            return idx, False, "compile_error"

        # Run model (from run_dir so output files go there)
        result = subprocess.run(
            [exe_path],
            capture_output=True, text=True,
            cwd=run_dir, timeout=600  # 10 min timeout
        )
        if result.returncode != 0:
            with open(os.path.join(run_dir, "runtime_error.log"), "w") as f:
                f.write(result.stdout + "\n" + result.stderr)
            return idx, False, "runtime_error"

        return idx, True, "success"

    except subprocess.TimeoutExpired:
        return idx, False, "timeout"
    except Exception as e:
        return idx, False, str(e)


def main():
    print("=" * 60)
    print(" NECO Monte Carlo Ensemble with LHS")
    print(f" Parameters: {len(PARAMS)}")
    print(f" Ensemble members: {N_ENSEMBLE}")
    print(f" Parallel cores: {N_CORES}")
    print("=" * 60)

    # Check source file
    if not os.path.exists(SOURCE_FILE):
        print(f"[ERROR] {SOURCE_FILE} not found!")
        sys.exit(1)

    # Check for ICfiles directory
    if not os.path.isdir("ICfiles"):
        print("[WARNING] ICfiles directory not found - model may need initial conditions")

    # Read source
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        source_text = f.read()

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate LHS samples
    print("\n[1/3] Generating Latin Hypercube Samples...")
    lhs_unit = generate_lhs_samples(N_ENSEMBLE, len(PARAMS), seed=42)

    # Check if scipy is available for Gaussian sampling
    has_scipy = True
    try:
        from scipy.stats import norm
    except ImportError:
        has_scipy = False
        print("[WARNING] scipy not found, using uniform for all (install: pip install scipy)")
        # Fall back to uniform for gaussian params
        for p in PARAMS:
            if p["dist"] == "gaussian":
                p["dist"] = "uniform"
                p["low"] = p["bounds"][0]
                p["high"] = p["bounds"][1]

    param_values = transform_samples(lhs_unit, PARAMS)

    # Save full parameter matrix
    param_names = [p["name"] for p in PARAMS]
    header = ",".join(param_names)
    np.savetxt(
        os.path.join(RESULTS_DIR, "parameter_matrix.csv"),
        param_values, delimiter=",", header=header, comments=""
    )
    print(f"  Parameter matrix saved: {RESULTS_DIR}/parameter_matrix.csv")

    # Copy ICfiles to each run directory will be handled by symlinks
    # First check if ICfiles exists
    icfiles_abs = os.path.abspath("ICfiles_mlz75") if os.path.isdir("ICfiles_mlz75") else None

    # Prepare tasks
    print(f"\n[2/3] Preparing {N_ENSEMBLE} ensemble runs...")
    tasks = []
    for i in range(N_ENSEMBLE):
        run_dir = os.path.join(RESULTS_DIR, f"run_{i:04d}")
        os.makedirs(run_dir, exist_ok=True)

        # Create ICfiles symlink or copy
        ic_dest = os.path.join(run_dir, "ICfiles_mlz75")
        if icfiles_abs and not os.path.exists(ic_dest):
            try:
                # Try symlink first (Windows may need admin)
                os.symlink(icfiles_abs, ic_dest)
            except (OSError, NotImplementedError):
                # Fall back to copy
                shutil.copytree(icfiles_abs, ic_dest)

        tasks.append((i, param_values[i], source_text, RESULTS_DIR, PARAMS))

    # Run in parallel
    print(f"\n[3/3] Running ensemble ({N_CORES} parallel processes)...")
    print("  This will take approximately 3-4 hours for 1000 runs.")
    print("  Progress updates every 50 completed runs.\n")

    start_time = time.time()
    success_count = 0
    fail_count = 0

    with Pool(processes=N_CORES) as pool:
        for i, (idx, success, msg) in enumerate(
            pool.imap_unordered(run_single, tasks), 1
        ):
            if success:
                success_count += 1
            else:
                fail_count += 1
                if fail_count <= 10:
                    print(f"  [FAIL] Run {idx:04d}: {msg}")

            if i % 50 == 0 or i == N_ENSEMBLE:
                elapsed = time.time() - start_time
                rate = i / elapsed * 60  # runs per minute
                remaining = (N_ENSEMBLE - i) / (i / elapsed) if i > 0 else 0
                print(
                    f"  Progress: {i}/{N_ENSEMBLE} "
                    f"(OK:{success_count} FAIL:{fail_count}) "
                    f"[{elapsed/60:.1f} min elapsed, "
                    f"~{remaining/60:.1f} min remaining, "
                    f"{rate:.1f} runs/min]"
                )

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f" Ensemble complete!")
    print(f" Successful: {success_count}/{N_ENSEMBLE}")
    print(f" Failed: {fail_count}/{N_ENSEMBLE}")
    print(f" Total time: {total_time/60:.1f} minutes")
    print(f" Results in: {RESULTS_DIR}/")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "n_ensemble": N_ENSEMBLE,
        "n_params": len(PARAMS),
        "n_success": success_count,
        "n_fail": fail_count,
        "total_time_minutes": round(total_time / 60, 1),
        "param_names": param_names,
    }
    with open(os.path.join(RESULTS_DIR, "ensemble_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
