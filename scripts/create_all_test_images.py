#!/usr/bin/env python3

import argparse
from glob import glob
import os
import subprocess
import sys
from pathlib import Path

# Always run from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "simulation_data" / "data"

DEFAULT_GRAD = "simulation_data/grad/grad_HCP.txt"

MODEL_GRAD = {
    "VERDICT": "simulation_data/grad/grad_verdict.txt",
    "SANDI": "simulation_data/grad/grad_sandi.txt",
    "IVIM": "simulation_data/grad/grad_ivim.txt",
    "BallStick": "simulation_data/grad/grad_HCP_with_deltas.txt",
    "ZeppelinZeppelin": "simulation_data/grad/grad_anisotropic_ivim.txt",
    "Ball": "simulation_data/grad/grad_HCP_with_deltas.txt",
    "Msdki": "simulation_data/grad/grad_HCP_with_deltas.txt",
    "Zeppelin": "simulation_data/grad/grad_HCP_with_deltas.txt",
    "Sphere": "simulation_data/grad/grad_verdict.txt",
    "Stick": "simulation_data/grad/grad_HCP_with_deltas.txt",
    "Astrosticks": "simulation_data/grad/grad_verdict.txt",
    "Ballt2Ballt2": "simulation_data/grad/grad_ivim_T2.txt",
}



def run_make_test_image(model_name: str, grad_path: str):
    cmd = [
        sys.executable,
        "-m", "src.utils.make_test_image",
        "-m", model_name,
        "-g", grad_path,
    ]

    print("\n>>> Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def get_image_and_mask(model_name: str):
    model_dir = os.path.join(DATA_ROOT, model_name)

    print(model_dir)

    data_files = list(glob(os.path.join(model_dir, model_name + "*_data.nii.gz")))
    mask_files = list(glob(os.path.join(model_dir, model_name + "*_mask.nii.gz")))

    print(data_files)
    print(mask_files)

    if len(data_files) != 1:
        raise RuntimeError(f"Expected 1 data file in {model_dir}, found {len(data_files)}")

    if len(mask_files) != 1:
        raise RuntimeError(f"Expected 1 mask file in {model_dir}, found {len(mask_files)}")

    return data_files[0], mask_files[0]


def run_fit(model_name: str, grad_path: str, image_path: Path, mask_path: Path):
    cmd = [
        sys.executable,
        "-m", "src.main",
        f"data.image={image_path}",
        f"data.mask={mask_path}",
        f"acquisition.grad={grad_path}",
        f"model.name={model_name}",
        f"plot.enabled=false",
    ]

    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run simulation + fitting pipeline.")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only a specific model (default: run all models)"
    )

    parser.add_argument(
        "--grad",
        type=str,
        default=None,
        help="Override gradient file path"
    )

    parser.add_argument(
        "--fit",
        action="store_true",
        help="Run fitting step as well"
    )

    args = parser.parse_args()

    # Determine which models to run
    if args.model:
        if args.model not in MODEL_GRAD:
            raise ValueError(f"Unknown model {args.model}")
        models_to_run = {args.model: MODEL_GRAD[args.model]}
    else:
        models_to_run = MODEL_GRAD

    for model, grad in models_to_run.items():
        grad_path = args.grad if args.grad else grad

        run_make_test_image(model, grad_path)
        image_path, mask_path = get_image_and_mask(model)

        if args.fit:
            run_fit(model, grad_path, image_path, mask_path)


if __name__ == "__main__":
    main()