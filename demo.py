"""Rebuild the public synthetic eye-tracking example and provenance manifest."""

import argparse
import json
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_ROOT / "demo"))
from eye_demo import run_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODULE_ROOT / "assets")
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()
    report = run_demo(args.output, args.seed)
    manifest = {
        "project": "eye-tracking",
        "data_type": "SYNTHETIC DEMO",
        "seed": args.seed,
        "command": "python demo.py",
        "generator": "Procedural grayscale eye ROIs with known pupil/glint coordinates",
        "detector": "Independent NumPy pixel detector; no access to ground truth",
        "evaluation": "90 calibration pairs, 32 held-out target pairs, 3 invalid-frame cases",
        "artifacts": [
            "assets/eye_tracking_demo.png",
            "assets/synthetic_metrics.json",
            "assets/synthetic_left_roi.png",
            "assets/synthetic_right_roi.png",
        ],
        "metrics": report["metrics"],
        "excluded_claims": [
            "real human eye accuracy",
            "device FPS",
            "original product measurements",
        ],
    }
    # When overriding the output directory, keep the manifest next to that output.
    directory = (
        MODULE_ROOT / "examples"
        if args.output.resolve() == (MODULE_ROOT / "assets").resolve()
        else args.output
    )
    directory.mkdir(parents=True, exist_ok=True)
    if directory == args.output:
        manifest["artifacts"] = [Path(name).name for name in manifest["artifacts"]]
    (directory / "demo_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
