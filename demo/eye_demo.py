"""Independent synthetic pupil/glint validation. No private detector or eye data.

The detector consumes pixels only; generator ground truth is used for evaluation.
This deliberately small baseline does not implement a production eye tracker.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Detection:
    valid: bool
    pupil: np.ndarray | None = None
    glints: np.ndarray | None = None
    axes: np.ndarray | None = None
    reason: str = ""

    @property
    def feature(self) -> np.ndarray:
        if not self.valid:
            raise ValueError("An invalid detection has no calibration feature")
        return self.pupil - self.glints.mean(axis=0)


def synthetic_eye(target, side, rng, blink=False, missing_glint=False):
    """Return a procedural grayscale ROI and evaluation-only ground truth."""
    y, x = np.mgrid[:128, :224]
    u, v = np.asarray(target, dtype=float) - 0.5
    if side == "left":
        pupil = np.array([112 + 42 * u + 7 * v, 65 + 4 * u + 32 * v])
        glints = np.array([[106.0, 58.0], [120.0, 59.0]])
    elif side == "right":
        pupil = np.array([110 + 39 * u - 6 * v, 65 - 3 * u + 34 * v])
        glints = np.array([[104.0, 58.0], [118.0, 59.0]])
    else:
        raise ValueError("side must be left or right")
    pupil += rng.normal(0, 0.28, 2)
    glints += rng.normal(0, 0.08, (2, 2))
    skin = 152 + 8 * np.cos(x / 27) + 6 * y / 128
    eye_open = ((x - 112) / 102) ** 2 + ((y - 64) / 45) ** 2 < 1
    image = np.where(eye_open, 187 + 3 * np.sin(x / 15), skin)
    iris = ((x - 112) / 58) ** 2 + ((y - 64) / 42) ** 2 < 1
    texture = 77 + 5 * np.sin(np.arctan2(y - 64, x - 112) * 35)
    image = np.where(iris & eye_open, texture, image)
    pupil_mask = ((x - pupil[0]) / 23) ** 2 + ((y - pupil[1]) / 18) ** 2 < 1
    image[pupil_mask & eye_open] = 22
    for index, (gx, gy) in enumerate(glints):
        if missing_glint and index == 1:
            continue
        image[(x - gx) ** 2 + (y - gy) ** 2 <= 3.0**2] = 248
    if blink:
        image = skin.copy()
        image[np.abs(y - (65 + 0.0008 * (x - 112) ** 2)) < 1.2] = 65
    image += rng.normal(0, 1.4, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), {"pupil": pupil, "glints": glints}


def components(mask):
    """Four-connected components; intentionally transparent NumPy/Python code."""
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    found = []
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        pixels = []
        while stack:
            py, px = stack.pop()
            pixels.append((py, px))
            for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        found.append(np.asarray(pixels))
    return found


def detect(image):
    """Threshold -> largest dark component -> row-span fill -> two bright blobs."""
    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        raise ValueError("Expected a finite, non-empty grayscale ROI")
    pupils = [c for c in components(image < 45) if len(c) >= 100]
    if not pupils:
        return Detection(False, reason="no_pupil")
    candidate = max(pupils, key=len)
    # Fill glint holes row by row, so bright reflections do not bias the center.
    filled = []
    for row in np.unique(candidate[:, 0]):
        xs = candidate[candidate[:, 0] == row, 1]
        filled.extend(
            (int(px), int(row)) for px in range(int(xs.min()), int(xs.max()) + 1)
        )
    xy = np.asarray(filled, dtype=float)
    pupil = xy.mean(axis=0)
    axes = 2 * np.sqrt(np.mean((xy - pupil) ** 2, axis=0))
    blobs = [c for c in components(image > 230) if 8 <= len(c) <= 80]
    centers = [c[:, ::-1].mean(axis=0) for c in blobs]
    near = [p for p in centers if np.linalg.norm(p - pupil) < 50]
    if len(near) != 2:
        return Detection(False, pupil=pupil, axes=axes, reason="expected_two_glints")
    glints = np.asarray(sorted(near, key=lambda p: p[0]))
    spacing = np.linalg.norm(glints[1] - glints[0])
    if not 7 <= spacing <= 25:
        return Detection(False, pupil=pupil, axes=axes, reason="invalid_glint_spacing")
    return Detection(True, pupil, glints, axes)


class AffineCalibration:
    """Four raw pupil-minus-glint features plus bias; no feature mixing."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (5, 2) or not np.isfinite(self.weights).all():
            raise ValueError("Calibration weights must be a finite (5, 2) array")

    @classmethod
    def fit(cls, features, targets):
        x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
        if x.ndim != 2 or x.shape[1] != 4 or y.shape != (len(x), 2) or len(x) < 9:
            raise ValueError("Need at least nine labeled stereo samples")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Training data must be finite")
        if np.linalg.matrix_rank(y - y.mean(axis=0)) < 2:
            raise ValueError("Calibration targets must span both screen axes")
        design = np.column_stack([x, np.ones(len(x))])
        weights, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 5:
            raise ValueError("Calibration features do not identify the affine model")
        return cls(weights)

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.shape != (4,) or not np.isfinite(x).all():
            raise ValueError("Expected four finite stereo features")
        return np.append(x, 1) @ self.weights


class GazeSession:
    """Reject unsynchronized/old frames; retain last point with explicit validity."""

    def __init__(self, calibration):
        self.calibration = calibration
        self.last = None
        self.timestamp = -1

    def process(self, left, right, left_timestamp_us, right_timestamp_us):
        if left_timestamp_us != right_timestamp_us:
            raise ValueError("Stereo timestamps must match")
        if left_timestamp_us <= self.timestamp:
            raise ValueError("Timestamps must increase monotonically")
        self.timestamp = left_timestamp_us
        if not left.valid or not right.valid:
            return {
                "valid": False,
                "held": self.last is not None,
                "gaze": None if self.last is None else self.last.tolist(),
                "timestamp_us": left_timestamp_us,
            }
        self.last = self.calibration.predict(np.r_[left.feature, right.feature])
        return {
            "valid": True,
            "held": False,
            "gaze": self.last.tolist(),
            "timestamp_us": left_timestamp_us,
        }


def font(size, bold=False):
    candidates = (
        [
            "C:/Windows/Fonts/seguisb.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for name in candidates:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def overlay(image, result, scale=3):
    rgb = (
        Image.fromarray(image)
        .convert("RGB")
        .resize((image.shape[1] * scale, image.shape[0] * scale))
    )
    draw = ImageDraw.Draw(rgb)
    if result.pupil is not None:
        px, py = result.pupil * scale
        ax, ay = result.axes * scale
        draw.ellipse((px - ax, py - ay, px + ax, py + ay), outline="#4FF3BF", width=3)
        draw.line((px - 8, py, px + 8, py), fill="#4FF3BF", width=2)
        draw.line((px, py - 8, px, py + 8), fill="#4FF3BF", width=2)
    if result.valid:
        a, b = result.glints * scale
        draw.line((*a, *b), fill="#FFC97B", width=2)
        for gx, gy in [a, b]:
            draw.ellipse(
                (gx - 10, gy - 10, gx + 10, gy + 10), outline="#FFC97B", width=2
            )
        midpoint = result.glints.mean(axis=0) * scale
        draw.line((*midpoint, *result.pupil * scale), fill="#6CBFFF", width=3)
    return rgb


def render_dashboard(
    out_dir, examples, metrics, target_points, predictions, train_targets
):
    page = Image.new("RGB", (1600, 1130), "#0D1525")
    d = ImageDraw.Draw(page)
    white, muted, mint = "#E9EFF7", "#96A7BD", "#4FF3BF"
    d.text((60, 38), "EYE TRACKING / ENGINEERING", fill=mint, font=font(19, True))
    d.text(
        (60, 75), "From eye pixels to gaze coordinates", fill=white, font=font(43, True)
    )
    d.rounded_rectangle((1215, 42, 1540, 88), radius=10, fill="#3A3022")
    d.text((1236, 50), "SYNTHETIC DEMO", fill="#FFD18A", font=font(26, True))
    d.text(
        (60, 136),
        "Independent pixel detector + explicit stereo state + nine-point affine calibration",
        fill=muted,
        font=font(23),
    )
    for index, (side, image, result) in enumerate(examples):
        x = 60 + index * 750
        d.rounded_rectangle((x, 192, x + 730, 673), radius=18, fill="#172237")
        d.text((x + 24, 213), side.upper() + " EYE", fill=white, font=font(23, True))
        d.text((x + 455, 216), "PIXEL DETECTION", fill=mint, font=font(17, True))
        panel = overlay(image, result)
        page.paste(panel, (x + 29, 264))
        d.text(
            (x + 30, 648),
            f"dx {result.feature[0]:+.2f} px    dy {result.feature[1]:+.2f} px",
            fill=white,
            font=font(16),
        )
    for x, label, color in [
        (60, "Pupil / filled-mask moments", mint),
        (500, "Two detected glints / midpoint", "#FFC97B"),
        (1040, "Raw pupil - midpoint feature", "#6CBFFF"),
    ]:
        d.ellipse((x, 695, x + 12, 707), fill=color)
        d.text((x + 23, 688), label, fill=muted, font=font(19))
    d.rounded_rectangle((60, 741, 905, 1050), radius=18, fill="#172237")
    d.text((86, 761), "HELD-OUT CALIBRATION CHECK", fill=white, font=font(23, True))
    x0, y0, pw, ph = 108, 818, 492, 200
    d.rectangle((x0, y0, x0 + pw, y0 + ph), outline="#3C4D65", width=2)

    def screen(p):
        return x0 + float(p[0]) * pw, y0 + float(p[1]) * ph

    for p in train_targets:
        xx, yy = screen(p)
        d.line((xx - 6, yy, xx + 6, yy), fill="#546982", width=2)
        d.line((xx, yy - 6, xx, yy + 6), fill="#546982", width=2)
    for t, p in zip(target_points, predictions):
        tx, ty = screen(t)
        px, py = screen(p)
        d.line((tx, ty, px, py), fill="#FFC97B", width=2)
        d.ellipse((tx - 5, ty - 5, tx + 5, ty + 5), outline="#6CBFFF", width=2)
        d.ellipse((px - 2, py - 2, px + 2, py + 2), fill=mint)
    d.text((633, 828), "+ Nine training targets", fill=muted, font=font(16))
    d.text((633, 862), "o Unseen target", fill="#6CBFFF", font=font(19))
    d.text((633, 896), "  Predicted point", fill=mint, font=font(19))
    d.text((633, 947), "Normalized screen", fill=muted, font=font(18))
    d.text((633, 976), "x, y in [0, 1]", fill=muted, font=font(18))
    d.rounded_rectangle((928, 741, 1540, 1050), radius=18, fill="#172237")
    values = [
        (
            f"{metrics['pupil_mean_error_px']:.2f} px",
            "Mean pupil error / synthetic ROI",
        ),
        (
            f"{100*metrics['heldout_gaze_rmse_normalized']:.2f}%",
            "Held-out gaze RMSE / normalized units",
        ),
        (f"{metrics['rejected_invalid_frames']} / 3", "Invalid stereo frames rejected"),
    ]
    for index, (value, label) in enumerate(values):
        yy = 760 + index * 91
        d.text((954, yy), value, fill=mint, font=font(33, True))
        d.text((954, yy + 45), label, fill=muted, font=font(18))
    d.text(
        (60, 1080),
        "Procedural data only. No real-subject accuracy, device benchmark, or historical FPS claim.",
        fill=muted,
        font=font(21),
    )
    page.save(out_dir / "eye_tracking_demo.png")


def run_demo(out_dir, seed=27):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    training_targets = [(x, y) for y in (0.15, 0.5, 0.85) for x in (0.15, 0.5, 0.85)]
    features, labels, pupil_errors = [], [], []
    for target in training_targets:
        for _ in range(10):
            detected = []
            for side in ("left", "right"):
                roi, truth = synthetic_eye(target, side, rng)
                result = detect(roi)
                if not result.valid:
                    raise RuntimeError("Synthetic calibration detection failed")
                detected.append(result)
                pupil_errors.append(
                    float(np.linalg.norm(result.pupil - truth["pupil"]))
                )
            features.append(np.r_[detected[0].feature, detected[1].feature])
            labels.append(target)
    model = AffineCalibration.fit(features, labels)
    session = GazeSession(model)
    targets = rng.uniform(0.18, 0.82, (32, 2))
    predictions = []
    records = []
    examples = []
    for index, target in enumerate(targets):
        pair = []
        for side in ("left", "right"):
            roi, truth = synthetic_eye(target, side, rng)
            result = detect(roi)
            pair.append(result)
            pupil_errors.append(float(np.linalg.norm(result.pupil - truth["pupil"])))
            if index == 0:
                examples.append((side, roi, result))
                Image.fromarray(roi).save(out_dir / f"synthetic_{side}_roi.png")
        packet = session.process(*pair, 1000 + index * 1000, 1000 + index * 1000)
        if not packet["valid"]:
            raise RuntimeError("Unexpected holdout detection failure")
        predictions.append(packet["gaze"])
        records.append({"target": target.tolist(), **packet})
    invalid = []
    for index, case in enumerate(("left_blink", "right_blink", "missing_glint")):
        left, _ = synthetic_eye(
            (0.5, 0.5),
            "left",
            rng,
            blink=case == "left_blink",
            missing_glint=case == "missing_glint",
        )
        right, _ = synthetic_eye((0.5, 0.5), "right", rng, blink=case == "right_blink")
        invalid.append(
            {
                "case": case,
                **session.process(
                    detect(left),
                    detect(right),
                    40000 + index * 1000,
                    40000 + index * 1000,
                ),
            }
        )
    predictions = np.asarray(predictions)
    distances = np.linalg.norm(predictions - targets, axis=1)
    metrics = {
        "pupil_mean_error_px": float(np.mean(pupil_errors)),
        "heldout_gaze_rmse_normalized": float(np.sqrt(np.mean(distances**2))),
        "heldout_gaze_p95_normalized": float(np.percentile(distances, 95)),
        "calibration_samples": len(features),
        "heldout_samples": len(targets),
        "rejected_invalid_frames": sum(not p["valid"] for p in invalid),
    }
    report = {
        "data_type": "SYNTHETIC DEMO",
        "seed": seed,
        "scope": "Independent Python concept validation; not original C++/device measurements",
        "metric_definition": "RMSE = sqrt(mean((pred_x-target_x)^2+(pred_y-target_y)^2)); coordinates in [0,1], no angular units",
        "detector": "fixed threshold + connected components + row-span fill + two glint centroids",
        "metrics": metrics,
        "model_weights": model.weights.tolist(),
        "training_targets": training_targets,
        "training_features": np.asarray(features).tolist(),
        "training_labels": labels,
        "heldout_predictions": records,
        "invalid_frame_cases": invalid,
    }
    (out_dir / "synthetic_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    render_dashboard(out_dir, examples, metrics, targets, predictions, training_targets)
    print(json.dumps(metrics, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "assets"
    )
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()
    run_demo(args.output, args.seed)
