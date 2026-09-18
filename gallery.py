"""Rebuild the synthetic detection, held-out evaluation and animation gallery.

Every pupil/glint overlay and gaze point comes from the public detector/model.
Ground truth only draws target markers and computes evaluation residuals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "demo"))
from eye_demo import (
    AffineCalibration,
    GazeSession,
    detect,
    font,
    overlay,
    run_demo,
    synthetic_eye,
)

BG, CARD, TEXT, MUTED = "#0D1525", "#172237", "#E9EFF7", "#96A7BD"
MINT, BLUE, GOLD, RED = "#4FF3BF", "#6CBFFF", "#FFC97B", "#FF829B"


def heading(canvas, eyebrow, title, subtitle):
    draw = ImageDraw.Draw(canvas)
    draw.text((52, 32), eyebrow, fill=MINT, font=font(20, True))
    draw.text((52, 68), title, fill=TEXT, font=font(40, True))
    draw.text((52, 123), subtitle, fill=MUTED, font=font(21))
    width = canvas.width
    draw.rounded_rectangle((width - 330, 30, width - 52, 70), 9, fill="#3A3022")
    draw.text((width - 315, 35), "SYNTHETIC DEMO", fill=GOLD, font=font(23, True))


def pixel_stages(output):
    canvas = Image.new("RGB", (1600, 1090), BG)
    heading(
        canvas,
        "PIXEL PIPELINE / DETECTION",
        "See what the detector actually uses",
        "Raw grayscale pixels -> threshold candidates -> detected pupil and two glints",
    )
    draw = ImageDraw.Draw(canvas)
    stage_names = ["01  INPUT ROI", "02  THRESHOLD CANDIDATES", "03  DETECTED GEOMETRY"]
    captions = [
        "Grayscale input / 224 x 128 pixels",
        "Dark < 45   |   bright > 230",
        "Filled-mask center + glint centroids",
    ]
    for column in range(3):
        x = 52 + column * 506
        draw.rounded_rectangle((x, 182, x + 482, 956), 16, fill=CARD)
        draw.text((x + 18, 204), stage_names[column], fill=TEXT, font=font(22, True))
        draw.text((x + 18, 242), captions[column], fill=MUTED, font=font(17))
    records = []
    for row, side in enumerate(("left", "right")):
        raw = np.asarray(Image.open(output / f"synthetic_{side}_roi.png").convert("L"))
        result = detect(raw)
        if not result.valid:
            raise RuntimeError("Expected saved sample to contain valid pupil/glints")
        mask = np.full((*raw.shape, 3), (20, 31, 48), dtype=np.uint8)
        mask[raw < 45] = (79, 243, 191)
        mask[raw > 230] = (255, 201, 123)
        panels = [
            Image.fromarray(raw).convert("RGB"),
            Image.fromarray(mask),
            overlay(raw, result, scale=2),
        ]
        y = 312 + row * 326
        for column, panel in enumerate(panels):
            x = 69 + column * 506
            draw.text(
                (x, y - 31), side.upper() + " EYE", fill=MUTED, font=font(18, True)
            )
            canvas.paste(panel.resize((448, 256), Image.Resampling.NEAREST), (x, y))
            if column == 2:
                draw.text(
                    (x, y + 267),
                    f"dx {result.feature[0]:+.2f} px   dy {result.feature[1]:+.2f} px",
                    fill=MINT,
                    font=font(20),
                )
        records.append(
            {
                "side": side,
                "pupil": result.pupil.tolist(),
                "glints": result.glints.tolist(),
                "feature": result.feature.tolist(),
                "dark_candidate_pixels": int((raw < 45).sum()),
                "bright_candidate_pixels": int((raw > 230).sum()),
            }
        )
    draw.text(
        (52, 984),
        "Mint: dark candidates / pupil    Gold: bright candidates / glints    Blue: pupil-to-midpoint vector",
        fill=MUTED,
        font=font(21),
    )
    draw.text(
        (52, 1023),
        "All overlays are computed from pixels. The threshold mask is shown before component selection and hole filling.",
        fill=MUTED,
        font=font(20),
    )
    canvas.save(output / "pixel_detection_stages.png")
    return records


def calibration_plot(output, report):
    # Use a plotting library for quantitative axes, units and error distributions.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = np.asarray([p["target"] for p in report["heldout_predictions"]])
    predicted = np.asarray([p["gaze"] for p in report["heldout_predictions"]])
    training = np.asarray(report["training_targets"])
    residual = predicted - target
    distances = np.linalg.norm(residual, axis=1)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": "#435570",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.facecolor": CARD,
            "figure.facecolor": BG,
            "grid.color": "#314058",
        }
    ):
        fig = plt.figure(figsize=(16, 10.2), dpi=125)
        fig.text(
            0.05,
            0.95,
            "CALIBRATION / HELD-OUT EVALUATION",
            color=MINT,
            fontsize=13,
            weight="bold",
        )
        fig.text(
            0.05,
            0.901,
            "Measured residuals on 32 unseen targets",
            fontsize=26,
            weight="bold",
        )
        fig.text(
            0.95,
            0.95,
            "SYNTHETIC DEMO",
            ha="right",
            color=GOLD,
            fontsize=13,
            weight="bold",
        )
        fig.text(
            0.05,
            0.86,
            "90 calibration pairs at 9 positions. These 32 target pairs are excluded from fitting.",
            color=MUTED,
            fontsize=13,
        )
        axes = [
            fig.add_axes([0.075, 0.44, 0.40, 0.34]),
            fig.add_axes([0.575, 0.44, 0.35, 0.34]),
            fig.add_axes([0.075, 0.14, 0.85, 0.205]),
        ]
        ax = axes[0]
        ax.scatter(
            training[:, 0],
            training[:, 1],
            marker="+",
            s=115,
            color="#8292A8",
            linewidths=1.6,
            label="Calibration target",
        )
        ax.scatter(
            target[:, 0],
            target[:, 1],
            facecolors="none",
            edgecolors=BLUE,
            s=64,
            linewidths=1.5,
            label="Unseen target",
        )
        ax.scatter(
            predicted[:, 0], predicted[:, 1], c=MINT, s=17, label="Prediction", zorder=4
        )
        for t, p in zip(target, predicted):
            ax.plot([t[0], p[0]], [t[1], p[1]], c=GOLD, linewidth=1)
        ax.set(
            xlim=(0, 1),
            ylim=(1, 0),
            xlabel="Normalized screen x",
            ylabel="Normalized screen y",
        )
        ax.set_title(
            "Target and prediction locations", loc="left", pad=15, weight="bold"
        )
        ax.grid(alpha=0.5)
        legend = ax.legend(
            loc="lower right",
            frameon=True,
            facecolor=BG,
            edgecolor="#435570",
            fontsize=9,
        )
        for text in legend.get_texts():
            text.set_color(TEXT)
        ax = axes[1]
        bound = max(0.02, float(np.abs(residual).max()) * 1.2)
        ax.scatter(residual[:, 0], residual[:, 1], c=MINT, s=35, alpha=0.85)
        ax.axhline(0, color="#8292A8", linewidth=1)
        ax.axvline(0, color="#8292A8", linewidth=1)
        ax.set(
            xlim=(-bound, bound),
            ylim=(-bound, bound),
            xlabel="Prediction x - target x",
            ylabel="Prediction y - target y",
        )
        ax.set_title(
            "Residuals / native normalized units", loc="left", pad=15, weight="bold"
        )
        ax.grid(alpha=0.5)
        ax = axes[2]
        ax.bar(np.arange(1, 33), distances, color=MINT, alpha=0.88, width=0.68)
        rmse = float(np.sqrt(np.mean(distances**2)))
        p95 = float(np.percentile(distances, 95))
        ax.axhline(rmse, c=BLUE, linewidth=1.7, label=f"RMSE = {rmse:.5f}")
        ax.axhline(
            p95,
            c=GOLD,
            linewidth=1.7,
            linestyle="--",
            label=f"P95 distance = {p95:.5f}",
        )
        ax.set(
            xlim=(0, 33),
            xlabel="Held-out sample index",
            ylabel="Euclidean distance error",
        )
        ax.set_title(
            "All 32 held-out positions / per-sample error",
            loc="left",
            pad=13,
            weight="bold",
        )
        ax.grid(axis="y", alpha=0.4)
        legend = ax.legend(
            loc="upper right", facecolor=BG, edgecolor="#435570", fontsize=10
        )
        for text in legend.get_texts():
            text.set_color(TEXT)
        fig.text(
            0.05,
            0.043,
            "Errors use x, y in [0, 1]. They are not angular accuracy or real-subject measurements.",
            color=MUTED,
            fontsize=12,
        )
        fig.savefig(output / "heldout_calibration_errors.png", facecolor=BG)
        plt.close(fig)
    return {
        "samples": len(target),
        "rmse_normalized": rmse,
        "p95_distance_normalized": p95,
        "errors": residual.tolist(),
        "distance_errors": distances.tolist(),
    }


def animate(output, report, seed):
    rng = np.random.default_rng(seed + 1000)
    session = GazeSession(AffineCalibration(report["model_weights"]))
    frames, records, trail = [], [], []
    frame_count, duration_ms = 60, 120
    phases = np.linspace(0, 2 * np.pi, frame_count, endpoint=False)
    targets = np.column_stack(
        [0.5 + 0.29 * np.cos(phases), 0.5 + 0.25 * np.sin(2 * phases + 0.4)]
    )

    for index, target in enumerate(targets):
        case = (
            "left_blink"
            if 17 <= index <= 20
            else "missing_glint" if 42 <= index <= 45 else "valid"
        )
        pair, crops = [], []
        for side in ("left", "right"):
            roi, _ = synthetic_eye(
                target,
                side,
                rng,
                blink=side == "left" and case == "left_blink",
                missing_glint=side == "right" and case == "missing_glint",
            )
            result = detect(roi)
            pair.append(result)
            crops.append(overlay(roi, result, scale=2))
        packet = session.process(*pair, (index + 1) * 120000, (index + 1) * 120000)
        if packet["valid"]:
            trail.append(packet["gaze"])
        frame = Image.new("RGB", (1200, 800), BG)
        heading(
            frame,
            "EYE TRACKING / FRAME BY FRAME",
            "Pixels in. Gaze out.",
            "Live algorithm replay on procedural eye ROIs / no recorded subject data",
        )
        draw = ImageDraw.Draw(frame)
        for eye, side in enumerate(("LEFT EYE", "RIGHT EYE")):
            y = 190 + eye * 292
            draw.text((40, y - 29), side, fill=TEXT, font=font(19, True))
            frame.paste(crops[eye], (40, y))
            state = (
                "DETECTED"
                if pair[eye].valid
                else pair[eye].reason.upper().replace("_", " ")
            )
            draw.rounded_rectangle((301, y - 31, 488, y - 4), 6, fill=CARD)
            draw.text(
                (310, y - 29),
                state,
                fill=MINT if pair[eye].valid else RED,
                font=font(13, True),
            )
        draw.text((548, 167), "NORMALIZED SCREEN", fill=TEXT, font=font(21, True))
        left, top, width, height = 550, 215, 598, 390
        draw.rounded_rectangle(
            (left, top, left + width, top + height),
            12,
            fill=CARD,
            outline="#435570",
            width=2,
        )

        def mapped(p):
            return left + p[0] * width, top + p[1] * height

        for a in (0.25, 0.5, 0.75):
            draw.line(
                (left + a * width, top + 1, left + a * width, top + height - 1),
                fill="#24344B",
            )
            draw.line(
                (left + 1, top + a * height, left + width - 1, top + a * height),
                fill="#24344B",
            )
        draw.line([mapped(p) for p in targets], fill="#32435D", width=2)
        if len(trail) > 1:
            draw.line([mapped(p) for p in trail[-16:]], fill="#278B79", width=3)
        tx, ty = mapped(target)
        draw.ellipse((tx - 12, ty - 12, tx + 12, ty + 12), outline=BLUE, width=3)
        if packet["gaze"] is not None:
            px, py = mapped(packet["gaze"])
            color = MINT if packet["valid"] else GOLD
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=color)
            draw.line((tx, ty, px, py), fill=GOLD, width=1)
        draw.text((550, 622), "o Target", fill=BLUE, font=font(20))
        draw.text((725, 622), "Predicted gaze / trail", fill=MINT, font=font(20))
        status = (
            "NEW VALID PREDICTION"
            if packet["valid"]
            else "INVALID FRAME / PREVIOUS POINT HELD"
        )
        draw.text(
            (550, 661),
            status,
            fill=MINT if packet["valid"] else GOLD,
            font=font(21, True),
        )
        draw.text(
            (550, 696),
            f"Frame {index + 1:02d}/{frame_count}   valid={str(packet['valid']).lower()}   held={str(packet['held']).lower()}",
            fill=MUTED,
            font=font(18),
        )
        draw.text(
            (40, 756),
            "Synthetic replay only. Playback speed is chosen for viewing; it is not detector throughput.",
            fill=MUTED,
            font=font(19),
        )
        frames.append(frame)
        records.append(
            {
                "frame": index + 1,
                "case": case,
                "target": target.tolist(),
                **packet,
                "left_valid": pair[0].valid,
                "right_valid": pair[1].valid,
                "left_feature": pair[0].feature.tolist() if pair[0].valid else None,
                "right_feature": pair[1].feature.tolist() if pair[1].valid else None,
            }
        )
    # One shared palette avoids changing palette colors across GIF frames.
    palette_source = Image.new("RGB", (1200, 800 * 4))
    for position, index in enumerate((0, 18, 35, 43)):
        palette_source.paste(frames[index], (0, position * 800))
    palette = palette_source.quantize(colors=128)
    quantized = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]
    quantized[0].save(
        output / "eye_tracking_replay.gif",
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=1,
    )
    # Validate the encoded animation, not only the source frames. A wrong GIF
    # disposal mode can silently erase static titles between cropped delta frames.
    with Image.open(output / "eye_tracking_replay.gif") as decoded:
        if decoded.n_frames != frame_count:
            raise RuntimeError("Encoded replay has an unexpected frame count")
        for index, expected in enumerate(quantized):
            decoded.seek(index)
            if not np.array_equal(
                np.asarray(decoded.convert("RGB")),
                np.asarray(expected.convert("RGB")),
            ):
                raise RuntimeError(
                    f"GIF frame {index + 1} differs from its rendered frame"
                )
    # Contact sheet contains actual frames, including both explicit failure cases.
    contact = Image.new("RGB", (1600, 1130), BG)
    heading(
        contact,
        "REPLAY / KEY FRAMES",
        "A missing eye does not invent a new gaze",
        "Computed detections, valid predictions and explicit held states from the animation",
    )
    d = ImageDraw.Draw(contact)
    for slot, index in enumerate((5, 18, 34, 43)):
        x, y = 40 + (slot % 2) * 790, 205 + (slot // 2) * 438
        content = frames[index].crop((0, 156, 1200, 740))
        contact.paste(content.resize((730, 359), Image.Resampling.LANCZOS), (x, y))
        label = f"FRAME {index + 1:02d} / {records[index]['case'].upper().replace('_', ' ')}"
        d.text(
            (x + 2, y - 28),
            label,
            fill=MINT if records[index]["valid"] else GOLD,
            font=font(19, True),
        )
    d.text(
        (40, 1080),
        "All frames are synthetic; the detector and calibration model run for every eye pair.",
        fill=MUTED,
        font=font(21),
    )
    contact.save(output / "replay_keyframes.png")
    # Check the visualization's state story against the actual computed packets.
    for index, record in enumerate(records):
        if record["case"] == "valid" and not record["valid"]:
            raise RuntimeError(
                f"Unexpected detection failure in replay frame {index + 1}"
            )
        if record["case"] != "valid":
            if (
                record["valid"]
                or not record["held"]
                or record["gaze"] != records[index - 1]["gaze"]
            ):
                raise RuntimeError(
                    f"Invalid frame {index + 1} did not hold the previous gaze"
                )
    (output / "replay_frames.json").write_text(
        json.dumps(
            {
                "data_type": "SYNTHETIC DEMO",
                "seed": seed + 1000,
                "frame_duration_ms": duration_ms,
                "playback_is_not_throughput": True,
                "frames": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "frames": frame_count,
        "duration_ms": frame_count * duration_ms,
        "valid_frames": sum(r["valid"] for r in records),
        "held_frames": sum(r["held"] for r in records),
        "keyframe_indices_1based": [6, 19, 35, 44],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--output", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    output = args.output
    report = run_demo(output, args.seed)
    detection = pixel_stages(output)
    calibration = calibration_plot(output, report)
    replay = animate(output, report, args.seed)
    names = [
        "pixel_detection_stages.png",
        "heldout_calibration_errors.png",
        "eye_tracking_replay.gif",
        "replay_keyframes.png",
        "replay_frames.json",
    ]
    manifest = {
        "project": "eye-tracking-reference",
        "data_type": "SYNTHETIC DEMO",
        "seed": args.seed,
        "command": f"python gallery.py --seed {args.seed}",
        "provenance": "Procedural grayscale ROIs; overlays from detect(); gaze from fitted AffineCalibration via GazeSession",
        "calibration": calibration,
        "pixel_stages": detection,
        "replay": replay,
        "artifacts": [
            {
                "path": name,
                "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
            }
            for name in names
        ],
        "limits": "Not real-subject data, original product measurements, or an FPS benchmark. GIF pace is editorial.",
    }
    destination = (
        ROOT / "examples" if output.resolve() == (ROOT / "assets").resolve() else output
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gallery": names, "replay": replay}, indent=2))


if __name__ == "__main__":
    main()
