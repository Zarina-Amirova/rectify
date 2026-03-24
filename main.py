#!/usr/bin/env python3
import argparse
import sys
import json
from pathlib import Path
import cv2
import numpy as np

from src.detector import get_facade_lines
from src.vanishing_point import find_vanishing_point
from src.rectifier import compute_homography, apply_rectification
from src.metrics import compute_all_metrics
from src.annotator import draw_debug_image, save_annotation


def process(input_path, output_path, debug=False, annotate=True, verbose=True):
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        print(f"Ошибка: не удалось открыть файл: {input_path}", file=sys.stderr)
        return False

    h, w = img_bgr.shape[:2]
    if verbose:
        print(f"\n{'─'*50}")
        print(f"  Файл: {input_path}  [{w}x{h}]")

    facade_lines, _ = get_facade_lines(img_bgr)
    if verbose:
        print(f"  Линий фасада: {len(facade_lines)}")

    vp, score = find_vanishing_point(facade_lines, h, w)
    if vp is None or score < 3:
        print("  Ошибка: точку схода найти не удалось.", file=sys.stderr)
        return False
    if verbose:
        print(f"  VP: ({vp[0]:.0f}, {vp[1]:.0f})  score={score}")

    H, pitch, yaw, K = compute_homography(vp[0], vp[1], w, h)
    rectified = apply_rectification(img_bgr, H)
    cv2.imwrite(str(output_path), rectified)
    if verbose:
        print(f"  Сохранено: {output_path}")
        print(f"  Pitch: {np.degrees(pitch):.1f}  Yaw: {np.degrees(yaw):.1f}")

    metrics = compute_all_metrics(img_bgr, rectified, vp)
    if verbose:
        for name, vals in metrics.items():
            b = f"{vals['before']:.2f}" if vals['before'] is not None else "N/A"
            a = f"{vals['after']:.2f}"  if vals['after']  is not None else "N/A"
            print(f"  {name:30s} до={b:8s}  после={a:8s}  ({vals['unit']})")

    if annotate:
        annotation = {
            "input": str(input_path),
            "output": str(output_path),
            "vanishing_point": {"x": float(vp[0]), "y": float(vp[1])},
            "vp_score": int(score),
            "camera": {
                "focal_length_px": float(w),
                "cx": float(w / 2),
                "cy": float(h / 2),
                "image_size": [w, h]
            },
            "camera_pose": {
                "pitch_deg": float(np.degrees(pitch)),
                "yaw_deg":   float(np.degrees(yaw))
            },
            "metrics": metrics
        }
        json_path = Path(output_path).with_suffix(".json")
        save_annotation(json_path, annotation)
        if verbose:
            print(f"  Аннотация: {json_path}")

    if debug:
        vis = draw_debug_image(img_bgr, facade_lines, vp,
                               np.degrees(pitch), np.degrees(yaw), score)
        dbg_path = Path(output_path).parent / (Path(output_path).stem + "_debug.jpg")
        cv2.imwrite(str(dbg_path), vis)
        if verbose:
            print(f"  Debug: {dbg_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Ректификация фото зданий по точке схода",
        epilog=(
            "Примеры:\n"
            "  python main.py data/raw/photo.jpg\n"
            "  python main.py data/raw/ -o data/output/ --debug\n"
            "  python main.py data/raw/ -o data/output/ --quiet"
        )
    )
    parser.add_argument("inputs", nargs="+", help="Файл(ы) или папка")
    parser.add_argument("-o", "--output", default="data/output",
                        help="Папка для результатов (по умолч. data/output)")
    parser.add_argument("--debug", action="store_true",
                        help="Сохранять debug-картинку с линиями и VP")
    parser.add_argument("--no-annotate", action="store_true",
                        help="Не сохранять JSON-аннотации")
    parser.add_argument("--quiet", action="store_true",
                        help="Минимальный вывод")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG"):
                files.extend(p.glob(ext))
        elif p.exists():
            files.append(p)
        else:
            print(f"Предупреждение: файл не найден: {inp}", file=sys.stderr)

    if not files:
        print("Ошибка: файлы не найдены.", file=sys.stderr)
        sys.exit(1)

    ok = fail = 0
    for f in files:
        out = out_dir / (f.stem + "_rectified" + f.suffix)
        if process(f, out,
                   debug=args.debug,
                   annotate=not args.no_annotate,
                   verbose=not args.quiet):
            ok += 1
        else:
            fail += 1

    if len(files) > 1:
        print(f"\n{'='*50}")
        print(f"  Обработано: {ok}/{len(files)}  Ошибок: {fail}")


if __name__ == "__main__":
    main()
