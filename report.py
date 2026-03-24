"""
Читает все JSON из папки output и показывает таблицу результатов
отсортированную по качеству ректификации

Использование:
    python report.py                             # папка по умолчанию data/outputall/
    python report.py -o data/outputall/
    python report.py --top 10
"""

import json
import argparse
from pathlib import Path


def load_results(output_dir):
    results = []
    for json_path in sorted(Path(output_dir).glob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        m = data.get("metrics", {})
        pose = data.get("camera_pose", {})
        vp = data.get("vanishing_point", {})

        vert_before = m.get("verticality_error", {}).get("before")
        vert_after  = m.get("verticality_error", {}).get("after")
        improvement = (vert_before - vert_after) if (vert_before and vert_after) else None

        results.append({
            "file":        Path(data.get("input", json_path.stem)).name,
            "pitch_deg":   pose.get("pitch_deg"),
            "vp_score":    data.get("vp_score"),
            "vert_before": vert_before,
            "vert_after":  vert_after,
            "improvement": improvement,
            "par_before":  m.get("parallelism_std", {}).get("before"),
            "par_after":   m.get("parallelism_std", {}).get("after"),
            "vp_residual_before": m.get("vp_residual_px", {}).get("before"),
            "vp_residual_after":  m.get("vp_residual_px", {}).get("after"),
            "json_path":   str(json_path),
        })

    return results


def grade(r):
    """Оценка качества: чем больше улучшение и выше score — тем лучше."""
    imp = r["improvement"] or 0
    score = r["vp_score"] or 0
    return imp * 0.7 + score * 0.3


def fmt(val, decimals=2):
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def print_report(results, top_n=None):
    if not results:
        print("JSON-файлы не найдены.")
        return

    results_sorted = sorted(results, key=grade, reverse=True)
    if top_n:
        results_sorted = results_sorted[:top_n]

    # Ширина колонок
    w_file = max(len(r["file"]) for r in results_sorted) + 2

    header = (
        f"{'Файл':<{w_file}} "
        f"{'pitch':>7} "
        f"{'score':>6} "
        f"{'vert_до':>8} "
        f"{'vert_после':>10} "
        f"{'улучш':>7} "
        f"{'std_до':>7} "
        f"{'std_после':>9} "
        f"{'оценка':>7}"
    )
    sep = "─" * len(header)

    print(f"\n{'='*len(header)}")
    print("  РЕЗУЛЬТАТЫ РЕКТИФИКАЦИИ")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for r in results_sorted:
        imp = r["improvement"]
        # Маркер качества
        if imp is None:
            mark = "?"
        elif imp > 5:
            mark = "+"
        elif imp > 0:
            mark = "~"
        else:
            mark = "-"

        print(
            f"{r['file']:<{w_file}} "
            f"{fmt(r['pitch_deg'],1):>7} "
            f"{fmt(r['vp_score'],0):>6} "
            f"{fmt(r['vert_before']):>8} "
            f"{fmt(r['vert_after']):>10} "
            f"{fmt(imp):>7} "
            f"{fmt(r['par_before']):>7} "
            f"{fmt(r['par_after']):>9} "
            f"[{mark}] {grade(r):>5.1f}"
        )

    print(sep)

    # Итоги
    with_improvement = [r for r in results if r["improvement"] and r["improvement"] > 0]
    print(f"\n  Всего фото:          {len(results)}")
    print(f"  Улучшение > 0:       {len(with_improvement)}")
    print(f"  Среднее улучшение:   "
          f"{sum(r['improvement'] for r in with_improvement) / max(len(with_improvement),1):.2f} deg")


def main():
    parser = argparse.ArgumentParser(description="Отчёт по результатам ректификации")
    parser.add_argument("-o", "--output", default="data/outputall/",
                        help="Папка с JSON-файлами (по умолч. data/outputall/)")
    parser.add_argument("--top", type=int, default=None,
                        help="Показать только топ-N фото")
    args = parser.parse_args()

    results = load_results(args.output)
    print_report(results, top_n=args.top)


if __name__ == "__main__":
    main()
