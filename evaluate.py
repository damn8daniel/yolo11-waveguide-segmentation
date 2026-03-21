"""
Оценка обученных моделей + генерация визуализаций для отчёта.

Создаёт:
  - Сравнительную таблицу моделей (n vs s vs m)
  - Визуальные примеры сегментации на тестовой выборке
  - JSON с метриками
  - Markdown-отчёт

Запуск:
    python evaluate.py                   # оценить все найденные модели
    python evaluate.py --samples 20      # 20 визуальных примеров
"""

import os
import sys
import json
import cv2
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RUNS_DIR, DATASET_YAML, CLASSES, CLASS_COLORS, RESULTS_DIR


MODELS_TO_CHECK = ["yolo11n-seg", "yolo11s-seg", "yolo11m-seg"]


def evaluate_all():
    """Оценивает все найденные модели."""
    from ultralytics import YOLO

    results = []

    for model_name in MODELS_TO_CHECK:
        best_path = os.path.join(RUNS_DIR, model_name, "weights", "best.pt")
        if not os.path.exists(best_path):
            print(f"  Пропуск {model_name} (веса не найдены)")
            continue

        print(f"\n  Оценка: {model_name}")
        model = YOLO(best_path)

        metrics = model.val(
            data=DATASET_YAML,
            split="test",
            project=os.path.join(RESULTS_DIR, model_name),
            name="eval",
            verbose=True,
            plots=True,
        )

        result = {
            "model": model_name,
            "weights": best_path,
            "seg_precision": float(metrics.seg.mp),
            "seg_recall": float(metrics.seg.mr),
            "seg_mAP50": float(metrics.seg.map50),
            "seg_mAP50_95": float(metrics.seg.map),
            "box_precision": float(metrics.box.mp),
            "box_recall": float(metrics.box.mr),
            "box_mAP50": float(metrics.box.map50),
            "box_mAP50_95": float(metrics.box.map),
            "speed_preprocess_ms": float(metrics.speed["preprocess"]),
            "speed_inference_ms": float(metrics.speed["inference"]),
            "speed_postprocess_ms": float(metrics.speed["postprocess"]),
        }

        # Per-class
        for i, name in CLASSES.items():
            if i < len(metrics.seg.p):
                result[f"precision_{name}"] = float(metrics.seg.p[i])
                result[f"recall_{name}"] = float(metrics.seg.r[i])
                result[f"ap50_{name}"] = float(metrics.seg.ap50[i])

        results.append(result)
        print(f"    mAP50-95(seg): {result['seg_mAP50_95']:.4f}")

    return results


def generate_predictions(num_samples=10):
    """Генерирует визуальные примеры сегментации."""
    from ultralytics import YOLO

    test_dir = os.path.join(os.path.dirname(DATASET_YAML), "images", "test")
    if not os.path.isdir(test_dir):
        print("  Тестовая выборка не найдена")
        return

    images = sorted([
        os.path.join(test_dir, f) for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:num_samples]

    print(f"\n  Визуализация: {len(images)} изображений")

    vis_dir = os.path.join(RESULTS_DIR, "visual_predictions")
    os.makedirs(vis_dir, exist_ok=True)

    for model_name in MODELS_TO_CHECK:
        best_path = os.path.join(RUNS_DIR, model_name, "weights", "best.pt")
        if not os.path.exists(best_path):
            continue

        model = YOLO(best_path)
        model_dir = os.path.join(vis_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        for img_path in images:
            results = model.predict(img_path, conf=0.5, verbose=False)
            annotated = results[0].plot()
            out_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(model_dir, out_name), annotated)

        print(f"    {model_name}: {len(images)} визуализаций")


def generate_report(results):
    """Генерирует Markdown-отчёт и JSON."""
    if not results:
        print("  Нет результатов для отчёта")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # JSON
    json_path = os.path.join(RESULTS_DIR, "evaluation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Markdown
    md_path = os.path.join(RESULTS_DIR, "evaluation.md")
    best = max(results, key=lambda x: x["seg_mAP50_95"])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Результаты оценки моделей YOLO11-seg\n\n")

        f.write("## Сегментация (маски)\n\n")
        f.write("| Модель | Precision | Recall | mAP50 | mAP50-95 | Inference (ms) |\n")
        f.write("|--------|-----------|--------|-------|----------|----------------|\n")
        for r in results:
            mark = " **" if r == best else ""
            f.write(f"| {r['model']}{mark} | {r['seg_precision']:.4f} | "
                    f"{r['seg_recall']:.4f} | {r['seg_mAP50']:.4f} | "
                    f"{r['seg_mAP50_95']:.4f} | {r['speed_inference_ms']:.2f} |\n")

        f.write("\n## Per-class (лучшая модель)\n\n")
        f.write("| Класс | Precision | Recall | AP50 |\n")
        f.write("|-------|-----------|--------|------|\n")
        for name in CLASSES.values():
            p = best.get(f"precision_{name}", 0)
            r_val = best.get(f"recall_{name}", 0)
            ap = best.get(f"ap50_{name}", 0)
            f.write(f"| {name} | {p:.4f} | {r_val:.4f} | {ap:.4f} |\n")

        f.write(f"\n## Лучшая модель\n\n")
        f.write(f"**{best['model']}** — mAP50-95 = {best['seg_mAP50_95']:.4f}\n\n")
        f.write(f"Веса: `{best['weights']}`\n")

    print(f"\n  Отчёт: {md_path}")
    print(f"  JSON: {json_path}")
    print(f"  Лучшая модель: {best['model']} (mAP50-95={best['seg_mAP50_95']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Оценка моделей")
    parser.add_argument("--samples", type=int, default=10,
                        help="Кол-во визуальных примеров")
    args = parser.parse_args()

    print("=" * 60)
    print("  ОЦЕНКА МОДЕЛЕЙ YOLO11-seg")
    print("=" * 60)

    results = evaluate_all()
    generate_predictions(args.samples)
    generate_report(results)

    print(f"\n{'='*60}")
    print(f"  Результаты в: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
