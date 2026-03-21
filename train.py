"""
Обучение моделей YOLO11-seg для сегментации объектов пайки волноводов.

Обучает выбранную модель (nano/small/medium) на подготовленном датасете.
По окончании запускает оценку на тестовой выборке.

Запуск:
    python train.py                    # обучить все модели
    python train.py --model small      # обучить только small
    python train.py --model nano --epochs 50  # nano, 50 эпох
    python train.py --resume           # продолжить прерванное обучение
"""

import os
import sys
import json
import time
import argparse
import yaml
import torch
from datetime import timedelta

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATASET_YAML, RUNS_DIR, MODEL_CONFIGS, TRAIN_PARAMS, CLASSES
)


def ensure_data_yaml():
    """
    Проверяет и автоматически исправляет путь 'path:' в data.yaml.

    Это нужно для портативности: data.yaml мог быть создан на другой
    машине с другим абсолютным путём. Функция обновляет path на
    фактическую директорию, где лежит data.yaml.
    """
    if not os.path.exists(DATASET_YAML):
        return False

    dataset_dir = os.path.dirname(os.path.abspath(DATASET_YAML))

    with open(DATASET_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    current_path = data.get("path", "")

    # Если path уже правильный — ничего не делаем
    if os.path.abspath(current_path) == dataset_dir:
        print(f"  data.yaml path OK: {dataset_dir}")
        return True

    # Исправляем path
    print(f"  data.yaml path исправлен:")
    print(f"    было:  {current_path}")
    print(f"    стало: {dataset_dir}")

    data["path"] = dataset_dir

    with open(DATASET_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    return True


def get_device():
    """Определяет доступное устройство для обучения."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  GPU: {name} ({vram:.1f} GB VRAM)")
        return 0, vram
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  GPU: Apple MPS")
        return "mps", 0
    else:
        print("  GPU: не найден, используется CPU")
        return "cpu", 0


def auto_batch_size(model_name, vram_gb):
    """
    Подбирает batch size под доступную VRAM.

    Базовые значения из config рассчитаны на ~12GB.
    RTX 5070 Ti (16GB) / RTX 4090 (24GB) позволяют увеличить.
    """
    base = MODEL_CONFIGS[model_name]["batch"]

    if vram_gb <= 0:  # CPU или MPS
        return max(base // 4, 2)

    # Масштабирование относительно 12GB базы
    if model_name == "nano":
        if vram_gb >= 20:
            return 64
        elif vram_gb >= 16:
            return 48
        elif vram_gb >= 12:
            return 32
        elif vram_gb >= 8:
            return 16
        else:
            return 8
    elif model_name == "small":
        if vram_gb >= 20:
            return 32
        elif vram_gb >= 16:
            return 24
        elif vram_gb >= 12:
            return 16
        elif vram_gb >= 8:
            return 8
        else:
            return 4
    else:  # medium
        if vram_gb >= 20:
            return 16
        elif vram_gb >= 16:
            return 12
        elif vram_gb >= 12:
            return 8
        elif vram_gb >= 8:
            return 4
        else:
            return 2


def train_model(model_name, config, device, vram_gb, custom_epochs=None,
                resume=False):
    """
    Обучает одну модель YOLO11-seg.

    Args:
        model_name: ключ из MODEL_CONFIGS (nano/small/medium)
        config: dict с pretrained, batch, epochs
        device: устройство (0, "mps", "cpu")
        vram_gb: объём VRAM (для авто-batch)
        custom_epochs: переопределение кол-ва эпох
        resume: продолжить прерванное обучение из last.pt

    Returns:
        (results, elapsed_seconds, eval_metrics)
    """
    from ultralytics import YOLO

    pretrained = config["pretrained"]
    batch = auto_batch_size(model_name, vram_gb)
    epochs = custom_epochs or config["epochs"]

    run_name = f"yolo11{model_name[0]}-seg"

    # Resume: продолжить с last.pt если есть
    if resume:
        last_path = os.path.join(RUNS_DIR, run_name, "weights", "last.pt")
        if os.path.exists(last_path):
            print(f"\n  Продолжение обучения с {last_path}")
            model = YOLO(last_path)
            start = time.time()
            results = model.train(
                resume=True,
                device=device,
                batch=batch,
            )
            elapsed = time.time() - start
            print(f"\n  Обучение завершено за {timedelta(seconds=int(elapsed))}")
            eval_metrics = evaluate_model(run_name)
            return results, elapsed, eval_metrics
        else:
            print(f"  last.pt не найден для {run_name}, начинаю с нуля")

    if not os.path.exists(pretrained):
        print(f"  Претрейн не найден: {pretrained}")
        return None, 0, None

    print(f"\n{'='*60}")
    print(f"  Модель:     {run_name}")
    print(f"  Претрейн:   {os.path.basename(pretrained)}")
    print(f"  Batch:      {batch} (auto, VRAM={vram_gb:.0f}GB)")
    print(f"  Epochs:     {epochs}")
    print(f"  Устройство: {device}")
    print(f"{'='*60}")

    model = YOLO(pretrained)

    start = time.time()

    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        batch=batch,
        project=RUNS_DIR,
        name=run_name,
        device=device,
        **TRAIN_PARAMS,
    )

    elapsed = time.time() - start
    print(f"\n  Обучение завершено за {timedelta(seconds=int(elapsed))}")

    # Оценка на тестовой выборке
    eval_metrics = evaluate_model(run_name)

    return results, elapsed, eval_metrics


def evaluate_model(run_name):
    """Оценивает обученную модель на тестовой выборке."""
    from ultralytics import YOLO

    best_path = os.path.join(RUNS_DIR, run_name, "weights", "best.pt")

    if not os.path.exists(best_path):
        print(f"  Веса не найдены: {best_path}")
        return None

    print(f"\n  Оценка {run_name} на тесте...")
    model = YOLO(best_path)

    metrics = model.val(
        data=DATASET_YAML,
        split="test",
        verbose=True,
    )

    result = {
        "model": run_name,
        "weights": best_path,
        # Сегментация (маски)
        "seg_precision": float(metrics.seg.mp),
        "seg_recall": float(metrics.seg.mr),
        "seg_mAP50": float(metrics.seg.map50),
        "seg_mAP50_95": float(metrics.seg.map),
        # Детекция (боксы)
        "box_mAP50": float(metrics.box.map50),
        "box_mAP50_95": float(metrics.box.map),
        # Скорость
        "speed_inference_ms": float(metrics.speed["inference"]),
    }

    # Per-class метрики
    for i, name in CLASSES.items():
        if i < len(metrics.seg.p):
            result[f"seg_precision_{name}"] = float(metrics.seg.p[i])
            result[f"seg_recall_{name}"] = float(metrics.seg.r[i])
            result[f"seg_ap50_{name}"] = float(metrics.seg.ap50[i])

    print(f"  mAP50(seg): {result['seg_mAP50']:.4f}")
    print(f"  mAP50-95(seg): {result['seg_mAP50_95']:.4f}")
    print(f"  Inference: {result['speed_inference_ms']:.1f}ms")

    return result


def compare_results(all_results):
    """Выводит сравнительную таблицу и сохраняет результаты."""
    if not all_results:
        return

    print(f"\n{'='*70}")
    print("  СРАВНЕНИЕ МОДЕЛЕЙ")
    print(f"{'='*70}")

    header = f"{'Модель':<18} {'Prec':>8} {'Recall':>8} {'mAP50':>8} {'mAP50-95':>10} {'ms':>8}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        print(f"{r['model']:<18} "
              f"{r['seg_precision']:>8.4f} "
              f"{r['seg_recall']:>8.4f} "
              f"{r['seg_mAP50']:>8.4f} "
              f"{r['seg_mAP50_95']:>10.4f} "
              f"{r['speed_inference_ms']:>8.1f}")

    best = max(all_results, key=lambda x: x["seg_mAP50_95"])
    print(f"\n  Лучшая модель: {best['model']} (mAP50-95 = {best['seg_mAP50_95']:.4f})")
    print(f"  Веса: {best['weights']}")

    # Сохраняем в JSON
    os.makedirs(RUNS_DIR, exist_ok=True)
    results_path = os.path.join(RUNS_DIR, "comparison.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  Результаты: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Обучение YOLO11-seg")
    parser.add_argument("--model", type=str, default="all",
                        choices=["nano", "small", "medium", "all"],
                        help="Какую модель обучать (default: all)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Переопределить кол-во эпох")
    parser.add_argument("--eval-only", action="store_true",
                        help="Только оценка (без обучения)")
    parser.add_argument("--resume", action="store_true",
                        help="Продолжить прерванное обучение с last.pt")
    parser.add_argument("--batch", type=int, default=None,
                        help="Вручную задать batch size (иначе авто)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ОБУЧЕНИЕ YOLO11-seg: сегментация пайки волноводов")
    print("=" * 60)
    print(f"  Датасет: {DATASET_YAML}")

    if not os.path.exists(DATASET_YAML):
        print(f"\n  Датасет не найден: {DATASET_YAML}")
        print("  Сначала подготовьте датасет.")
        sys.exit(1)

    # Автоисправление пути в data.yaml
    ensure_data_yaml()

    device, vram_gb = get_device()

    models_to_train = (
        list(MODEL_CONFIGS.keys()) if args.model == "all"
        else [args.model]
    )

    all_results = []

    for model_name in models_to_train:
        config = MODEL_CONFIGS[model_name]

        if args.eval_only:
            run_name = f"yolo11{model_name[0]}-seg"
            result = evaluate_model(run_name)
        else:
            _, elapsed, result = train_model(
                model_name, config, device, vram_gb,
                custom_epochs=args.epochs,
                resume=args.resume,
            )
            if result:
                result["training_time_sec"] = int(elapsed)

        if result:
            all_results.append(result)

    compare_results(all_results)

    print(f"\n{'='*60}")
    print("  Готово!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
