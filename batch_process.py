"""
Пакетная обработка видео: сегментация + определение этапов без GUI.

Прогоняет все видео из папки, сохраняет:
  - Аннотированные видео с масками и этапами
  - JSON-лог с таймлайном этапов по каждому видео
  - Сводную статистику

Запуск:
    python batch_process.py                          # все видео из 162___10
    python batch_process.py --source path/to/videos  # указать папку
    python batch_process.py --no-video               # только лог, без видео
"""

import cv2
import numpy as np
import os
import sys
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CLASS_COLORS, CLASSES, STAGES, STAGE_COLORS,
    PREPROCESS, INFERENCE_CONF, find_best_weights, VIDEO_DIR, RESULTS_DIR
)
from stage_detector import StageDetector
from inference import FramePreprocessor, draw_masks, draw_labels, draw_stage_panel, draw_timeline


def process_video(video_path, model, preprocessor, detector, conf,
                  save_video=True, output_dir=None):
    """Обрабатывает одно видео целиком."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name = os.path.basename(video_path)

    detector.reset()

    writer = None
    if save_video and output_dir:
        out_path = os.path.join(output_dir, f"seg_{name}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Предобработка + инференс
        processed = preprocessor.process(frame)
        results = model.predict(processed, conf=conf, verbose=False, stream=False)

        r = results[0]
        masks = r.masks.data.cpu().numpy() if r.masks is not None else None
        classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.array([])
        confs_arr = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.array([])
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None

        # Определяем этап
        stage_info = detector.analyze_frame(frame, masks, classes, confs_arr)

        # Визуализация (если сохраняем видео)
        if writer:
            vis = frame.copy()
            vis = draw_masks(vis, masks, classes, confs_arr)
            vis = draw_labels(vis, boxes, classes, confs_arr)
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            vis = draw_stage_panel(vis, stage_info, avg_fps, 0, os.path.basename(str(model.model)))
            vis = draw_timeline(vis, detector.get_timeline())
            writer.write(vis)

        # Прогресс
        if frame_count % 100 == 0:
            pct = frame_count / total * 100 if total > 0 else 0
            print(f"    {name}: {frame_count}/{total} ({pct:.0f}%) — этап {stage_info['stage']}")

    cap.release()
    if writer:
        writer.release()

    elapsed = time.time() - start_time
    timeline = detector.get_timeline()

    result = {
        "video": name,
        "total_frames": frame_count,
        "processing_time_sec": round(elapsed, 1),
        "avg_fps": round(frame_count / elapsed, 1) if elapsed > 0 else 0,
        "timeline": timeline,
        "stage_distribution": {
            STAGES[s]: timeline.count(s) for s in range(4)
        },
        "stage_percentages": {
            STAGES[s]: round(timeline.count(s) / len(timeline) * 100, 1) if timeline else 0
            for s in range(4)
        },
    }

    if save_video and output_dir:
        result["output_video"] = os.path.join(output_dir, f"seg_{name}.mp4")

    return result


def main():
    parser = argparse.ArgumentParser(description="Пакетная обработка видео")
    parser.add_argument("--source", type=str, default=None,
                        help=f"Папка с видео (default: {VIDEO_DIR})")
    parser.add_argument("--model", type=str, default=None,
                        help="Путь к весам .pt")
    parser.add_argument("--conf", type=float, default=INFERENCE_CONF)
    parser.add_argument("--no-video", action="store_true",
                        help="Не сохранять видео (только лог)")
    parser.add_argument("--output", type=str, default=None,
                        help="Папка для результатов")
    args = parser.parse_args()

    from ultralytics import YOLO

    model_path = args.model or find_best_weights()
    if model_path is None:
        print("  Модель не найдена. Запустите train.py или укажите --model")
        sys.exit(1)

    source = args.source or VIDEO_DIR
    output_dir = args.output or os.path.join(RESULTS_DIR, "batch")
    os.makedirs(output_dir, exist_ok=True)

    # Список видео
    video_files = sorted([
        os.path.join(source, f) for f in os.listdir(source)
        if f.upper().endswith((".MOV", ".MP4", ".AVI")) and not f.startswith("._")
    ])

    print("=" * 60)
    print("  ПАКЕТНАЯ ОБРАБОТКА ВИДЕО")
    print("=" * 60)
    print(f"  Модель:  {model_path}")
    print(f"  Видео:   {len(video_files)} файлов")
    print(f"  Выход:   {output_dir}")
    print(f"  Видео:   {'нет' if args.no_video else 'да'}")

    model = YOLO(model_path)
    preprocessor = FramePreprocessor(PREPROCESS)
    detector = StageDetector(history_size=30)

    all_results = []

    for i, video_path in enumerate(video_files, 1):
        name = os.path.basename(video_path)
        print(f"\n  [{i}/{len(video_files)}] {name}")

        result = process_video(
            video_path, model, preprocessor, detector, args.conf,
            save_video=not args.no_video, output_dir=output_dir
        )

        if result:
            all_results.append(result)
            print(f"    Готово: {result['total_frames']} кадров за {result['processing_time_sec']}с")
            print(f"    Этапы: {result['stage_percentages']}")

    # Сохраняем общий лог
    log_path = os.path.join(output_dir, "batch_results.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Обработано: {len(all_results)} видео")
    print(f"  Лог: {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
