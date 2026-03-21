"""
Инференс-пайплайн: сегментация + определение этапов пайки в реальном времени.

Запуск:
    # Видеофайл
    python inference.py --source ../162___10/MVI_6279.MOV

    # Камера
    python inference.py --source 0

    # Папка с видео + сохранение результата
    python inference.py --source ../162___10/ --save

    # Без предобработки
    python inference.py --source video.mov --no-preprocess

    # Конкретная модель
    python inference.py --source video.mov --model path/to/best.pt
"""

import cv2
import numpy as np
import os
import sys
import time
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CLASS_COLORS, CLASSES, STAGES, STAGE_COLORS,
    PREPROCESS, INFERENCE_CONF, find_best_weights, VIDEO_DIR
)
from stage_detector import StageDetector


# ============================================================================
# ПРЕДОБРАБОТКА КАДРОВ
# ============================================================================

class FramePreprocessor:
    """Предобработка кадров для улучшения качества инференса."""

    def __init__(self, params=None):
        self.params = params or PREPROCESS
        self.enabled = self.params.get("enable", True)

        if self.enabled:
            from cv2 import xphoto
            self.wb = xphoto.createGrayworldWB()
            self.wb.setSaturationThreshold(self.params.get("wb_saturation", 0.8))

    def process(self, frame):
        if not self.enabled:
            return frame

        out = frame.copy()

        # Баланс белого
        out = self.wb.balanceWhite(out)

        # Яркость / контраст
        alpha = self.params.get("alpha", 1.2)
        beta = self.params.get("beta", 5)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

        # CLAHE
        clip = self.params.get("clahe_clip", 2.0)
        tile = self.params.get("clahe_tile", 8)
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_Lab2BGR)

        return out


# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

def draw_masks(frame, masks, classes, confs, alpha=0.4):
    """Рисует маски сегментации на кадре."""
    if masks is None or len(masks) == 0:
        return frame

    overlay = frame.copy()
    h, w = frame.shape[:2]

    for mask, cls_id, conf in zip(masks, classes, confs):
        cls_id = int(cls_id)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        if mask.shape != (h, w):
            mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
        else:
            mask_resized = mask.astype(np.float32)

        mask_bool = mask_resized > 0.5
        overlay[mask_bool] = color

        # Контуры
        mask_u8 = (mask_resized * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2)

    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Подписи
    for cls_id, conf in zip(classes, confs):
        cls_id = int(cls_id)
        name = CLASSES.get(cls_id, f"cls_{cls_id}")
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

    return frame


def draw_labels(frame, boxes, classes, confs):
    """Рисует подписи классов над боксами."""
    if boxes is None:
        return frame

    for box, cls_id, conf in zip(boxes, classes, confs):
        cls_id = int(cls_id)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        name = CLASSES.get(cls_id, f"cls_{cls_id}")
        x1, y1 = int(box[0]), int(box[1])

        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def draw_stage_panel(frame, stage_info, fps, proc_time, model_name):
    """Рисует панель с информацией об этапе и метриках."""
    h, w = frame.shape[:2]

    # Верхняя панель — этап
    stage = stage_info["stage"]
    stage_name = stage_info["stage_name"]
    stage_conf = stage_info["confidence"]
    stage_color = STAGE_COLORS.get(stage, (200, 200, 200))

    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.rectangle(frame, (0, 0), (15, 50), stage_color, -1)

    text = f"Этап {stage}: {stage_name}"
    cv2.putText(frame, text, (25, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    conf_text = f"Уверенность: {stage_conf:.0%}"
    cv2.putText(frame, conf_text, (25, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Метрики справа
    metrics = f"FPS: {fps:.0f} | {proc_time:.0f}ms | {model_name}"
    (mw, _), _ = cv2.getTextSize(metrics, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, metrics, (w - mw - 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Площади
    areas = stage_info["areas"]
    area_text = f"W:{areas['waveguide']:.3f} F:{areas['flux']:.3f} S:{areas['solder']:.3f}"
    cv2.putText(frame, area_text, (w - mw - 10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    # Нижняя легенда классов
    y_legend = h - 12
    x_off = 10
    for cls_id, cls_name in CLASSES.items():
        color = CLASS_COLORS[cls_id]
        cv2.rectangle(frame, (x_off, y_legend - 10), (x_off + 12, y_legend + 2), color, -1)
        cv2.putText(frame, cls_name, (x_off + 16, y_legend),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        x_off += 110

    # Легенда этапов
    x_off = w - 380
    for s_id, s_name in STAGES.items():
        s_color = STAGE_COLORS[s_id]
        marker = ">>>" if s_id == stage else "   "
        cv2.putText(frame, f"{marker} {s_id}:{s_name[:12]}", (x_off, y_legend),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    s_color if s_id == stage else (100, 100, 100),
                    1, cv2.LINE_AA)
        x_off += 95

    return frame


def draw_timeline(frame, stage_history, max_width=None):
    """Рисует мини-таймлайн этапов внизу кадра."""
    h, w = frame.shape[:2]
    if not stage_history:
        return frame

    bar_h = 6
    bar_y = h - 30
    max_width = max_width or (w - 20)
    bar_x = 10

    # Рисуем полоску с цветами этапов
    n = len(stage_history)
    if n > max_width:
        # Сэмплируем
        indices = np.linspace(0, n - 1, max_width, dtype=int)
        sampled = [stage_history[i] for i in indices]
    else:
        sampled = stage_history

    for i, s in enumerate(sampled):
        color = STAGE_COLORS.get(s, (100, 100, 100))
        x = bar_x + i
        cv2.line(frame, (x, bar_y), (x, bar_y + bar_h), color, 1)

    return frame


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run(source, model_path, conf=0.5, preprocess=True, save=False, output=None):
    """Запускает пайплайн сегментации + определения этапов."""
    from ultralytics import YOLO

    # Загрузка модели
    print(f"  Модель: {model_path}")
    model = YOLO(model_path)
    model_name = os.path.basename(model_path)

    # Предобработка
    preprocessor = FramePreprocessor(
        PREPROCESS if preprocess else {"enable": False}
    )

    # Детектор этапов
    detector = StageDetector(history_size=30)

    # Список видео
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        video_files = [int(source)]
    elif os.path.isdir(source):
        video_files = sorted([
            os.path.join(source, f) for f in os.listdir(source)
            if f.upper().endswith((".MOV", ".MP4", ".AVI"))
        ])
        print(f"  Папка: {len(video_files)} видео")
    else:
        video_files = [source]

    window = "Segmentation + Stage Detection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    print("\n  Управление:")
    print("    q/Esc — выход | Space — пауза | p — предобработка")
    print("    +/- — порог conf | s — скриншот\n")

    all_stage_logs = []

    for video_src in video_files:
        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            print(f"  Не удалось открыть: {video_src}")
            continue

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_name = os.path.basename(str(video_src))
        print(f"  Воспроизведение: {src_name} ({total_frames} кадров)")

        detector.reset()

        writer = None
        if save:
            out_path = output or os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "results", f"result_{src_name}.mp4"
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            w_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_video, (w_v, h_v))

        paused = False
        frame_count = 0
        fps_counter = 0
        fps_start = time.time()
        current_fps = 0.0
        frame = None

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

            if frame is None:
                continue

            t0 = time.perf_counter()

            # Предобработка
            processed = preprocessor.process(frame)

            # Инференс YOLO
            results = model.predict(processed, conf=conf, verbose=False, stream=False)

            t1 = time.perf_counter()
            proc_time = (t1 - t0) * 1000

            # Извлекаем данные
            r = results[0]
            masks = r.masks.data.cpu().numpy() if r.masks is not None else None
            classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.array([])
            confs_arr = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.array([])
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None

            # Определение этапа
            stage_info = detector.analyze_frame(frame, masks, classes, confs_arr)

            # Визуализация
            vis = frame.copy()
            vis = draw_masks(vis, masks, classes, confs_arr)
            vis = draw_labels(vis, boxes, classes, confs_arr)
            vis = draw_stage_panel(vis, stage_info, current_fps, proc_time, model_name)
            vis = draw_timeline(vis, detector.get_timeline())

            # Прогресс-бар
            if total_frames > 0:
                progress = frame_count / total_frames
                bw = int(vis.shape[1] * 0.6)
                bx = int(vis.shape[1] * 0.2)
                by = vis.shape[0] - 3
                cv2.rectangle(vis, (bx, by - 2), (bx + bw, by), (50, 50, 50), -1)
                cv2.rectangle(vis, (bx, by - 2), (bx + int(bw * progress), by), (0, 180, 0), -1)

            cv2.imshow(window, vis)

            if writer:
                writer.write(vis)

            # FPS
            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

            # Управление
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                cap.release()
                if writer:
                    writer.release()
                cv2.destroyAllWindows()
                return all_stage_logs
            elif key == 32:
                paused = not paused
            elif key == ord("p"):
                preprocessor.enabled = not preprocessor.enabled
                print(f"    Предобработка: {'ВКЛ' if preprocessor.enabled else 'ВЫКЛ'}")
            elif key == ord("+") or key == ord("="):
                conf = min(0.95, conf + 0.05)
                print(f"    Порог: {conf:.2f}")
            elif key == ord("-"):
                conf = max(0.05, conf - 0.05)
                print(f"    Порог: {conf:.2f}")
            elif key == ord("s"):
                ss_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "results", f"screenshot_{src_name}_{frame_count}.png"
                )
                os.makedirs(os.path.dirname(ss_path), exist_ok=True)
                cv2.imwrite(ss_path, vis)
                print(f"    Скриншот: {ss_path}")

        # Логируем таймлайн для этого видео
        timeline = detector.get_timeline()
        all_stage_logs.append({
            "video": src_name,
            "total_frames": frame_count,
            "timeline": timeline,
            "stage_distribution": {
                STAGES[s]: timeline.count(s) for s in range(4)
            },
        })

        cap.release()
        if writer:
            writer.release()
            print(f"    Сохранено: {out_path}")

    cv2.destroyAllWindows()

    # Сохраняем лог этапов
    if all_stage_logs:
        log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results", "stage_log.json"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(all_stage_logs, f, indent=2, ensure_ascii=False)
        print(f"\n  Лог этапов: {log_path}")

    print("\n  Пайплайн завершён.")
    return all_stage_logs


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Сегментация пайки + определение этапов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python inference.py --source ../162___10/MVI_6279.MOV
  python inference.py --source 0
  python inference.py --source ../162___10/ --save
  python inference.py --source video.mov --no-preprocess
        """,
    )
    parser.add_argument("--source", type=str, default="0",
                        help="Камера (0), видеофайл, или папка")
    parser.add_argument("--model", type=str, default=None,
                        help="Путь к весам .pt (авто-поиск если не указан)")
    parser.add_argument("--conf", type=float, default=INFERENCE_CONF,
                        help=f"Порог уверенности (default: {INFERENCE_CONF})")
    parser.add_argument("--save", action="store_true",
                        help="Сохранить выходное видео")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь для сохранения видео")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Отключить предобработку")
    args = parser.parse_args()

    # Найти модель
    model_path = args.model or find_best_weights()
    if model_path is None:
        print("  Обученная модель не найдена!")
        print("  Сначала запустите: python train.py")
        print("  Или укажите путь: --model path/to/best.pt")
        sys.exit(1)

    print("=" * 60)
    print("  СЕГМЕНТАЦИЯ + ОПРЕДЕЛЕНИЕ ЭТАПОВ ПАЙКИ")
    print("=" * 60)

    run(
        source=args.source,
        model_path=model_path,
        conf=args.conf,
        preprocess=not args.no_preprocess,
        save=args.save,
        output=args.output,
    )


if __name__ == "__main__":
    main()
