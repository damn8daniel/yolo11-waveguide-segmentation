"""Демонстрация определения этапов пайки на видео."""
import cv2, numpy as np, json, sys
from ultralytics import YOLO
from config import find_best_weights
from stage_detector import StageDetector

model = YOLO(find_best_weights())
detector = StageDetector()

video = "../162___10/MVI_6265.MOV"
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total} frames, {fps:.0f} fps, {total/fps:.1f} sec")

SKIP = 30  # каждый 30-й кадр (~1 раз в секунду)
stage_log = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % SKIP != 0:
        frame_idx += 1
        continue

    r = model(frame, conf=0.5, verbose=False)[0]
    masks = r.masks.data.cpu().numpy() if r.masks is not None else None
    classes = r.boxes.cls.int().cpu().numpy() if r.boxes is not None else np.array([])
    confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.array([])
    result = detector.analyze_frame(frame, masks, classes, confs)

    stage_log.append({
        "frame": frame_idx,
        "sec": round(frame_idx / fps, 1),
        "stage": result["stage"],
        "name": result["stage_name"],
        "flux": round(result.get("areas", {}).get("flux", 0) * 100, 2),
        "solder": round(result.get("areas", {}).get("solder", 0) * 100, 2),
        "waveguide": round(result.get("areas", {}).get("waveguide", 0) * 100, 2),
    })

    frame_idx += 1

cap.release()

# Выводим таблицу
print(f"\n{'Кадр':>6} {'Сек':>6} {'Этап':<25} {'Flux%':>7} {'Solder%':>8} {'Waveguide%':>10}")
print("-" * 70)

prev_stage = -1
for entry in stage_log:
    marker = " <<<" if entry["stage"] != prev_stage else ""
    print(f"{entry['frame']:>6} {entry['sec']:>6.1f} {entry['name']:<25} {entry['flux']:>7.2f} {entry['solder']:>8.2f} {entry['waveguide']:>10.2f}{marker}")
    prev_stage = entry["stage"]

# Сводка переходов
print("\n=== ПЕРЕХОДЫ МЕЖДУ ЭТАПАМИ ===")
prev = None
for entry in stage_log:
    if entry["stage"] != prev:
        print(f"  {entry['sec']:>6.1f}s (кадр {entry['frame']:>5}): -> {entry['name']}")
        prev = entry["stage"]

with open("results/stage_timeline.json", "w") as f:
    json.dump(stage_log, f, ensure_ascii=False, indent=2)
print("\nСохранено: results/stage_timeline.json")
