"""Визуализация этапов пайки — сохраняет ключевые кадры с масками и этапами."""
import cv2, numpy as np, os
from ultralytics import YOLO
from config import find_best_weights, CLASSES, CLASS_COLORS
from stage_detector import StageDetector

model = YOLO(find_best_weights())
detector = StageDetector()

cap = cv2.VideoCapture("../162___10/MVI_6265.MOV")
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total} frames, {fps:.0f} fps")

# Визуализируем каждый 30-й кадр (1 раз в секунду), сохраняем все
SKIP = 30
out_dir = "results/stage_frames"
os.makedirs(out_dir, exist_ok=True)

stage_colors_bgr = {
    0: (200, 200, 200),
    1: (0, 165, 255),
    2: (0, 0, 255),
    3: (0, 200, 0),
}

frame_idx = 0
saved = 0
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

    vis = frame.copy()
    h_frame, w_frame = frame.shape[:2]

    if masks is not None:
        for i, cls_id in enumerate(classes):
            mask = cv2.resize(masks[i], (w_frame, h_frame))
            mask_bool = mask > 0.5
            color = np.array(CLASS_COLORS.get(int(cls_id), (255, 255, 255)))
            overlay = vis.copy()
            overlay[mask_bool] = (color * 0.4 + overlay[mask_bool] * 0.6).astype(np.uint8)
            vis = overlay
            contours, _ = cv2.findContours(
                (mask_bool * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color.tolist(), 2)
            if len(contours) > 0:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"{CLASSES[int(cls_id)]} {confs[i]:.2f}"
                    cv2.putText(vis, label, (cx - 40, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    st = result["stage"]
    sec = frame_idx / fps
    banner_color = stage_colors_bgr[st]
    cv2.rectangle(vis, (0, 0), (w_frame, 55), banner_color, -1)
    line1 = f"Frame {frame_idx} | {sec:.1f}s | Stage {st}: {result['stage_name']}"
    cv2.putText(vis, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    line2 = f"conf={result['confidence']:.2f} | {result.get('details', '')[:60]}"
    cv2.putText(vis, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    areas = result.get("areas", {})
    info = f"flux={areas.get('flux', 0) * 100:.1f}%  solder={areas.get('solder', 0) * 100:.1f}%  waveguide={areas.get('waveguide', 0) * 100:.1f}%"
    cv2.putText(vis, info, (10, h_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    fname = f"{out_dir}/frame_{frame_idx:04d}_s{st}.jpg"
    cv2.imwrite(fname, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    saved += 1
    print(f"  [{saved}] frame {frame_idx} ({sec:.1f}s) -> stage {st}: {result['stage_name']}")

    frame_idx += 1

cap.release()
print(f"\nSaved {saved} frames to {out_dir}/")
