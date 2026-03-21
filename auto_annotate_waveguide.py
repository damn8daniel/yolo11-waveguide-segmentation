"""
Автоматическая аннотация волновода (класс 0) для датасета.

Волновод — яркая металлическая прямоугольная деталь в центре кадра.
Используем анализ яркости + цвета + морфологию для детекции.
Учитываем аугментации Roboflow (повороты → белый/серый фон в углах).

Добавляет аннотацию waveguide (class 0) в существующие .txt файлы
(которые уже содержат flux=1 и solder=2).

Запуск:
    python auto_annotate_waveguide.py                     # обработать весь датасет
    python auto_annotate_waveguide.py --preview 10        # визуализировать 10 примеров
    python auto_annotate_waveguide.py --dry-run            # не менять файлы
"""

import cv2
import numpy as np
import os
import sys
import argparse
import glob

# Пути к датасету
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "dataset")


def get_existing_annotations_center(label_path, img_w, img_h):
    """
    Находит центр масс существующих аннотаций (flux/solder).
    Волновод всегда рядом с ними — используем как подсказку.
    """
    centers = []
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = [float(x) for x in parts[1:]]
            xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
            ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
            centers.append((np.mean(xs), np.mean(ys)))

    if not centers:
        return None
    return (np.mean([c[0] for c in centers]), np.mean([c[1] for c in centers]))


def create_augmentation_mask(image):
    """
    Маскирует области заливки от аугментации Roboflow (повороты).
    Эти области обычно однородного серого/белого цвета в углах.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Проверяем углы на однородность
    corner_size = min(h, w) // 8
    corners = [
        gray[0:corner_size, 0:corner_size],           # top-left
        gray[0:corner_size, w-corner_size:w],          # top-right
        gray[h-corner_size:h, 0:corner_size],          # bottom-left
        gray[h-corner_size:h, w-corner_size:w],        # bottom-right
    ]

    # Если угол очень однороден (std < 5), это заливка
    fill_values = []
    for corner in corners:
        if np.std(corner) < 5:
            fill_values.append(int(np.mean(corner)))

    if not fill_values:
        return np.ones((h, w), dtype=np.uint8) * 255  # Нет заливки

    # Создаём маску: пиксели близкие к цвету заливки = исключаем
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for fv in fill_values:
        fill_mask = np.abs(gray.astype(int) - fv) < 15
        # Проверяем также низкую текстурность
        local_std = cv2.blur((gray.astype(float) - cv2.blur(gray.astype(float), (5, 5)))**2, (15, 15))
        local_std = np.sqrt(np.maximum(local_std, 0))
        uniform_mask = local_std < 8
        bad_pixels = fill_mask & uniform_mask
        mask[bad_pixels] = 0

    return mask


def has_texture(image, contour, min_std=10):
    """Проверяет что область контура содержит текстуру (не однородная заливка)."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask > 0]
    if len(pixels) < 50:
        return False

    return np.std(pixels) > min_std


def detect_waveguide(image, label_path=None, debug=False):
    """
    Детектирует волновод на изображении пайки.

    Улучшенная версия:
    - Маскирует заливку аугментации (Roboflow повороты)
    - Использует позицию flux/solder как подсказку
    - Проверяет текстурность области (исключает однородный фон)
    """
    h, w = image.shape[:2]
    img_area = h * w

    # Маска для исключения заливки аугментации
    aug_mask = create_augmentation_mask(image)

    # Центр существующих аннотаций (подсказка)
    anno_center = None
    if label_path:
        anno_center = get_existing_annotations_center(label_path, w, h)

    # Конвертации
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    h_ch, s_ch, v_ch = cv2.split(hsv)
    l_ch = lab[:, :, 0]

    # Определяем: grayscale аугментация? (очень низкая насыщенность по всему кадру)
    is_grayscale = np.mean(s_ch) < 20

    if is_grayscale:
        # Для grayscale: только по яркости
        combined = (v_ch > 160).astype(np.uint8) * 255
    else:
        # === Стратегия 1: Яркий металл (высокая яркость, низкая насыщенность) ===
        bright_mask = v_ch > 170
        low_sat_mask = s_ch < 90
        metal_mask = (bright_mask & low_sat_mask).astype(np.uint8) * 255

        # === Стратегия 2: LAB — яркие области ===
        bright_lab = (l_ch > 160).astype(np.uint8) * 255

        # Комбинируем
        combined = cv2.bitwise_or(metal_mask, bright_lab)

    # Применяем маску аугментации (убираем заливку)
    combined = cv2.bitwise_and(combined, aug_mask)

    # Морфология
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    combined = cv2.dilate(combined, kernel_small, iterations=1)

    # Контуры
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, combined if debug else None

    # Референсный центр: аннотации > центр изображения
    ref_cx = anno_center[0] if anno_center else w / 2
    ref_cy = anno_center[1] if anno_center else h / 2

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / img_area
        if area_ratio < 0.015 or area_ratio > 0.35:
            continue

        # Aspect ratio
        rect = cv2.minAreaRect(cnt)
        rect_w, rect_h = rect[1]
        if rect_w == 0 or rect_h == 0:
            continue
        aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
        if aspect > 4.0:
            continue

        # Solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.35:
            continue

        # Текстура — исключаем однородные области (заливка аугментации)
        if not has_texture(image, cnt, min_std=8):
            continue

        # Центр контура
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = w / 2, h / 2

        # Расстояние до референсного центра
        max_dist = np.sqrt((w/2)**2 + (h/2)**2)
        dist_to_ref = np.sqrt((cx - ref_cx)**2 + (cy - ref_cy)**2) / max_dist

        # Скоринг
        score = area_ratio * (1 - 0.7 * dist_to_ref) * solidity

        candidates.append((score, cnt, area_ratio, cx, cy))

    if not candidates:
        return _detect_waveguide_fallback(image, gray, img_area, ref_cx, ref_cy, aug_mask, debug)

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_cnt = candidates[0][1]

    if debug:
        return best_cnt, combined
    return best_cnt, None


def _detect_waveguide_fallback(image, gray, img_area, ref_cx, ref_cy, aug_mask, debug):
    """Запасная стратегия: Otsu + проверка текстуры."""
    h, w = image.shape[:2]

    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Применяем маску аугментации
    otsu = cv2.bitwise_and(otsu, aug_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=3)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0
    max_dist = np.sqrt((w/2)**2 + (h/2)**2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / img_area
        if area_ratio < 0.015 or area_ratio > 0.35:
            continue

        if not has_texture(image, cnt, min_std=8):
            continue

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            continue

        dist_to_ref = np.sqrt((cx - ref_cx)**2 + (cy - ref_cy)**2) / max_dist
        score = area_ratio * (1 - 0.7 * dist_to_ref)

        if score > best_score:
            best_score = score
            best = cnt

    if debug:
        return best, otsu
    return best, None


def contour_to_yolo_polygon(contour, img_w, img_h, simplify_epsilon=2.0):
    """Конвертирует контур OpenCV в YOLO polygon format."""
    epsilon = simplify_epsilon * cv2.arcLength(contour, True) / 1000
    epsilon = max(epsilon, 1.0)
    simplified = cv2.approxPolyDP(contour, epsilon, True)

    if len(simplified) < 3:
        simplified = contour

    max_points = 50
    if len(simplified) > max_points:
        indices = np.linspace(0, len(simplified) - 1, max_points, dtype=int)
        simplified = simplified[indices]

    points = simplified.reshape(-1, 2).astype(float)
    points[:, 0] /= img_w
    points[:, 1] /= img_h
    points = np.clip(points, 0.0, 1.0)

    parts = ["0"]
    for x, y in points:
        parts.append(f"{x:.6f}")
        parts.append(f"{y:.6f}")

    return " ".join(parts)


def process_split(split_name, dataset_dir, preview=0, dry_run=False):
    """Обрабатывает один split (train/val/test)."""
    images_dir = os.path.join(dataset_dir, "images", split_name)
    labels_dir = os.path.join(dataset_dir, "labels", split_name)

    if not os.path.isdir(images_dir):
        print(f"  Пропуск {split_name}: {images_dir} не найден")
        return 0, 0, 0

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) +
                         glob.glob(os.path.join(images_dir, "*.jpeg")) +
                         glob.glob(os.path.join(images_dir, "*.png")))

    success = 0
    failed = 0
    skipped = 0
    preview_count = 0

    preview_dir = None
    if preview > 0:
        preview_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results", "annotation_preview")
        os.makedirs(preview_dir, exist_ok=True)

    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, basename + ".txt")

        if not os.path.exists(label_path):
            print(f"    WARN: нет label для {basename}")
            skipped += 1
            continue

        with open(label_path, "r") as f:
            existing_lines = f.readlines()

        has_waveguide = any(line.strip().startswith("0 ") for line in existing_lines)
        if has_waveguide:
            skipped += 1
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"    WARN: не удалось прочитать {basename}")
            failed += 1
            continue

        img_h, img_w = image.shape[:2]

        do_debug = preview_count < preview
        contour, debug_mask = detect_waveguide(image, label_path=label_path, debug=do_debug)

        if contour is None:
            print(f"    FAIL: волновод не найден в {basename}")
            failed += 1
            continue

        area = cv2.contourArea(contour)
        if area < 100:
            print(f"    FAIL: слишком маленький контур в {basename}")
            failed += 1
            continue

        yolo_line = contour_to_yolo_polygon(contour, img_w, img_h)

        if not dry_run:
            with open(label_path, "w") as f:
                f.write(yolo_line + "\n")
                for line in existing_lines:
                    f.write(line)

        success += 1

        # Визуализация
        if preview_dir and preview_count < preview:
            vis = image.copy()
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)

            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(vis, "waveguide", (cx - 40, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for line in existing_lines:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                pts = np.array([(coords[i] * img_w, coords[i+1] * img_h)
                               for i in range(0, len(coords), 2)], dtype=np.int32)
                color = (255, 165, 0) if cls_id == 1 else (0, 0, 255)
                label = "flux" if cls_id == 1 else "solder"
                cv2.drawContours(vis, [pts], -1, color, 2)
                if len(pts) > 0:
                    cv2.putText(vis, label, (pts[0][0], pts[0][1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out_path = os.path.join(preview_dir, f"{split_name}_{basename}.jpg")
            cv2.imwrite(out_path, vis)

            if debug_mask is not None:
                mask_path = os.path.join(preview_dir, f"{split_name}_{basename}_mask.jpg")
                cv2.imwrite(mask_path, debug_mask)

            preview_count += 1

    return success, failed, skipped


def verify_annotations(dataset_dir):
    """Проверяет распределение классов после аннотации."""
    print("\n  === ВЕРИФИКАЦИЯ АННОТАЦИЙ ===")

    for split in ["train", "val", "test"]:
        labels_dir = os.path.join(dataset_dir, "labels", split)
        if not os.path.isdir(labels_dir):
            continue

        class_counts = {0: 0, 1: 0, 2: 0}
        total_files = 0
        files_with_waveguide = 0

        for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
            total_files += 1
            has_wg = False
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id in class_counts:
                            class_counts[cls_id] += 1
                        if cls_id == 0:
                            has_wg = True
            if has_wg:
                files_with_waveguide += 1

        print(f"\n  {split.upper()}: {total_files} файлов")
        print(f"    waveguide (0): {class_counts[0]} ({files_with_waveguide}/{total_files} файлов)")
        print(f"    flux      (1): {class_counts[1]}")
        print(f"    solder    (2): {class_counts[2]}")


def main():
    parser = argparse.ArgumentParser(description="Авто-аннотация волновода")
    parser.add_argument("--preview", type=int, default=0,
                        help="Визуализировать N примеров")
    parser.add_argument("--dry-run", action="store_true",
                        help="Не менять файлы, только проверить")
    parser.add_argument("--dataset", type=str, default=DATASET_DIR,
                        help="Путь к датасету")
    parser.add_argument("--verify-only", action="store_true",
                        help="Только проверить существующие аннотации")
    args = parser.parse_args()

    print("=" * 60)
    print("  АВТО-АННОТАЦИЯ ВОЛНОВОДА (класс 0)")
    print("=" * 60)
    print(f"  Датасет: {args.dataset}")
    print(f"  Режим:   {'dry-run' if args.dry_run else 'запись'}")

    if args.verify_only:
        verify_annotations(args.dataset)
        return

    total_success = 0
    total_failed = 0
    total_skipped = 0

    for split in ["train", "val", "test"]:
        print(f"\n  --- {split.upper()} ---")
        success, failed, skipped = process_split(
            split, args.dataset,
            preview=args.preview if split == "train" else 0,
            dry_run=args.dry_run
        )
        total_success += success
        total_failed += failed
        total_skipped += skipped
        print(f"    OK: {success}, FAIL: {failed}, SKIP: {skipped}")

    print(f"\n{'=' * 60}")
    print(f"  ИТОГО: {total_success} аннотировано, {total_failed} ошибок, {total_skipped} пропущено")
    print(f"{'=' * 60}")

    if not args.dry_run:
        verify_annotations(args.dataset)


if __name__ == "__main__":
    main()
