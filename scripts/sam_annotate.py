"""
SAM-based annotation for waveguide soldering frames.
Uses Segment Anything Model with point prompts to segment waveguide, flux, solder.

Strategy:
1. First find the approximate center of the waveguide using simple CV (dark cavity detection)
2. Use SAM with point prompts at:
   - Cavity center → waveguide mask
   - Left/right of waveguide → flux masks
   - Bottom of waveguide → solder mask
"""
import cv2
import numpy as np
import os
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# Paths
PROJECT = Path("/Users/damn8daniel/Desktop/НИР/ДаняБоряНир")
DATASET = Path("/Users/damn8daniel/Desktop/НИР/data/dataset")
OUTPUT = PROJECT / "results" / "sam_annotations"
OUTPUT.mkdir(exist_ok=True)

SAM_CHECKPOINT = PROJECT / "models" / "sam_vit_b.pth"
MODEL_TYPE = "vit_b"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print("Loading SAM model...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
sam.to(device=DEVICE)
predictor = SamPredictor(sam)
print("SAM loaded!")


def find_cavity_center(img: np.ndarray):
    """Find the dark cavity center using simple thresholding.
    Returns (cx, cy) in image coordinates, or None.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Focus on center region
    margin_x = w // 4
    margin_y = h // 4
    roi = gray[margin_y:h - margin_y, margin_x:w - margin_x]

    # Find dark regions
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # Try multiple thresholds to find the cavity
    for thresh_val in [50, 60, 70, 80]:
        _, dark_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find best rectangular dark region
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 150 or area > (roi.shape[0] * roi.shape[1] * 0.3):
                continue
            rect = cv2.minAreaRect(cnt)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area == 0:
                continue
            rectangularity = area / rect_area
            if rectangularity > 0.4:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + margin_x
                    cy = int(M["m01"] / M["m00"]) + margin_y
                    candidates.append((cx, cy, area, rectangularity))

        if candidates:
            # Pick the most rectangular one, preferring larger area
            candidates.sort(key=lambda c: c[2] * c[3], reverse=True)
            return candidates[0][0], candidates[0][1]

    # Fallback: center of image
    return w // 2, h // 2


def estimate_waveguide_bbox(img, cavity_cx, cavity_cy):
    """Estimate bounding box of waveguide based on cavity center."""
    h, w = img.shape[:2]
    # Waveguide is roughly 80-120px in a 640x640 image
    wg_half_w = int(w * 0.09)  # ~58px for 640
    wg_half_h = int(h * 0.08)  # ~51px for 640

    return (
        max(0, cavity_cx - wg_half_w),
        max(0, cavity_cy - wg_half_h),
        min(w, cavity_cx + wg_half_w),
        min(h, cavity_cy + wg_half_h)
    )


def segment_with_sam(img, point_coords, point_labels, multimask=True):
    """Run SAM prediction with point prompts.
    point_coords: Nx2 array of (x,y) points
    point_labels: N array of 1 (foreground) or 0 (background)
    Returns best mask.
    """
    predictor.set_image(img)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=multimask,
    )

    # Pick highest scoring mask
    best_idx = np.argmax(scores)
    return masks[best_idx].astype(np.uint8) * 255


def annotate_frame(img_path: str):
    """Annotate a single frame using SAM with point prompts."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Step 1: Find cavity center
    cavity_cx, cavity_cy = find_cavity_center(img)
    print(f"  Cavity center: ({cavity_cx}, {cavity_cy})")

    # Step 2: Estimate waveguide region
    wg_x1, wg_y1, wg_x2, wg_y2 = estimate_waveguide_bbox(img, cavity_cx, cavity_cy)
    wg_w = wg_x2 - wg_x1
    wg_h = wg_y2 - wg_y1

    # Step 3: SAM - Waveguide
    # Give multiple points: cavity center + 4 corners of estimated waveguide
    wg_points = np.array([
        [cavity_cx, cavity_cy],                    # Cavity center (definitely waveguide)
        [cavity_cx, cavity_cy - int(wg_h * 0.35)], # Top wall
        [cavity_cx, cavity_cy + int(wg_h * 0.35)], # Bottom wall
        [cavity_cx - int(wg_w * 0.35), cavity_cy], # Left wall
        [cavity_cx + int(wg_w * 0.35), cavity_cy], # Right wall
        # Background points (NOT waveguide)
        [cavity_cx - int(wg_w * 1.5), cavity_cy],  # Far left (inductor)
        [cavity_cx + int(wg_w * 1.5), cavity_cy],  # Far right (inductor)
        [cavity_cx, cavity_cy - int(wg_h * 1.5)],  # Far above (inductor)
    ])
    wg_labels = np.array([1, 1, 1, 1, 1, 0, 0, 0])

    # Clip to image bounds
    wg_points[:, 0] = np.clip(wg_points[:, 0], 0, w - 1)
    wg_points[:, 1] = np.clip(wg_points[:, 1], 0, h - 1)

    predictor.set_image(img_rgb)

    # Use BOX prompt to constrain SAM to the waveguide region
    # The box should tightly contain the waveguide + small margin
    box_margin = int(max(wg_w, wg_h) * 0.15)
    sam_box = np.array([
        max(0, wg_x1 - box_margin),
        max(0, wg_y1 - box_margin),
        min(w, wg_x2 + box_margin),
        min(h, wg_y2 + box_margin)
    ])

    # Use box + center point for best results
    wg_center_point = np.array([[cavity_cx, cavity_cy]])
    wg_center_label = np.array([1])

    wg_masks, wg_scores, _ = predictor.predict(
        point_coords=wg_center_point,
        point_labels=wg_center_label,
        box=sam_box,
        multimask_output=True,
    )

    # Pick best waveguide mask — should be small, not the whole inductor
    best_wg_idx = -1
    best_wg_score = -1
    for i, (mask, score) in enumerate(zip(wg_masks, wg_scores)):
        mask_area = mask.sum()
        img_area = h * w
        # Waveguide should be 1-10% of image, not more
        area_ratio = mask_area / img_area
        if 0.003 < area_ratio < 0.12:
            if score > best_wg_score:
                best_wg_score = score
                best_wg_idx = i

    if best_wg_idx == -1:
        # Fallback: pick smallest mask
        areas = [m.sum() for m in wg_masks]
        best_wg_idx = np.argmin(areas)

    waveguide_mask = wg_masks[best_wg_idx].astype(np.uint8) * 255
    print(f"  Waveguide mask area: {waveguide_mask.sum() // 255}px, score: {wg_scores[best_wg_idx]:.3f}")

    # Step 4: SAM - Flux (left and right patches)
    flux_offset_x = int(wg_w * 0.6)  # Flux is just outside waveguide sides

    # Left flux point
    left_flux_x = max(5, cavity_cx - flux_offset_x)
    right_flux_x = min(w - 5, cavity_cx + flux_offset_x)

    flux_masks_combined = np.zeros((h, w), dtype=np.uint8)

    for fx in [left_flux_x, right_flux_x]:
        flux_points = np.array([
            [fx, cavity_cy],
            # Background: waveguide center and far away
            [cavity_cx, cavity_cy],
            [cavity_cx, max(5, cavity_cy - int(wg_h * 1.0))],
        ])
        flux_labels = np.array([1, 0, 0])

        flux_points[:, 0] = np.clip(flux_points[:, 0], 0, w - 1)
        flux_points[:, 1] = np.clip(flux_points[:, 1], 0, h - 1)

        predictor.set_image(img_rgb)
        f_masks, f_scores, _ = predictor.predict(
            point_coords=flux_points,
            point_labels=flux_labels,
            multimask_output=True,
        )

        # Pick smallest reasonable mask (flux is small)
        best_f_idx = -1
        best_f_score = -1
        for i, (mask, score) in enumerate(zip(f_masks, f_scores)):
            area_ratio = mask.sum() / (h * w)
            if 0.001 < area_ratio < 0.08:
                if score > best_f_score:
                    best_f_score = score
                    best_f_idx = i

        if best_f_idx >= 0:
            flux_masks_combined = cv2.bitwise_or(
                flux_masks_combined,
                f_masks[best_f_idx].astype(np.uint8) * 255
            )

    # Remove overlap with waveguide
    flux_mask = cv2.bitwise_and(flux_masks_combined, cv2.bitwise_not(waveguide_mask))
    print(f"  Flux mask area: {flux_mask.sum() // 255}px")

    # Step 5: SAM - Solder (bottom of waveguide, U-shape)
    solder_y = cavity_cy + int(wg_h * 0.4)  # Below cavity
    solder_points = np.array([
        [cavity_cx, min(h - 5, solder_y)],                          # Bottom center
        [max(5, cavity_cx - int(wg_w * 0.3)), min(h - 5, solder_y)], # Bottom left
        [min(w - 5, cavity_cx + int(wg_w * 0.3)), min(h - 5, solder_y)], # Bottom right
        # Background
        [cavity_cx, cavity_cy],  # Cavity center (not solder)
        [cavity_cx, max(5, cavity_cy - int(wg_h * 0.5))],  # Above (not solder)
    ])
    solder_labels = np.array([1, 1, 1, 0, 0])

    solder_points[:, 0] = np.clip(solder_points[:, 0], 0, w - 1)
    solder_points[:, 1] = np.clip(solder_points[:, 1], 0, h - 1)

    predictor.set_image(img_rgb)
    s_masks, s_scores, _ = predictor.predict(
        point_coords=solder_points,
        point_labels=solder_labels,
        multimask_output=True,
    )

    best_s_idx = -1
    best_s_score = -1
    for i, (mask, score) in enumerate(zip(s_masks, s_scores)):
        area_ratio = mask.sum() / (h * w)
        if 0.001 < area_ratio < 0.10:
            if score > best_s_score:
                best_s_score = score
                best_s_idx = i

    solder_mask = np.zeros((h, w), dtype=np.uint8)
    if best_s_idx >= 0:
        solder_mask = s_masks[best_s_idx].astype(np.uint8) * 255
        # Remove overlap with waveguide
        solder_mask = cv2.bitwise_and(solder_mask, cv2.bitwise_not(waveguide_mask))
    print(f"  Solder mask area: {solder_mask.sum() // 255}px")

    return img, waveguide_mask, flux_mask, solder_mask


def visualize(img, waveguide_mask, flux_mask, solder_mask):
    """Create visualization with colored overlays and contours."""
    overlay = img.copy()

    # Waveguide: yellow overlay
    wg_overlay = overlay.copy()
    wg_overlay[waveguide_mask > 0] = (0, 230, 255)  # Yellow
    overlay = cv2.addWeighted(overlay, 0.55, wg_overlay, 0.45, 0)

    # Flux: cyan overlay
    flux_overlay = overlay.copy()
    flux_overlay[flux_mask > 0] = (255, 230, 0)  # Cyan
    overlay = cv2.addWeighted(overlay, 0.5, flux_overlay, 0.5, 0)

    # Solder: green overlay
    solder_overlay = overlay.copy()
    solder_overlay[solder_mask > 0] = (0, 230, 0)  # Green
    overlay = cv2.addWeighted(overlay, 0.5, solder_overlay, 0.5, 0)

    # Contours
    for mask, color, thickness in [
        (waveguide_mask, (0, 200, 255), 2),
        (flux_mask, (255, 200, 0), 2),
        (solder_mask, (0, 200, 0), 2),
    ]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, color, thickness)

    # Legend
    cv2.putText(overlay, "Waveguide", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay, "Flux", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(overlay, "Solder", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return overlay


# Process sample frames
test_dir = DATASET / "images" / "test"
test_images = sorted(os.listdir(test_dir))

# Pick 4 diverse frames
samples = []
for prefix in ["MVI_6265", "MVI_6268", "MVI_6270", "MVI_6273"]:
    for f in test_images:
        if prefix in f:
            samples.append(f)
            break

if len(samples) < 4:
    samples = test_images[:4]

print(f"\nProcessing {len(samples)} frames with SAM...")
output_files = []
for i, name in enumerate(samples):
    print(f"\n--- Frame {i+1}: {name[:20]}... ---")
    img_path = str(test_dir / name)

    img, wg_mask, flux_mask, solder_mask = annotate_frame(img_path)
    if img is None:
        continue

    annotated = visualize(img, wg_mask, flux_mask, solder_mask)
    combined = np.hstack([img, annotated])

    out_path = str(OUTPUT / f"sam_{i+1}_{name[:12]}.jpg")
    cv2.imwrite(out_path, combined)
    output_files.append(out_path)
    print(f"  Saved: {out_path}")

print(f"\nDone! {len(output_files)} files saved to {OUTPUT}")
