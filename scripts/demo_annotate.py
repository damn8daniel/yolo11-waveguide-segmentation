"""
Demo annotation script - segments waveguide, flux, solder on sample frames.
Uses color-based and spatial analysis approach.
"""
import cv2
import numpy as np
import os
from pathlib import Path

# Paths
PROJECT = Path("/Users/damn8daniel/Desktop/НИР/ДаняБоряНир")
DATASET = Path("/Users/damn8daniel/Desktop/НИР/data/dataset")
OUTPUT = PROJECT / "results" / "demo_annotations"
OUTPUT.mkdir(exist_ok=True)

# Pick a few diverse test frames
test_dir = DATASET / "images" / "test"
test_images = sorted(os.listdir(test_dir))
# Pick frames from different videos
sample_names = [f for f in test_images if "MVI_6265" in f][:1] + \
               [f for f in test_images if "MVI_6268" in f][:1] + \
               [f for f in test_images if "MVI_6270" in f][:1] + \
               [f for f in test_images if "MVI_6273" in f][:1]

if len(sample_names) < 4:
    sample_names = test_images[:4]


def find_waveguide_region(img: np.ndarray):
    """Find the waveguide (small rectangular component) in the center of the frame.

    Strategy:
    1. Look for the dark cavity (very dark rectangular region in center area)
    2. Expand slightly to get the walls around it
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Focus on center region (waveguide is always roughly centered)
    cx, cy = w // 2, h // 2
    roi_size = min(w, h) // 3
    x1 = max(0, cx - roi_size)
    y1 = max(0, cy - roi_size // 2)
    x2 = min(w, cx + roi_size)
    y2 = min(h, cy + roi_size)

    roi = gray[y1:y2, x1:x2]

    # Step 1: Find the dark cavity using adaptive threshold
    # The cavity is the darkest rectangular region
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # Very dark pixels = cavity
    _, dark_mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    # Clean up
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    # Find contours of dark regions
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # Find the largest dark contour that's roughly rectangular (the cavity)
    best_contour = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:  # too small
            continue
        # Check if roughly rectangular
        rect = cv2.minAreaRect(cnt)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area == 0:
            continue
        rectangularity = area / rect_area
        if rectangularity > 0.5 and area > best_area:
            best_area = area
            best_contour = cnt

    if best_contour is None:
        return None, None

    # Get bounding rect of cavity
    bx, by, bw, bh = cv2.boundingRect(best_contour)

    # Cavity mask (in full image coords)
    cavity_mask = np.zeros((h, w), dtype=np.uint8)
    shifted_contour = best_contour.copy()
    shifted_contour[:, :, 0] += x1
    shifted_contour[:, :, 1] += y1
    cv2.drawContours(cavity_mask, [shifted_contour], -1, 255, -1)

    # Step 2: Expand cavity to get walls (waveguide = cavity + walls)
    # Walls are the bright metallic border around the cavity
    # Expand by ~40-60% in each direction
    expand_x = int(bw * 0.5)
    expand_y = int(bh * 0.45)

    wg_x1 = max(0, x1 + bx - expand_x)
    wg_y1 = max(0, y1 + by - expand_y)
    wg_x2 = min(w, x1 + bx + bw + expand_x)
    wg_y2 = min(h, y1 + by + bh + expand_y)

    # Create waveguide mask as rectangle (will refine with edges)
    waveguide_mask = np.zeros((h, w), dtype=np.uint8)

    # Use edge detection to find the actual waveguide boundaries
    wg_roi = gray[wg_y1:wg_y2, wg_x1:wg_x2]
    wg_blurred = cv2.GaussianBlur(wg_roi, (3, 3), 0)
    edges = cv2.Canny(wg_blurred, 50, 150)

    # Dilate edges to connect them
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Fill from center (cavity center should be inside waveguide)
    flood_mask = np.zeros((wg_y2 - wg_y1 + 2, wg_x2 - wg_x1 + 2), dtype=np.uint8)
    center_local = (bw // 2 + bx - (wg_x1 - x1), bh // 2 + by - (wg_y1 - y1))
    center_local = (max(1, min(center_local[0], wg_x2 - wg_x1 - 1)),
                    max(1, min(center_local[1], wg_y2 - wg_y1 - 1)))

    # Use the expanded rectangle as waveguide shape, refined
    # Simple approach: use a slightly rounded rectangle
    cv2.rectangle(waveguide_mask, (wg_x1, wg_y1), (wg_x2, wg_y2), 255, -1)

    # Refine: use the actual bright region around cavity
    # The waveguide walls are brighter than the dark background of the inductor
    wg_region = img[wg_y1:wg_y2, wg_x1:wg_x2]
    hsv_wg = cv2.cvtColor(wg_region, cv2.COLOR_BGR2HSV)

    # Walls are metallic/bright, cavity is dark
    # Create mask of "not dark background"
    v_channel = hsv_wg[:, :, 2]
    _, bright_mask = cv2.threshold(v_channel, 80, 255, cv2.THRESH_BINARY)

    # Combine with cavity
    local_cavity = cavity_mask[wg_y1:wg_y2, wg_x1:wg_x2]
    combined = cv2.bitwise_or(bright_mask, local_cavity)

    # Clean
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Find largest contour in combined
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        # Convex hull for cleaner shape
        hull = cv2.convexHull(largest)
        waveguide_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_hull = hull.copy()
        shifted_hull[:, :, 0] += wg_x1
        shifted_hull[:, :, 1] += wg_y1
        cv2.drawContours(waveguide_mask, [shifted_hull], -1, 255, -1)

    return waveguide_mask, (wg_x1, wg_y1, wg_x2, wg_y2)


def find_flux(img: np.ndarray, waveguide_bbox):
    """Find flux - white/bright powder patches on left and right sides of waveguide."""
    h, w = img.shape[:2]

    if waveguide_bbox is None:
        return np.zeros((h, w), dtype=np.uint8)

    wg_x1, wg_y1, wg_x2, wg_y2 = waveguide_bbox
    wg_cx = (wg_x1 + wg_x2) // 2
    wg_cy = (wg_y1 + wg_y2) // 2
    wg_w = wg_x2 - wg_x1
    wg_h = wg_y2 - wg_y1

    # Flux is on the LEFT and RIGHT sides of waveguide
    # Search area: slightly wider than waveguide, same height range
    search_expand_x = int(wg_w * 0.8)
    search_expand_y = int(wg_h * 0.3)

    sx1 = max(0, wg_x1 - search_expand_x)
    sy1 = max(0, wg_y1 - search_expand_y)
    sx2 = min(w, wg_x2 + search_expand_x)
    sy2 = min(h, wg_y2 + search_expand_y)

    search_roi = img[sy1:sy2, sx1:sx2]

    # Flux is WHITE/bright powder - high value, low saturation in HSV
    hsv = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)

    # White/bright: high V, low S
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 80, 255))

    # Also check LAB - flux appears very bright
    lab = cv2.cvtColor(search_roi, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    _, bright = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)

    flux_raw = cv2.bitwise_or(white_mask, bright)

    # Remove the waveguide area from flux (flux is AROUND, not ON the waveguide)
    wg_local_x1 = wg_x1 - sx1
    wg_local_y1 = wg_y1 - sy1
    wg_local_x2 = wg_x2 - sx1
    wg_local_y2 = wg_y2 - sy1

    # Shrink waveguide exclusion zone slightly (flux touches edges)
    shrink = 5
    ex1 = max(0, wg_local_x1 + shrink)
    ey1 = max(0, wg_local_y1 + shrink)
    ex2 = min(flux_raw.shape[1], wg_local_x2 - shrink)
    ey2 = min(flux_raw.shape[0], wg_local_y2 - shrink)
    flux_raw[ey1:ey2, ex1:ex2] = 0

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    flux_raw = cv2.morphologyEx(flux_raw, cv2.MORPH_OPEN, kernel, iterations=1)
    flux_raw = cv2.morphologyEx(flux_raw, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only regions close to waveguide sides (left and right)
    flux_mask = np.zeros((h, w), dtype=np.uint8)
    flux_mask[sy1:sy2, sx1:sx2] = flux_raw

    # Filter: keep only blobs that are to the left or right of waveguide center
    cnts, _ = cv2.findContours(flux_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flux_final = np.zeros((h, w), dtype=np.uint8)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx_blob = int(M["m10"] / M["m00"])
        # Should be to the left or right of waveguide, not above/below
        if abs(cx_blob - wg_cx) > wg_w * 0.15:  # offset from center
            cv2.drawContours(flux_final, [cnt], -1, 255, -1)

    return flux_final


def find_solder(img: np.ndarray, waveguide_bbox, waveguide_mask):
    """Find solder - U-shaped metallic strip at the bottom of waveguide."""
    h, w = img.shape[:2]

    if waveguide_bbox is None:
        return np.zeros((h, w), dtype=np.uint8)

    wg_x1, wg_y1, wg_x2, wg_y2 = waveguide_bbox
    wg_w = wg_x2 - wg_x1
    wg_h = wg_y2 - wg_y1

    # Solder is at the BOTTOM of waveguide, U-shaped
    # Search in the lower half of waveguide and slightly below
    sy1 = wg_y1 + int(wg_h * 0.4)  # lower 60% of waveguide
    sy2 = min(h, wg_y2 + int(wg_h * 0.3))  # extend slightly below
    sx1 = max(0, wg_x1 - int(wg_w * 0.1))
    sx2 = min(w, wg_x2 + int(wg_w * 0.1))

    search_roi = img[sy1:sy2, sx1:sx2]
    gray_roi = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)

    # Solder is metallic gray, medium brightness
    # It's brighter than cavity but not as white as flux
    hsv_roi = cv2.cvtColor(search_roi, cv2.COLOR_BGR2HSV)

    # Metallic: low saturation, medium-high value (but not white like flux)
    metallic = cv2.inRange(hsv_roi, (0, 0, 100), (180, 60, 200))

    # Remove very dark (cavity) and very bright (flux) areas
    _, not_dark = cv2.threshold(gray_roi, 70, 255, cv2.THRESH_BINARY)
    _, not_bright = cv2.threshold(gray_roi, 210, 255, cv2.THRESH_BINARY_INV)

    solder_raw = cv2.bitwise_and(metallic, not_dark)
    solder_raw = cv2.bitwise_and(solder_raw, not_bright)

    # Clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    solder_raw = cv2.morphologyEx(solder_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    solder_raw = cv2.morphologyEx(solder_raw, cv2.MORPH_OPEN, kernel, iterations=1)

    solder_mask = np.zeros((h, w), dtype=np.uint8)
    solder_mask[sy1:sy2, sx1:sx2] = solder_raw

    # Remove overlap with waveguide interior (keep only edges/bottom)
    # Erode waveguide mask and subtract
    wg_eroded = cv2.erode(waveguide_mask,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
    solder_mask = cv2.bitwise_and(solder_mask, cv2.bitwise_not(wg_eroded))

    return solder_mask


def visualize_annotations(img: np.ndarray, waveguide_mask, flux_mask, solder_mask):
    """Create visualization with colored overlays."""
    overlay = img.copy()

    # Colors (BGR): waveguide=yellow+purple, flux=cyan, solder=green
    # Waveguide: yellow overlay
    wg_colored = np.zeros_like(img)
    wg_colored[:] = (0, 255, 255)  # Yellow in BGR
    mask_3ch = cv2.merge([waveguide_mask, waveguide_mask, waveguide_mask])
    overlay = np.where(mask_3ch > 0,
                       cv2.addWeighted(overlay, 0.6, wg_colored, 0.4, 0),
                       overlay)

    # Flux: cyan overlay
    flux_colored = np.zeros_like(img)
    flux_colored[:] = (255, 255, 0)  # Cyan in BGR
    mask_3ch = cv2.merge([flux_mask, flux_mask, flux_mask])
    overlay = np.where(mask_3ch > 0,
                       cv2.addWeighted(overlay, 0.5, flux_colored, 0.5, 0),
                       overlay)

    # Solder: green overlay
    solder_colored = np.zeros_like(img)
    solder_colored[:] = (0, 255, 0)  # Green in BGR
    mask_3ch = cv2.merge([solder_mask, solder_mask, solder_mask])
    overlay = np.where(mask_3ch > 0,
                       cv2.addWeighted(overlay, 0.5, solder_colored, 0.5, 0),
                       overlay)

    # Draw contours for clarity
    wg_cnts, _ = cv2.findContours(waveguide_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, wg_cnts, -1, (0, 200, 255), 2)  # Yellow outline

    flux_cnts, _ = cv2.findContours(flux_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, flux_cnts, -1, (255, 200, 0), 2)  # Cyan outline

    solder_cnts, _ = cv2.findContours(solder_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, solder_cnts, -1, (0, 200, 0), 2)  # Green outline

    # Add legend
    legend_y = 30
    cv2.putText(overlay, "Waveguide", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(overlay, "Flux", (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(overlay, "Solder", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return overlay


def process_frame(img_path: str, output_path: str):
    """Process a single frame and save annotated result."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read: {img_path}")
        return

    # Find components
    waveguide_mask, wg_bbox = find_waveguide_region(img)

    if waveguide_mask is None:
        print(f"No waveguide found in {img_path}")
        return

    flux_mask = find_flux(img, wg_bbox)
    solder_mask = find_solder(img, wg_bbox, waveguide_mask)

    # Create side-by-side: original + annotated
    annotated = visualize_annotations(img, waveguide_mask, flux_mask, solder_mask)

    combined = np.hstack([img, annotated])
    cv2.imwrite(output_path, combined)
    print(f"Saved: {output_path}")


# Process sample frames
print(f"Processing {len(sample_names)} sample frames...")
for i, name in enumerate(sample_names):
    img_path = str(test_dir / name)
    out_path = str(OUTPUT / f"demo_{i+1}_{name[:12]}.jpg")
    process_frame(img_path, out_path)

print("Done!")
print(f"Results saved to: {OUTPUT}")
