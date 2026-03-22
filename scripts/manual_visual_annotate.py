"""
Manually annotate 3 frames by drawing polygons based on visual analysis.
I can see the images and will define polygon coordinates for each object.
"""
import cv2
import numpy as np
from pathlib import Path

PROJECT = Path("/Users/damn8daniel/Desktop/НИР/ДаняБоряНир")
FRAMES = PROJECT / "manual_annotation" / "frames"
OUTPUT = PROJECT / "results" / "my_annotations"
OUTPUT.mkdir(exist_ok=True)

# Colors
COLORS = {
    "waveguide": (0, 230, 255),   # Yellow
    "flux": (255, 230, 0),         # Cyan
    "solder": (0, 230, 0),         # Green
}

def draw_annotation(img, polygons):
    """Draw polygons with colored overlays on image."""
    overlay = img.copy()
    for cls, pts in polygons:
        pts_arr = np.array(pts, dtype=np.int32)
        color = COLORS[cls]
        # Fill
        fill_color = tuple(int(c * 0.5) for c in color)
        cv2.fillPoly(overlay, [pts_arr], fill_color)
        # Contour
        cv2.polylines(overlay, [pts_arr], True, color, 2)

    result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

    # Legend
    cv2.putText(result, "Waveguide", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(result, "Flux", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(result, "Solder", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return result

def save_yolo_label(label_path, polygons, img_w=640, img_h=640):
    """Save YOLO-seg format label."""
    class_map = {"waveguide": 0, "flux": 1, "solder": 2}
    lines = []
    for cls, pts in polygons:
        cid = class_map[cls]
        coords = []
        for x, y in pts:
            coords.append(f"{x/img_w:.6f}")
            coords.append(f"{y/img_h:.6f}")
        lines.append(f"{cid} " + " ".join(coords))
    with open(label_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ========== Frame 1: MVI_6268_MOV-0005 ==========
# Looking at the image: waveguide is centered around (330, 380), slightly left of center
# Cavity (dark hole) is roughly at (320-370, 340-375)
# Walls extend around it
# Flux: white patches visible on left side ~(280, 370) and slight on right ~(385, 360)
# Solder: U-shape at bottom of waveguide ~(300-380, 390-430)

frame1_name = "MVI_6268_MOV-0005_jpg.rf.89f23872dffa9e326b2f5d0c345dbcf0.jpg"
frame1_polys = [
    # Waveguide (cavity + walls as one polygon)
    ("waveguide", [
        (295, 330), (375, 330), (385, 340), (385, 400),
        (375, 415), (295, 415), (285, 400), (285, 340)
    ]),
    # Flux left
    ("flux", [
        (265, 345), (290, 345), (290, 410), (275, 415),
        (260, 410), (255, 380)
    ]),
    # Flux right (small yellowish patch)
    ("flux", [
        (390, 340), (415, 340), (420, 360), (415, 380),
        (390, 385), (385, 365)
    ]),
    # Solder (U-shape bottom)
    ("solder", [
        (290, 400), (300, 420), (295, 440), (300, 450),
        (370, 450), (375, 440), (370, 420), (380, 400),
        (385, 415), (385, 455), (280, 455), (280, 415)
    ]),
]

# ========== Frame 2: MVI_6270_MOV-0008 ==========
# Waveguide centered around (325, 395), similar layout
# Slightly more visible flux (white patches on sides)

frame2_name = "MVI_6270_MOV-0008_jpg.rf.6a0436a536ab5c2b6d19c3d838039ac8.jpg"
frame2_polys = [
    # Waveguide
    ("waveguide", [
        (290, 350), (370, 350), (380, 360), (380, 420),
        (370, 430), (290, 430), (280, 420), (280, 360)
    ]),
    # Flux left
    ("flux", [
        (255, 365), (280, 360), (280, 425), (265, 430),
        (250, 420), (245, 390)
    ]),
    # Flux right
    ("flux", [
        (382, 355), (410, 350), (415, 375), (410, 400),
        (385, 405), (380, 380)
    ]),
    # Solder
    ("solder", [
        (285, 420), (290, 440), (285, 460), (375, 460),
        (380, 440), (375, 420), (385, 430), (390, 465),
        (275, 465), (275, 430)
    ]),
]

# ========== Frame 3: MVI_6273_MOV-0013 ==========
# Waveguide is a bit lower and more centered, ~(310, 420)
# Flux: white patches on sides
# This frame has slightly different angle

frame3_name = "MVI_6273_MOV-0013_jpg.rf.018c867928ac979dd4ddae55ecc5c4d5.jpg"
frame3_polys = [
    # Waveguide
    ("waveguide", [
        (275, 370), (355, 370), (365, 380), (365, 435),
        (355, 445), (275, 445), (265, 435), (265, 380)
    ]),
    # Flux left
    ("flux", [
        (240, 385), (265, 380), (265, 440), (250, 445),
        (235, 435), (230, 410)
    ]),
    # Flux right (less visible on this frame)
    ("flux", [
        (368, 378), (390, 375), (395, 395), (390, 420),
        (370, 425), (365, 400)
    ]),
    # Solder
    ("solder", [
        (270, 435), (275, 455), (270, 475), (360, 475),
        (365, 455), (360, 435), (370, 445), (375, 480),
        (260, 480), (260, 445)
    ]),
]

# Process all 3
frames_data = [
    (frame1_name, frame1_polys),
    (frame2_name, frame2_polys),
    (frame3_name, frame3_polys),
]

for fname, polys in frames_data:
    img_path = FRAMES / fname
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read {fname}")
        continue

    h, w = img.shape[:2]
    print(f"\n{fname}: {w}x{h}")
    print(f"  Polygons: {len(polys)}")

    # Draw annotated version
    annotated = draw_annotation(img, polys)
    combined = np.hstack([img, annotated])
    out_path = OUTPUT / f"annotated_{fname}"
    cv2.imwrite(str(out_path), combined)

    # Save YOLO label
    label_name = fname.rsplit(".", 1)[0] + ".txt"
    save_yolo_label(OUTPUT / label_name, polys, w, h)

    print(f"  Saved: {out_path}")

print(f"\nDone! Check {OUTPUT}")
