"""
Convert manual annotations JSON to YOLO-seg format labels.
Merges multiple polygons of same class into single entries.
Creates dataset structure ready for YOLO training.
"""
import json
import os
import shutil
from pathlib import Path

PROJECT = Path("/Users/damn8daniel/Desktop/НИР/ДаняБоряНир")
ANNOTATION_DIR = PROJECT / "manual_annotation"
FRAMES_DIR = ANNOTATION_DIR / "frames"

# Output dataset
OUTPUT = PROJECT / "dataset_v2"
IMAGES_TRAIN = OUTPUT / "images" / "train"
LABELS_TRAIN = OUTPUT / "labels" / "train"
IMAGES_VAL = OUTPUT / "images" / "val"
LABELS_VAL = OUTPUT / "labels" / "val"

for d in [IMAGES_TRAIN, LABELS_TRAIN, IMAGES_VAL, LABELS_VAL]:
    d.mkdir(parents=True, exist_ok=True)

# Load annotations
with open(ANNOTATION_DIR / "annotations.json") as f:
    annotations = json.load(f)

# Frame list (same order as in annotator.html)
frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])

CLASS_MAP = {"waveguide": 0, "flux": 1, "solder": 2}

# Image size (all frames are 640x640)
IMG_W = 640
IMG_H = 640

total_labels = 0
annotated_frames = []

for idx_str, polys in annotations.items():
    idx = int(idx_str)
    if idx >= len(frame_files):
        continue

    fname = frame_files[idx]
    label_name = os.path.splitext(fname)[0] + ".txt"

    # Convert polygons to YOLO-seg format
    # Each line: class_id x1 y1 x2 y2 ... (normalized 0-1)
    lines = []
    for poly in polys:
        cls = poly["class"]
        points = poly["points"]

        if cls not in CLASS_MAP:
            continue
        if len(points) < 3:
            continue  # Need at least 3 points for a polygon

        class_id = CLASS_MAP[cls]

        # Normalize coordinates
        coords = []
        for px, py in points:
            nx = max(0.0, min(1.0, px / IMG_W))
            ny = max(0.0, min(1.0, py / IMG_H))
            coords.append(f"{nx:.6f}")
            coords.append(f"{ny:.6f}")

        line = f"{class_id} " + " ".join(coords)
        lines.append(line)

    if lines:
        annotated_frames.append((fname, label_name, lines))
        total_labels += len(lines)

print(f"Converted {len(annotated_frames)} frames with {total_labels} total polygons")

# Split: 8 train, 2 val
val_count = max(1, len(annotated_frames) // 5)
train_frames = annotated_frames[:-val_count]
val_frames = annotated_frames[-val_count:]

print(f"Train: {len(train_frames)}, Val: {len(val_frames)}")

for frame_list, img_dir, lbl_dir in [
    (train_frames, IMAGES_TRAIN, LABELS_TRAIN),
    (val_frames, IMAGES_VAL, LABELS_VAL),
]:
    for fname, label_name, lines in frame_list:
        # Copy image
        shutil.copy2(FRAMES_DIR / fname, img_dir / fname)
        # Write label
        with open(lbl_dir / label_name, "w") as f:
            f.write("\n".join(lines) + "\n")

# Create data.yaml
data_yaml = f"""path: {OUTPUT}
train: images/train
val: images/val

names:
  0: waveguide
  1: flux
  2: solder

nc: 3
"""

with open(OUTPUT / "data.yaml", "w") as f:
    f.write(data_yaml)

print(f"\nDataset created at: {OUTPUT}")
print(f"data.yaml: {OUTPUT / 'data.yaml'}")

# Summary
for split, frames in [("train", train_frames), ("val", val_frames)]:
    print(f"\n{split}:")
    for fname, _, lines in frames:
        classes = [l.split()[0] for l in lines]
        class_names = {0: "wg", 1: "fx", 2: "sd"}
        names = [class_names[int(c)] for c in classes]
        print(f"  {fname[:30]}... → {', '.join(names)}")
