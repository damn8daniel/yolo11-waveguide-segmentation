"""
Setup manual annotation:
1. Select 40 diverse frames (10 per video)
2. Copy them to a dedicated folder
3. Generate Label Studio JSON import file
4. Print instructions
"""
import os
import json
import shutil
import random
from pathlib import Path

DATASET = Path("/Users/damn8daniel/Desktop/НИР/data/dataset")
PROJECT = Path("/Users/damn8daniel/Desktop/НИР/ДаняБоряНир")
ANNOTATION_DIR = PROJECT / "manual_annotation"
FRAMES_DIR = ANNOTATION_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Select 10 frames per video, spread across train/val/test
VIDEOS = ["MVI_6265", "MVI_6268", "MVI_6270", "MVI_6273"]
FRAMES_PER_VIDEO = 10

selected = []
for vid in VIDEOS:
    all_frames = []
    for split in ["train", "val", "test"]:
        img_dir = DATASET / "images" / split
        if img_dir.exists():
            for f in sorted(os.listdir(img_dir)):
                if vid in f and f.endswith(".jpg"):
                    all_frames.append((split, f))

    # Spread evenly
    if len(all_frames) <= FRAMES_PER_VIDEO:
        chosen = all_frames
    else:
        step = len(all_frames) // FRAMES_PER_VIDEO
        chosen = [all_frames[i * step] for i in range(FRAMES_PER_VIDEO)]

    selected.extend(chosen)

print(f"Selected {len(selected)} frames:")
for vid in VIDEOS:
    count = sum(1 for s, f in selected if vid in f)
    print(f"  {vid}: {count} frames")

# Copy frames
for split, fname in selected:
    src = DATASET / "images" / split / fname
    # Simplify filename
    dst = FRAMES_DIR / fname
    shutil.copy2(src, dst)

# Generate Label Studio tasks JSON
tasks = []
for i, (split, fname) in enumerate(selected):
    tasks.append({
        "id": i + 1,
        "data": {
            "image": f"/data/local-files/?d=frames/{fname}"
        },
        "meta": {
            "original_split": split,
            "original_filename": fname
        }
    })

tasks_file = ANNOTATION_DIR / "tasks.json"
with open(tasks_file, "w") as f:
    json.dump(tasks, f, indent=2)

# Also create a simpler HTML-based annotation viewer
# Create Label Studio config for polygon segmentation
ls_config = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>

  <PolygonLabels name="label" toName="image" strokeWidth="3" pointSize="small" opacity="0.5">
    <Label value="waveguide" background="#FFD700"/>
    <Label value="flux" background="#00CED1"/>
    <Label value="solder" background="#32CD32"/>
  </PolygonLabels>
</View>"""

config_file = ANNOTATION_DIR / "label_config.xml"
with open(config_file, "w") as f:
    f.write(ls_config)

print(f"\nFiles created:")
print(f"  Frames: {FRAMES_DIR} ({len(selected)} images)")
print(f"  Tasks JSON: {tasks_file}")
print(f"  Label config: {config_file}")
print(f"\nTo start Label Studio:")
print(f"  label-studio start --port 8080")
print(f"  Then import tasks from: {tasks_file}")
print(f"  Use labeling config from: {config_file}")
print(f"  Set local storage root to: {ANNOTATION_DIR}")
