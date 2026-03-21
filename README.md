# YOLO11 Waveguide Soldering Segmentation

Automatic instance segmentation of **waveguides**, **flux**, and **solder** during induction soldering of space-grade waveguide assemblies, powered by [YOLO11-seg](https://docs.ultralytics.com/).

The system also detects the **technological stage** of the soldering process in real time (preheating, flux melting, solder melting, stabilization).

## Features

- **Multi-model comparison** -- trains and evaluates YOLO11 nano / small / medium variants to find the best accuracy-speed trade-off
- **Automatic batch tuning** -- selects the largest batch size that fits GPU memory
- **CLAHE preprocessing** -- contrast-limited adaptive histogram equalization + white-balance correction for robust detection under varying lighting
- **Real-time inference** -- processes live camera or video feeds with overlay visualization of masks and stage labels
- **Stage detection** -- rule-based classifier on top of segmentation masks identifies the current soldering phase
- **Colab-ready** -- includes a notebook for training on Google Colab (Tesla T4)

## Results

| Model | Seg mAP50 | Seg mAP50-95 | Speed (ms) |
|---|---|---|---|
| YOLO11n-seg | 0.840 | 0.623 | 21.7 |
| YOLO11s-seg | 0.864 | 0.644 | 26.3 |
| **YOLO11m-seg** | **0.885** | **0.661** | 43.5 |

Best model: **YOLO11m-seg** (trained on Google Colab, Tesla T4, 150 epochs).

## Segmentation Classes

| ID | Class | Description |
|----|-----------|-------------------------------|
| 0 | waveguide | Metal waveguide component |
| 1 | flux | Flux paste/liquid |
| 2 | solder | Molten solder metal |

## Soldering Stages

| Stage | Name | Visual Cues |
|-------|---------------------|----------------------------------------------|
| 0 | Preheating | Flux is white/matte |
| 1 | Flux melting | Flux becomes transparent, vapors visible |
| 2 | Solder melting | Solder liquefies, fills joints |
| 3 | Stabilization | Solder solidifies (5-15 seconds hold) |

## Project Structure

```
├── config.py                 # Paths, classes, training hyperparameters
├── train.py                  # Training pipeline (single or multi-model)
├── evaluate.py               # Evaluation on test set
├── inference.py              # Real-time inference with stage detection
├── stage_detector.py         # Soldering stage classifier
├── auto_annotate_waveguide.py# Semi-automatic annotation helper
├── batch_process.py          # Batch video processing
├── train_colab.ipynb         # Google Colab training notebook
├── requirements.txt          # Python dependencies
└── results/                  # Prediction previews & comparison JSON
```

## Tech Stack

- **Python 3.10+**
- **[Ultralytics](https://github.com/ultralytics/ultralytics)** (YOLO11)
- **PyTorch**
- **OpenCV** (image preprocessing, visualization)
- **scikit-image**

## Quick Start

```bash
# Clone
git clone https://github.com/damn888daniel/yolo11-waveguide-segmentation.git
cd yolo11-waveguide-segmentation

# Install dependencies
pip install -r requirements.txt

# Train (requires dataset in ../data/dataset/)
python train.py --model medium --epochs 150

# Run inference on a video
python inference.py --source path/to/video.mov

# Run inference on camera
python inference.py --source 0
```

## Dataset

The dataset is not included in this repository. It consists of annotated frames from induction soldering recordings with three segmentation classes: `waveguide`, `flux`, and `solder`. The dataset should be placed at `../data/dataset/` relative to the project root, following the standard Ultralytics YOLO format.

## Research Context

This project is part of a research effort on automating quality control in space waveguide induction soldering. The goal is to enable real-time monitoring and stage detection during the soldering process using computer vision.

## License

This project is provided for research and educational purposes.
