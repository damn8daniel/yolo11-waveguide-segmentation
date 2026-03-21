"""
Конфигурация проекта.
Сегментация объектов индукционной пайки волноводов + определение этапов техпроцесса.
"""

import os

# ============================================================================
# ПУТИ
# ============================================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PROJECT_DIR)  # корень НИР/

# Данные
VIDEO_DIR = os.path.join(PARENT_DIR, "162___10")
DATASET_DIR = os.path.join(PARENT_DIR, "data", "dataset")
DATASET_YAML = os.path.join(DATASET_DIR, "data.yaml")

# Модели — свои веса в ДаняБоряНир/runs/, старые в корне
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")
OLD_RUNS_DIR = os.path.join(PARENT_DIR, "runs")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Претрейн
PRETRAINED = {
    "nano": os.path.join(PARENT_DIR, "yolo11n-seg.pt"),
    "small": os.path.join(PARENT_DIR, "yolo11s-seg.pt"),
    "medium": os.path.join(PARENT_DIR, "yolo11m-seg.pt"),
}

# ============================================================================
# КЛАССЫ СЕГМЕНТАЦИИ
# ============================================================================

CLASSES = {0: "waveguide", 1: "flux", 2: "solder"}
NUM_CLASSES = len(CLASSES)

# Цвета для визуализации (BGR)
CLASS_COLORS = {
    0: (107, 107, 255),   # waveguide — красный
    1: (196, 205, 78),    # flux — бирюзовый
    2: (77, 230, 255),    # solder — жёлтый
}

# ============================================================================
# ЭТАПЫ ТЕХНОЛОГИЧЕСКОГО ПРОЦЕССА
# ============================================================================

STAGES = {
    0: "Предварительный нагрев",
    1: "Плавление флюса",
    2: "Плавление припоя",
    3: "Стабилизация",
}

STAGE_COLORS = {
    0: (200, 200, 200),   # серый
    1: (255, 165, 0),     # оранжевый (BGR)
    2: (0, 0, 255),       # красный
    3: (0, 200, 0),       # зелёный
}

# ============================================================================
# ПАРАМЕТРЫ ОБУЧЕНИЯ
# ============================================================================

TRAIN_PARAMS = {
    "imgsz": 640,
    "patience": 30,
    "lr0": 0.01,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 5,
    "mosaic": 1.0,
    "mixup": 0.15,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "flipud": 0.5,
    "fliplr": 0.5,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "workers": 8,
    "exist_ok": True,
    "verbose": True,
    "save": True,
    "save_period": 25,
}

MODEL_CONFIGS = {
    "nano":   {"pretrained": PRETRAINED["nano"],   "batch": 32, "epochs": 150},
    "small":  {"pretrained": PRETRAINED["small"],  "batch": 16, "epochs": 150},
    "medium": {"pretrained": PRETRAINED["medium"], "batch": 8,  "epochs": 150},
}

# ============================================================================
# РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (Google Colab, Tesla T4, 2026-03-21)
# ============================================================================

TRAINING_RESULTS = {
    "yolo11n-seg": {
        "seg_mAP50": 0.8404, "seg_mAP50_95": 0.6227, "speed_ms": 21.7,
        "ap50_waveguide": 0.778, "ap50_flux": 0.932, "ap50_solder": 0.946,
    },
    "yolo11s-seg": {
        "seg_mAP50": 0.8639, "seg_mAP50_95": 0.6438, "speed_ms": 26.3,
    },
    "yolo11m-seg": {
        "seg_mAP50": 0.8854, "seg_mAP50_95": 0.6606, "speed_ms": 43.5,
        "ap50_waveguide": 0.778, "ap50_flux": 0.932, "ap50_solder": 0.946,
    },
    "best_model": "yolo11m-seg",
}

# ============================================================================
# ПАРАМЕТРЫ ПРЕДОБРАБОТКИ
# ============================================================================

PREPROCESS = {
    "enable": True,
    "alpha": 1.2,       # контраст
    "beta": 5,          # яркость
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "wb_saturation": 0.8,
}

# ============================================================================
# ИНФЕРЕНС
# ============================================================================

INFERENCE_CONF = 0.5
INFERENCE_IOU = 0.7


def find_best_weights():
    """Находит лучшие обученные веса в runs/."""
    candidates = []
    if os.path.isdir(RUNS_DIR):
        for name in os.listdir(RUNS_DIR):
            best = os.path.join(RUNS_DIR, name, "weights", "best.pt")
            if os.path.exists(best):
                candidates.append((name, best))
    # Приоритет: medium > small > nano
    priority = {"yolo11m-seg": 3, "yolo11s-seg": 2, "yolo11n-seg": 1}
    candidates.sort(key=lambda x: priority.get(x[0], 0), reverse=True)
    if candidates:
        return candidates[0][1]
    return None
