"""
Модуль определения этапов технологического процесса индукционной пайки
на основе результатов сегментации YOLO.

Этапы:
  0 — Предварительный нагрев: флюс белый/матовый, припой твёрдый
  1 — Плавление флюса (~300°C): флюс становится прозрачным, испарения
  2 — Плавление припоя (~580-600°C): припой переходит в жидкое состояние
  3 — Стабилизация: припой затвердевает, температура фиксируется

Логика определения:
  - Анализируем площади масок (waveguide, flux, solder)
  - Анализируем цветовые характеристики внутри масок
  - Отслеживаем динамику изменений между кадрами
"""

import numpy as np
import cv2
from collections import deque


class StageDetector:
    """Определяет текущий этап пайки по результатам сегментации."""

    def __init__(self, history_size=30):
        """
        Args:
            history_size: количество кадров для скользящего анализа
        """
        self.history_size = history_size
        self.area_history = deque(maxlen=history_size)
        self.color_history = deque(maxlen=history_size)
        self.stage_history = deque(maxlen=history_size)
        self.current_stage = 0
        self.frame_count = 0

    def analyze_frame(self, frame, masks, classes, confs):
        """
        Анализирует один кадр и возвращает определённый этап.

        Args:
            frame: исходный кадр (BGR, numpy)
            masks: массив масок сегментации [N, H, W]
            classes: массив классов [N]
            confs: массив уверенностей [N]

        Returns:
            dict с информацией:
                stage: номер этапа (0-3)
                stage_name: название этапа
                confidence: уверенность в определении
                areas: площади масок по классам
                details: подробности анализа
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        total_area = h * w

        # Извлекаем площади масок по классам
        areas = {"waveguide": 0.0, "flux": 0.0, "solder": 0.0}
        colors = {"flux_brightness": 0.0, "solder_brightness": 0.0,
                  "flux_saturation": 0.0, "solder_saturation": 0.0}

        class_map = {0: "waveguide", 1: "flux", 2: "solder"}

        if masks is not None and len(masks) > 0:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for mask, cls_id, conf in zip(masks, classes, confs):
                cls_name = class_map.get(int(cls_id), None)
                if cls_name is None:
                    continue

                # Ресайз маски к размеру кадра
                if mask.shape != (h, w):
                    mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
                else:
                    mask_resized = mask.astype(np.float32)

                mask_bool = mask_resized > 0.5
                pixel_count = np.sum(mask_bool)

                # Площадь относительно всего кадра
                areas[cls_name] += pixel_count / total_area

                # Цветовой анализ внутри маски
                if pixel_count > 100 and cls_name in ("flux", "solder"):
                    masked_hsv = hsv_frame[mask_bool]
                    colors[f"{cls_name}_brightness"] = float(np.mean(masked_hsv[:, 2]))
                    colors[f"{cls_name}_saturation"] = float(np.mean(masked_hsv[:, 1]))

        self.area_history.append(areas)
        self.color_history.append(colors)

        # Определяем этап
        stage, confidence, details = self._classify_stage(areas, colors)

        self.current_stage = stage
        self.stage_history.append(stage)

        # Стабилизация: если последние N кадров показывают другой этап,
        # переключаемся; иначе оставляем текущий
        stable_stage = self._get_stable_stage()

        return {
            "stage": stable_stage,
            "stage_name": self._stage_name(stable_stage),
            "raw_stage": stage,
            "confidence": confidence,
            "areas": areas,
            "colors": colors,
            "details": details,
        }

    def _classify_stage(self, areas, colors):
        """
        Классифицирует этап на основе текущих площадей и цветов.

        Returns:
            (stage, confidence, details_str)
        """
        flux_area = areas["flux"]
        solder_area = areas["solder"]
        waveguide_area = areas["waveguide"]

        flux_brightness = colors["flux_brightness"]
        solder_brightness = colors["solder_brightness"]
        flux_saturation = colors["flux_saturation"]
        solder_saturation = colors["solder_saturation"]

        # Динамика площади припоя (растёт ли)
        solder_growing = self._is_growing("solder")
        flux_shrinking = self._is_shrinking("flux")

        # --- Правила определения этапов ---

        # Нет сегментации → этап 0 (начало)
        if flux_area < 0.005 and solder_area < 0.005:
            return 0, 0.5, "Объекты не обнаружены"

        # Этап 3: Стабилизация
        # Припой виден, площадь стабильна, яркость снижается
        if solder_area > 0.02 and not solder_growing and self._is_stable("solder"):
            return 3, 0.8, f"Припой стабилен ({solder_area:.3f}), не растёт"

        # Этап 2: Плавление припоя
        # Припой виден и его площадь растёт, яркость высокая
        if solder_area > 0.01 and (solder_growing or solder_brightness > 150):
            conf = min(0.95, 0.6 + solder_area * 5)
            return 2, conf, f"Припой плавится ({solder_area:.3f}), яркость={solder_brightness:.0f}"

        # Этап 1: Плавление флюса
        # Флюс виден, его яркость высокая (прозрачный/блестящий)
        if flux_area > 0.01 and flux_brightness > 120:
            conf = min(0.9, 0.5 + flux_area * 3)
            return 1, conf, f"Флюс плавится ({flux_area:.3f}), яркость={flux_brightness:.0f}"

        # Этап 0: Предварительный нагрев
        # Флюс виден, но матовый (низкая яркость)
        if flux_area > 0.005:
            return 0, 0.7, f"Флюс виден ({flux_area:.3f}), матовый, яркость={flux_brightness:.0f}"

        return 0, 0.5, "По умолчанию: предварительный нагрев"

    def _is_growing(self, class_name, window=10, threshold=0.002):
        """Проверяет, растёт ли площадь класса за последние N кадров."""
        if len(self.area_history) < window:
            return False
        recent = [h[class_name] for h in list(self.area_history)[-window:]]
        if len(recent) < 2:
            return False
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        return (second_half - first_half) > threshold

    def _is_shrinking(self, class_name, window=10, threshold=0.002):
        """Проверяет, уменьшается ли площадь класса."""
        if len(self.area_history) < window:
            return False
        recent = [h[class_name] for h in list(self.area_history)[-window:]]
        if len(recent) < 2:
            return False
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        return (first_half - second_half) > threshold

    def _is_stable(self, class_name, window=15, threshold=0.003):
        """Проверяет, стабильна ли площадь класса."""
        if len(self.area_history) < window:
            return False
        recent = [h[class_name] for h in list(self.area_history)[-window:]]
        return np.std(recent) < threshold

    def _get_stable_stage(self, min_votes=5):
        """Возвращает стабильный этап по голосованию последних кадров."""
        if len(self.stage_history) < min_votes:
            return self.current_stage
        recent = list(self.stage_history)[-min_votes:]
        # Большинство голосов
        counts = {}
        for s in recent:
            counts[s] = counts.get(s, 0) + 1
        return max(counts, key=counts.get)

    @staticmethod
    def _stage_name(stage):
        names = {
            0: "Предварительный нагрев",
            1: "Плавление флюса",
            2: "Плавление припоя",
            3: "Стабилизация",
        }
        return names.get(stage, "Неизвестно")

    def reset(self):
        """Сбрасывает состояние детектора (для нового видео)."""
        self.area_history.clear()
        self.color_history.clear()
        self.stage_history.clear()
        self.current_stage = 0
        self.frame_count = 0

    def get_timeline(self):
        """Возвращает историю этапов для визуализации таймлайна."""
        return list(self.stage_history)
