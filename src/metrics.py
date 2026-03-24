"""
Метрики качества ректификации.

Выбранные метрики оценивают насколько хорошо линии здания
стали параллельными после выравнивания
"""
import numpy as np
import cv2
from src.detector import get_facade_lines


def verticality_error(lines):
    """
    Метрика 1: Средняя ошибка вертикальности (градусы).
    Для каждой линии считаем угол отклонения от вертикали (90°).
    Идеально = 0°. Меньше = лучше.
    стандартная оценка VP через угловую точность.
    """
    if not lines:
        return None
    errors = []
    for l in lines:
        dx = abs(l[2] - l[0]); dy = abs(l[3] - l[1])
        angle = abs(np.degrees(np.arctan2(dy, max(dx, 1))))
        errors.append(abs(90 - angle))
    return float(np.mean(errors))


def parallelism_std(lines):
    """
    Метрика 2: Стандартное отклонение углов линий (градусы).
    Если линии параллельны — все углы одинаковы, std ≈ 0.
    standard deviation of corrected line angles
    как мера качества перспективной коррекции.
    """
    if not lines:
        return None
    angles = []
    for l in lines:
        dx = l[2] - l[0]; dy = l[3] - l[1]
        angles.append(np.degrees(np.arctan2(dy, max(abs(dx), 1))))
    return float(np.std(angles))


def vp_residual(lines, vp):
    """
    Метрика 3: Среднее расстояние от VP до каждой линии (пикселей).
    Если VP найдена точно — все линии проходят через неё, расстояние ≈ 0.
    тандартная inlier-метрика в RANSAC.
    """
    if not lines or vp is None:
        return None
    vp_x, vp_y = vp
    dists = []
    for l in lines:
        x1,y1,x2,y2 = l
        a = y2-y1; b = x1-x2; c = x2*y1-x1*y2
        n = np.sqrt(a**2 + b**2 + 1e-10)
        dists.append(abs(a*vp_x + b*vp_y + c) / n)
    return float(np.mean(dists))


def horizon_distance(vp, img_h, img_w):
    """
    Метрика 4: Расстояние VP от центра кадра (нормированное).
    Чем дальше VP за верхним краем — тем сильнее был наклон камеры.
    После ректификации VP должна уйти в бесконечность (линии параллельны).
    """
    if vp is None:
        return None
    vp_x, vp_y = vp
    cx, cy = img_w / 2, img_h / 2
    dist = np.sqrt((vp_x - cx)**2 + (vp_y - cy)**2)
    return float(dist / img_h)


def compute_all_metrics(img_bgr_before, img_bgr_after, vp_before):
    """Считает все метрики ДО и ПОСЛЕ, возвращает словарь."""
    lines_before, _ = get_facade_lines(img_bgr_before)
    lines_after, _  = get_facade_lines(img_bgr_after)

    h, w = img_bgr_before.shape[:2]
    ha, wa = img_bgr_after.shape[:2]

    # После ректификации линии должны стать вертикальными, ищем VP на ректифицированном изображении
    from src.vanishing_point import find_vanishing_point
    vp_after, _ = find_vanishing_point(lines_after, ha, wa)

    m = {
        "verticality_error": {
            "before": verticality_error(lines_before),
            "after":  verticality_error(lines_after),
            "unit": "degrees, lower=better"
        },
        "parallelism_std": {
            "before": parallelism_std(lines_before),
            "after":  parallelism_std(lines_after),
            "unit": "degrees, lower=better"
        },
        "vp_residual_px": {
            "before": vp_residual(lines_before, vp_before),
            "after":  vp_residual(lines_after, vp_after),
            "unit": "pixels, lower=better"
        },
        "vp_horizon_distance": {
            "before": horizon_distance(vp_before, h, w),
            "after":  horizon_distance(vp_after, ha, wa),
            "unit": "normalized by height, higher=more tilt"
        }
    }
    return m
