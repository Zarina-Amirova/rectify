import cv2
import numpy as np


def find_all_lines(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.equalizeHist(gray), 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=80, minLineLength=60, maxLineGap=20)
    return [l[0].tolist() for l in lines] if lines is not None else []


def is_facade_line(l, img_h, img_w):
    x1, y1, x2, y2 = l
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Слишком короткая линия
    if length < img_h * 0.04:
        return False

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    angle = abs(np.degrees(np.arctan2(dy, max(dx, 1))))

    # Только наклонные линии фасада: не горизонтали, не вертикали
    if angle < 15 or angle > 80:
        return False

    # Длинная диагональная линия через весь кадр — это провод, не фасад
    if length > img_w * 0.5 and 20 < angle < 70:
        return False

    # Линия не должна начинаться в небе (верхние 15% кадра)
    if min(y1, y2) < img_h * 0.15:
        return False

    # Линия не должна пересекать больше 60% ширины кадра — тоже признак провода
    if abs(x2 - x1) > img_w * 0.6:
        return False

    # Продолжение линии должно уходить в разумную зону выше кадра (к VP)
    if abs(x2 - x1) > 1:
        t = (0 - y1) / max(y2 - y1, 1e-5)
        x_at_top = x1 + t * (x2 - x1)
        if not (-img_w * 0.5 < x_at_top < img_w * 1.5):
            return False

    return True


def get_facade_lines(img_bgr):
    h, w = img_bgr.shape[:2]
    all_lines = find_all_lines(img_bgr)
    facade = [l for l in all_lines if is_facade_line(l, h, w)]

    # Если строгий фильтр ничего не оставил — смягчаем только угол и длину
    if len(facade) < 4:
        facade = [l for l in all_lines
                  if 15 < abs(np.degrees(np.arctan2(
                      abs(l[3] - l[1]), max(abs(l[2] - l[0]), 1)))) < 80
                  and np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2) > 50
                  and min(l[1], l[3]) > h * 0.15
                  and abs(l[2] - l[0]) < w * 0.6]

    return facade, all_lines
