import cv2
import numpy as np


class Detector:
    def __init__(self, img_bgr):
        self.img = img_bgr
        self.h, self.w = img_bgr.shape[:2]

    def find_all_lines(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.equalizeHist(gray), 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=80, minLineLength=60, maxLineGap=20)
        return [l[0].tolist() for l in lines] if lines is not None else []

    def is_facade_line(self, l):
        x1, y1, x2, y2 = l
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if length < self.h * 0.04:
            return False
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        angle = abs(np.degrees(np.arctan2(dy, max(dx, 1))))
        if angle < 15 or angle > 80:
            return False
        if length > self.w * 0.5 and 20 < angle < 70:
            return False
        if min(y1, y2) < self.h * 0.15:
            return False
        if abs(x2 - x1) > self.w * 0.6:
            return False
        if abs(x2 - x1) > 1:
            t = (0 - y1) / max(y2 - y1, 1e-5)
            x_at_top = x1 + t * (x2 - x1)
            if not (-self.w * 0.5 < x_at_top < self.w * 1.5):
                return False
        return True

    def get_facade_lines(self):
        all_lines = self.find_all_lines()
        facade = [l for l in all_lines if self.is_facade_line(l)]

        if len(facade) < 4:
            facade = [l for l in all_lines
                      if 15 < abs(np.degrees(np.arctan2(
                          abs(l[3] - l[1]), max(abs(l[2] - l[0]), 1)))) < 80
                      and np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2) > 50
                      and min(l[1], l[3]) > self.h * 0.15
                      and abs(l[2] - l[0]) < self.w * 0.6]

        return facade, all_lines
