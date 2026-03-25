import numpy as np
from src.detector import Detector
from src.vanishing_point import VanishingPointFinder


class MetricsCalculator:
    def __init__(self, img_before, img_after, vp_before):
        self.img_before = img_before
        self.img_after = img_after
        self.vp_before = vp_before
        self.h, self.w = img_before.shape[:2]
        self.ha, self.wa = img_after.shape[:2]

    @staticmethod
    def verticality_error(lines):
        if not lines:
            return None
        errors = []
        for l in lines:
            dx = abs(l[2] - l[0])
            dy = abs(l[3] - l[1])
            angle = abs(np.degrees(np.arctan2(dy, max(dx, 1))))
            errors.append(abs(90 - angle))
        return float(np.mean(errors))

    @staticmethod
    def parallelism_std(lines):
        if not lines:
            return None
        angles = []
        for l in lines:
            dx = l[2] - l[0]
            dy = l[3] - l[1]
            angles.append(np.degrees(np.arctan2(dy, max(abs(dx), 1))))
        return float(np.std(angles))

    @staticmethod
    def vp_residual(lines, vp):
        if not lines or vp is None:
            return None
        vp_x, vp_y = vp
        dists = []
        for l in lines:
            x1, y1, x2, y2 = l
            a = y2-y1; b = x1-x2; c = x2*y1-x1*y2
            n = np.sqrt(a**2 + b**2 + 1e-10)
            dists.append(abs(a*vp_x + b*vp_y + c) / n)
        return float(np.mean(dists))

    @staticmethod
    def horizon_distance(vp, img_h, img_w):
        if vp is None:
            return None
        vp_x, vp_y = vp
        cx, cy = img_w / 2, img_h / 2
        dist = np.sqrt((vp_x - cx)**2 + (vp_y - cy)**2)
        return float(dist / img_h)

    def compute_all(self):
        lines_before, _ = Detector(self.img_before).get_facade_lines()
        lines_after, _  = Detector(self.img_after).get_facade_lines()
        vp_after, _ = VanishingPointFinder(lines_after, self.ha, self.wa).find()

        return {
            "verticality_error": {
                "before": self.verticality_error(lines_before),
                "after":  self.verticality_error(lines_after),
                "unit": "degrees, lower=better"
            },
            "parallelism_std": {
                "before": self.parallelism_std(lines_before),
                "after":  self.parallelism_std(lines_after),
                "unit": "degrees, lower=better"
            },
            "vp_residual_px": {
                "before": self.vp_residual(lines_before, self.vp_before),
                "after":  self.vp_residual(lines_after, vp_after),
                "unit": "pixels, lower=better"
            },
            "vp_horizon_distance": {
                "before": self.horizon_distance(self.vp_before, self.h, self.w),
                "after":  self.horizon_distance(vp_after, self.ha, self.wa),
                "unit": "normalized by height, higher=more tilt"
            }
        }
