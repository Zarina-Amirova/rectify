import numpy as np
import random


class VanishingPointFinder:
    def __init__(self, facade_lines, img_h, img_w, n_iter=2000, thresh=12):
        self.lines = facade_lines
        self.img_h = img_h
        self.img_w = img_w
        self.n_iter = n_iter
        self.thresh = thresh
        self.abc_list = [self._line_abc(l) for l in facade_lines]

    @staticmethod
    def _line_abc(l):
        x1, y1, x2, y2 = l
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        n = np.sqrt(a ** 2 + b ** 2 + 1e-10)
        return a / n, b / n, c / n

    @staticmethod
    def _intersect(l1, l2):
        a1 = l1[3]-l1[1]; b1 = l1[0]-l1[2]; c1 = l1[2]*l1[1]-l1[0]*l1[3]
        a2 = l2[3]-l2[1]; b2 = l2[0]-l2[2]; c2 = l2[2]*l2[1]-l2[0]*l2[3]
        D = a1*b2 - a2*b1
        if abs(D) < 1e-8:
            return None
        return (b1*c2 - b2*c1) / D, (a2*c1 - a1*c2) / D

    def find(self):
        if len(self.lines) < 2:
            return None, 0

        best_vp, best_score = None, 0

        for _ in range(self.n_iter):
            i, j = random.sample(range(len(self.lines)), 2)
            pt = self._intersect(self.lines[i], self.lines[j])
            if pt is None:
                continue

            px, py = pt

            if py > self.img_h * 0.35:
                continue
            if not (-self.img_h * 30 < px < self.img_w * 30):
                continue

            score = sum(1 for abc in self.abc_list
                        if abs(abc[0]*px + abc[1]*py + abc[2]) < self.thresh)

            if score > best_score:
                best_score = score
                best_vp = (px, py)

        return best_vp, best_score
