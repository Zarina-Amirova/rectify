import cv2
import json
import numpy as np
from pathlib import Path


class Annotator:
    def __init__(self, img_bgr):
        self.img = img_bgr

    def draw_debug_image(self, facade_lines, vp, pitch_deg, yaw_deg, score):
        vis = self.img.copy()
        for l in facade_lines:
            cv2.line(vis, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1)
        if vp:
            h, w = self.img.shape[:2]
            pt = (int(np.clip(vp[0], 0, w-1)), int(np.clip(vp[1], 0, h-1)))
            cv2.circle(vis, pt, 20, (0, 0, 255), -1)
            cv2.putText(vis,
                        f"p={pitch_deg:.0f} y={yaw_deg:.0f} s={score}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return vis

    @staticmethod
    def save_annotation(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
