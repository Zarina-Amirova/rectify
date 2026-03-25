import cv2
import numpy as np


class Rectifier:
    def __init__(self, img_bgr):
        self.img = img_bgr
        self.h, self.w = img_bgr.shape[:2]
        self.H = None
        self.pitch = None
        self.yaw = None
        self.K = None

    def compute_homography(self, vp_x, vp_y):
        cx, cy = self.w / 2, self.h / 2
        f = float(self.w)
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        Kinv = np.linalg.inv(self.K)

        d = Kinv @ np.array([vp_x, vp_y, 1.0])
        d /= np.linalg.norm(d)
        self.pitch = np.arctan2(d[1], d[2])
        self.yaw = np.arctan2(d[0], d[2])

        R_p = np.array([[1, 0, 0],
                        [0, np.cos(-self.pitch), -np.sin(-self.pitch)],
                        [0, np.sin(-self.pitch),  np.cos(-self.pitch)]])
        R_y = np.array([[np.cos(-self.yaw), 0, np.sin(-self.yaw)],
                        [0, 1, 0],
                        [-np.sin(-self.yaw), 0, np.cos(-self.yaw)]])
        self.H = self.K @ (R_y @ R_p) @ Kinv
        return self.H, self.pitch, self.yaw, self.K

    def apply_rectification(self):
        corners = np.float32([[0, 0], [self.w, 0],
                               [self.w, self.h], [0, self.h]]).reshape(-1, 1, 2)
        tp = cv2.perspectiveTransform(corners, self.H)
        min_x, min_y = tp[:, :, 0].min(), tp[:, :, 1].min()
        max_x, max_y = tp[:, :, 0].max(), tp[:, :, 1].max()
        T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
        return cv2.warpPerspective(self.img, T @ self.H,
                                   (int(max_x - min_x), int(max_y - min_y)))
