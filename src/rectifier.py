import cv2
import numpy as np


def compute_homography(vp_x, vp_y, img_w, img_h):
    cx, cy = img_w / 2, img_h / 2
    f = float(img_w)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    Kinv = np.linalg.inv(K)

    d = Kinv @ np.array([vp_x, vp_y, 1.0])
    d /= np.linalg.norm(d)
    pitch = np.arctan2(d[1], d[2])
    yaw = np.arctan2(d[0], d[2])

    R_p = np.array([[1, 0, 0],
                    [0, np.cos(-pitch), -np.sin(-pitch)],
                    [0, np.sin(-pitch), np.cos(-pitch)]])
    R_y = np.array([[np.cos(-yaw), 0, np.sin(-yaw)],
                    [0, 1, 0],
                    [-np.sin(-yaw), 0, np.cos(-yaw)]])
    H = K @ (R_y @ R_p) @ Kinv

    return H, pitch, yaw, K


def apply_rectification(img_bgr, H):
    h, w = img_bgr.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    tp = cv2.perspectiveTransform(corners, H)
    min_x, min_y = tp[:,:,0].min(), tp[:,:,1].min()
    max_x, max_y = tp[:,:,0].max(), tp[:,:,1].max()
    T = np.array([[1,0,-min_x],[0,1,-min_y],[0,0,1]], dtype=np.float64)
    result = cv2.warpPerspective(img_bgr, T @ H,
                                  (int(max_x - min_x), int(max_y - min_y)))
    return result
