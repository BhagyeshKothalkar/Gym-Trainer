import cv2
import numpy as np


SKELETON_CONNECTIONS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def calculate_angle_2d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def draw_skeleton_on_image(image_array, kpts, confidence_threshold=0.3):
    img_copy = image_array.copy()

    for idx, (x, y) in enumerate(kpts):
        if 5 <= idx <= 16:
            cv2.circle(img_copy, (int(x), int(y)), 4, (0, 255, 0), -1)

    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx < len(kpts) and end_idx < len(kpts):
            pt1 = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
            pt2 = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
            cv2.line(img_copy, pt1, pt2, (0, 0, 255), 2)

    return img_copy


def get_kp(kpts, idx):
    if idx >= len(kpts):
        return np.array([0.0, 0.0])
    if hasattr(kpts[idx], "cpu"):
        return kpts[idx].cpu().numpy()
    return np.array(kpts[idx])


def keypoints_to_feature_vector(kpts):
    return [
        calculate_angle_2d(get_kp(kpts, 6), get_kp(kpts, 8), get_kp(kpts, 10)),
        calculate_angle_2d(get_kp(kpts, 5), get_kp(kpts, 7), get_kp(kpts, 9)),
        calculate_angle_2d(get_kp(kpts, 8), get_kp(kpts, 6), get_kp(kpts, 12)),
        calculate_angle_2d(get_kp(kpts, 7), get_kp(kpts, 5), get_kp(kpts, 11)),
        calculate_angle_2d(get_kp(kpts, 6), get_kp(kpts, 12), get_kp(kpts, 14)),
        calculate_angle_2d(get_kp(kpts, 5), get_kp(kpts, 11), get_kp(kpts, 13)),
        calculate_angle_2d(get_kp(kpts, 12), get_kp(kpts, 14), get_kp(kpts, 16)),
        calculate_angle_2d(get_kp(kpts, 11), get_kp(kpts, 13), get_kp(kpts, 15)),
    ]
