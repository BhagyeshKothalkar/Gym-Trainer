from .common import flux_image

with flux_image.imports():
    import base64
    from io import BytesIO

    import cv2
    import numpy as np

SKELETON_LABELS = [
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_hip"),
    ("r_shoulder", "r_hip"),
    ("l_hip", "r_hip"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
]
LABEL_TO_INDEX = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}
angle_triplets = [
    ("r_shoulder", "r_elbow", "r_wrist"),  # R_Elbow
    ("l_shoulder", "l_elbow", "l_wrist"),  # L_Elbow
    ("r_elbow", "r_shoulder", "r_hip"),  # R_Shoulder
    ("l_elbow", "l_shoulder", "l_hip"),  # L_Shoulder
    ("r_shoulder", "r_hip", "r_knee"),  # R_Hip
    ("l_shoulder", "l_hip", "l_knee"),  # L_Hip
    ("r_hip", "r_knee", "r_ankle"),  # R_Knee
    ("l_hip", "l_knee", "l_ankle"),  # L_Knee
]


def calculate_angle_2d(a, b, c):
    """
    calculates angle between the given three points a, b, c
    """

    if a is None or b is None or c is None:
        return np.nan

    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def draw_skeleton_on_image(image_array, pose_result, confidence_threshold=0.1):
    """
    Draws the skeleton on a numpy image array.
    """
    print(
        f"[][draw_skeleton_on_image] image_shape={image_array.shape}, pose_result_keys={list(pose_result.keys())}, confidence_threshold={confidence_threshold}"
    )
    img_copy = image_array.copy()

    for start_label, end_label in SKELETON_LABELS:
        pt1 = get_kp(pose_result, start_label, conf_threshold=confidence_threshold)
        pt2 = get_kp(pose_result, end_label, conf_threshold=confidence_threshold)

        if pt1 is not None and pt2 is not None:
            print(pt1, pt2)
            cv2.line(img_copy, pt1, pt2, (0, 0, 255), 2)
    unique_labels = set([lbl for connection in SKELETON_LABELS for lbl in connection])

    for label in unique_labels:
        kp = get_kp(pose_result, label, conf_threshold=confidence_threshold)
        if kp is not None:
            center = tuple(kp)
            cv2.circle(img_copy, center, 4, (0, 255, 0), -1)

    return img_copy


def find_angles(results):
    print(f"[][find_angles] results_len={len(results) if results else 0}")
    if not results:
        return np.zeros(len(angle_triplets))

    person_pose = results[0]

    angles = [
        calculate_angle_2d(
            get_kp(person_pose, p1),
            get_kp(person_pose, p2),
            get_kp(person_pose, p3),
        )
        for p1, p2, p3 in angle_triplets
    ]
    return np.array(angles)


def get_kp(pose_results, label, conf_threshold=0.1):
    try:
        iabel_id = LABEL_TO_INDEX[label]
        idx = pose_results["labels"].index(iabel_id)
    except ValueError:
        return None
    if pose_results["scores"][idx] > conf_threshold:
        return pose_results["keypoints"][idx]
    else:
        return None


def get_perspective_transform(src_pose, dst_pose, confidence_threshold=0.3):
    """
    Calculates the 3x3 transformation matrix to align the torso of src_pose to dst_pose.
    """
    torso_labels = ["l_shoulder", "r_shoulder", "l_hip", "r_hip"]

    def extract_torso_points(pose_data):

        points = []
        for label in torso_labels:
            kp = get_kp(pose_data, label)
            points.append(kp)
        if len(points) < 4:
            return None
        return np.array(points, dtype=np.float32)

    src_pts = extract_torso_points(src_pose)
    dst_pts = extract_torso_points(dst_pose)

    print(
        f"[][get_perspective_transform] src_pts_shape={src_pts.shape if src_pts is not None else 'None'}, dst_pts_shape={dst_pts.shape if dst_pts is not None else 'None'}"
    )

    if src_pts is not None and dst_pts is not None:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return M

    return np.eye(3, dtype=np.float32)


def image_to_base64(img_array):
    print(f"[][image_to_base64] img_array_shape={img_array.shape}")
    _, buffer = cv2.imencode(".jpg", img_array)
    return base64.b64encode(buffer).decode("utf-8")


def pil_to_base64(image, format="JPEG"):
    print(f"[][pil_to_base64] image_size={image.size}, format={format}")
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
