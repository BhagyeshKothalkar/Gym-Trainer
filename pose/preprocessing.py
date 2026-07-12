import cv2
import numpy as np
from PIL import Image

from config import CONFIG


def sample_frame_indices(total_frames: int, target_frames: int):
    return np.linspace(0, total_frames - 1, target_frames, dtype=int)


def resize_frame(frame):
    return cv2.resize(frame, (CONFIG.pose.resize_width, CONFIG.pose.resize_height))


def frame_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def decode_video_path(video_path: str, target_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total_frames, target_frames)
    frames = []

    for frame_id, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = resize_frame(frame)
        frames.append(
            {
                "frame_id": frame_id,
                "image": frame,
                "pil_image": frame_to_pil(frame),
                "source_frame_index": int(frame_idx),
            }
        )

    cap.release()
    return frames
