import os
import tempfile

import modal
import torch

from config import CONFIG
from modal_backend import app, huggingface_secret, pose_image, pose_volumes
from pose.feature_extraction import keypoints_to_feature_vector
from pose.preprocessing import decode_video_path
from shared.schemas import PoseFrame, PoseResult


class PoseEstimator:
    def __init__(self, device=None):
        from transformers import (
            AutoProcessor,
            RTDetrForObjectDetection,
            VitPoseForPoseEstimation,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.det_processor = AutoProcessor.from_pretrained(CONFIG.pose.detector_model)
        self.det_model = RTDetrForObjectDetection.from_pretrained(
            CONFIG.pose.detector_model,
            device_map=self.device,
        )
        self.pose_processor = AutoProcessor.from_pretrained(CONFIG.pose.pose_model)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(
            CONFIG.pose.pose_model,
            device_map=self.device,
        )

    def warmup(self):
        return None

    def detect_person_box(self, pil_image):
        det_inputs = self.det_processor(images=pil_image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            det_outputs = self.det_model(**det_inputs)

        det_results = self.det_processor.post_process_object_detection(
            det_outputs,
            target_sizes=torch.tensor([(pil_image.height, pil_image.width)]),
            threshold=CONFIG.pose.detection_threshold,
        )
        person_boxes = det_results[0]["boxes"][det_results[0]["labels"] == 0]
        if len(person_boxes) == 0:
            return None

        box = person_boxes[0].cpu().numpy()
        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

    def process_pose_batch(
        self, batch_images, batch_boxes, batch_indices, results_container
    ):
        if not batch_images:
            return

        pose_inputs = self.pose_processor(
            images=batch_images,
            boxes=batch_boxes,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            pose_outputs = self.pose_model(
                **pose_inputs,
                dataset_index=torch.tensor([0]).to(self.device),
            )

        pose_results = self.pose_processor.post_process_pose_estimation(
            pose_outputs,
            boxes=batch_boxes,
        )

        for i, result in enumerate(pose_results):
            original_idx = batch_indices[i]

            if len(result) > 0:
                kpts = result[0]["keypoints"]
                raw_kpts_np = kpts.cpu().numpy()
                results_container[original_idx] = {
                    "exists": True,
                    "kpts": raw_kpts_np,
                    "features": keypoints_to_feature_vector(kpts),
                }
            else:
                results_container[original_idx] = {
                    "exists": False,
                    "kpts": [],
                    "features": [0] * 8,
                }

    def process_video_path(self, video_path, target_frames=None):
        frames = decode_video_path(
            video_path, target_frames or CONFIG.pose.target_frames
        )
        results_container = [None] * len(frames)
        batch_imgs, batch_boxes, batch_idxs = [], [], []

        for i, frame in enumerate(frames):
            box_coco = self.detect_person_box(frame["pil_image"])

            if box_coco is None:
                results_container[i] = {
                    "exists": False,
                    "kpts": [],
                    "features": [0] * 8,
                }
            else:
                batch_imgs.append(frame["pil_image"])
                batch_boxes.append([box_coco])
                batch_idxs.append(i)

            if len(batch_imgs) >= CONFIG.pose.batch_size:
                self.process_pose_batch(
                    batch_imgs, batch_boxes, batch_idxs, results_container
                )
                batch_imgs, batch_boxes, batch_idxs = [], [], []

        if batch_imgs:
            self.process_pose_batch(
                batch_imgs, batch_boxes, batch_idxs, results_container
            )

        processed_data = []
        for i, frame in enumerate(frames):
            result = results_container[i]
            if result is not None:
                processed_data.append(
                    {
                        "image": frame["image"],
                        "features": result["features"],
                        "kpts": result["kpts"],
                        "exists": result["exists"],
                        "metadata": {
                            "frame_id": frame["frame_id"],
                            "source_frame_index": frame["source_frame_index"],
                        },
                    }
                )

        return processed_data

    def process_video_bytes(self, video_bytes: bytes, target_frames=None) -> PoseResult:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            video_path = tmp.name

        try:
            frames = self.process_video_path(video_path, target_frames)
            return PoseResult(
                frames=[
                    PoseFrame(
                        frame_id=frame["metadata"]["frame_id"],
                        exists=frame["exists"],
                        keypoints=frame["kpts"].tolist()
                        if hasattr(frame["kpts"], "tolist")
                        else frame["kpts"],
                        features=frame["features"],
                        metadata=frame["metadata"],
                    )
                    for frame in frames
                ]
            )
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)


_local_estimator = None


def get_local_estimator():
    global _local_estimator
    if _local_estimator is None:
        _local_estimator = PoseEstimator()
    return _local_estimator


def process_video_path(video_path, target_frames):
    return get_local_estimator().process_video_path(video_path, target_frames)


@app.cls(
    image=pose_image,
    gpu=CONFIG.hardware.pose_gpu,
    secrets=[huggingface_secret],
    volumes=pose_volumes,
    min_containers=CONFIG.pose_scaling.min_containers,
    max_containers=CONFIG.pose_scaling.max_containers,
    scaledown_window=CONFIG.pose_scaling.idle_timeout,
)
@modal.enable_memory_snapshot()
class PoseService:
    @modal.enter()
    def enter(self):
        self.estimator = PoseEstimator(device="cuda")
        self.estimator.warmup()

    @modal.method()
    def infer(self, video_bytes: bytes, target_frames=None):
        return self.estimator.process_video_bytes(video_bytes, target_frames)
