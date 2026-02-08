from .common import flux_image
from .utils import find_angles

with flux_image.imports():
    import tempfile

    import cv2
    import numpy as np
    import torch
    from transformers import (
        AutoProcessor,
        RTDetrForObjectDetection,
        VitPoseForPoseEstimation,
    )


class VideoProcessor:
    def enter(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 64
        self.height = 480
        self.width = 640

        print(f"[VideoProcessor][__init__] initializing on device={self.device}")

        print(f"Loading AI Models on {self.device}...")
        self.det_processor = AutoProcessor.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365"
        )
        self.det_model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365", device_map=self.device
        )

        self.pose_processor = AutoProcessor.from_pretrained(
            "usyd-community/vitpose-plus-small"
        )
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-plus-small", device_map=self.device
        )
        print("Models loaded successfully.")

    def pose_detection(self, batch):
        print(f"[VideoProcessor][pose_detection] processing batch of size {len(batch)}")

        det_inputs = self.det_processor(images=batch, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            det_outputs = self.det_model(**det_inputs)

        det_results = self.det_processor.post_process_object_detection(
            det_outputs,
            target_sizes=torch.tensor(
                [(self.height, self.width) for _ in range(len(batch))]
            ),
            threshold=0.3,
        )

        # first box per image in the batch
        boxes = [  # batch
            det_results[i]["boxes"][det_results[i]["labels"] == 0][0]
            for i in range(len(det_results))
        ]
        # processor exects [batch of images, person, box]
        coco_boxes = [  # batch
            [  # the processor expects a list for person
                [  # box
                    boxes[i][0].item(),
                    boxes[i][1].item(),
                    (boxes[i][2] - boxes[i][0]).item(),
                    (boxes[i][3] - boxes[i][1]).item(),
                ]
            ]
            for i in range(len(boxes))
        ]

        pose_inputs = self.pose_processor(
            batch, boxes=coco_boxes, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            pose_outputs = self.pose_model(
                **pose_inputs, dataset_index=torch.tensor([0]).to(self.device)
            )
        pose_results = self.pose_processor.post_process_pose_estimation(
            pose_outputs, boxes=coco_boxes
        )
        print(pose_results)
        for image in pose_results:
            for person_pose in image:
                for k, v in person_pose.items():
                    if isinstance(v, torch.Tensor):
                        person_pose[k] = v.cpu().round().to(torch.int32).tolist()
        return pose_results

    def get_video_batches(self, video_bytes, record_fps=8, batch_size=-1):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as u_tmp:
            u_tmp.write(video_bytes)
            u_tmp.flush()
            video_path = u_tmp.name

        if batch_size < 0:
            batch_size = self.batch_size
        print(
            f"[VideoProcessor][get_video_batches] video_path={video_path}, record_fps={record_fps}, batch_size={batch_size}"
        )
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS: {fps}")
        frame_interval = max(1, int(fps / record_fps))
        batches = []
        batch = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_interval != 0:
                continue
            frame = cv2.resize(frame, (self.width, self.height))
            batch.append(frame)
            if len(batch) == batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
        cap.release()
        print(f"[VideoProcessor][get_video_batches] Found {len(batches)} batches")
        return batches

    def process_video_path(self, video_bytes):
        print("[VideoProcessor][process_video_path] start processing ")
        pose_agg = []
        angles_list = []
        images = []

        batches = self.get_video_batches(video_bytes)
        for batch in batches:
            result_batch = self.pose_detection(batch)
            pose_agg.extend(result_batch)

        for batch in batches:
            images.extend(batch)

        for result_batch in pose_agg:
            angles = find_angles(result_batch)
            angles_list.append(angles)

        if angles_list:
            angles_agg = np.vstack(angles_list)
            print(
                f"[VideoProcessor][process_video_path] compiled angles shape: {angles_agg.shape}"
            )
        else:
            angles_agg = np.array([]).reshape(0, 8)

        return pose_agg, angles_agg, images
