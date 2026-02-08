from .common import flux_image
from .utils import draw_skeleton_on_image, get_perspective_transform

with flux_image.imports():
    import cv2
    import numpy as np
    import torch
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )


class ControlImagePipeline:
    def enter(self):
        print("[ControlImagePipeline][__init__] initializing")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(
            self.device
        )
        self.sam_processor = Sam2Processor.from_pretrained(
            "facebook/sam2.1-hiera-large"
        )
        self.grounding_processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        )
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(self.device)

    def mask(self, trainer_image, prompt):
        print(
            f"[ControlImagePipeline][mask] prompt={prompt}, trainer_image_size={trainer_image.size}"
        )
        inputs = self.grounding_processor(
            images=trainer_image, text=prompt, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        # results = self.grounding_processor.post_process_grounded_object_detection(
        #     outputs,
        #     inputs.input_ids,
        #     # box_threshold=0.35,
        #     text_threshold=0.25,
        #     target_sizes=[trainer_image.size[::-1]],
        # )

        # if not results[0]["boxes"].shape[0]:
        #     raise ValueError(f"No object found for prompt: {prompt}")

        # box = results[0]["boxes"][0].cpu().numpy()
        height, width, channels = trainer_image.shape
        box = torch.Tensor([0, 0, trainer_image.shape[0], trainer_image.shape[1]])

        inputs = self.sam_processor(
            images=trainer_image, input_boxes=[[box.tolist()]], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        masks = self.sam_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]

        final_mask = masks[0][0].numpy().astype(np.uint8)
        return final_mask

    def overlay_and_warp(
        self, final_mask, trainer_image, user_image, trainer_pose, user_pose
    ):
        print(
            f"[ControlImagePipeline][overlay_and_warp] final_mask_shape={final_mask.shape}, trainer_image_size={trainer_image.size}, user_image_size={user_image.size}"
        )
        trainer_cv = np.array(trainer_image)
        masked_img = cv2.bitwise_and(trainer_cv, trainer_cv, mask=final_mask)

        edges = cv2.Canny(masked_img, 100, 200)
        kernel = np.ones((2, 2), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=1)
        base = np.zeros(trainer_cv.shape)

        skeleton_layer = draw_skeleton_on_image(base, trainer_pose)

        combined_map = skeleton_layer.copy()

        edge_indices = thick_edges > 0
        combined_map[edge_indices] = [255, 255, 255]

        transform_matrix = get_perspective_transform(trainer_pose, user_pose)

        user_h, user_w, user_c = user_image.shape
        warped_img = cv2.warpPerspective(
            combined_map,
            transform_matrix,
            (user_w, user_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return warped_img

    def __call__(self, trainer_image, user_image, trainer_pose, user_pose, prompt):
        print(f"[ControlImagePipeline][__call__] prompt={prompt}")
        print(f"user image: {type(user_image)}, {user_image.shape}")
        print(f"tranier image: {type(trainer_image)}, {trainer_image.shape}")

        final_mask = self.mask(trainer_image, prompt)
        warped_img = self.overlay_and_warp(
            final_mask, trainer_image, user_image, trainer_pose, user_pose
        )

        return warped_img
