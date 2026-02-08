from .common import flux_image
from .control_image_pipeline import ControlImagePipeline

with flux_image.imports():
    import time
    from io import BytesIO

    import numpy as np
    import torch
    from diffusers import FluxKontextPipeline
    from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
    from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
    from nunchaku.utils import get_precision
    from PIL import Image


class ImageGenPipeline:
    def enter(self):
        print("[ImageGenPipeline][__init__] initializing")
        self.num_inference_steps = 30  # use ~50 for [dev], smaller for [schnell]
        self.control_pipe = ControlImagePipeline()
        self.control_pipe.enter()
        precision = get_precision()
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-{precision}_r32-flux.1-kontext-dev.safetensors",
            cache_dir="/cache",
        )
        transformer.update_lora_params(
            "thedeoxen/refcontrol-flux-kontext-reference-pose-lora/refcontrol_pose.safetensors"
        )
        text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
            "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors",
            cache_dir="/cache",
        )
        self.pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        apply_cache_on_pipe(self.pipe, residual_diff_threshold=0.12)

    def generate_correct_image(
        self,
        trainer_image,
        user_image,
        trainer_pose,
        user_pose,
        flux_prompt,
        gdino_prompt,
    ):
        print(
            f"[ImageGenPipeline][generate_correct_image] flux_prompt={flux_prompt}, gdino_prompt={gdino_prompt}"
        )

        control_image = self.control_pipe(
            trainer_image, user_image, trainer_pose, user_pose, gdino_prompt
        )
        flux_input = np.hstack([user_image, control_image]).astype(np.uint8)
        flux_input = Image.fromarray(flux_input)
        # flux_input = flux_input.transpose(2, 0, 1)
        print("input size: ", flux_input.size)
        flux_output = self.inference("refcontrolpose " + flux_prompt, flux_input)
        return flux_output

    def inference(self, prompt: str, input_image) -> bytes:
        print(f"[ImageGenPipeline][inference] prompt={prompt}")
        print("🎨 processing image-to-image...")
        start_time = time.time()
        out = self.pipe(
            prompt=prompt,
            image=input_image,
            num_inference_steps=self.num_inference_steps,
        ).images[0]
        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        end_time = time.time()
        duration = end_time - start_time
        print(f"⏱️ Generation finished in {duration:.2f}s")
        return byte_stream.getvalue()
