import torch
import torch.nn as nn

from config import CONFIG
from shared.utils import pil_image_to_base64


class ModelWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, x, x_ids, timesteps, ctx, ctx_ids, guidance):
        return self.transformer(
            x,
            x_ids.contiguous().to(torch.bfloat16),
            timesteps.contiguous().to(torch.bfloat16),
            ctx.contiguous().to(torch.bfloat16),
            ctx_ids.contiguous().to(torch.bfloat16),
            guidance.contiguous().to(torch.bfloat16),
        )


class FluxRuntime:
    def __init__(self):
        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not found")

    def load(self):
        from flux2.util import FLUX2_MODEL_INFO, load_ae, load_text_encoder

        self.model_info = FLUX2_MODEL_INFO[CONFIG.flux.model_name]
        self.package_path = CONFIG.flux.package_path(
            CONFIG.volumes,
            CONFIG.hardware,
            CONFIG.compilation,
        )
        self.loaded_model = ModelWrapper(
            torch._inductor.aoti_load_package(self.package_path)
        )
        self.tokenizer = None
        self.text_encoder = load_text_encoder(CONFIG.flux.model_name, self.device)
        self.scheduler = None
        self.ae = load_ae(CONFIG.flux.model_name)

        self.loaded_model.eval()
        self.ae.eval()
        self.text_encoder.eval()

        defaults = self.model_info.get("defaults", {})
        self.num_steps = defaults.get("num_steps", CONFIG.flux.num_steps)
        self.guidance = defaults.get("guidance", CONFIG.flux.guidance)

    def warmup(self):
        return None

    def generate(self, prompt: str, cond_image_b64: str):
        import base64
        import random
        import tempfile
        import time
        from typing import List

        from einops import rearrange
        from flux2.sampling import (
            batched_prc_img,
            batched_prc_txt,
            denoise,
            encode_image_refs,
            get_schedule,
            scatter_ids,
        )
        from PIL import Image

        t0 = time.perf_counter()
        if not prompt or prompt.strip() == "":
            prompt = CONFIG.flux.prompt

        cond_image_bytes = base64.b64decode(cond_image_b64)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            tmp.write(cond_image_bytes)
            tmp.flush()
            img = Image.open(tmp.name)

            batch_size = CONFIG.flux.batch_size
            img_ctx: List[Image.Image] = [img] * batch_size
            prompt_batch = [prompt] * batch_size
            seed = (
                CONFIG.flux.seed
                if CONFIG.flux.seed is not None
                else random.randrange(2**31)
            )

            t1 = time.perf_counter()
            print(f"Configuration took {t1 - t0:.4f}s")

            with torch.no_grad():
                ref_tokens, ref_ids = encode_image_refs(self.ae, img_ctx)

                if ref_tokens is not None and ref_ids is not None:
                    if ref_tokens.shape[0] != batch_size:
                        ref_tokens = ref_tokens.expand(batch_size, -1, -1).contiguous()
                        ref_ids = ref_ids.expand(batch_size, -1, -1).contiguous()

                ctx = self.text_encoder(prompt_batch).to(torch.bfloat16)
                ctx, ctx_ids = batched_prc_txt(ctx)

                shape = (
                    batch_size,
                    128,
                    CONFIG.flux.height // 16,
                    CONFIG.flux.width // 16,
                )
                generator = torch.Generator(device="cuda").manual_seed(seed)
                randn = torch.randn(
                    shape,
                    generator=generator,
                    dtype=torch.bfloat16,
                    device="cuda",
                )

                x, x_ids = batched_prc_img(randn)
                timesteps = get_schedule(self.num_steps, x.shape[1])

                t2 = time.perf_counter()
                print(f"Pre-denoising processing took {t2 - t1:.4f}s")

                x = denoise(
                    self.loaded_model,
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=self.guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )

                t3 = time.perf_counter()
                print(f"Denoising took {t3 - t2:.4f}s")

                x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
                x = self.ae.decode(x).float()
                x = x.clamp(-1, 1)
                x = rearrange(x[0], "c h w -> h w c")

                img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
                image_b64 = pil_image_to_base64(img)

                t4 = time.perf_counter()
                print(f"Total generation took {t4 - t0:.4f}s")

                return image_b64
