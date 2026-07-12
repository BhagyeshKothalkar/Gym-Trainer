import modal

from config import CONFIG
from flux.runtime import FluxRuntime
from modal_backend import app, flux_image, flux_volumes, huggingface_secret


@app.cls(
    image=flux_image,
    gpu=CONFIG.hardware.flux_gpu,
    secrets=[huggingface_secret],
    volumes=flux_volumes,
    min_containers=CONFIG.flux_scaling.min_containers,
    max_containers=CONFIG.flux_scaling.max_containers,
    scaledown_window=CONFIG.flux_scaling.idle_timeout,
)
@modal.enable_memory_snapshot()
class FluxService:
    @modal.enter()
    def enter(self):
        self.runtime = FluxRuntime()
        self.runtime.load()
        self.runtime.warmup()

    @modal.method()
    def infer(self, prompt: str, cond_image_b64: str):
        return self.runtime.generate(prompt, cond_image_b64)
