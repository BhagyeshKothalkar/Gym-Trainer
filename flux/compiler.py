import os

import modal
import torch

from config import CONFIG
from modal_backend import (
    app,
    ckpts_vol,
    flux_image,
    flux_volumes,
    huggingface_secret,
    inductor_cache_vol,
    inductor_vol,
    nv_cache_vol,
    triton_cache_vol,
)


def configure_inductor():
    import multiprocessing

    import torch._inductor.config as inductor_config

    inductor_config.compile_threads = multiprocessing.cpu_count()
    inductor_config.fx_graph_cache = True
    inductor_config.autotune_local_cache = True
    inductor_config.disable_progress = False
    inductor_config.max_autotune = True
    inductor_config.freezing = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.layout_optimization = True
    inductor_config.triton.cudagraphs = True
    inductor_config.triton.cudagraph_trees = False
    inductor_config.aot_inductor.compile_wrapper_opt_level = CONFIG.compilation.suffix
    inductor_config.cuda.enable_cuda_lto = True
    inductor_config.aot_inductor.emit_multi_arch_kernel = False
    inductor_config.coordinate_descent_check_all_directions = True
    inductor_config.epilogue_fusion = True
    inductor_config.triton.multi_kernel = 0
    inductor_config.triton.store_cubin = True
    inductor_config.aot_inductor.package = True

    os.environ["TORCH_INDUCTOR_CPP_VEC_ISA"] = "avx2"
    inductor_config.cpp.vec_isa_ok = False


def flux_quantization_filter(module: torch.nn.Module, fqn: str) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False

    parts = fqn.split(".")

    if len(parts) == 4 and parts[0] == "double_blocks":
        _, _, block_type, layer_name = parts
        if block_type in ("img_attn", "txt_attn"):
            return layer_name in ("proj", "qkv")
        if block_type in ("img_mlp", "txt_mlp"):
            return layer_name in ("0", "2")

    if len(parts) == 3 and parts[0] == "single_blocks":
        _, _, layer_name = parts
        return layer_name in ("linear1", "linear2")

    return False


@app.cls(
    image=flux_image,
    gpu=CONFIG.hardware.flux_gpu,
    secrets=[huggingface_secret],
    volumes=flux_volumes,
    timeout=CONFIG.compilation.timeout,
)
class FluxCompiler:
    @modal.enter()
    def enter(self):
        from flux2.util import load_flow_model

        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not found")

        self.model = load_flow_model(CONFIG.flux.model_name, device=self.device)
        ckpts_vol.commit()
        self.model.eval()

        batch_size = CONFIG.flux.batch_size
        x = torch.rand(
            (batch_size, 4096, 128), device=self.device, dtype=torch.bfloat16
        )
        x_ids = torch.rand(
            (batch_size, 4096, 4), device=self.device, dtype=torch.bfloat16
        )
        ctx = torch.rand(
            (batch_size, 512, 7680), device=self.device, dtype=torch.bfloat16
        )
        ctx_ids = torch.rand(
            (batch_size, 512, 4), device=self.device, dtype=torch.bfloat16
        )
        timesteps = torch.rand((batch_size,), device=self.device, dtype=torch.bfloat16)
        guidance = torch.full(
            (batch_size,), 1.0, device=self.device, dtype=torch.bfloat16
        )
        ref_tokens = torch.rand(
            (batch_size, 4096, 128), device=self.device, dtype=torch.bfloat16
        )
        ref_ids = torch.rand(
            (batch_size, 4096, 4), device=self.device, dtype=torch.bfloat16
        )

        self.dummy_args = (
            torch.cat((x, ref_tokens), dim=1),
            torch.cat((x_ids, ref_ids), dim=1),
            timesteps,
            ctx,
            ctx_ids,
            guidance,
        )
        self.package_path = CONFIG.flux.package_path(
            CONFIG.volumes,
            CONFIG.hardware,
            CONFIG.compilation,
        )
        os.makedirs(os.path.dirname(self.package_path), exist_ok=True)
        configure_inductor()

    @modal.method()
    def compile(self):
        from torch.export import export
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            PerTensor,
            quantize_,
        )

        print("Starting Flux compilation")
        with torch.no_grad():
            print("Quantizing...")
            quantize_(
                self.model,
                Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                filter_fn=flux_quantization_filter,
            )

            print("Exporting...")
            exported_program = export(self.model, self.dummy_args, strict=False)

            print("AOT compiling to package...")
            output_path = torch._inductor.aoti_compile_and_package(
                exported_program,
                package_path=self.package_path,
            )

            print("Committing volumes...")
            inductor_vol.commit()
            nv_cache_vol.commit()
            triton_cache_vol.commit()
            inductor_cache_vol.commit()

            print(f"Compilation finished. Saved to: {output_path}")
            return output_path
