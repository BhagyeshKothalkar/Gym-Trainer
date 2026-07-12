import modal

from config import CONFIG


app = modal.App("gym-trainer-modal-backend")

ckpts_vol = modal.Volume.from_name(
    CONFIG.volumes.flux_checkpoints_name,
    create_if_missing=True,
)
inductor_vol = modal.Volume.from_name(
    CONFIG.volumes.flux_artifacts_name,
    create_if_missing=True,
)
inductor_cache_vol = modal.Volume.from_name(
    CONFIG.volumes.inductor_cache_name,
    create_if_missing=True,
)
nv_cache_vol = modal.Volume.from_name(
    CONFIG.volumes.nv_cache_name,
    create_if_missing=True,
)
triton_cache_vol = modal.Volume.from_name(
    CONFIG.volumes.triton_cache_name,
    create_if_missing=True,
)
pose_models_vol = modal.Volume.from_name(
    CONFIG.volumes.pose_models_name,
    create_if_missing=True,
)

analysis_queue = modal.Queue.from_name(
    CONFIG.queues.analysis_queue_name,
    create_if_missing=True,
)
analysis_events = modal.Queue.from_name(
    CONFIG.queues.analysis_events_name,
    create_if_missing=True,
)
analysis_jobs = modal.Dict.from_name(
    CONFIG.queues.analysis_jobs_name,
    create_if_missing=True,
)
analysis_results = modal.Dict.from_name(
    CONFIG.queues.analysis_results_name,
    create_if_missing=True,
)

huggingface_secret = modal.Secret.from_name("huggingface-secret")

flux_image = (
    modal.Image.from_registry("pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel")
    .apt_install("git", "curl")
    .uv_pip_install("git+https://github.com/BhagyeshKothalkar/flux2")
    .env(
        {
            "HF_HUB_CACHE": CONFIG.volumes.flux_checkpoints_path,
            "TORCHINDUCTOR_CACHE_DIR": "/root/.inductor-cache",
            "TRITON_CACHE_DIR": "/root/.triton",
            "CUDA_CACHE_PATH": "/root/.nv",
        }
    )
    .uv_pip_install("kernels<0.13.0", "torchao")
)

pose_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .uv_pip_install(
        "opencv-python-headless",
        "numpy",
        "pillow",
        "torch",
        "transformers",
    )
    .env({"HF_HUB_CACHE": CONFIG.volumes.pose_models_path})
)

fastapi_image = modal.Image.debian_slim().uv_pip_install(
    "fastapi",
    "python-multipart",
    "numpy",
    "scipy",
    "opencv-python-headless",
    "langchain-groq",
    "langchain-core",
    "langgraph",
    "langchain-exa",
    "langchain-openai",
    "psycopg[binary]",
    "pgvector",
    "cloudinary",
    "dtw",
)

flux_volumes = {
    CONFIG.volumes.flux_checkpoints_path: ckpts_vol,
    CONFIG.volumes.flux_artifacts_path: inductor_vol,
    "/root/.nv": nv_cache_vol,
    "/root/.triton": triton_cache_vol,
    "/root/.inductor-cache": inductor_cache_vol,
}

pose_volumes = {
    CONFIG.volumes.pose_models_path: pose_models_vol,
}
