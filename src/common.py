import modal

cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.10"
).entrypoint([])

flux_image = (
    cuda_dev_image.apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .uv_pip_install(
        "torch==2.10.0",
        "numpy==2.2.2",
        "protobuf==5.29.3",
        "transformers==5.0.0",
        "accelerate==1.12.0",
        "safetensors==0.7.0",
        "sentencepiece==0.2.1",
        "huggingface-hub==1.3.7",
        "invisible-watermark==0.2.0",
        "diffusers==0.36.0",
        "https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu12.8torch2.10-cp310-cp310-linux_x86_64.whl",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_CACHE": "/cache"})
)
flux_image = (
    flux_image.uv_pip_install(
        "opencv-python==4.13.0.92",
        "pillow==12.1.0",
        "python-dotenv==1.2.1",
        "pydantic==2.12.5",
        "fastapi[standard]==0.128.4",
        "langchain==1.2.9",
        "langchain-groq==1.1.2",
    )
    .apt_install(
        "libatlas-base-dev",
    )
    .uv_pip_install(
        "pybind11==3.0.1", "cython==3.2.4", "pythran==0.18.1", "scipy==1.15.3"
    )
)
