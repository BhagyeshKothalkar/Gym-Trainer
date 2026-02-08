import os

import modal
from fastapi import File, UploadFile

from .common import flux_image

app = modal.App("example-flux", image=flux_image)


os.environ["HF_HOME"] = "/cache"

MINUTES = 60


@app.cls(
    gpu="A100",  # fast GPU with strong software support
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,  # leave plenty of time for compilation
    volumes={  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("groq-secret"),
    ],
)
class GymTrainer:
    @modal.enter()
    def enter(self):
        print("[GymTrainer][enter] initializing app components")

        from .alignment import Alignment
        from .image_gen_model import ImageGenPipeline
        from .process_video import VideoProcessor

        self.video_processor = VideoProcessor()
        self.video_processor.enter()
        self.image_gen_pipeline = ImageGenPipeline()
        self.image_gen_pipeline.enter()
        self.alignment = Alignment()
        self.alignment.enter()

    @modal.method()
    def analyze_movement(self, u_bytes, t_bytes, exercise_name):
        print(
            f"[GymTrainer][analyze_movement] processing request for exercise: {exercise_name}"
        )
        print("Processing User Video:")
        u_poses, u_angles, u_images = self.video_processor.process_video_path(u_bytes)
        print("Processing Trainer Video:")
        t_poses, t_angles, t_images = self.video_processor.process_video_path(t_bytes)

        critical_frames = self.alignment.find_critical_frames(u_angles, t_angles)
        results = self.alignment.feedback(
            critical_frames, exercise_name, u_images, t_images, u_poses, t_poses
        )
        u_idx, t_idx = critical_frames[0][1], critical_frames[0][2]
        flux_prompt = f"performing {exercise_name}"
        gdino_prompt = "Human, exercising equipment"

        image = self.image_gen_pipeline.generate_correct_image(
            t_images[t_idx],
            u_images[u_idx],
            t_poses[t_idx][0],
            u_poses[u_idx][0],
            flux_prompt,
            gdino_prompt,
        )

        return {"status": "success", "analysis": results}, image


@app.function(
    image=flux_image, volumes={"/cache": modal.Volume.from_name("hf-hub-cache")}
)
@modal.fastapi_endpoint(method="POST", docs=True)
async def analyze_movement(
    trainer_video: UploadFile = File(...),
    user_video: UploadFile = File(...),
    exercise_name: str = "Exercise",
):
    u_bytes = await user_video.read()
    t_bytes = await trainer_video.read()
    print(f"[][analyze_movement] received request: exercise_name={exercise_name}")

    analyzer = GymTrainer()
    result, image = analyzer.analyze_movement.remote(u_bytes, t_bytes, exercise_name)
    return result
