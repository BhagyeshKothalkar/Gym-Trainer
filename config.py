import os
from dataclasses import dataclass, field
from typing import Optional


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


@dataclass
class VolumeConfig:
    flux_checkpoints_name: str = "flux2_ckpts"
    flux_artifacts_name: str = "inductor_aot_models"
    inductor_cache_name: str = "inductor-cache"
    nv_cache_name: str = "nv-cache"
    triton_cache_name: str = "triton-cache"
    pose_models_name: str = "pose-models"
    flux_checkpoints_path: str = "/checkpoints"
    flux_artifacts_path: str = "/artifacts"
    pose_models_path: str = "/pose-models"


@dataclass
class HardwareConfig:
    flux_gpu: str = "L40S"
    pose_gpu: str = "L40S"


@dataclass
class CompilationConfig:
    suffix: str = "O3"
    timeout: int = 3000


@dataclass
class ScalingConfig:
    min_containers: int = 0
    max_containers: int = 1
    idle_timeout: int = 300


@dataclass
class QueueConfig:
    analysis_queue_name: str = "gym-trainer-analysis-queue"
    analysis_events_name: str = "gym-trainer-analysis-events"
    analysis_jobs_name: str = "gym-trainer-analysis-jobs"
    analysis_results_name: str = "gym-trainer-analysis-results"


@dataclass
class AnalysisConfig:
    dtw_distance_threshold: float = 1.0
    max_error_frames: int = 3
    frame_cluster_gap: int = 10
    peak_merge_window: int = 6
    movement_score_floor: float = 0.0


@dataclass
class DatabaseConfig:
    url: str = os.environ.get("DATABASE_URL", "")
    embedding_dimension: int = int(
        os.environ.get("FEEDBACK_EMBEDDING_DIMENSION", "1536")
    )


@dataclass
class CloudinaryConfig:
    cloud_name: str = os.environ.get("CLOUDINARY_CLOUD_NAME", "")
    api_key: str = os.environ.get("CLOUDINARY_API_KEY", "")
    api_secret: str = os.environ.get("CLOUDINARY_API_SECRET", "")
    folder: str = os.environ.get("CLOUDINARY_FOLDER", "gym-trainer")


@dataclass
class ToolBudgetConfig:
    cloudinary_uploads: int = 8
    database_reads: int = 6
    database_writes: int = 24
    exa_searches: int = 3
    vlm_calls: int = 4
    llm_calls: int = 5
    embedding_calls: int = 2


@dataclass
class AgentConfig:
    feedback_model: str = os.environ.get("FEEDBACK_MODEL", "llama-3.3-70b-versatile")
    vision_model: str = os.environ.get(
        "VISION_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"
    )
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    exa_api_key: str = os.environ.get("EXA_API_KEY", "")
    tool_budgets: ToolBudgetConfig = field(default_factory=ToolBudgetConfig)


@dataclass
class SnapshotConfig:
    enabled: bool = True


@dataclass
class FluxConfig:
    model_name: str = "flux.2-klein-4b"
    safe_model_name: str = "flux_2_klein_4b"
    prompt: str = "A high-quality image"
    seed: Optional[int] = None
    width: int = 1024
    height: int = 1024
    num_steps: int = 4
    guidance: float = 2.8
    batch_size: int = 1

    def package_path(
        self,
        volumes: VolumeConfig,
        hardware: HardwareConfig,
        compilation: CompilationConfig,
    ) -> str:
        return os.path.join(
            volumes.flux_artifacts_path,
            self.safe_model_name,
            f"{self.safe_model_name}_{hardware.flux_gpu}_{compilation.suffix}.pt2",
        )


@dataclass
class PoseConfig:
    detector_model: str = "PekingU/rtdetr_r50vd_coco_o365"
    pose_model: str = "usyd-community/vitpose-plus-small"
    target_frames: int = 100
    batch_size: int = 64
    detection_threshold: float = 0.3
    resize_width: int = 640
    resize_height: int = 480


@dataclass
class AppConfig:
    volumes: VolumeConfig = field(default_factory=VolumeConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    compilation: CompilationConfig = field(default_factory=CompilationConfig)
    queues: QueueConfig = field(default_factory=QueueConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cloudinary: CloudinaryConfig = field(default_factory=CloudinaryConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    fastapi_scaling: ScalingConfig = field(default_factory=ScalingConfig)
    flux_scaling: ScalingConfig = field(default_factory=ScalingConfig)
    pose_scaling: ScalingConfig = field(default_factory=ScalingConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    flux: FluxConfig = field(default_factory=FluxConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)


CONFIG = AppConfig()
