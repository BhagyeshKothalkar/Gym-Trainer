from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PoseFrame:
    frame_id: int
    exists: bool
    keypoints: List[List[float]]
    features: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoseResult:
    frames: List[PoseFrame]


@dataclass
class FluxRequest:
    prompt: str
    cond_image_b64: str


@dataclass
class FluxResult:
    image_b64: str
    seed: Optional[int] = None
