from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import zscore

from config import CONFIG


JOINT_NAMES = [
    "right_elbow",
    "left_elbow",
    "right_shoulder",
    "left_shoulder",
    "right_hip",
    "left_hip",
    "right_knee",
    "left_knee",
]


def detected_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [frame for frame in frames if frame["exists"]]


def normalized_angle_matrix(frames: List[Dict[str, Any]]) -> np.ndarray:
    values = np.array(
        [frame["features"] for frame in detected_frames(frames)], dtype=float
    )
    return np.nan_to_num(zscore(values, axis=0))


def raw_angle_matrix(frames: List[Dict[str, Any]]) -> np.ndarray:
    return np.array(
        [frame["features"] for frame in detected_frames(frames)], dtype=float
    )


def phase_for_index(index: int, total: int) -> str:
    if total <= 1:
        return "single_frame"
    ratio = index / max(total - 1, 1)
    if ratio < 0.25:
        return "setup"
    if ratio < 0.5:
        return "eccentric"
    if ratio < 0.75:
        return "bottom_or_transition"
    return "concentric"


def dtw_alignment(
    user_features: np.ndarray, reference_features: np.ndarray
) -> Tuple[List[int], List[int]]:
    from dtw import dtw

    alignment = dtw(
        user_features, reference_features, dist_method="cityblock", keep_internals=True
    )
    return list(alignment.index1), list(alignment.index2)


def affected_joints(user_vector: np.ndarray, reference_vector: np.ndarray) -> List[str]:
    deltas = np.abs(user_vector - reference_vector)
    if deltas.size == 0:
        return []
    threshold = max(float(np.mean(deltas) + np.std(deltas)), 0.1)
    return [JOINT_NAMES[idx] for idx, delta in enumerate(deltas) if delta >= threshold]


def merge_error_regions(
    distance_curve: List[float],
    alignment: List[Dict[str, Any]],
    user_features: np.ndarray,
    reference_features: np.ndarray,
) -> List[Dict[str, Any]]:
    threshold = CONFIG.analysis.dtw_distance_threshold
    regions = []
    current = []

    for idx, distance in enumerate(distance_curve):
        if distance > threshold:
            current.append(idx)
        elif current:
            regions.append(current)
            current = []

    if current:
        regions.append(current)

    merged = []
    for region in regions:
        if not merged:
            merged.append(region)
            continue
        if region[0] - merged[-1][-1] <= CONFIG.analysis.frame_cluster_gap:
            merged[-1].extend(region)
        else:
            merged.append(region)

    output = []
    for region_index, region in enumerate(merged):
        local_distances = [distance_curve[idx] for idx in region]
        peak_offset = int(np.argmax(local_distances))
        peak_idx = region[peak_offset]
        pair = alignment[peak_idx]
        user_idx = pair["user_index"]
        reference_idx = pair["reference_index"]
        output.append(
            {
                "region_index": region_index,
                "start": int(region[0]),
                "end": int(region[-1]),
                "peak": int(peak_idx),
                "peak_distance": float(distance_curve[peak_idx]),
                "mean_distance": float(np.mean(local_distances)),
                "affected_joints": affected_joints(
                    user_features[user_idx], reference_features[reference_idx]
                ),
                "movement_phase": phase_for_index(user_idx, len(user_features)),
            }
        )

    output.sort(key=lambda item: item["peak_distance"], reverse=True)
    return output


def select_critical_frames(
    error_regions: List[Dict[str, Any]], alignment: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    selected = []
    for region in error_regions[: CONFIG.analysis.max_error_frames]:
        pair = alignment[region["peak"]]
        selected.append(
            {
                "region_index": region["region_index"],
                "severity": region["peak_distance"],
                "phase": region["movement_phase"],
                "distance": region["peak_distance"],
                "user_frame_index": pair["user_index"],
                "reference_frame_index": pair["reference_index"],
                "user_frame_id": pair["user_frame_id"],
                "reference_frame_id": pair["reference_frame_id"],
                "affected_joints": region["affected_joints"],
            }
        )
    return selected


def movement_score(distance_curve: List[float]) -> float:
    if not distance_curve:
        return 100.0
    mean_distance = float(np.mean(distance_curve))
    score = 100.0 - (mean_distance * 12.5)
    return max(CONFIG.analysis.movement_score_floor, min(100.0, score))


def build_visualization_data(
    user_raw: np.ndarray,
    reference_raw: np.ndarray,
    user_normalized: np.ndarray,
    reference_normalized: np.ndarray,
    alignment: List[Dict[str, Any]],
    distance_curve: List[float],
) -> Dict[str, Any]:
    joint_angle_curves = {}
    per_joint_deviation = {}
    range_of_motion = {}
    symmetry = {}

    for idx, joint in enumerate(JOINT_NAMES):
        user_curve = user_raw[:, idx].tolist()
        reference_curve = reference_raw[:, idx].tolist()
        joint_angle_curves[joint] = {
            "user": user_curve,
            "reference": reference_curve,
        }
        aligned_deltas = [
            abs(
                float(
                    user_normalized[pair["user_index"], idx]
                    - reference_normalized[pair["reference_index"], idx]
                )
            )
            for pair in alignment
        ]
        per_joint_deviation[joint] = {
            "mean": float(np.mean(aligned_deltas)) if aligned_deltas else 0.0,
            "max": float(np.max(aligned_deltas)) if aligned_deltas else 0.0,
        }
        range_of_motion[joint] = {
            "user": float(np.max(user_raw[:, idx]) - np.min(user_raw[:, idx])),
            "reference": float(
                np.max(reference_raw[:, idx]) - np.min(reference_raw[:, idx])
            ),
        }

    symmetry["elbow_balance"] = float(
        abs(
            per_joint_deviation["left_elbow"]["mean"]
            - per_joint_deviation["right_elbow"]["mean"]
        )
    )
    symmetry["shoulder_balance"] = float(
        abs(
            per_joint_deviation["left_shoulder"]["mean"]
            - per_joint_deviation["right_shoulder"]["mean"]
        )
    )
    symmetry["hip_balance"] = float(
        abs(
            per_joint_deviation["left_hip"]["mean"]
            - per_joint_deviation["right_hip"]["mean"]
        )
    )
    symmetry["knee_balance"] = float(
        abs(
            per_joint_deviation["left_knee"]["mean"]
            - per_joint_deviation["right_knee"]["mean"]
        )
    )

    return {
        "dtw_distance_curve": distance_curve,
        "joint_angle_curves": joint_angle_curves,
        "range_of_motion": range_of_motion,
        "symmetry": symmetry,
        "phase_timeline": [
            phase_for_index(pair["user_index"], len(user_raw)) for pair in alignment
        ],
        "movement_score": movement_score(distance_curve),
        "per_joint_deviation": per_joint_deviation,
        "stability": {
            "user_angle_variance": np.var(user_raw, axis=0).tolist(),
            "reference_angle_variance": np.var(reference_raw, axis=0).tolist(),
        },
        "error_heatmap": [
            {
                "alignment_index": idx,
                "distance": distance,
                "phase": phase_for_index(alignment[idx]["user_index"], len(user_raw)),
            }
            for idx, distance in enumerate(distance_curve)
        ],
    }


def compute_movement_analysis(
    user_frames: List[Dict[str, Any]], reference_frames: List[Dict[str, Any]]
) -> Dict[str, Any]:
    user_detected = detected_frames(user_frames)
    reference_detected = detected_frames(reference_frames)
    if not user_detected or not reference_detected:
        raise ValueError("No person detected in one of the videos.")

    user_raw = raw_angle_matrix(user_frames)
    reference_raw = raw_angle_matrix(reference_frames)
    user_normalized = normalized_angle_matrix(user_frames)
    reference_normalized = normalized_angle_matrix(reference_frames)
    user_indices, reference_indices = dtw_alignment(
        user_normalized, reference_normalized
    )

    alignment = []
    distance_curve = []
    for user_idx, reference_idx in zip(user_indices, reference_indices):
        distance = float(
            cdist(
                [user_normalized[user_idx]],
                [reference_normalized[reference_idx]],
                metric="cityblock",
            )[0][0]
        )
        distance_curve.append(distance)
        alignment.append(
            {
                "alignment_index": len(alignment),
                "user_index": int(user_idx),
                "reference_index": int(reference_idx),
                "user_frame_id": int(user_detected[user_idx]["frame_id"]),
                "reference_frame_id": int(
                    reference_detected[reference_idx]["frame_id"]
                ),
                "distance": distance,
            }
        )

    error_regions = merge_error_regions(
        distance_curve, alignment, user_normalized, reference_normalized
    )
    critical_frames = select_critical_frames(error_regions, alignment)
    visualization_data = build_visualization_data(
        user_raw,
        reference_raw,
        user_normalized,
        reference_normalized,
        alignment,
        distance_curve,
    )
    total_cost = float(np.sum(distance_curve))
    score = movement_score(distance_curve)

    joint_metrics = visualization_data["per_joint_deviation"]
    per_frame_metrics = [
        {
            "alignment_index": item["alignment_index"],
            "user_frame_id": item["user_frame_id"],
            "reference_frame_id": item["reference_frame_id"],
            "distance": item["distance"],
            "phase": phase_for_index(item["user_index"], len(user_raw)),
        }
        for item in alignment
    ]

    return {
        "summary": {
            "movement_score": score,
            "dtw_cost": total_cost,
            "mean_distance": float(np.mean(distance_curve)) if distance_curve else 0.0,
            "max_distance": float(np.max(distance_curve)) if distance_curve else 0.0,
            "alignment_length": len(alignment),
            "error_region_count": len(error_regions),
        },
        "alignment": alignment,
        "per_frame_metrics": per_frame_metrics,
        "joint_metrics": joint_metrics,
        "error_regions": error_regions,
        "critical_frames": critical_frames,
        "visualization_data": visualization_data,
    }
