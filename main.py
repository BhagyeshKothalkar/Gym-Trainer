import base64
import time
import traceback
import uuid
from typing import Any, Dict, List
from urllib.request import Request, urlopen

import cv2
import modal
import numpy as np

from agents import embed_feedback, run_feedback_agent, run_prompt_generation_agent
from analysis import compute_movement_analysis
from cloudinary_client import CloudinaryClient
from config import CONFIG
from database import Database
from flux.inference import FluxService
from modal_backend import (
    analysis_events,
    analysis_jobs,
    analysis_queue,
    analysis_results,
    app,
    fastapi_image,
)
from pose.inference import PoseService


def download_url_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Gym-Trainer/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def pose_result_to_frames(pose_result) -> List[Dict[str, Any]]:
    frames = (
        pose_result.frames if hasattr(pose_result, "frames") else pose_result["frames"]
    )
    normalized = []
    for frame in frames:
        if hasattr(frame, "exists"):
            normalized.append(
                {
                    "frame_id": frame.frame_id,
                    "exists": frame.exists,
                    "keypoints": frame.keypoints,
                    "features": frame.features,
                    "metadata": frame.metadata,
                }
            )
        else:
            normalized.append(
                {
                    "frame_id": frame["frame_id"],
                    "exists": frame["exists"],
                    "keypoints": frame["keypoints"],
                    "features": frame["features"],
                    "metadata": frame.get("metadata", {}),
                }
            )
    return normalized


def sampled_frame_bytes(video_bytes: bytes, target_frames: int, frame_id: int) -> bytes:
    temp_name = f"/tmp/{uuid.uuid4()}.mp4"
    with open(temp_name, "wb") as temp_file:
        temp_file.write(video_bytes)

    try:
        cap = cv2.VideoCapture(temp_name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[frame_id]))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not extract sampled frame {frame_id}.")
        frame = cv2.resize(frame, (CONFIG.pose.resize_width, CONFIG.pose.resize_height))
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError(f"Could not encode sampled frame {frame_id}.")
        return buffer.tobytes()
    finally:
        import os

        if os.path.exists(temp_name):
            os.remove(temp_name)


def emit_job_event(
    job_id: str, status: str, detail: Dict[str, Any] | None = None
) -> None:
    event = {
        "job_id": job_id,
        "status": status,
        "timestamp": time.time(),
        "detail": detail or {},
    }
    analysis_events.put(event)


def create_storage_hierarchy(
    db: Database, payload: Dict[str, Any], movement_analysis: Dict[str, Any]
) -> Dict[str, str]:
    user_id = db.upsert_user(
        external_id=payload.get("user_external_id")
        or payload.get("email")
        or "anonymous",
        email=payload.get("email", ""),
        metadata={"source": "analyze_movement"},
    )
    exercise_id = db.upsert_exercise(
        user_id=user_id, name=payload.get("exercise_name", "Exercise")
    )
    session_id = db.create_session(
        exercise_id=exercise_id,
        user_video_url=payload["user_video_url"],
        reference_video_url=payload["reference_video_url"],
        metadata={"job_id": payload["job_id"]},
    )
    summary = f"{payload.get('exercise_name', 'Exercise')} scored {movement_analysis['summary']['movement_score']:.1f}."
    movement_id = db.create_movement(
        session_id=session_id,
        summary=summary,
        movement_score=movement_analysis["summary"]["movement_score"],
        dtw_cost=movement_analysis["summary"]["dtw_cost"],
        aggregate_metrics=movement_analysis["visualization_data"],
    )
    dtw_analysis_id = db.create_dtw_analysis(
        movement_id=movement_id, analysis=movement_analysis
    )
    return {
        "user_id": user_id,
        "exercise_id": exercise_id,
        "session_id": session_id,
        "movement_id": movement_id,
        "dtw_analysis_id": dtw_analysis_id,
    }


def upload_critical_frames(
    cloudinary_client: CloudinaryClient,
    critical_frames: List[Dict[str, Any]],
    user_video_bytes: bytes,
    reference_video_bytes: bytes,
    ids: Dict[str, str],
) -> List[Dict[str, Any]]:
    uploaded = []
    for frame in critical_frames:
        base_metadata = {
            "feedback_id": "",
            "critical_frame_id": "",
            "movement_id": ids["movement_id"],
            "exercise_id": ids["exercise_id"],
            "session_id": ids["session_id"],
        }
        user_bytes = sampled_frame_bytes(
            user_video_bytes, CONFIG.pose.target_frames, frame["user_frame_id"]
        )
        reference_bytes = sampled_frame_bytes(
            reference_video_bytes,
            CONFIG.pose.target_frames,
            frame["reference_frame_id"],
        )
        user_upload = cloudinary_client.upload_image_bytes(
            user_bytes,
            f"{ids['session_id']}/critical_{frame['region_index']}_user",
            base_metadata,
        )
        reference_upload = cloudinary_client.upload_image_bytes(
            reference_bytes,
            f"{ids['session_id']}/critical_{frame['region_index']}_reference",
            base_metadata,
        )
        uploaded.append(
            {
                **frame,
                "user_image_url": user_upload["url"],
                "reference_image_url": reference_upload["url"],
                "metadata": {
                    "affected_joints": frame["affected_joints"],
                    "user_frame_id": frame["user_frame_id"],
                    "reference_frame_id": frame["reference_frame_id"],
                },
            }
        )
    return uploaded


def analyze_and_store_feedback(
    db: Database,
    movement_analysis: Dict[str, Any],
    frame: Dict[str, Any],
    payload: Dict[str, Any],
    ids: Dict[str, str],
) -> Dict[str, Any]:
    critical_frame_id = db.create_critical_frame(ids["dtw_analysis_id"], frame)
    feedback = run_feedback_agent(
        {
            "exercise_name": payload.get("exercise_name", "Exercise"),
            "movement_analysis": movement_analysis,
            "critical_frame": frame,
            "user_image_url": frame["user_image_url"],
            "reference_image_url": frame["reference_image_url"],
        }
    )
    embedding = embed_feedback(feedback)
    feedback_id = db.create_feedback(critical_frame_id, feedback, embedding)
    return {
        "critical_frame_id": critical_frame_id,
        "feedback_id": feedback_id,
        "feedback": feedback,
        "frame": frame,
    }


@app.function(
    image=fastapi_image,
    min_containers=CONFIG.fastapi_scaling.min_containers,
    max_containers=CONFIG.fastapi_scaling.max_containers,
    scaledown_window=CONFIG.fastapi_scaling.idle_timeout,
)
def run_generation_job(job_id: str, feedback_id: str) -> Dict[str, Any]:
    db = Database()
    cloudinary_client = CloudinaryClient()
    try:
        feedback_record = db.get_feedback_for_generation(feedback_id)
        if not feedback_record:
            raise ValueError(f"Feedback {feedback_id} not found")

        prompt_result = run_prompt_generation_agent(feedback_id)
        prompt = prompt_result["prompt"]

        image_bytes = download_url_bytes(feedback_record["user_image_url"])
        cond_image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        generated_b64 = FluxService().infer.remote(prompt, cond_image_b64)
        generated_bytes = base64.b64decode(generated_b64)

        upload = cloudinary_client.upload_image_bytes(
            generated_bytes,
            f"generated_{feedback_id}",
            {"feedback_id": feedback_id},
        )

        db.update_generated_image(
            job_id,
            upload["url"],
            prompt,
            {"cloudinary_public_id": upload["public_id"]},
        )

        return {
            "generated_image_id": job_id,
            "generated_image_url": upload["url"],
        }
    except Exception:
        db.fail_generation_job(job_id)
        raise


@app.function(
    image=fastapi_image,
    min_containers=CONFIG.fastapi_scaling.min_containers,
    max_containers=CONFIG.fastapi_scaling.max_containers,
    scaledown_window=CONFIG.fastapi_scaling.idle_timeout,
)
def run_analysis_job(job_id: str) -> Dict[str, Any]:
    payload = analysis_jobs[job_id]["payload"]
    analysis_jobs[job_id] = {
        **analysis_jobs[job_id],
        "status": "running",
        "started_at": time.time(),
    }
    emit_job_event(job_id, "running")

    try:
        db = Database()
        db.initialize_schema()
        cloudinary_client = CloudinaryClient()

        reference_video_bytes = download_url_bytes(payload["reference_video_url"])
        user_video_bytes = download_url_bytes(payload["user_video_url"])
        emit_job_event(job_id, "videos_downloaded")

        pose_service = PoseService()
        reference_pose_call = pose_service.infer.spawn(
            reference_video_bytes, CONFIG.pose.target_frames
        )
        user_pose_call = pose_service.infer.spawn(
            user_video_bytes, CONFIG.pose.target_frames
        )

        reference_frames = pose_result_to_frames(reference_pose_call.get())
        user_frames = pose_result_to_frames(user_pose_call.get())
        emit_job_event(job_id, "pose_complete")

        movement_analysis = compute_movement_analysis(user_frames, reference_frames)
        emit_job_event(
            job_id,
            "movement_analysis_complete",
            {
                "movement_score": movement_analysis["summary"]["movement_score"],
                "critical_frame_count": len(movement_analysis["critical_frames"]),
            },
        )

        ids = create_storage_hierarchy(db, payload, movement_analysis)
        critical_frames = upload_critical_frames(
            cloudinary_client,
            movement_analysis["critical_frames"],
            user_video_bytes,
            reference_video_bytes,
            ids,
        )
        movement_analysis["critical_frames"] = critical_frames
        emit_job_event(job_id, "critical_frames_uploaded")

        feedback_items = []
        for frame in critical_frames:
            feedback_payload = analyze_and_store_feedback(
                db, movement_analysis, frame, payload, ids
            )
            feedback_items.append(feedback_payload)

        db.complete_session(ids["session_id"])
        result = {
            "job_id": job_id,
            "status": "complete",
            "movement_id": ids["movement_id"],
            "session_id": ids["session_id"],
            "exercise_id": ids["exercise_id"],
            "movement_summary": movement_analysis["summary"],
            "visualization_metrics": movement_analysis["visualization_data"],
            "dtw": movement_analysis,
            "critical_frames": critical_frames,
            "feedback": [
                {
                    "feedback_id": item["feedback_id"],
                    "critical_frame_id": item["critical_frame_id"],
                    "feedback": item["feedback"],
                }
                for item in feedback_items
            ],
        }

        analysis_results[job_id] = result
        analysis_jobs[job_id] = {
            **analysis_jobs[job_id],
            "status": "complete",
            "completed_at": time.time(),
        }
        emit_job_event(job_id, "complete")
        return result

    except Exception as exc:
        error = {
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        analysis_jobs[job_id] = {
            **analysis_jobs[job_id],
            "status": "failed",
            "failed_at": time.time(),
            "error": error,
        }
        emit_job_event(job_id, "failed", {"message": str(exc)})
        raise


@app.function(
    image=fastapi_image,
    min_containers=CONFIG.fastapi_scaling.min_containers,
    max_containers=CONFIG.fastapi_scaling.max_containers,
    scaledown_window=CONFIG.fastapi_scaling.idle_timeout,
)
@modal.fastapi_endpoint(method="POST")
async def analyze_movement(
    user_video_url: str,
    reference_video_url: str,
    exercise_name: str = "Exercise",
    user_external_id: str = "anonymous",
    email: str = "",
) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "user_video_url": user_video_url,
        "reference_video_url": reference_video_url,
        "exercise_name": exercise_name,
        "user_external_id": user_external_id,
        "email": email,
    }
    analysis_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "payload": payload,
    }
    analysis_queue.put({"job_id": job_id, "payload": payload})
    run_analysis_job.spawn(job_id)
    emit_job_event(job_id, "queued")
    return {
        "job_id": job_id,
        "status": "queued",
    }


@app.function(image=fastapi_image)
@modal.fastapi_endpoint(method="GET")
async def analysis_status(job_id: str) -> Dict[str, Any]:
    try:
        job = analysis_jobs[job_id]
    except KeyError:
        return {
            "job_id": job_id,
            "status": "not_found",
        }

    response = {
        "job_id": job_id,
        "status": job["status"],
    }
    if job["status"] == "complete":
        try:
            response["result"] = analysis_results[job_id]
        except KeyError:
            pass
    if job["status"] == "failed":
        response["error"] = job.get("error", {})
    return response


@app.function(
    image=fastapi_image,
    min_containers=CONFIG.fastapi_scaling.min_containers,
    max_containers=CONFIG.fastapi_scaling.max_containers,
    scaledown_window=CONFIG.fastapi_scaling.idle_timeout,
)
@modal.fastapi_endpoint(method="POST")
async def generate_correction(payload: Dict[str, str]) -> Dict[str, Any]:
    feedback_id = payload.get("feedback_id")
    if not feedback_id:
        return {"error": "feedback_id is required"}

    db = Database()
    job_id = db.create_generation_job(feedback_id)
    run_generation_job.spawn(job_id, feedback_id)

    return {"generation_job_id": job_id, "status": "queued"}


@app.function(image=fastapi_image)
@modal.fastapi_endpoint(method="GET")
async def generation_status(job_id: str) -> Dict[str, Any]:
    db = Database()
    job = db.get_generation_job(job_id)
    if not job:
        return {"status": "not_found"}

    status = job["generation_status"]
    if status == "complete":
        return {
            "status": "complete",
            "generated_image_id": job_id,
            "generated_image_url": job["image_url"],
        }
    return {"status": status}
