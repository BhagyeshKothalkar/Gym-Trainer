# Gym-Trainer Technical Documentation

This document outlines the technical architecture, data flows, API responses, database schema, CV pipeline, and agent toolchain for the Gym-Trainer backend system.

## 1. API Architecture & Frontend Responses

The system is orchestrated via FastAPI endpoints running on Modal. The architecture follows an asynchronous job pattern.

### `POST /analyze_movement`
Initiates a new analysis job for a user and reference video.
**Returns:**
```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

### `POST /generate_correction`
Initiates a new asynchronous image generation job based on existing feedback.
**Returns:**
```json
{
  "generation_job_id": "uuid",
  "status": "queued"
}
```

### `GET /generation_status?job_id=...`
Polls the current status of an image generation job.
**Returns (when `status` == "complete"):**
```json
{
  "status": "complete",
  "generated_image_id": "uuid",
  "generated_image_url": "https://res.cloudinary.com/..."
}
```

### `GET /analysis_status?job_id=...`
Polls the current status of an analysis job.
**Returns (when `status` == "complete"):**
```json
{
  "job_id": "uuid",
  "status": "complete",
  "result": {
    "movement_id": "uuid",
    "session_id": "uuid",
    "exercise_id": "uuid",
    "movement_summary": {
      "movement_score": 85.5,
      "dtw_cost": 142.3,
      "mean_distance": 1.2,
      "max_distance": 3.4,
      "alignment_length": 105,
      "error_region_count": 2
    },
    "visualization_metrics": {
      "dtw_distance_curve": [...],
      "joint_angle_curves": { "right_knee": { "user": [...], "reference": [...] }, ... },
      "range_of_motion": { "right_knee": { "user": 120.5, "reference": 125.0 } },
      "symmetry": { "knee_balance": 2.5, "hip_balance": 1.1, ... },
      "phase_timeline": ["setup", "eccentric", "bottom_or_transition", "concentric"],
      "movement_score": 85.5,
      "per_joint_deviation": { "right_knee": { "mean": 0.5, "max": 1.2 } },
      "stability": { "user_angle_variance": [...], "reference_angle_variance": [...] },
      "error_heatmap": [{ "alignment_index": 0, "distance": 1.2, "phase": "setup" }]
    },
    "dtw": {
      "alignment": [...],
      "per_frame_metrics": [...],
      "error_regions": [...]
    },
    "critical_frames": [
      {
        "region_index": 0,
        "severity": 3.4,
        "phase": "eccentric",
        "distance": 3.4,
        "user_frame_id": 42,
        "reference_frame_id": 45,
        "affected_joints": ["right_knee", "right_hip"],
        "user_image_url": "https://res.cloudinary.com/...",
        "reference_image_url": "https://res.cloudinary.com/..."
      }
    ],
    "feedback": [
      {
        "feedback_id": "uuid",
        "critical_frame_id": "uuid",
        "feedback": {
          "summary": "...",
          "technical_analysis": "...",
          "body_level_analysis": "...",
          "primary_issue": "Knee Valgus",
          "secondary_issues": ["Hip Shift"],
          "risk_level": "moderate",
          "research": {...},
          "model_version": "llama-3.3-70b-versatile"
        }
      }
    ]
  }
}
```
*Note: The frontend acts purely as a renderer. All metrics, symmetries, phases, and error heatmaps are precomputed by the backend.*

---

## 2. Database Storage Schema

Data is stored in **PostgreSQL** with the `pgvector` extension for semantic retrieval. The hierarchy is strictly relational.

### Table Hierarchy
1. **`users`**: Stores user identity and lightweight metadata.
2. **`exercises`**: Maps to users (e.g., "Squat", "Deadlift").
3. **`sessions`**: Represents a specific analysis run (contains source video URLs).
4. **`movements`**: Stores top-level `movement_score`, `dtw_cost`, and aggregated visualization arrays natively as JSONB.
5. **`dtw_analyses`**: Stores the complete serialized DTW path, distance curves, and raw visualization structs as JSONB.
6. **`critical_frames`**: Represents isolated error regions (peak distance, phase, severity, affected joints, Cloudinary URLs).
7. **`feedback`**: Contains the LLM's narrative response (`technical_analysis`, `body_level_analysis`, `primary_issue`). Includes a **vector embedding** representing the text for historical RAG retrieval.
8. **`generated_images`**: Stores references to the generated Flux overlay images, the prompt used (`generation_prompt`), job statuses (`generation_status`, `generation_started`, `generation_finished`), and the Cloudinary public IDs.

*Note: Cloudinary metadata contains only lightweight reference IDs (e.g., `feedback_id`, `movement_id`), not the actual text analysis.*

---

## 3. Computer Vision (CV) Pipeline

The CV pipeline processes videos synchronously before LLM analysis begins.

### Pose Inference (Modal)
- Detects the subject using **RT-DETR**.
- Extracts keypoints using **ViTPose+**.
- Outputs normalized coordinates, joint angles, and confidence matrices.

### Dynamic Time Warping (DTW) & Alignment
- **Normalization**: Applies Z-score normalization to frame feature vectors.
- **DTW**: Uses the Python `dtw` package with `cityblock` (L1) distance to temporally align user and reference angle matrices.
- **Phase Mapping**: Assigns functional phases (`setup`, `eccentric`, `bottom_or_transition`, `concentric`) to the user timeline.
- **Error Regions**: Threshold-based local maxima detection groups sequential errors into regions (no KMeans). Adjacent peaks within a strict temporal gap are merged. Ranks top 3 regions by peak severity.

### Flux Image Generation (Modal)
- Image generation is decoupled from the analysis pipeline and triggered via a separate `/generate_correction` endpoint.
- The Prompt Generation Agent retrieves the feedback and images from the database, runs VLM logic to build a prompt.
- The generated prompt and a base64-encoded comparison image are sent to a remote Modal instance.
- **Flux.2-klein-4b** executes inference with a tuned guidance scale (2.8) and 4 steps, preserving identity while drawing visual cues from the coaching feedback.

---

## 4. Agents and Tools

The analysis intelligence is implemented using **LangGraph**, relying on fixed, deterministic state graphs. All tools execute via **LangChain Structured Tools** with strict budget tracking (calls used, budget remaining, execution time, confidence).

### Feedback Agent (LangGraph)
**Graph Flow:**
`Movement Analysis` ➔ `Evidence Collector` ➔ `Movement Analyst` ➔ `Pattern Researcher` ➔ `Coach` ➔ `Verifier` ➔ `Structured Feedback`

- **Evidence Collector**: Accumulates raw DTW metrics, frame IDs, and historical context. No NLG occurs here.
- **Movement Analyst**: Concatenates user and reference critical frames in memory. Uses a Vision-Language Model (VLM) to correlate visible biomechanical deviations with the numeric DTW alerts.
- **Pattern Researcher**: Orchestrates RAG using the identified deviations (uses the Database Retrieval and Exa Search tools).
- **Coach**: Translates pure biomechanics and evidence into body-level, actionable coaching cues. Never exposes raw DTW numbers to the user.
- **Verifier**: Audits the Coach's output against the ground-truth Evidence Collector data. Trims unsupported claims or hallucinations.

### Prompt Generation Agent (LangGraph)
**Graph Flow:**
`VLM Scene Representation` ➔ `Flux Prompt Construction`

- Now decoupled and takes `feedback_id` as input to dynamically query postgres for context.
- **VLM Scene Representation**: Generates a structured JSON mapping of the environment, user appearance, equipment, and required identity preservation constraints.
- **Flux Prompt Construction**: Merges the environment schema with the `Coach` feedback to output a highly specific image-editing prompt that targets the biomechanical failure while preserving the scene.

### Available Tools
1. **`database_retrieval`**
   - **Mechanism**: Calculates OpenAI embeddings of the current query and executes a pgvector `ivfflat` cosine similarity search over historical `feedback` rows. Filters by `exercise_name` and `affected_joints`.
2. **`exa_search`**
   - **Mechanism**: Semantic web search using Exa (via LangChain). Pulls active coaching cues and physiological research. Must be queried with semantic descriptions (e.g., "knees collapsing inward"), not raw angles.
3. **VLM Compare / LLM Call / Embedding Call**
   - Tracked internally as separate tools to enforce LLM budgeting per request (limits defined in `config.py`).
