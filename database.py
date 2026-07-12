import json
import uuid
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from config import CONFIG


def new_id() -> str:
    return str(uuid.uuid4())


def vector_literal(values: List[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


class Database:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or CONFIG.database.url
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required for PostgreSQL storage.")

    def connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def initialize_schema(self) -> None:
        embedding_dim = CONFIG.database.embedding_dimension
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            """
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                external_id TEXT UNIQUE NOT NULL,
                email TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS exercises (
                id UUID PRIMARY KEY,
                user_id UUID NOT NULL REFERENCES users(id),
                name TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(user_id, name)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id UUID PRIMARY KEY,
                exercise_id UUID NOT NULL REFERENCES exercises(id),
                user_video_url TEXT NOT NULL,
                reference_video_url TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS movements (
                id UUID PRIMARY KEY,
                session_id UUID NOT NULL REFERENCES sessions(id),
                summary TEXT NOT NULL,
                movement_score DOUBLE PRECISION NOT NULL,
                dtw_cost DOUBLE PRECISION NOT NULL,
                aggregate_metrics JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dtw_analyses (
                id UUID PRIMARY KEY,
                movement_id UUID NOT NULL REFERENCES movements(id),
                analysis JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS critical_frames (
                id UUID PRIMARY KEY,
                dtw_analysis_id UUID NOT NULL REFERENCES dtw_analyses(id),
                region_index INTEGER NOT NULL,
                severity DOUBLE PRECISION NOT NULL,
                phase TEXT NOT NULL,
                distance DOUBLE PRECISION NOT NULL,
                user_image_url TEXT NOT NULL,
                reference_image_url TEXT NOT NULL,
                metadata JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS feedback (
                id UUID PRIMARY KEY,
                critical_frame_id UUID NOT NULL REFERENCES critical_frames(id),
                summary TEXT NOT NULL,
                technical_analysis TEXT NOT NULL,
                body_level_analysis TEXT NOT NULL,
                primary_issue TEXT NOT NULL,
                secondary_issues JSONB NOT NULL,
                risk_level TEXT NOT NULL,
                research JSONB NOT NULL,
                model_version TEXT NOT NULL,
                embedding vector({embedding_dim}),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS generated_images (
                id UUID PRIMARY KEY,
                feedback_id UUID NOT NULL REFERENCES feedback(id),
                image_url TEXT,
                generation_prompt TEXT,
                metadata JSONB NOT NULL,
                generation_status TEXT NOT NULL,
                generation_started TIMESTAMPTZ,
                generation_finished TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            "CREATE INDEX IF NOT EXISTS feedback_embedding_idx ON feedback USING ivfflat (embedding vector_cosine_ops)",
        ]

        with self.connect() as conn:
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
            conn.commit()

    def upsert_user(
        self,
        external_id: str,
        email: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        user_id = new_id()
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO users (id, external_id, email, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (external_id) DO UPDATE
                SET email = EXCLUDED.email,
                    metadata = users.metadata || EXCLUDED.metadata
                RETURNING id
                """,
                (user_id, external_id, email, json.dumps(metadata or {})),
            ).fetchone()
            conn.commit()
        return str(row["id"])

    def upsert_exercise(
        self, user_id: str, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        exercise_id = new_id()
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO exercises (id, user_id, name, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, name) DO UPDATE
                SET metadata = exercises.metadata || EXCLUDED.metadata
                RETURNING id
                """,
                (exercise_id, user_id, name, json.dumps(metadata or {})),
            ).fetchone()
            conn.commit()
        return str(row["id"])

    def create_session(
        self,
        exercise_id: str,
        user_video_url: str,
        reference_video_url: str,
        metadata: Dict[str, Any],
    ) -> str:
        session_id = new_id()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, exercise_id, user_video_url, reference_video_url, status, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    exercise_id,
                    user_video_url,
                    reference_video_url,
                    "running",
                    json.dumps(metadata),
                ),
            )
            conn.commit()
        return session_id

    def create_movement(
        self,
        session_id: str,
        summary: str,
        movement_score: float,
        dtw_cost: float,
        aggregate_metrics: Dict[str, Any],
    ) -> str:
        movement_id = new_id()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO movements (id, session_id, summary, movement_score, dtw_cost, aggregate_metrics)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    movement_id,
                    session_id,
                    summary,
                    movement_score,
                    dtw_cost,
                    json.dumps(aggregate_metrics),
                ),
            )
            conn.commit()
        return movement_id

    def create_dtw_analysis(self, movement_id: str, analysis: Dict[str, Any]) -> str:
        dtw_analysis_id = new_id()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO dtw_analyses (id, movement_id, analysis) VALUES (%s, %s, %s)",
                (dtw_analysis_id, movement_id, json.dumps(analysis)),
            )
            conn.commit()
        return dtw_analysis_id

    def create_critical_frame(self, dtw_analysis_id: str, frame: Dict[str, Any]) -> str:
        critical_frame_id = new_id()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO critical_frames (
                    id, dtw_analysis_id, region_index, severity, phase, distance,
                    user_image_url, reference_image_url, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    critical_frame_id,
                    dtw_analysis_id,
                    frame["region_index"],
                    frame["severity"],
                    frame["phase"],
                    frame["distance"],
                    frame["user_image_url"],
                    frame["reference_image_url"],
                    json.dumps(frame.get("metadata", {})),
                ),
            )
            conn.commit()
        return critical_frame_id

    def create_feedback(
        self, critical_frame_id: str, feedback: Dict[str, Any], embedding: List[float]
    ) -> str:
        feedback_id = new_id()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (
                    id, critical_frame_id, summary, technical_analysis, body_level_analysis,
                    primary_issue, secondary_issues, risk_level, research, model_version,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                """,
                (
                    feedback_id,
                    critical_frame_id,
                    feedback.get("summary", ""),
                    feedback.get("technical_analysis", ""),
                    feedback.get("body_level_analysis", ""),
                    feedback.get("primary_issue", ""),
                    json.dumps(feedback.get("secondary_issues", [])),
                    feedback.get("risk_level", "unknown"),
                    json.dumps(feedback.get("research", {})),
                    feedback.get("model_version", CONFIG.agents.feedback_model),
                    vector_literal(embedding),
                ),
            )
            conn.commit()
        return feedback_id

    def create_generation_job(self, feedback_id: str) -> str:
        job_id = new_id()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO generated_images (id, feedback_id, metadata, generation_status, generation_started)
                VALUES (%s, %s, %s, %s, NOW())
                """,
                (job_id, feedback_id, json.dumps({}), "queued"),
            )
            conn.commit()
        return job_id

    def update_generated_image(
        self, job_id: str, image_url: str, prompt: str, metadata: Dict[str, Any]
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE generated_images 
                SET image_url = %s, generation_prompt = %s, metadata = %s, generation_status = %s, generation_finished = NOW()
                WHERE id = %s
                """,
                (image_url, prompt, json.dumps(metadata), "complete", job_id),
            )
            conn.commit()

    def fail_generation_job(self, job_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE generated_images SET generation_status = %s, generation_finished = NOW() WHERE id = %s",
                ("failed", job_id),
            )
            conn.commit()

    def get_generation_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM generated_images WHERE id = %s", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_feedback_for_generation(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT 
                    f.*, 
                    cf.user_image_url, 
                    cf.reference_image_url,
                    da.analysis->>'summary' AS movement_analysis_summary
                FROM feedback f
                JOIN critical_frames cf ON f.critical_frame_id = cf.id
                JOIN dtw_analyses da ON cf.dtw_analysis_id = da.id
                WHERE f.id = %s
                """,
                (feedback_id,),
            ).fetchone()
        return dict(row) if row else None

    def complete_session(self, session_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET status = 'complete', completed_at = NOW() WHERE id = %s",
                (session_id,),
            )
            conn.commit()

    def search_feedback_embeddings(
        self,
        embedding: List[float],
        exercise_name: str = "",
        affected_joints: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        filters = []
        params: List[Any] = [vector_literal(embedding)]
        if exercise_name:
            filters.append("e.name = %s")
            params.append(exercise_name)
        if affected_joints:
            filters.append("cf.metadata->'affected_joints' ?| %s")
            params.append(affected_joints)
        where_clause = "WHERE " + " AND ".join(filters) if filters else ""
        params.extend([vector_literal(embedding), limit])
        query = f"""
            SELECT
                f.id AS feedback_id,
                f.summary,
                f.technical_analysis,
                f.body_level_analysis,
                f.primary_issue,
                f.risk_level,
                e.name AS exercise_name,
                m.id AS movement_id,
                1 - (f.embedding <=> %s::vector) AS similarity
            FROM feedback f
            JOIN critical_frames cf ON cf.id = f.critical_frame_id
            JOIN dtw_analyses da ON da.id = cf.dtw_analysis_id
            JOIN movements m ON m.id = da.movement_id
            JOIN sessions s ON s.id = m.session_id
            JOIN exercises e ON e.id = s.exercise_id
            {where_clause}
            ORDER BY f.embedding <=> %s::vector
            LIMIT %s
        """
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
