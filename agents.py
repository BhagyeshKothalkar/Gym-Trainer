import base64
import json
import time
from io import BytesIO
from typing import Any, Callable, Dict, List, TypedDict
from urllib.request import Request, urlopen

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, StateGraph
from PIL import Image

from config import CONFIG
from database import Database


class FeedbackState(TypedDict, total=False):
    exercise_name: str
    movement_analysis: Dict[str, Any]
    critical_frame: Dict[str, Any]
    user_image_url: str
    reference_image_url: str
    historical_context: List[Dict[str, Any]]
    evidence: Dict[str, Any]
    analyst_findings: Dict[str, Any]
    research: Dict[str, Any]
    coached_feedback: Dict[str, Any]
    verified_feedback: Dict[str, Any]
    structured_feedback: Dict[str, Any]
    tool_budgets: Dict[str, int]
    tool_usage: Dict[str, int]


class PromptState(TypedDict, total=False):
    user_image_url: str
    reference_image_url: str
    feedback: Dict[str, Any]
    movement_analysis: Dict[str, Any]
    scene: Dict[str, Any]
    flux_prompt: str
    tool_budgets: Dict[str, int]
    tool_usage: Dict[str, int]


def default_tool_budgets() -> Dict[str, int]:
    budgets = CONFIG.agents.tool_budgets
    return {
        "database_retrieval": budgets.database_reads,
        "exa_search": budgets.exa_searches,
        "vlm_compare": budgets.vlm_calls,
        "llm_call": budgets.llm_calls,
        "embedding": budgets.embedding_calls,
    }


def tool_result(
    output: Any,
    calls_used: int,
    calls_remaining: int,
    start_time: float,
    cache_hit: bool = False,
    confidence: float = 0.7,
) -> Dict[str, Any]:
    return {
        "output": output,
        "calls_used": calls_used,
        "calls_remaining": calls_remaining,
        "execution_time": time.perf_counter() - start_time,
        "cache_hit": cache_hit,
        "confidence": confidence,
    }


def guarded_tool_call(
    state: Dict[str, Any],
    tool_name: str,
    call: Callable[[], Any],
    confidence: float = 0.7,
) -> Dict[str, Any]:
    budgets = state.setdefault("tool_budgets", default_tool_budgets())
    usage = state.setdefault("tool_usage", {})
    remaining = budgets.get(tool_name, 0)
    start = time.perf_counter()
    if remaining <= 0:
        return tool_result(
            {"error": f"{tool_name} call budget exhausted"},
            usage.get(tool_name, 0),
            0,
            start,
            confidence=0.0,
        )
    budgets[tool_name] = remaining - 1
    usage[tool_name] = usage.get(tool_name, 0) + 1
    output = call()
    return tool_result(
        output, usage[tool_name], budgets[tool_name], start, confidence=confidence
    )


def download_image(url: str) -> Image.Image:
    request = Request(url, headers={"User-Agent": "Gym-Trainer/1.0"})
    with urlopen(request, timeout=60) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


def concatenate_images(user_image_url: str, reference_image_url: str) -> str:
    user_image = download_image(user_image_url)
    reference_image = download_image(reference_image_url)
    height = max(user_image.height, reference_image.height)
    user_image = user_image.resize(
        (int(user_image.width * height / user_image.height), height)
    )
    reference_image = reference_image.resize(
        (int(reference_image.width * height / reference_image.height), height)
    )
    canvas = Image.new(
        "RGB", (user_image.width + reference_image.width, height), "white"
    )
    canvas.paste(user_image, (0, 0))
    canvas.paste(reference_image, (user_image.width, 0))
    buffer = BytesIO()
    canvas.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_json(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback


def llm() -> ChatGroq:
    return ChatGroq(
        model=CONFIG.agents.feedback_model,
        temperature=0.2,
        api_key=CONFIG.GROQ_API_KEY if hasattr(CONFIG, "GROQ_API_KEY") else None,
    )


def vlm() -> ChatGroq:
    return ChatGroq(
        model=CONFIG.agents.vision_model,
        temperature=0.0,
        api_key=CONFIG.GROQ_API_KEY if hasattr(CONFIG, "GROQ_API_KEY") else None,
    )


def embed_text(text: str) -> List[float]:
    embeddings = OpenAIEmbeddings(model=CONFIG.agents.embedding_model)
    return embeddings.embed_query(text)


def database_retrieval_tool(
    query: str, exercise_name: str, affected_joints: List[str]
) -> List[Dict[str, Any]]:
    embedding = embed_text(query)
    return Database().search_feedback_embeddings(
        embedding=embedding,
        exercise_name=exercise_name,
        affected_joints=affected_joints,
        limit=5,
    )


def exa_search_tool(query: str) -> List[Dict[str, Any]]:
    try:
        from langchain_exa import ExaSearchRetriever

        retriever = ExaSearchRetriever(k=5, api_key=CONFIG.agents.exa_api_key)
        docs = retriever.invoke(query)
        return [
            {
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("url", ""),
                "content": doc.page_content,
            }
            for doc in docs
        ]
    except Exception as exc:
        return [{"error": str(exc), "query": query}]


database_retrieval = StructuredTool.from_function(database_retrieval_tool)
exa_search = StructuredTool.from_function(exa_search_tool)


def movement_analysis_node(state: FeedbackState) -> FeedbackState:
    state["tool_budgets"] = state.get("tool_budgets", default_tool_budgets())
    state["tool_usage"] = state.get("tool_usage", {})
    return state


def evidence_collector_node(state: FeedbackState) -> FeedbackState:
    movement = state["movement_analysis"]
    critical_frame = state["critical_frame"]
    state["evidence"] = {
        "movement_summary": movement["summary"],
        "joint_metrics": movement["joint_metrics"],
        "critical_frame": critical_frame,
        "error_region": next(
            (
                region
                for region in movement["error_regions"]
                if region["region_index"] == critical_frame["region_index"]
            ),
            {},
        ),
        "historical_context": state.get("historical_context", []),
    }
    return state


def movement_analyst_node(state: FeedbackState) -> FeedbackState:
    comparison_b64 = concatenate_images(
        state["user_image_url"], state["reference_image_url"]
    )

    def call_vlm():
        messages = [
            SystemMessage(
                content="You are a biomechanics analyst. Compare the left user frame against the right reference frame."
            ),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Return JSON with biomechanical_deviations, image_observations, "
                            "dtw_correlations, affected_joints, and research_queries. "
                            "Use movement descriptions, not raw angle numbers, for research queries."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{comparison_b64}"
                        },
                    },
                    {"type": "text", "text": json.dumps(state["evidence"])},
                ]
            ),
        ]
        response = vlm().invoke(messages)
        return parse_json(
            response.content,
            {
                "biomechanical_deviations": [],
                "image_observations": response.content,
                "dtw_correlations": [],
                "affected_joints": state["critical_frame"].get("affected_joints", []),
                "research_queries": [],
            },
        )

    result = guarded_tool_call(state, "vlm_compare", call_vlm, confidence=0.75)
    state["analyst_findings"] = {
        "tool_result": result,
        **(result["output"] if isinstance(result["output"], dict) else {}),
    }
    return state


def pattern_researcher_node(state: FeedbackState) -> FeedbackState:
    findings = state.get("analyst_findings", {})
    affected = findings.get("affected_joints") or state["critical_frame"].get(
        "affected_joints", []
    )
    queries = (
        findings.get("research_queries")
        or findings.get("biomechanical_deviations")
        or []
    )
    semantic_query = (
        "; ".join(str(query) for query in queries[:2])
        or "movement deviation during exercise"
    )

    db_result = guarded_tool_call(
        state,
        "database_retrieval",
        lambda: database_retrieval.invoke(
            {
                "query": semantic_query,
                "exercise_name": state["exercise_name"],
                "affected_joints": affected,
            }
        ),
        confidence=0.65,
    )
    exa_result = guarded_tool_call(
        state,
        "exa_search",
        lambda: exa_search.invoke({"query": semantic_query}),
        confidence=0.55,
    )
    state["research"] = {
        "database": db_result,
        "exa": exa_result,
        "semantic_query": semantic_query,
    }
    return state


def coach_node(state: FeedbackState) -> FeedbackState:
    def call_llm():
        messages = [
            SystemMessage(
                content="You are a strength coach. Convert evidence into body-level coaching without exposing raw DTW numbers."
            ),
            HumanMessage(
                content=(
                    "Return JSON with summary, technical_analysis, body_level_analysis, "
                    "primary_issue, secondary_issues, risk_level, and research.\n\n"
                    f"Evidence:\n{json.dumps(state['evidence'])}\n\n"
                    f"Analyst findings:\n{json.dumps(state.get('analyst_findings', {}))}\n\n"
                    f"Research:\n{json.dumps(state.get('research', {}))}"
                )
            ),
        ]
        response = llm().invoke(messages)
        return parse_json(
            response.content,
            {
                "summary": response.content,
                "technical_analysis": "",
                "body_level_analysis": "",
                "primary_issue": "",
                "secondary_issues": [],
                "risk_level": "unknown",
                "research": state.get("research", {}),
            },
        )

    result = guarded_tool_call(state, "llm_call", call_llm, confidence=0.7)
    state["coached_feedback"] = result["output"]
    return state


def verifier_node(state: FeedbackState) -> FeedbackState:
    def call_llm():
        messages = [
            SystemMessage(
                content="Verify feedback against the evidence. Remove unsupported claims. Return corrected JSON only."
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "evidence": state["evidence"],
                        "feedback": state["coached_feedback"],
                    }
                )
            ),
        ]
        response = llm().invoke(messages)
        return parse_json(response.content, state["coached_feedback"])

    result = guarded_tool_call(state, "llm_call", call_llm, confidence=0.8)
    state["verified_feedback"] = result["output"]
    return state


def structured_feedback_node(state: FeedbackState) -> FeedbackState:
    feedback = state.get("verified_feedback", {})
    state["structured_feedback"] = {
        "summary": feedback.get("summary", ""),
        "technical_analysis": feedback.get("technical_analysis", ""),
        "body_level_analysis": feedback.get("body_level_analysis", ""),
        "primary_issue": feedback.get("primary_issue", ""),
        "secondary_issues": feedback.get("secondary_issues", []),
        "risk_level": feedback.get("risk_level", "unknown"),
        "research": feedback.get("research", state.get("research", {})),
        "model_version": CONFIG.agents.feedback_model,
        "tool_usage": state.get("tool_usage", {}),
        "tool_budgets_remaining": state.get("tool_budgets", {}),
    }
    return state


def build_feedback_graph():
    graph = StateGraph(FeedbackState)
    graph.add_node("Movement Analysis", movement_analysis_node)
    graph.add_node("Evidence Collector", evidence_collector_node)
    graph.add_node("Movement Analyst", movement_analyst_node)
    graph.add_node("Pattern Researcher", pattern_researcher_node)
    graph.add_node("Coach", coach_node)
    graph.add_node("Verifier", verifier_node)
    graph.add_node("Structured Feedback", structured_feedback_node)
    graph.set_entry_point("Movement Analysis")
    graph.add_edge("Movement Analysis", "Evidence Collector")
    graph.add_edge("Evidence Collector", "Movement Analyst")
    graph.add_edge("Movement Analyst", "Pattern Researcher")
    graph.add_edge("Pattern Researcher", "Coach")
    graph.add_edge("Coach", "Verifier")
    graph.add_edge("Verifier", "Structured Feedback")
    graph.add_edge("Structured Feedback", END)
    return graph.compile()


def prompt_scene_node(state: PromptState) -> PromptState:
    comparison_b64 = concatenate_images(
        state["user_image_url"], state["reference_image_url"]
    )

    def call_vlm():
        messages = [
            SystemMessage(
                content="Describe the user and reference frames for faithful Flux image editing."
            ),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Return JSON with user_appearance, body_features, posture, expression, movement, "
                            "equipment, environment, identity_preservation, and user_vs_reference_comparison."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{comparison_b64}"
                        },
                    },
                ]
            ),
        ]
        response = vlm().invoke(messages)
        return parse_json(response.content, {"description": response.content})

    result = guarded_tool_call(state, "vlm_compare", call_vlm, confidence=0.8)
    state["scene"] = result["output"]
    return state


def flux_prompt_node(state: PromptState) -> PromptState:
    def call_llm():
        messages = [
            SystemMessage(
                content="Create a Flux image editing prompt from structured scene data and verified coaching feedback."
            ),
            HumanMessage(
                content=(
                    "Return only the final prompt. Preserve identity, equipment, environment, and camera context. "
                    "Modify only the biomechanical issue described by the feedback.\n\n"
                    f"Scene:\n{json.dumps(state['scene'])}\n\n"
                    f"Feedback:\n{json.dumps(state['feedback'])}\n\n"
                    f"Movement analysis:\n{json.dumps(state['movement_analysis']['summary'])}"
                )
            ),
        ]
        return llm().invoke(messages).content

    result = guarded_tool_call(state, "llm_call", call_llm, confidence=0.75)
    state["flux_prompt"] = result["output"]
    return state


def build_prompt_graph():
    graph = StateGraph(PromptState)
    graph.add_node("VLM Scene Representation", prompt_scene_node)
    graph.add_node("Flux Prompt Construction", flux_prompt_node)
    graph.set_entry_point("VLM Scene Representation")
    graph.add_edge("VLM Scene Representation", "Flux Prompt Construction")
    graph.add_edge("Flux Prompt Construction", END)
    return graph.compile()


def run_feedback_agent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    state = {
        **inputs,
        "tool_budgets": default_tool_budgets(),
        "tool_usage": {},
    }
    return build_feedback_graph().invoke(state)["structured_feedback"]


def run_prompt_generation_agent(feedback_id: str) -> Dict[str, Any]:
    db = Database()
    feedback_record = db.get_feedback_for_generation(feedback_id)
    if not feedback_record:
        raise ValueError(f"Feedback {feedback_id} not found")

    movement_analysis = {
        "summary": json.loads(feedback_record["movement_analysis_summary"])
    }

    state = {
        "user_image_url": feedback_record["user_image_url"],
        "reference_image_url": feedback_record["reference_image_url"],
        "feedback": {
            "summary": feedback_record["summary"],
            "technical_analysis": feedback_record["technical_analysis"],
            "body_level_analysis": feedback_record["body_level_analysis"],
            "primary_issue": feedback_record["primary_issue"],
            "secondary_issues": feedback_record["secondary_issues"],
            "risk_level": feedback_record["risk_level"],
            "research": feedback_record["research"],
        },
        "movement_analysis": movement_analysis,
        "tool_budgets": default_tool_budgets(),
        "tool_usage": {},
    }
    result = build_prompt_graph().invoke(state)
    return {
        "prompt": result["flux_prompt"],
        "scene": result.get("scene", {}),
        "tool_usage": result.get("tool_usage", {}),
        "tool_budgets_remaining": result.get("tool_budgets", {}),
    }


def embed_feedback(feedback: Dict[str, Any]) -> List[float]:
    text = "\n".join(
        [
            feedback.get("technical_analysis", ""),
            feedback.get("body_level_analysis", ""),
            feedback.get("summary", ""),
        ]
    )
    return embed_text(text)
