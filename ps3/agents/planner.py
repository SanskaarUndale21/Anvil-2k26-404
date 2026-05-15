import json
from state import AgentState
from llm import generate

_SYSTEM = (
    "You are a research planner. Decompose a research query into exactly 4 "
    "specific, independently searchable sub-tasks. "
    "Return ONLY a valid JSON array of strings. No markdown fences, no extra text."
)


def planner_node(state: AgentState) -> dict:
    raw = generate(_SYSTEM, f"Research query: {state['query']}\n\nReturn exactly 4 sub-tasks as a JSON array.")

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    sub_tasks = json.loads(raw.strip())

    return {
        "sub_tasks": sub_tasks[:4],
        "current_agent": "planner",
        "agent_log": [f"[Planner] Decomposed into {len(sub_tasks)} sub-tasks: {sub_tasks[:2]}..."],
    }
