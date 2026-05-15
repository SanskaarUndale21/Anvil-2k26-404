import json
from state import AgentState
from llm import generate

_SYSTEM = (
    "You are a fact extractor. Given web search results for multiple sub-tasks, "
    "extract the most important and verifiable facts for each sub-task. "
    "Return ONLY a valid JSON array where each item has: "
    '{"sub_task": str, "key_facts": [str], "sources": [str]}. '
    "Aim for 3-5 key facts per sub-task. No markdown fences, no extra text."
)


def extractor_node(state: AgentState) -> dict:
    search_dump = json.dumps(state["search_results"], indent=2)
    raw = generate(
        _SYSTEM,
        f"Original query: {state['query']}\n\nSearch results:\n{search_dump}",
    )

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    extracted_facts = json.loads(raw.strip())

    total_facts = sum(len(item.get("key_facts", [])) for item in extracted_facts)
    return {
        "extracted_facts": extracted_facts,
        "current_agent": "extractor",
        "agent_log": [
            f"[Extractor] Extracted {total_facts} key facts across {len(extracted_facts)} sub-tasks"
        ],
    }
