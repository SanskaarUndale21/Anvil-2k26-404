import json
from state import AgentState
from llm import generate

_SYSTEM = (
    "You are a research synthesizer. Given structured facts extracted from multiple sub-tasks, "
    "write a coherent, well-connected synthesis that answers the original query. "
    "Identify patterns, conflicts, and key insights. Write in flowing prose (not bullet points). "
    "Aim for 400-600 words."
)


def synthesizer_node(state: AgentState) -> dict:
    facts_dump = json.dumps(state["extracted_facts"], indent=2)
    synthesis = generate(
        _SYSTEM,
        f"Original research query: {state['query']}\n\nExtracted facts:\n{facts_dump}",
    )

    return {
        "synthesis": synthesis,
        "current_agent": "synthesizer",
        "agent_log": [f"[Synthesizer] Generated synthesis ({len(synthesis)} chars)"],
    }
