import json
from datetime import datetime
from state import AgentState
from llm import generate

_SYSTEM = (
    "You are a professional report writer. "
    "Format a research report in clean Markdown. "
    "Structure: Title, Executive Summary (3 bullets), "
    "then one section per sub-task with findings and source URLs, "
    "then a Conclusion with actionable insights. Be precise and professional."
)


def writer_node(state: AgentState) -> dict:
    facts_dump = json.dumps(state["extracted_facts"], indent=2)
    date_str = datetime.utcnow().strftime("%B %d, %Y")

    final_report = generate(
        _SYSTEM,
        f"Query: {state['query']}\n"
        f"Date: {date_str}\n\n"
        f"Synthesis:\n{state['synthesis']}\n\n"
        f"Extracted facts with sources:\n{facts_dump}",
    )

    return {
        "final_report": final_report,
        "current_agent": "writer",
        "agent_log": [f"[Writer] Formatted final report ({len(final_report.splitlines())} lines)"],
    }
