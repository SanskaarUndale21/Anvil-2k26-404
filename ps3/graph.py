from langgraph.graph import StateGraph, START, END
from state import AgentState
from agents import (
    planner_node,
    searcher_node,
    extractor_node,
    synthesizer_node,
    writer_node,
    delivery_node,
)


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("searcher", searcher_node)
    g.add_node("extractor", extractor_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("writer", writer_node)
    g.add_node("delivery", delivery_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "searcher")
    g.add_edge("searcher", "extractor")
    g.add_edge("extractor", "synthesizer")
    g.add_edge("synthesizer", "writer")
    g.add_edge("writer", "delivery")
    g.add_edge("delivery", END)

    return g.compile()


compiled_graph = build_graph()


def run_research_pipeline(
    task_id: str,
    query: str,
    recipient_email: str = None,
    recipient_slack: str = None,
    event_callback=None,
) -> dict:
    """
    Runs the full research pipeline synchronously.
    event_callback(agent_name, log_msg) is called after each node if provided.
    """
    initial_state: AgentState = {
        "query": query,
        "task_id": task_id,
        "recipient_email": recipient_email,
        "recipient_slack": recipient_slack,
        "sub_tasks": [],
        "search_results": [],
        "extracted_facts": [],
        "synthesis": "",
        "final_report": "",
        "delivery_status": "",
        "errors": [],
        "agent_log": [],
        "current_agent": "init",
    }

    final_state = compiled_graph.invoke(initial_state)

    if event_callback:
        for log in final_state.get("agent_log", []):
            event_callback(log)

    return final_state
