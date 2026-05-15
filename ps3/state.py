from typing import TypedDict, List, Optional, Annotated
import operator


class AgentState(TypedDict):
    # Input
    query: str
    task_id: str
    recipient_email: Optional[str]
    recipient_slack: Optional[str]

    # Pipeline data
    sub_tasks: List[str]
    search_results: Annotated[List[dict], operator.add]
    extracted_facts: List[dict]
    synthesis: str
    final_report: str

    # Metadata
    delivery_status: str
    errors: Annotated[List[str], operator.add]
    agent_log: Annotated[List[str], operator.add]
    current_agent: str
