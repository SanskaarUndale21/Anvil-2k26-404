import os
import re
from datetime import datetime
from state import AgentState
from config import (
    REPORTS_DIR,
    RESEND_API_KEY,
    SLACK_BOT_TOKEN,
    SLACK_CHANNEL,
)


def _save_to_disk(task_id: str, query: str, report: str) -> str:
    safe_q = re.sub(r"[^\w\s-]", "", query)[:40].strip().replace(" ", "_")
    filename = f"{task_id[:8]}_{safe_q}.md"
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path


def _send_email(recipient: str, query: str, report: str) -> str:
    try:
        import resend
        resend.api_key = RESEND_API_KEY
        resend.Emails.send(
            {
                "from": "Research Agent <reports@yourdomain.com>",
                "to": [recipient],
                "subject": f"Research Report: {query[:60]}",
                "text": report,
            }
        )
        return f"email sent to {recipient}"
    except Exception as e:
        return f"email failed: {e}"


def _send_slack(query: str, report: str) -> str:
    try:
        from slack_sdk import WebClient

        sc = WebClient(token=SLACK_BOT_TOKEN)
        # Post summary; attach full report as snippet
        sc.chat_postMessage(
            channel=SLACK_CHANNEL,
            text=f":mag: *Research Report Ready*\n*Query:* {query}\n\n{report[:2000]}...",
        )
        return f"slack sent to {SLACK_CHANNEL}"
    except Exception as e:
        return f"slack failed: {e}"


def delivery_node(state: AgentState) -> dict:
    statuses = []

    # Always save to disk
    path = _save_to_disk(state["task_id"], state["query"], state["final_report"])
    statuses.append(f"saved to {path}")

    if state.get("recipient_email") and RESEND_API_KEY:
        statuses.append(_send_email(state["recipient_email"], state["query"], state["final_report"]))

    if SLACK_BOT_TOKEN:
        statuses.append(_send_slack(state["query"], state["final_report"]))

    delivery_status = " | ".join(statuses)
    return {
        "delivery_status": delivery_status,
        "current_agent": "delivery",
        "agent_log": [f"[Delivery] {delivery_status}"],
    }
