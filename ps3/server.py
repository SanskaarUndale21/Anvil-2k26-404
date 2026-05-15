import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graph import run_research_pipeline

app = FastAPI(title="Research Automation Agent", version="1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory task store. Use Redis for prod.
tasks: dict = {}


class ResearchRequest(BaseModel):
    query: str
    recipient_email: Optional[str] = None
    recipient_slack: Optional[str] = None


def _run_pipeline(task_id: str, req: ResearchRequest):
    def on_event(log: str):
        tasks[task_id]["events"].append(
            {"type": "log", "message": log, "ts": datetime.utcnow().isoformat()}
        )

    try:
        tasks[task_id]["status"] = "running"
        final_state = run_research_pipeline(
            task_id=task_id,
            query=req.query,
            recipient_email=req.recipient_email,
            recipient_slack=req.recipient_slack,
            event_callback=on_event,
        )
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = {
            "final_report": final_state.get("final_report", ""),
            "delivery_status": final_state.get("delivery_status", ""),
            "sub_tasks": final_state.get("sub_tasks", []),
            "agent_log": final_state.get("agent_log", []),
        }
        tasks[task_id]["events"].append(
            {"type": "completed", "ts": datetime.utcnow().isoformat()}
        )
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["events"].append(
            {"type": "error", "message": str(e), "ts": datetime.utcnow().isoformat()}
        )


@app.post("/webhook/research")
async def research_webhook(req: ResearchRequest, background_tasks: BackgroundTasks):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "query": req.query,
        "events": [],
        "created_at": datetime.utcnow().isoformat(),
    }

    background_tasks.add_task(_run_pipeline, task_id, req)
    return {"task_id": task_id, "status": "queued"}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.get("/tasks/{task_id}/stream")
async def stream_task(task_id: str):
    """SSE endpoint -- clients listen here for live agent updates."""

    async def generator():
        last_idx = 0
        while True:
            task = tasks.get(task_id)
            if not task:
                yield f"data: {json.dumps({'type': 'error', 'message': 'task not found'})}\n\n"
                break

            events = task.get("events", [])
            for event in events[last_idx:]:
                yield f"data: {json.dumps(event)}\n\n"
            last_idx = len(events)

            if task["status"] in ("completed", "failed"):
                yield f"data: {json.dumps({'type': 'done', 'status': task['status']})}\n\n"
                break

            await asyncio.sleep(0.4)

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/tasks")
async def list_tasks():
    return [
        {
            "task_id": t["task_id"],
            "status": t["status"],
            "query": t["query"],
            "created_at": t["created_at"],
        }
        for t in tasks.values()
    ]


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
