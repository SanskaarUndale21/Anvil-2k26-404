# PS3 -- Research Automation Agent

Multi-agent pipeline that autonomously researches any topic and delivers a formatted report.

## Architecture

```
Webhook/UI trigger
  -> Planner    (Claude: decompose query into 4 sub-tasks)
  -> Searcher   (Tavily: parallel async web search for each sub-task)
  -> Extractor  (Claude: extract key facts from all search results)
  -> Synthesizer(Claude: merge facts into coherent narrative)
  -> Writer     (Claude: format as professional Markdown report)
  -> Delivery   (Save to disk + optional email via Resend + Slack)
```

LangGraph manages state and orchestration. FastAPI serves the webhook and SSE dashboard.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in GOOGLE_API_KEY and TAVILY_API_KEY at minimum
```

## Run

```bash
python server.py
# Open http://localhost:8000 for the live dashboard
```

## API

```bash
# Trigger a research pipeline
curl -X POST http://localhost:8000/webhook/research \
  -H "Content-Type: application/json" \
  -d '{"query": "State of LLM reasoning in 2025", "recipient_email": "you@example.com"}'

# Monitor via SSE
curl http://localhost:8000/tasks/<task_id>/stream

# Get result
curl http://localhost:8000/tasks/<task_id>
```

## Env vars

| Key | Required | Purpose |
|-----|----------|---------|
| `GOOGLE_API_KEY` | Yes | Gemini 2.0 Flash for planning/extraction/synthesis/writing |
| `TAVILY_API_KEY` | Yes | Web search |
| `RESEND_API_KEY` | No | Email delivery |
| `SLACK_BOT_TOKEN` | No | Slack delivery |
| `SLACK_CHANNEL` | No | Target Slack channel |

Reports are always saved locally to `reports/` regardless of email/Slack config.

## Evaluation alignment

| Axis | Implementation |
|------|---------------|
| Autonomous execution | Zero human intervention after trigger; crash recovery via FastAPI background tasks |
| Multi-agent workflow | 6 specialized agents with explicit handoffs tracked in `agent_log` |
| Tooling + integrations | Tavily (live web), Resend/Slack (delivery side-effects) |
| Async / fanout | 4 parallel Tavily searches via `asyncio.gather` in searcher node |
| Observability | Real-time SSE dashboard; all agent logs in state |
