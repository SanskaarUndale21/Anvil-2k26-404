import asyncio
from tavily import AsyncTavilyClient
from config import TAVILY_API_KEY, MAX_SEARCH_RESULTS
from state import AgentState


async def _search_one(client: AsyncTavilyClient, task: str) -> dict:
    try:
        results = await client.search(
            query=task,
            max_results=MAX_SEARCH_RESULTS,
            include_raw_content=False,
            include_answer=True,
        )
        return {
            "task": task,
            "answer": results.get("answer", ""),
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:600],
                    "score": r.get("score", 0),
                }
                for r in results.get("results", [])
            ],
        }
    except Exception as e:
        return {"task": task, "answer": "", "results": [], "error": str(e)}


def searcher_node(state: AgentState) -> dict:
    """Runs all sub-task searches in parallel (async fanout)."""

    async def run_all():
        async with AsyncTavilyClient(api_key=TAVILY_API_KEY) as client:
            coros = [_search_one(client, task) for task in state["sub_tasks"]]
            return await asyncio.gather(*coros)

    search_results = asyncio.run(run_all())

    total_results = sum(len(r["results"]) for r in search_results)
    return {
        "search_results": list(search_results),
        "current_agent": "searcher",
        "agent_log": [
            f"[Searcher] Ran {len(search_results)} parallel searches, got {total_results} total results"
        ],
    }
