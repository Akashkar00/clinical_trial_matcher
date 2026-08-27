# pipeline/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pipeline.state import PipelineState
from pipeline.nodes import (
    extract_node,
    fetch_node,
    ingest_node,
    retrieve_node,
    score_node
)


def should_retry_or_fail(state: PipelineState) -> str:
    """Conditional edge after extract node."""
    if state.get("error") and state.get("retry_count", 0) < 2:
        return "retry"
    elif state.get("error"):
        return "fail"
    return "continue"


def check_error(state: PipelineState) -> str:
    """Generic error check after any node."""
    if state.get("error"):
        return "fail"
    return "continue"


def build_graph():
    graph = StateGraph(PipelineState)

    # add nodes
    graph.add_node("extract", extract_node)
    graph.add_node("fetch", fetch_node)
    graph.add_node("ingest", ingest_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("score", score_node)

    # entry
    graph.set_entry_point("extract")

    # conditional: retry extraction up to 2x
    graph.add_conditional_edges(
        "extract",
        should_retry_or_fail,
        {
            "retry": "extract",
            "fail": END,
            "continue": "fetch"
        }
    )

    # linear after fetch
    graph.add_conditional_edges(
        "fetch", check_error,
        {"continue": "ingest", "fail": END}
    )
    graph.add_conditional_edges(
        "ingest", check_error,
        {"continue": "retrieve", "fail": END}
    )
    graph.add_conditional_edges(
        "retrieve", check_error,
        {"continue": "score", "fail": END}
    )

    graph.add_edge("score", END)

    # MemorySaver: in-process RAM checkpointer for multi-turn session memory.
    # Each invocation must pass config={"configurable": {"thread_id": session_id}}.
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# expose compiled graph
pipeline = build_graph()