"""
Task 2-7: LangGraph Crash Recovery via SqliteSaver checkpointing.

Modifications from task2_6:
- Uses langgraph.checkpoint.sqlite.SqliteSaver to persist graph state
  after every node execution to a local SQLite database.
- Every run uses the same thread_id, so if the process is killed and
  restarted, LangGraph loads the last checkpoint and resumes seamlessly.
- The conversation history, verbose flag, and all state fields survive
  crashes with no loss.

How to test crash recovery:
  1. Run: python task2_7_checkpointing.py
  2. Have a few turns of conversation.
  3. Press Ctrl+C to kill the process mid-run.
  4. Run: python task2_7_checkpointing.py again.
  5. The program picks up exactly where it left off — history intact.

Key LangGraph checkpointing concepts:
  - SqliteSaver stores a snapshot of AgentState after every node completes.
  - thread_id identifies which conversation to resume.
  - If a node was interrupted mid-execution, LangGraph re-runs that node
    (idempotency matters for nodes with side effects like LLM calls, but
    for our I/O-bound get_user_input node this just re-prompts the user).

Usage:
    python task2_7_checkpointing.py [--reset]   (--reset clears saved state)
"""

import re
import sys
import sqlite3
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List
import operator

CHECKPOINT_DB = "task2_7_checkpoints.db"
THREAD_ID     = "main-conversation"   # identifies this conversation across restarts

LLAMA_SYSTEM = (
    "You are Llama, an AI assistant in a group conversation with a Human and Qwen. "
    "Messages from the Human appear as 'Human: ...'. "
    "Messages from Qwen appear as 'Qwen: ...'. "
    "Respond naturally, helpfully, and concisely."
)
QWEN_SYSTEM = (
    "You are Qwen, an AI assistant in a group conversation with a Human and Llama. "
    "Messages from the Human appear as 'Human: ...'. "
    "Messages from Llama appear as 'Llama: ...'. "
    "Respond naturally, helpfully, and concisely."
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_input: str
    should_exit: bool
    last_response: str
    use_qwen: bool
    verbose: bool


def create_llm(model_id, label):
    device = get_device()
    print(f"Loading {label} ({model_id}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device == "mps":
        model = model.to(device)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=pipe)


def format_history_for(model_name: str, history: List[BaseMessage]) -> List[BaseMessage]:
    prefix = f"{model_name}: "
    return [
        AIMessage(content=m.content) if m.content.startswith(prefix)
        else HumanMessage(content=m.content)
        for m in history
    ]


def create_graph(llama_llm, qwen_llm, checkpointer):

    def get_user_input(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        n_turns = len(state.get("messages", []))
        if verbose:
            print(f"[TRACE] >>> Node: get_user_input | history: {n_turns} msgs")
        print("\n" + "=" * 60)
        print(f"You [{n_turns} messages so far] (quit / verbose / quiet):")
        print('  Prefix "Hey Qwen" to talk to Qwen, otherwise Llama responds.')
        print("=" * 60)
        user_input = input("> ").strip()
        if verbose:
            print(f"[TRACE] Input: '{user_input}'")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True, "verbose": verbose}
        if user_input.lower() == "verbose":
            print("[INFO] Verbose ON")
            return {"user_input": "", "should_exit": False, "verbose": True}
        if user_input.lower() == "quiet":
            print("[INFO] Verbose OFF")
            return {"user_input": "", "should_exit": False, "verbose": False}
        use_qwen = bool(re.match(r'^hey\s+qwen\b', user_input, re.IGNORECASE))
        clean = re.sub(r'^hey\s+qwen\s*', '', user_input, flags=re.IGNORECASE).strip()
        actual = clean if use_qwen else user_input
        return {"user_input": actual, "should_exit": False, "use_qwen": use_qwen, "verbose": verbose}

    def call_llama(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print(f"[TRACE] >>> Node: call_llama")
        formatted = format_history_for("Llama", state.get("messages", []))
        msgs = [SystemMessage(content=LLAMA_SYSTEM)] + formatted + [HumanMessage(content=f"Human: {state['user_input']}")]
        response = llama_llm.invoke(msgs)
        text = response.content if isinstance(response, BaseMessage) else str(response)
        if verbose:
            print(f"[TRACE] Llama: '{text[:80]}'")
        return {
            "last_response": f"Llama: {text}",
            "messages": [HumanMessage(content=f"Human: {state['user_input']}"),
                         HumanMessage(content=f"Llama: {text}")],
        }

    def call_qwen(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print(f"[TRACE] >>> Node: call_qwen")
        formatted = format_history_for("Qwen", state.get("messages", []))
        msgs = [SystemMessage(content=QWEN_SYSTEM)] + formatted + [HumanMessage(content=f"Human: {state['user_input']}")]
        response = qwen_llm.invoke(msgs)
        text = response.content if isinstance(response, BaseMessage) else str(response)
        if verbose:
            print(f"[TRACE] Qwen: '{text[:80]}'")
        return {
            "last_response": f"Qwen: {text}",
            "messages": [HumanMessage(content=f"Human: {state['user_input']}"),
                         HumanMessage(content=f"Qwen: {text}")],
        }

    def print_response(state: AgentState) -> dict:
        response = state.get("last_response", "")
        speaker = "Llama" if response.startswith("Llama:") else "Qwen"
        print(f"\n{'-'*60}\n{speaker}:\n{'-'*60}")
        print(response[len(speaker)+2:].strip())
        return {}

    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END
        if not state.get("user_input", "").strip():
            print("[INFO] Empty input — please type something.")
            return "get_user_input"
        return "route_to_model"

    def route_to_model(state: AgentState) -> str:
        return "call_qwen" if state.get("use_qwen", False) else "call_llama"

    builder = StateGraph(AgentState)
    builder.add_node("get_user_input", get_user_input)
    builder.add_node("route_to_model", route_to_model)
    builder.add_node("call_llama", call_llama)
    builder.add_node("call_qwen", call_qwen)
    builder.add_node("print_response", print_response)

    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges("get_user_input", route_after_input,
        {"route_to_model": "route_to_model", "get_user_input": "get_user_input", END: END})
    builder.add_conditional_edges("route_to_model", route_to_model,
        {"call_llama": "call_llama", "call_qwen": "call_qwen"})
    builder.add_edge("call_llama", "print_response")
    builder.add_edge("call_qwen", "print_response")
    builder.add_edge("print_response", "get_user_input")

    # ── Compile with checkpointer ─────────────────────────────────────────────
    return builder.compile(checkpointer=checkpointer)


def check_existing_checkpoint(db_path: str, thread_id: str) -> bool:
    """Return True if a checkpoint exists for this thread."""
    if not Path(db_path).exists():
        return False
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM checkpoints WHERE thread_id=?", (thread_id,))
        count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


def main():
    # Handle --reset flag
    if "--reset" in sys.argv:
        Path(CHECKPOINT_DB).unlink(missing_ok=True)
        print(f"[INFO] Checkpoint database '{CHECKPOINT_DB}' cleared.")

    print("=" * 60)
    print("Task 2-7: Multi-Agent Chat with Crash Recovery")
    print(f"Checkpoint DB: {CHECKPOINT_DB}  |  Thread: {THREAD_ID}")
    print("=" * 60)

    # Detect resume vs fresh start
    resuming = check_existing_checkpoint(CHECKPOINT_DB, THREAD_ID)
    if resuming:
        print("[INFO] Existing checkpoint found — RESUMING previous conversation.")
    else:
        print("[INFO] No checkpoint found — starting fresh conversation.")

    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", "Llama")
    qwen_llm  = create_llm("Qwen/Qwen2.5-0.5B-Instruct", "Qwen")

    # ── Create SqliteSaver checkpointer ───────────────────────────────────────
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = create_graph(llama_llm, qwen_llm, checkpointer)

        try:
            png = graph.get_graph(xray=True).draw_mermaid_png()
            with open("task2_7_graph.png", "wb") as f:
                f.write(png)
            print("Graph saved to task2_7_graph.png")
        except Exception as e:
            print(f"Could not save graph: {e}")

        # Config identifies this conversation; LangGraph uses it to load checkpoints
        config = {"configurable": {"thread_id": THREAD_ID}}

        initial_state = {
            "messages": [],
            "user_input": "",
            "should_exit": False,
            "last_response": "",
            "use_qwen": False,
            "verbose": False,
        }

        # If resuming, LangGraph ignores initial_state and loads from checkpoint
        graph.invoke(initial_state, config=config)


if __name__ == "__main__":
    main()
