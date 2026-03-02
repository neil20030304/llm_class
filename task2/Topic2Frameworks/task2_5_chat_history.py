"""
Task 2-5: Add chat history using the LangChain Message API.

Modifications from task2_4:
- Qwen is DISABLED (Llama only) so we can test history in isolation.
- AgentState gains a `messages` field typed as Annotated[list, operator.add].
  LangGraph uses operator.add to APPEND new messages rather than replacing
  the whole list — this is the standard LangGraph accumulator pattern.
- call_llm now passes the full message history to the LLM:
      [SystemMessage] + history + [HumanMessage(current_input)]
- After the LLM responds, call_llm returns the two NEW messages
  (HumanMessage + AIMessage) which get appended to state.messages.

Message roles used:
  system    — persistent instructions / persona
  human     — user's turn
  ai        — Llama's turn

Usage:
    python task2_5_chat_history.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
import operator

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Llama, a helpful and concise AI assistant. "
    "You remember the full conversation history provided to you."
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Annotated with operator.add: LangGraph APPENDS returned messages
    # instead of replacing the list.
    messages: Annotated[List[BaseMessage], operator.add]
    user_input: str
    should_exit: bool
    llm_response: str
    verbose: bool


def create_llm():
    device = get_device()
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"Loading {model_id} on {device}...")
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
    print("Model loaded!")
    return HuggingFacePipeline(pipeline=pipe)


def create_graph(llm):

    # ── NODE 1: get_user_input ────────────────────────────────────────────────
    def get_user_input(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        history_len = len(state.get("messages", []))
        if verbose:
            print(f"[TRACE] >>> Node: get_user_input | history length: {history_len}")

        print("\n" + "=" * 55)
        print(f"You [{history_len // 2} turns so far] (quit / verbose / quiet):")
        print("=" * 55)
        user_input = input("> ").strip()

        if verbose:
            print(f"[TRACE] Input: '{user_input}'")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True, "verbose": verbose}
        if user_input.lower() == "verbose":
            print("[INFO] Verbose tracing ON")
            return {"user_input": "", "should_exit": False, "verbose": True}
        if user_input.lower() == "quiet":
            print("[INFO] Verbose tracing OFF")
            return {"user_input": "", "should_exit": False, "verbose": False}

        return {"user_input": user_input, "should_exit": False, "verbose": verbose}

    # ── NODE 2: call_llm ──────────────────────────────────────────────────────
    def call_llm(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        user_input = state["user_input"]
        history: List[BaseMessage] = state.get("messages", [])

        if verbose:
            print(f"[TRACE] >>> Node: call_llm")
            print(f"[TRACE] History has {len(history)} messages")
            for i, m in enumerate(history):
                role = type(m).__name__
                print(f"[TRACE]   [{i}] {role}: {m.content[:60]}...")

        # Build the full prompt: system + accumulated history + current input
        llm_messages = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + history
            + [HumanMessage(content=user_input)]
        )

        if verbose:
            print(f"[TRACE] Sending {len(llm_messages)} messages to Llama")

        response = llm.invoke(llm_messages)

        # Extract plain text from response
        if isinstance(response, BaseMessage):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)

        if verbose:
            print(f"[TRACE] Response: '{response_text[:80]}'")
            print(f"[TRACE] <<< Node: call_llm")

        # Return the two new messages to be APPENDED to state.messages
        return {
            "llm_response": response_text,
            "messages": [
                HumanMessage(content=user_input),
                AIMessage(content=response_text),
            ],
        }

    # ── NODE 3: print_response ────────────────────────────────────────────────
    def print_response(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print("[TRACE] >>> Node: print_response")
        print("\n" + "-" * 55)
        print("Llama:")
        print("-" * 55)
        print(state["llm_response"])
        return {}

    # ── Router ────────────────────────────────────────────────────────────────
    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END
        if not state.get("user_input", "").strip():
            print("[INFO] Empty input — please type something.")
            return "get_user_input"
        return "call_llm"

    # ── Build graph ───────────────────────────────────────────────────────────
    builder = StateGraph(AgentState)
    builder.add_node("get_user_input", get_user_input)
    builder.add_node("call_llm", call_llm)
    builder.add_node("print_response", print_response)

    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges(
        "get_user_input", route_after_input,
        {"call_llm": "call_llm", "get_user_input": "get_user_input", END: END},
    )
    builder.add_edge("call_llm", "print_response")
    builder.add_edge("print_response", "get_user_input")
    return builder.compile()


def main():
    print("=" * 55)
    print("Task 2-5: Llama Chat Agent with Message History")
    print("=" * 55)
    llm = create_llm()
    graph = create_graph(llm)
    try:
        png = graph.get_graph(xray=True).draw_mermaid_png()
        with open("task2_5_graph.png", "wb") as f:
            f.write(png)
        print("Graph saved to task2_5_graph.png")
    except Exception as e:
        print(f"Could not save graph: {e}")

    graph.invoke({
        "messages": [],        # start with empty history
        "user_input": "",
        "should_exit": False,
        "llm_response": "",
        "verbose": False,
    })


if __name__ == "__main__":
    main()
