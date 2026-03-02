"""
Task 2-6: Multi-agent chat — shared history + Hey Qwen routing.

Three participants: Human, Llama, Qwen.
LangChain Message API only has roles: system / human / ai / tool.
We encode the 3-entity conversation by storing ALL messages with a
speaker prefix in the content, and re-mapping roles when calling each model:

  Storage (shared history):
    HumanMessage(content="Human: <text>")
    HumanMessage(content="Llama: <text>")    ← stored as HumanMessage
    HumanMessage(content="Qwen: <text>")     ← stored as HumanMessage

  When calling Llama — reformat history so:
    "Llama: ..." → AIMessage  (Llama's own previous turns = assistant)
    everything else → HumanMessage

  When calling Qwen — reformat history so:
    "Qwen: ..."  → AIMessage  (Qwen's own previous turns = assistant)
    everything else → HumanMessage

This matches the format described in the task:
  [{role:"user", content:"Human: What is the best ice cream flavor?"},
   {role:"assistant", content:"Llama: There is no one best flavor..."}]

Routing:
  Input starting with "Hey Qwen" → Qwen node
  Everything else               → Llama node

Usage:
    python task2_6_multi_agent_chat.py
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
import operator

LLAMA_SYSTEM = (
    "You are Llama, an AI assistant in a group conversation with a Human and Qwen (another AI). "
    "Messages from the Human appear as 'Human: ...'. "
    "Messages from Qwen appear as 'Qwen: ...'. "
    "Respond naturally, helpfully, and concisely."
)

QWEN_SYSTEM = (
    "You are Qwen, an AI assistant in a group conversation with a Human and Llama (another AI). "
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
    use_qwen: bool          # routing flag
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
    """
    Re-map a shared history (all HumanMessage with speaker prefixes) into
    the correct roles for `model_name` ("Llama" or "Qwen").

    Rule:
      content starts with "<model_name>: " → AIMessage (assistant role)
      everything else                      → HumanMessage (user role)
    """
    formatted = []
    prefix = f"{model_name}: "
    for msg in history:
        content = msg.content if hasattr(msg, "content") else str(msg)
        if content.startswith(prefix):
            formatted.append(AIMessage(content=content))
        else:
            formatted.append(HumanMessage(content=content))
    return formatted


def create_graph(llama_llm, qwen_llm):

    # ── NODE: get_user_input ──────────────────────────────────────────────────
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

        # Detect "Hey Qwen" prefix
        use_qwen = bool(re.match(r'^hey\s+qwen\b', user_input, re.IGNORECASE))
        clean_input = re.sub(r'^hey\s+qwen\s*', '', user_input, flags=re.IGNORECASE).strip()
        actual_input = clean_input if use_qwen else user_input

        if verbose:
            target = "Qwen" if use_qwen else "Llama"
            print(f"[TRACE] Routing to {target} | cleaned input: '{actual_input}'")

        return {
            "user_input": actual_input,
            "should_exit": False,
            "use_qwen": use_qwen,
            "verbose": verbose,
        }

    # ── NODE: call_llama ──────────────────────────────────────────────────────
    def call_llama(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        user_input = state["user_input"]
        history = state.get("messages", [])

        if verbose:
            print(f"[TRACE] >>> Node: call_llama | history: {len(history)} msgs")

        formatted = format_history_for("Llama", history)
        llm_messages = (
            [SystemMessage(content=LLAMA_SYSTEM)]
            + formatted
            + [HumanMessage(content=f"Human: {user_input}")]
        )

        if verbose:
            print(f"[TRACE] Sending {len(llm_messages)} messages to Llama")

        response = llama_llm.invoke(llm_messages)
        response_text = response.content if isinstance(response, BaseMessage) else str(response)

        if verbose:
            print(f"[TRACE] Llama: '{response_text[:80]}'")

        return {
            "last_response": f"Llama: {response_text}",
            # Append both Human turn and Llama response to shared history
            "messages": [
                HumanMessage(content=f"Human: {user_input}"),
                HumanMessage(content=f"Llama: {response_text}"),
            ],
        }

    # ── NODE: call_qwen ───────────────────────────────────────────────────────
    def call_qwen(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        user_input = state["user_input"]
        history = state.get("messages", [])

        if verbose:
            print(f"[TRACE] >>> Node: call_qwen | history: {len(history)} msgs")

        formatted = format_history_for("Qwen", history)
        llm_messages = (
            [SystemMessage(content=QWEN_SYSTEM)]
            + formatted
            + [HumanMessage(content=f"Human: {user_input}")]
        )

        if verbose:
            print(f"[TRACE] Sending {len(llm_messages)} messages to Qwen")

        response = qwen_llm.invoke(llm_messages)
        response_text = response.content if isinstance(response, BaseMessage) else str(response)

        if verbose:
            print(f"[TRACE] Qwen: '{response_text[:80]}'")

        return {
            "last_response": f"Qwen: {response_text}",
            "messages": [
                HumanMessage(content=f"Human: {user_input}"),
                HumanMessage(content=f"Qwen: {response_text}"),
            ],
        }

    # ── NODE: print_response ──────────────────────────────────────────────────
    def print_response(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print("[TRACE] >>> Node: print_response")
        response = state.get("last_response", "")
        speaker = "Llama" if response.startswith("Llama:") else "Qwen"
        print(f"\n{'-' * 60}")
        print(f"{speaker}:")
        print(f"{'-' * 60}")
        print(response[len(speaker) + 2:].strip())
        return {}

    # ── Routers ───────────────────────────────────────────────────────────────
    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END
        if not state.get("user_input", "").strip():
            print("[INFO] Empty input — please type something.")
            return "get_user_input"
        return "route_to_model"

    def route_to_model(state: AgentState) -> str:
        verbose = state.get("verbose", False)
        use_qwen = state.get("use_qwen", False)
        if verbose:
            print(f"[TRACE] Router: use_qwen={use_qwen}")
        return "call_qwen" if use_qwen else "call_llama"

    # ── Build graph ───────────────────────────────────────────────────────────
    builder = StateGraph(AgentState)
    builder.add_node("get_user_input", get_user_input)
    builder.add_node("route_to_model", route_to_model)      # dummy node for routing
    builder.add_node("call_llama", call_llama)
    builder.add_node("call_qwen", call_qwen)
    builder.add_node("print_response", print_response)

    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges(
        "get_user_input", route_after_input,
        {"route_to_model": "route_to_model", "get_user_input": "get_user_input", END: END},
    )
    builder.add_conditional_edges(
        "route_to_model", route_to_model,
        {"call_llama": "call_llama", "call_qwen": "call_qwen"},
    )
    builder.add_edge("call_llama", "print_response")
    builder.add_edge("call_qwen", "print_response")
    builder.add_edge("print_response", "get_user_input")

    return builder.compile()


def main():
    print("=" * 60)
    print("Task 2-6: Multi-Agent Chat (Llama + Qwen, shared history)")
    print("=" * 60)
    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", "Llama")
    qwen_llm  = create_llm("Qwen/Qwen2.5-0.5B-Instruct", "Qwen")
    graph = create_graph(llama_llm, qwen_llm)
    try:
        png = graph.get_graph(xray=True).draw_mermaid_png()
        with open("task2_6_graph.png", "wb") as f:
            f.write(png)
        print("Graph saved to task2_6_graph.png")
    except Exception as e:
        print(f"Could not save graph: {e}")

    graph.invoke({
        "messages": [],
        "user_input": "",
        "should_exit": False,
        "last_response": "",
        "use_qwen": False,
        "verbose": False,
    })


if __name__ == "__main__":
    main()
