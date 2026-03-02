"""
Task 2-2: Handle empty input with a 3-way conditional branch.

Modifications from task2_1:
- route_after_input now has THREE branches:
    1. should_exit=True  -> END
    2. user_input empty  -> back to get_user_input (loop, no LLM call)
    3. valid input       -> call_llm
- This is the "LangGraph spirit" approach: encode logic in the graph topology,
  not in an imperative loop.

Observation about empty input and small LLMs:
- Without this fix, an empty prompt causes Llama-3.2-1B to hallucinate
  a full conversation from nothing, sometimes generating gibberish or
  repeating the same phrase in a loop.
- On the first empty input, the model might output "Assistant: " followed
  by a made-up dialog.  On subsequent empty inputs the behavior can differ
  because the model's sampling is stochastic—demonstrating that small
  instruction-tuned models have no reliable handling of empty prompts.

Usage:
    python task2_2_empty_input.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AgentState(TypedDict):
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
    return HuggingFacePipeline(pipeline=pipe)


def create_graph(llm):

    def get_user_input(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print("[TRACE] >>> Node: get_user_input")

        print("\n" + "=" * 50)
        print("Enter text (quit / verbose / quiet):")
        print("=" * 50)
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

        if verbose:
            print("[TRACE] <<< Node: get_user_input")
        return {"user_input": user_input, "should_exit": False, "verbose": verbose}

    def call_llm(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print(f"[TRACE] >>> Node: call_llm | input='{state['user_input']}'")
        prompt = f"User: {state['user_input']}\nAssistant:"
        print("\nProcessing...")
        response = llm.invoke(prompt)
        if verbose:
            print(f"[TRACE] <<< Node: call_llm")
        return {"llm_response": response}

    def print_response(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print("[TRACE] >>> Node: print_response")
        print("\n" + "-" * 50)
        print("LLM Response:")
        print("-" * 50)
        print(state["llm_response"])
        return {}

    # ── NEW: 3-way router ────────────────────────────────────────────────────
    def route_after_input(state: AgentState) -> str:
        verbose = state.get("verbose", False)
        if state.get("should_exit", False):
            if verbose:
                print("[TRACE] Route -> END")
            return END
        if not state.get("user_input", "").strip():
            if verbose:
                print("[TRACE] Route -> get_user_input (empty, looping back)")
            print("[INFO] Empty input — please type something.")
            return "get_user_input"          # ← third branch: loop back
        if verbose:
            print("[TRACE] Route -> call_llm")
        return "call_llm"

    builder = StateGraph(AgentState)
    builder.add_node("get_user_input", get_user_input)
    builder.add_node("call_llm", call_llm)
    builder.add_node("print_response", print_response)

    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges(
        "get_user_input", route_after_input,
        {"call_llm": "call_llm", "get_user_input": "get_user_input", END: END}
    )
    builder.add_edge("call_llm", "print_response")
    builder.add_edge("print_response", "get_user_input")
    return builder.compile()


def main():
    print("=" * 50)
    print("Task 2-2: 3-Way Conditional Branch (Empty Input Handling)")
    print("=" * 50)
    llm = create_llm()
    graph = create_graph(llm)
    try:
        png = graph.get_graph(xray=True).draw_mermaid_png()
        with open("task2_2_graph.png", "wb") as f:
            f.write(png)
        print("Graph saved to task2_2_graph.png")
    except Exception as e:
        print(f"Could not save graph: {e}")
    graph.invoke({"user_input": "", "should_exit": False, "llm_response": "", "verbose": False})


if __name__ == "__main__":
    main()
