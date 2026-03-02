"""
Task 2-1: Add verbose/quiet tracing to the original LangGraph agent.

Modifications from original langgraph_simple_llama_agent.py:
- If user types "verbose", each node prints [TRACE] information to stdout.
- If user types "quiet", tracing is suppressed.
- verbose/quiet are handled in get_user_input and stored in state.

Usage:
    python task2_1_verbose_quiet.py
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
    verbose: bool          # NEW: tracing flag


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
            print(f"[TRACE] >>> Entering node: get_user_input")

        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit, 'verbose'/'quiet' to toggle tracing):")
        print("=" * 50)
        user_input = input("> ").strip()

        if verbose:
            print(f"[TRACE] Received input: '{user_input}'")

        if user_input.lower() in ["quit", "exit", "q"]:
            if verbose:
                print("[TRACE] User requested exit")
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True, "verbose": verbose}

        # NEW: toggle tracing on/off without sending to LLM
        if user_input.lower() == "verbose":
            print("[INFO] Verbose tracing ENABLED")
            return {"user_input": "", "should_exit": False, "verbose": True}
        if user_input.lower() == "quiet":
            print("[INFO] Verbose tracing DISABLED")
            return {"user_input": "", "should_exit": False, "verbose": False}

        if verbose:
            print(f"[TRACE] <<< Exiting node: get_user_input")
        return {"user_input": user_input, "should_exit": False, "verbose": verbose}

    def call_llm(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print(f"[TRACE] >>> Entering node: call_llm")
            print(f"[TRACE] user_input = '{state['user_input']}'")

        prompt = f"User: {state['user_input']}\nAssistant:"
        print("\nProcessing...")
        response = llm.invoke(prompt)

        if verbose:
            preview = str(response)[:80]
            print(f"[TRACE] LLM response (first 80 chars): '{preview}'")
            print(f"[TRACE] <<< Exiting node: call_llm")
        return {"llm_response": response}

    def print_response(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print(f"[TRACE] >>> Entering node: print_response")
        print("\n" + "-" * 50)
        print("LLM Response:")
        print("-" * 50)
        print(state["llm_response"])
        if verbose:
            print(f"[TRACE] <<< Exiting node: print_response")
        return {}

    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END
        return "call_llm"

    builder = StateGraph(AgentState)
    builder.add_node("get_user_input", get_user_input)
    builder.add_node("call_llm", call_llm)
    builder.add_node("print_response", print_response)

    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges("get_user_input", route_after_input,
                                  {"call_llm": "call_llm", END: END})
    builder.add_edge("call_llm", "print_response")
    builder.add_edge("print_response", "get_user_input")
    return builder.compile()


def main():
    print("=" * 50)
    print("Task 2-1: LangGraph Agent with Verbose/Quiet Tracing")
    print("=" * 50)
    llm = create_llm()
    graph = create_graph(llm)
    try:
        png = graph.get_graph(xray=True).draw_mermaid_png()
        with open("task2_1_graph.png", "wb") as f:
            f.write(png)
        print("Graph saved to task2_1_graph.png")
    except Exception as e:
        print(f"Could not save graph: {e}")
    graph.invoke({"user_input": "", "should_exit": False, "llm_response": "", "verbose": False})


if __name__ == "__main__":
    main()
