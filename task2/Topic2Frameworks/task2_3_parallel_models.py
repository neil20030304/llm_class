"""
Task 2-3: Run Llama and Qwen in parallel, print both responses.

Modifications from task2_2:
- Load two models: Llama-3.2-1B-Instruct and Qwen2.5-0.5B-Instruct.
- After get_user_input, fan out to call_llama AND call_qwen in parallel.
- A collect_responses node receives both results and prints them side-by-side.

Graph topology:
    START -> get_user_input -> [3-way conditional]
                                  |-> END
                                  |-> get_user_input (empty)
                                  |-> call_llama ──┐
                                                    ├─> collect_responses -> get_user_input
                                  |-> call_qwen  ──┘

Note: LangGraph executes nodes with no dependency on each other in parallel
when both are reachable from the same conditional branch edge targets.

Usage:
    python task2_3_parallel_models.py
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
    llama_response: str
    qwen_response: str
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


def create_graph(llama_llm, qwen_llm):

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
        return {"user_input": user_input, "should_exit": False, "verbose": verbose}

    def call_llama(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print("[TRACE] >>> Node: call_llama")
        prompt = f"User: {state['user_input']}\nAssistant:"
        response = llama_llm.invoke(prompt)
        if verbose:
            print("[TRACE] <<< Node: call_llama")
        return {"llama_response": str(response)}

    def call_qwen(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print("[TRACE] >>> Node: call_qwen")
        prompt = f"User: {state['user_input']}\nAssistant:"
        response = qwen_llm.invoke(prompt)
        if verbose:
            print("[TRACE] <<< Node: call_qwen")
        return {"qwen_response": str(response)}

    def collect_responses(state: AgentState) -> dict:
        verbose = state.get("verbose", False)
        if verbose:
            print("[TRACE] >>> Node: collect_responses")
        print("\n" + "=" * 50)
        print("LLAMA RESPONSE:")
        print("-" * 50)
        print(state.get("llama_response", "(none)"))
        print("\n" + "=" * 50)
        print("QWEN RESPONSE:")
        print("-" * 50)
        print(state.get("qwen_response", "(none)"))
        return {}

    def route_after_input(state: AgentState) -> list:
        """
        Fan-out: return a LIST of node names to run in parallel.
        LangGraph runs them concurrently and waits for both before continuing.
        """
        if state.get("should_exit", False):
            return [END]
        if not state.get("user_input", "").strip():
            print("[INFO] Empty input — please type something.")
            return ["get_user_input"]
        return ["call_llama", "call_qwen"]   # ← parallel fan-out

    builder = StateGraph(AgentState)
    builder.add_node("get_user_input", get_user_input)
    builder.add_node("call_llama", call_llama)
    builder.add_node("call_qwen", call_qwen)
    builder.add_node("collect_responses", collect_responses)

    builder.add_edge(START, "get_user_input")
    builder.add_conditional_edges("get_user_input", route_after_input)
    builder.add_edge("call_llama", "collect_responses")
    builder.add_edge("call_qwen", "collect_responses")
    builder.add_edge("collect_responses", "get_user_input")
    return builder.compile()


def main():
    print("=" * 50)
    print("Task 2-3: Parallel Llama + Qwen Models")
    print("=" * 50)
    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", "Llama")
    qwen_llm  = create_llm("Qwen/Qwen2.5-0.5B-Instruct", "Qwen")
    graph = create_graph(llama_llm, qwen_llm)
    try:
        png = graph.get_graph(xray=True).draw_mermaid_png()
        with open("task2_3_graph.png", "wb") as f:
            f.write(png)
        print("Graph saved to task2_3_graph.png")
    except Exception as e:
        print(f"Could not save graph: {e}")
    graph.invoke({
        "user_input": "", "should_exit": False,
        "llama_response": "", "qwen_response": "", "verbose": False
    })


if __name__ == "__main__":
    main()
