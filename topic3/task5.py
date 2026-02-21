"""
Task 5: LangGraph conversation with checkpointing and recovery.

This script replaces manual Python iteration with a LangGraph workflow that:
- Runs a single long multi-turn conversation in one graph invocation
- Uses manual tool-calling via ToolNode
- Persists state with checkpointing
- Demonstrates recovery/resume from a saved checkpoint
"""

from __future__ import annotations

import json
import math
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Use tools whenever a tool can help solve the request."
)


def _get_weather_data(location: str) -> str:
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F",
    }
    return weather_data.get(location, f"Weather data not available for {location}")


def _calculator(operation: str, params: str) -> str:
    try:
        args = json.loads(params)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON parameters: {exc}"})

    try:
        if operation == "add":
            result = args["a"] + args["b"]
        elif operation == "subtract":
            result = args["a"] - args["b"]
        elif operation == "multiply":
            result = args["a"] * args["b"]
        elif operation == "divide":
            if args["b"] == 0:
                return json.dumps({"error": "Division by zero"})
            result = args["a"] / args["b"]
        elif operation == "power":
            result = math.pow(args["base"], args["exponent"])
        elif operation == "sqrt":
            if args["value"] < 0:
                return json.dumps({"error": "Cannot take square root of negative number"})
            result = math.sqrt(args["value"])
        elif operation == "sin":
            angle = args["angle"]
            if args.get("unit", "radians") == "degrees":
                angle = math.radians(angle)
            result = math.sin(angle)
        elif operation == "cos":
            angle = args["angle"]
            if args.get("unit", "radians") == "degrees":
                angle = math.radians(angle)
            result = math.cos(angle)
        elif operation == "tan":
            angle = args["angle"]
            if args.get("unit", "radians") == "degrees":
                angle = math.radians(angle)
            result = math.tan(angle)
        else:
            return json.dumps({"error": f"Unknown operation: {operation}"})

        return json.dumps(
            {
                "operation": operation,
                "params": args,
                "result": round(result, 10) if isinstance(result, float) else result,
            }
        )
    except KeyError as exc:
        return json.dumps({"error": f"Missing required parameter: {exc}"})
    except Exception as exc:  # pragma: no cover - defensive guard
        return json.dumps({"error": f"Calculation error: {exc}"})


def _count_letter_occurrences(text: str, letter: str, case_sensitive: bool = False) -> str:
    if not letter or len(letter) != 1:
        return json.dumps({"error": "Parameter 'letter' must be exactly one character"})

    search_text = text if case_sensitive else text.lower()
    search_letter = letter if case_sensitive else letter.lower()
    count = search_text.count(search_letter)

    return json.dumps(
        {
            "text": text,
            "letter": letter,
            "case_sensitive": case_sensitive,
            "count": count,
        }
    )


def _text_insights(text: str) -> str:
    words = [word for word in text.split() if word.strip()]
    longest_word = max(words, key=len) if words else ""

    return json.dumps(
        {
            "text": text,
            "character_count": len(text),
            "word_count": len(words),
            "longest_word": longest_word,
        }
    )


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return _get_weather_data(location)


@tool
def calculator(operation: str, params: str) -> str:
    """Calculator for arithmetic and trig operations. Params must be a JSON string."""
    return _calculator(operation, params)


@tool
def count_letter_occurrences(text: str, letter: str, case_sensitive: bool = False) -> str:
    """Count occurrences of a single character in text."""
    return _count_letter_occurrences(text=text, letter=letter, case_sensitive=case_sensitive)


@tool
def text_insights(text: str) -> str:
    """Return text stats: character count, word count, and longest word."""
    return _text_insights(text=text)


TOOLS = [get_weather, calculator, count_letter_occurrences, text_insights]


# =============================================================================
# STATE
# =============================================================================


class ConversationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_inputs: list[str]
    current_input: str
    turn_index: int
    finished: bool
    traces: list[str]


# =============================================================================
# GRAPH NODES
# =============================================================================


def dequeue_input_node(state: ConversationState) -> dict:
    queue = list(state.get("remaining_inputs", []))
    if not queue:
        return {"finished": True}

    next_input = queue.pop(0)
    turn_index = state.get("turn_index", 0) + 1
    trace = f"[TURN {turn_index}] User: {next_input}"
    print(trace)

    traces = list(state.get("traces", []))
    traces.append(trace)

    return {
        "remaining_inputs": queue,
        "current_input": next_input,
        "turn_index": turn_index,
        "finished": False,
        "traces": traces,
    }


def add_user_message_node(state: ConversationState) -> dict:
    return {"messages": [HumanMessage(content=state["current_input"])]}


def call_model_node(state: ConversationState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_tools = llm.bind_tools(TOOLS)
    response = model_with_tools.invoke(state["messages"])

    if getattr(response, "tool_calls", None):
        names = [tool_call["name"] for tool_call in response.tool_calls]
        print(f"  Model requested tools: {names}")
    else:
        preview = (response.content or "").replace("\n", " ")
        print(f"  Model final response preview: {preview[:100]}")

    return {"messages": [response]}


def summarize_turn_node(state: ConversationState) -> dict:
    last_ai = None
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and message.content:
            last_ai = message
            break

    tool_lines: list[str] = []
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage):
            tool_lines.append(f"  Tool result ({message.name}): {message.content}")
    tool_lines.reverse()

    traces = list(state.get("traces", []))
    for line in tool_lines:
        print(line)
        traces.append(line)

    assistant_line = f"Assistant: {last_ai.content if last_ai else '(no assistant content)'}"
    print(assistant_line)
    traces.append(assistant_line)

    return {"traces": traces, "current_input": ""}


# =============================================================================
# ROUTERS
# =============================================================================


def route_after_dequeue(state: ConversationState) -> Literal["add_user_message", "__end__"]:
    return END if state.get("finished", False) else "add_user_message"


def route_after_model(state: ConversationState) -> Literal["tools", "summarize_turn"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"
    return "summarize_turn"


def route_after_summary(state: ConversationState) -> Literal["dequeue_input", "__end__"]:
    return "dequeue_input" if state.get("remaining_inputs") else END


# =============================================================================
# GRAPH FACTORY + CHECKPOINTER
# =============================================================================


def create_checkpointer():
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver.from_conn_string("topic3/task5_checkpoints.sqlite"), "sqlite"
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver(), "memory"


def build_graph(checkpointer, interrupt_after: list[str] | None = None):
    workflow = StateGraph(ConversationState)

    workflow.add_node("dequeue_input", dequeue_input_node)
    workflow.add_node("add_user_message", add_user_message_node)
    workflow.add_node("call_model", call_model_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("summarize_turn", summarize_turn_node)

    workflow.add_edge(START, "dequeue_input")
    workflow.add_conditional_edges(
        "dequeue_input",
        route_after_dequeue,
        {"add_user_message": "add_user_message", END: END},
    )
    workflow.add_edge("add_user_message", "call_model")
    workflow.add_conditional_edges(
        "call_model",
        route_after_model,
        {"tools": "tools", "summarize_turn": "summarize_turn"},
    )
    workflow.add_edge("tools", "call_model")
    workflow.add_conditional_edges(
        "summarize_turn",
        route_after_summary,
        {"dequeue_input": "dequeue_input", END: END},
    )

    compile_kwargs = {"checkpointer": checkpointer}
    if interrupt_after:
        compile_kwargs["interrupt_after"] = interrupt_after
    return workflow.compile(**compile_kwargs)


def initial_state(user_inputs: list[str]) -> ConversationState:
    return {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "remaining_inputs": list(user_inputs),
        "current_input": "",
        "turn_index": 0,
        "finished": False,
        "traces": [],
    }


# =============================================================================
# DEMOS
# =============================================================================


def save_mermaid_text(app, output_path: str) -> None:
    mermaid_text = app.get_graph().draw_mermaid()
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(mermaid_text)
    print(f"Saved Mermaid graph text to: {output_path}")


def run_full_conversation_demo(app) -> dict:
    print("\n" + "=" * 80)
    print("DEMO 1: Single long conversation (no Python agent loop)")
    print("=" * 80)
    inputs = [
        "What's the weather in Tokyo?",
        "How many s are in Mississippi riverboats?",
        "Now count i in the same phrase and compute the sine in degrees of i-count minus s-count.",
        "Run text_insights on the same phrase and tell me the longest word.",
        "Using that longest word length, calculate 2 to that power.",
    ]
    config = {"configurable": {"thread_id": "task5-full-demo"}}
    final_state = app.invoke(initial_state(inputs), config=config)
    return final_state


def run_recovery_demo(checkpointer) -> dict:
    print("\n" + "=" * 80)
    print("DEMO 2: Checkpoint + recovery")
    print("=" * 80)
    inputs = [
        "What's the weather in London?",
        "How many s are in Mississippi riverboats?",
        "Use the same phrase and count i as well.",
        "Compute sine of 0 degrees.",
    ]
    config = {"configurable": {"thread_id": "task5-recovery-demo"}}

    # First run interrupts after one completed turn to simulate interruption.
    interrupted_app = build_graph(checkpointer, interrupt_after=["summarize_turn"])
    interrupted_state = interrupted_app.invoke(initial_state(inputs), config=config)
    print(
        f"Interrupted. Remaining queued inputs: {len(interrupted_state.get('remaining_inputs', []))}"
    )

    # Recovery: rebuild graph normally and resume same thread_id from checkpoint.
    resumed_app = build_graph(checkpointer)
    try:
        recovered_state = resumed_app.invoke(None, config=config)
    except Exception:
        recovered_state = resumed_app.invoke({}, config=config)

    print(
        f"Recovered and completed. Remaining queued inputs: {len(recovered_state.get('remaining_inputs', []))}"
    )
    return recovered_state


def main():
    checkpointer, checkpointer_name = create_checkpointer()
    print(f"Using checkpointer backend: {checkpointer_name}")

    app = build_graph(checkpointer)
    save_mermaid_text(app, "topic3/task5_mermaid_diagram.mmd")

    full_state = run_full_conversation_demo(app)
    recovery_state = run_recovery_demo(checkpointer)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Full demo turns: {full_state.get('turn_index', 0)}")
    print(f"Recovery demo turns: {recovery_state.get('turn_index', 0)}")
    print("Task 5 run completed.")


if __name__ == "__main__":
    main()
