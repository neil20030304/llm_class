# =============================================================================
# exercise1.py
# Multi-turn image chat agent using LangGraph + Ollama (LLaVA)
#
# Usage:
#   python exercise1.py
#   (requires: pip install ollama langgraph langchain-core Pillow grandalf)
#   (requires: ollama pull llava)
#
# Architecture (LangGraph style, same patterns as task2/langraph_llama_agent.py):
#
#   START
#     → upload_image        (one-shot setup: prompt for file, resize for speed)
#     → get_user_input      (prompt for text question each turn)
#         ↓ [route_after_input]
#     → END                 (quit / exit / q)
#     → get_user_input      (empty input  → loop back)
#     → call_vlm            (valid input  → call LLaVA)
#         → print_response
#         → get_user_input  (loop for next turn)
#
# Context management:
#   - state["messages"] accumulates the full conversation as a list of
#     ollama-format dicts: {'role': ..., 'content': ..., ['images': ...]}
#   - The image is attached ONLY to the first user message.  Subsequent
#     turns are text-only; the model keeps the image in its context window.
#   - operator.add (via Annotated) appends new messages without overwriting
#     history, identical to the pattern in task2.
#
# Reducing slowness:
#   - Set MAX_IMAGE_SIDE to resize the image before sending to the model.
#     Smaller images (e.g. 384 px) process faster; default is 512 px.
# =============================================================================

import os
import operator
from pathlib import Path
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, START, END


# =============================================================================
# CONFIGURATION
# =============================================================================
OLLAMA_MODEL = "llava"      # Vision model served by Ollama
MAX_IMAGE_SIDE = 512        # Resize longest side to this many pixels.
                            # Lower (e.g. 384) for speed; higher for quality.


# =============================================================================
# STATE DEFINITION
# =============================================================================
class AgentState(TypedDict):
    """
    State object that flows through every node in the graph.

    Fields:
        messages    - Accumulated conversation history.
                      Each entry is an ollama-format dict:
                        {'role': 'user'|'assistant', 'content': str,
                         optionally 'images': [path_or_b64]}
                      The Annotated[list, operator.add] annotation tells
                      LangGraph to *append* new messages rather than replace
                      the whole list (same pattern as task2's BaseMessage list).
        image_path  - Path to the (possibly resized) image file on disk.
                      Set once by upload_image, read by call_vlm.
        user_input  - Raw text entered by the user in the current turn.
        should_exit - True when the user types quit / exit / q.
        verbose     - If True, print [TRACE] node-level tracing to stdout.
    """
    messages:    Annotated[List[dict], operator.add]
    image_path:  str
    user_input:  str
    should_exit: bool
    verbose:     bool


# =============================================================================
# IMAGE HELPER
# =============================================================================
def resize_image(src_path: str, max_side: int = MAX_IMAGE_SIDE) -> str:
    """
    Resize the image so its longest dimension is at most max_side pixels.
    Preserves aspect ratio.  Saves the result next to the original with a
    '_resized' suffix.  Returns the new path (or src_path if no resize needed).

    Why resize?  VLMs process images as patches; a 3000-pixel photo generates
    hundreds of patch tokens and slows inference dramatically.  512 px is a
    reasonable quality/speed balance for most discussion tasks.

    Requires: pip install Pillow
    """
    try:
        from PIL import Image
    except ImportError:
        print("⚠  Pillow not installed – skipping resize.  Run: pip install Pillow")
        return src_path

    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    if max(w, h) <= max_side:
        print(f"  Image is {w}×{h} px – no resize needed.")
        return src_path

    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    src = Path(src_path)
    out_path = str(src.parent / f"{src.stem}_resized{src.suffix}")
    img.save(out_path)
    print(f"  Resized {w}×{h} → {new_w}×{new_h} px, saved as: {out_path}")
    return out_path


# =============================================================================
# GRAPH FACTORY
# =============================================================================
def create_graph():
    """
    Build and return the compiled LangGraph.

    All nodes are defined as closures inside this function, following the
    same pattern as task2/langraph_llama_agent.py.  The ollama import is
    done once at the top of create_graph() so every node can use it.
    """
    import ollama  # pip install ollama  (and: ollama pull llava)

    # =========================================================================
    # NODE 1: upload_image
    # =========================================================================
    # Runs once at startup (START → upload_image).
    # Prompts the user for an image path, validates it, resizes it for speed,
    # and stores the path in state.  Does NOT yet add any messages.
    #
    # State written:
    #   image_path ← path of the (possibly resized) image file
    #   messages   ← [] (initialise to empty so Annotated append starts fresh)
    def upload_image(state: AgentState) -> dict:
        """
        One-shot setup node: pick an image, validate, resize, store path.

        Reads state:  verbose
        Writes state: image_path, messages (resets to [])
        """
        verbose = state.get("verbose", False)
        if verbose:
            print("[TRACE] Entering node: upload_image")

        print("\n" + "=" * 60)
        print("  IMAGE CHAT AGENT  (LLaVA via Ollama + LangGraph)")
        print("=" * 60)
        print("  Commands: 'quit' or 'q' to exit | 'verbose' / 'quiet'")
        print("=" * 60)

        while True:
            print("\nEnter the path to your image file:")
            print("> ", end="", flush=True)
            path = input().strip()

            if not path:
                print("⚠  No path entered. Please try again.")
                continue

            if not os.path.isfile(path):
                print(f"⚠  File not found: {path!r}")
                continue

            # Resize for faster VLM inference
            resized = resize_image(path, MAX_IMAGE_SIDE)
            print(f"✓  Image ready: {resized}")

            if verbose:
                print(f"[TRACE] Exiting upload_image, image_path={resized!r}")

            # Return image_path and reset messages to an empty list.
            # Returning messages=[] is harmless: operator.add([], []) == [].
            return {
                "image_path": resized,
                "messages": [],
            }

    # =========================================================================
    # NODE 2: get_user_input
    # =========================================================================
    # Runs at the start of every conversation turn.
    # Handles special commands: quit, verbose, quiet.
    # Stores the text in state["user_input"].
    #
    # State written:
    #   user_input  ← user's text (or "" for commands that don't go to the LLM)
    #   should_exit ← True if the user wants to quit
    #   verbose     ← may be toggled by 'verbose' / 'quiet' commands
    def get_user_input(state: AgentState) -> dict:
        """
        Node that prompts the user for a question each turn.

        Reads state:  messages (to show turn number), verbose
        Writes state: user_input, should_exit, verbose
        """
        verbose = state.get("verbose", False)
        if verbose:
            print("[TRACE] Entering node: get_user_input")

        # Count completed turns to show a helpful turn number
        n_turns = sum(1 for m in state.get("messages", []) if m.get("role") == "user")

        print("\n" + "-" * 60)
        if n_turns == 0:
            print("Ask your first question about the image  (or 'quit' to exit):")
        else:
            print(f"Turn {n_turns + 1} – ask another question  (or 'quit' to exit):")
        print("> ", end="", flush=True)

        user_input = input().strip()

        # --- Command: quit ---
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True}

        # --- Command: verbose ---
        if user_input.lower() == "verbose":
            print("Verbose mode ON – [TRACE] messages will appear.")
            return {"user_input": "", "should_exit": False, "verbose": True}

        # --- Command: quiet ---
        if user_input.lower() == "quiet":
            print("Quiet mode ON.")
            return {"user_input": "", "should_exit": False, "verbose": False}

        if verbose:
            print(f"[TRACE] user_input={user_input!r}")
            print("[TRACE] Exiting node: get_user_input")

        return {"user_input": user_input, "should_exit": False}

    # =========================================================================
    # ROUTER: route_after_input
    # =========================================================================
    # Three-way branch (same pattern as task2's route_after_input):
    #   END             – user typed quit
    #   "get_user_input" – user typed nothing (loop back for another prompt)
    #   "call_vlm"       – valid text → proceed to the vision model
    def route_after_input(state: AgentState) -> str:
        """
        Examines state:
            should_exit  → END
            empty input  → "get_user_input"  (loop)
            valid input  → "call_vlm"
        """
        verbose = state.get("verbose", False)
        if verbose:
            print(f"[TRACE] Router route_after_input – "
                  f"should_exit={state.get('should_exit')}, "
                  f"user_input={state.get('user_input')!r}")

        if state.get("should_exit", False):
            return END

        if not state.get("user_input", "").strip():
            print("⚠  Empty input. Please type a question.")
            return "get_user_input"

        return "call_vlm"

    # =========================================================================
    # NODE 3: call_vlm
    # =========================================================================
    # Core node: builds the full message list for ollama and calls LLaVA.
    #
    # Context management strategy:
    #   - On the FIRST user turn, include the image in the message via
    #     the 'images' key.  LLaVA will encode the image into its context.
    #   - On SUBSEQUENT turns, send text-only messages.  The model retains
    #     the image encoding in its KV-cache / context window as long as the
    #     full history is replayed.  This avoids re-encoding the image bytes
    #     on every turn, which would slow things down significantly.
    #
    # State written:
    #   messages ← [new_user_msg, new_assistant_msg]  (operator.add appends)
    def call_vlm(state: AgentState) -> dict:
        """
        Build the conversation history and call the VLM via ollama.

        Reads state:  messages, image_path, user_input, verbose
        Writes state: messages (two new entries appended via operator.add)
        """
        verbose = state.get("verbose", False)
        user_input = state["user_input"]
        image_path = state["image_path"]
        history    = list(state.get("messages", []))

        if verbose:
            print("[TRACE] Entering node: call_vlm")
            print(f"[TRACE] History length before this turn: {len(history)}")
            print(f"[TRACE] user_input={user_input!r}")

        # Is this the first user turn?  Attach the image only then.
        first_turn = not any(m.get("role") == "user" for m in history)

        if first_turn:
            # First turn: include the image so the model can see it
            new_user_msg = {
                "role": "user",
                "content": user_input,
                "images": [image_path],
            }
            if verbose:
                print(f"[TRACE] First turn – attaching image: {image_path!r}")
        else:
            # Follow-up turn: text only (image already in model's context)
            new_user_msg = {
                "role": "user",
                "content": user_input,
            }
            if verbose:
                print("[TRACE] Follow-up turn – text only (image stays in context)")

        # Full message list to send to ollama (history + this new message)
        send_messages = history + [new_user_msg]

        if verbose:
            print(f"[TRACE] Sending {len(send_messages)} message(s) to {OLLAMA_MODEL}")

        print("\n[Thinking…]", flush=True)
        response = ollama.chat(model=OLLAMA_MODEL, messages=send_messages)
        assistant_content = response["message"]["content"]

        if verbose:
            print(f"[TRACE] Response received ({len(assistant_content)} chars)")
            print("[TRACE] Exiting node: call_vlm")

        # Both messages are appended to state["messages"] via operator.add
        new_assistant_msg = {"role": "assistant", "content": assistant_content}
        return {
            "messages": [new_user_msg, new_assistant_msg],
        }

    # =========================================================================
    # NODE 4: print_response
    # =========================================================================
    # Reads the most recent assistant message from history and prints it.
    # Does not modify state (returns {}).
    def print_response(state: AgentState) -> dict:
        """
        Print the latest assistant reply to stdout.

        Reads state:  messages, verbose
        Writes state: nothing
        """
        verbose = state.get("verbose", False)
        if verbose:
            print("[TRACE] Entering node: print_response")

        messages = state.get("messages", [])
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

        if assistant_msgs:
            reply = assistant_msgs[-1]["content"]
            print("\n" + "=" * 60)
            print(f"  LLaVA:")
            print("=" * 60)
            print(reply)
            print("=" * 60)
        else:
            print("(no response received)")

        if verbose:
            print("[TRACE] Exiting node: print_response")

        return {}   # No state changes

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("upload_image",   upload_image)
    builder.add_node("get_user_input", get_user_input)
    builder.add_node("call_vlm",       call_vlm)
    builder.add_node("print_response", print_response)

    # Edges
    # 1. START → upload_image  (one-shot setup)
    builder.add_edge(START, "upload_image")

    # 2. upload_image → get_user_input  (move to first prompt after setup)
    builder.add_edge("upload_image", "get_user_input")

    # 3. get_user_input → [conditional] → END | get_user_input | call_vlm
    builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            END:              END,
            "get_user_input": "get_user_input",
            "call_vlm":       "call_vlm",
        },
    )

    # 4. call_vlm → print_response  (always print after calling the model)
    builder.add_edge("call_vlm", "print_response")

    # 5. print_response → get_user_input  (loop back for the next turn)
    builder.add_edge("print_response", "get_user_input")

    return builder.compile()


# =============================================================================
# GRAPH VISUALISATION
# =============================================================================
def save_graph_image(graph, filename="vlm_chat_graph.png"):
    """
    Save a Mermaid PNG of the graph structure.
    Requires: pip install grandalf
    """
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph diagram saved to: {filename}")
    except Exception as e:
        print(f"Could not save graph diagram: {e}")
        print("  Install with: pip install grandalf")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """
    Entry point.
    1. Build the LangGraph.
    2. Save a visualisation of the graph.
    3. Invoke the graph with the initial state (the graph loops internally).
    """
    print("Building LangGraph VLM chat agent…")
    graph = create_graph()
    save_graph_image(graph)

    # Initial state – upload_image fills in image_path and resets messages.
    initial_state: AgentState = {
        "messages":    [],
        "image_path":  "",
        "user_input":  "",
        "should_exit": False,
        "verbose":     False,
    }

    # Single invoke – the graph loops internally until the user types 'quit'.
    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
