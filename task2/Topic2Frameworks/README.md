# Topic 2 Frameworks — LangGraph Agent Progression

Each file builds on the previous one, adding one feature at a time.
All programs use `meta-llama/Llama-3.2-1B-Instruct` as the base model
and run on Apple Silicon (MPS) or CPU; CUDA is auto-detected.

## Table of Contents

| File | Task | Key Feature Added |
|---|---|---|
| [`task2_1_verbose_quiet.py`](#task-1-verbosquiet-tracing) | 1 | `verbose` / `quiet` commands toggle `[TRACE]` output |
| [`task2_2_empty_input.py`](#task-2-empty-input-handling) | 2 | 3-way conditional branch — empty input loops back without calling LLM |
| [`task2_3_parallel_models.py`](#task-3-parallel-models) | 3 | Fan-out to Llama + Qwen in parallel, `collect_responses` merges results |
| [`task2_4_hey_qwen_routing.py`](#task-4-hey-qwen-routing) | 4 | `"Hey Qwen"` prefix routes to Qwen; otherwise Llama |
| [`task2_5_chat_history.py`](#task-5-chat-history) | 5 | Full conversation history via `Annotated[list, operator.add]` + Message API (Llama only) |
| [`task2_6_multi_agent_chat.py`](#task-6-multi-agent-chat) | 6 | 3-entity history (Human / Llama / Qwen) with role remapping per model |
| [`task2_7_checkpointing.py`](#task-7-crash-recovery) | 7 | `SqliteSaver` checkpointing — kill and restart with no conversation loss |

### Session Outputs

| File | Description |
|---|---|
| `session_task2_2_empty_input.txt` | Terminal output showing empty-input behavior and the original LLM hallucination |
| `session_task2_6_multi_agent.txt` | Example ice cream conversation across Llama and Qwen |
| `session_task2_7_checkpointing.txt` | Crash-and-resume demonstration |

---

## Task 1: Verbose/Quiet Tracing

**File:** `task2_1_verbose_quiet.py`

Added `verbose: bool` to `AgentState`. Typing `verbose` enables `[TRACE]` prints at the start and end of every node; `quiet` suppresses them. The `verbose`/`quiet` inputs are consumed by `get_user_input` without being forwarded to the LLM.

---

## Task 2: Empty Input Handling

**File:** `task2_2_empty_input.py`

### What happens with empty input (original code)?

When an empty string is passed to Llama-3.2-1B, the model generates text from an unconstrained distribution. On the first empty input it often fabricates a complete fake conversation ("User: What's the weather? Assistant: It's sunny..."). On the second empty input the output is completely different — sometimes whitespace, sometimes a partial sentence.

**This reveals** that small instruction-tuned models have no well-defined behavior for empty prompts. They lack the reasoning capacity to recognize "I was given nothing to respond to" and simply sample from whatever distribution the empty prompt implies.

### Fix: 3-way conditional branch

`route_after_input` now returns one of three values:
- `END` — user typed quit
- `"get_user_input"` — empty input, loop back *(new edge)*
- `"call_llm"` — valid input, proceed

This is the LangGraph-idiomatic approach: encode control flow in the graph topology rather than an imperative `while` loop.

---

## Task 3: Parallel Models

**File:** `task2_3_parallel_models.py`

The router returns a **list** `["call_llama", "call_qwen"]` — LangGraph interprets this as a parallel fan-out. Both nodes run concurrently and `collect_responses` waits for both before printing.

---

## Task 4: Hey Qwen Routing

**File:** `task2_4_hey_qwen_routing.py`

A `route_to_models` node checks for the `"Hey Qwen"` prefix (case-insensitive, regex), strips it, and sets a routing flag. The conditional edge then sends to `call_qwen` or `call_llama`. Only one model runs per turn.

---

## Task 5: Chat History (Llama only)

**File:** `task2_5_chat_history.py`

### Key implementation

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # accumulates across turns
    ...
```

`operator.add` tells LangGraph to **append** the list returned by a node to the existing `messages` rather than replacing it. `call_llm` sends:

```
[SystemMessage(system_prompt)] + history + [HumanMessage(current_input)]
```

and returns `{"messages": [HumanMessage(input), AIMessage(response)]}` — two new messages get appended automatically.

Qwen is disabled; this version tests history in isolation.

---

## Task 6: Multi-Agent Chat with Shared History

**File:** `task2_6_multi_agent_chat.py`

### The 3-entity problem

LangChain's Message API only supports roles: `system / human / ai / tool`. With three participants (Human, Llama, Qwen), we encode speaker identity in the message **content** using prefixes and store all messages as `HumanMessage`:

```
HumanMessage(content="Human: What is the best ice cream flavor?")
HumanMessage(content="Llama: There is no one best flavor...")
HumanMessage(content="Qwen: No way, chocolate is the best!")
```

When calling a model, `format_history_for(model_name, history)` remaps roles:

| Stored content | Called for Llama | Called for Qwen |
|---|---|---|
| `"Llama: ..."` | `AIMessage` (assistant) | `HumanMessage` (user) |
| `"Qwen: ..."`  | `HumanMessage` (user) | `AIMessage` (assistant) |
| `"Human: ..."` | `HumanMessage` (user) | `HumanMessage` (user) |

This matches the format described in the task specification exactly.

---

## Task 7: Crash Recovery

**File:** `task2_7_checkpointing.py`

### How it works

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("task2_7_checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "main-conversation"}}
    graph.invoke(initial_state, config=config)
```

LangGraph saves a snapshot of `AgentState` to SQLite **after every node completes**. On restart with the same `thread_id`, it loads the last checkpoint and resumes from there — the conversation history, verbose flag, and all state survive the crash.

### Testing crash recovery

```bash
python task2_7_checkpointing.py   # start, have a conversation, Ctrl+C
python task2_7_checkpointing.py   # resumes — history fully intact
python task2_7_checkpointing.py --reset   # clear checkpoint and start fresh
```

See `session_task2_7_checkpointing.txt` for a recorded example.

### Discussion (from class)

LangGraph checkpointing is valuable for long-running agentic workflows (web search, code execution, multi-step planning) where a crash in step 15 of 20 would otherwise lose all intermediate results. The checkpoint granularity (per-node) means at most one node's work is repeated on recovery. For nodes with side effects (emails sent, files written), idempotency must be considered separately.

---

## Setup

```bash
conda activate llm-class
pip install langgraph langchain-huggingface transformers torch datasets accelerate
huggingface-cli login
```
