# langgraph_simple_agent.py
# Program demonstrates use of LangGraph for a very simple agent.
# It writes to stdout and asks the user to enter a line of text through stdin.
# It passes the line to the LLM llama-3.2-1B-Instruct, then prints the
# what the LLM returns as text to stdout.
# The LLM should use Cuda if available, if not then if mps is available then use that,
# otherwise use cpu.
# After the LangGraph graph is created but before it executes, the program
# uses the Mermaid library to write a image of the graph to the file lg_graph.png
# The program gets the LLM llama-3.2-1B-Instruct from Hugging Face and wraps
# it for LangChain using HuggingFacePipeline.
# The code is commented in detail so a reader can understand each step.

# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Determine the best available device for inference
# Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
def get_device():
    """
    Detect and return the best available compute device.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"

# =============================================================================
# STATE DEFINITION
# =============================================================================
# The state is a TypedDict that flows through all nodes in the graph.
# Each node can read from and write to specific fields in the state.
# LangGraph automatically merges the returned dict from each node into the state.

class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - messages: List of messages that accumulates conversation history
      Each message is tagged with the speaker (Human, Llama, or Qwen) in the content
    - should_exit: Boolean flag indicating if user wants to quit
    - verbose: Boolean flag indicating if tracing should be printed
    - last_model: String indicating which model was used last ("llama" or "qwen")

    State Flow:
    1. Initial state: messages list is empty
    2. After get_user_input: HumanMessage added with "Human: " prefix
    3. After call_llm: AIMessage added with "Llama: " prefix
    4. After call_qwen: AIMessage added with "Qwen: " prefix
    5. After print_response: state unchanged (node only reads, doesn't write)

    The graph loops continuously:
        get_user_input -> [conditional] -> call_llm OR call_qwen -> print_response -> get_user_input
                              |
                              +-> END (if user wants to quit)
    
    Chat history format:
    - Human messages: "Human: <content>" as HumanMessage
    - Llama responses: "Llama: <content>" as AIMessage
    - Qwen responses: "Qwen: <content>" as AIMessage
    """
    user_input: str
    should_exit: bool
    llm_response: str
    qwen_response: str
    verbose: bool

def create_llm(model_id="meta-llama/Llama-3.2-1B-Instruct", model_name="Llama"):
    """
    Create and configure an LLM using HuggingFace's transformers library.
    Downloads the specified model from HuggingFace Hub and wraps it
    for use with LangChain via HuggingFacePipeline.
    
    Args:
        model_id: HuggingFace model identifier
        model_name: Display name for the model
    """
    # Get the optimal device for inference
    device = get_device()

    print(f"Loading {model_name} model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    # Load the tokenizer - converts text to tokens the model understands
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the model itself with appropriate settings for the device
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    # Move model to MPS device if using Apple Silicon
    if device == "mps":
        model = model.to(device)

    # Create a text generation pipeline that combines model and tokenizer
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  # Maximum tokens to generate in response
        do_sample=True,      # Enable sampling for varied responses
        temperature=0.7,     # Controls randomness (lower = more deterministic)
        top_p=0.95,          # Nucleus sampling parameter
        pad_token_id=tokenizer.eos_token_id,  # Suppress pad_token_id warning
    )

    # Wrap the HuggingFace pipeline for use with LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    print(f"{model_name} model loaded successfully!")
    return llm

def create_llama_llm():
    """Create the Llama model."""
    return create_llm("meta-llama/Llama-3.2-1B-Instruct", "Llama")

def create_qwen_llm():
    """Create the Qwen model (tiny version)."""
    return create_llm("Qwen/Qwen2.5-0.5B-Instruct", "Qwen")

def create_graph(llama_llm, qwen_llm):
    """
    Create the LangGraph state graph with three separate nodes:
    1. get_user_input: Reads input from stdin
    2. call_llm: Sends input to the LLM and gets response
    3. print_response: Prints the LLM's response to stdout

    Graph structure with conditional routing and internal loop:
        START -> get_user_input -> [conditional] -> call_llm -> print_response -+
                       ^                 |                                       |
                       |                 +-> END (if user wants to quit)         |
                       |                                                         |
                       +---------------------------------------------------------+

    The graph runs continuously until the user types 'quit', 'exit', or 'q'.
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    # This node reads a line of text from stdin and updates the state.
    # State changes:
    #   - user_input: Set to the text entered by the user
    #   - should_exit: Set to True if user typed quit/exit/q, False otherwise
    #   - verbose: Set based on "verbose" or "quiet" commands
    def get_user_input(state: AgentState) -> dict:
        """
        Node that prompts the user for input via stdin.

        Reads state: verbose (to determine if tracing should be printed)
        Updates state:
            - user_input: The raw text entered by the user
            - should_exit: True if user wants to quit, False otherwise
            - verbose: True if user typed "verbose", False if "quiet"
        """
        verbose = state.get("verbose", False)
        
        if verbose:
            print("[TRACE] Entering node: get_user_input")
            print(f"[TRACE] Current state - verbose: {verbose}, should_exit: {state.get('should_exit', False)}")
        
        # Display banner before each prompt
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()
        
        if verbose:
            print(f"[TRACE] Received user input: '{user_input}'")

        # Check if user wants to exit
        if user_input.lower() in ['quit', 'exit', 'q']:
            if verbose:
                print("[TRACE] User requested exit")
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True,        # Signal to exit the graph
                "verbose": verbose          # Preserve verbose setting
            }

        # Check for verbose/quiet commands
        if user_input.lower() == "verbose":
            if verbose:
                print("[TRACE] Setting verbose mode to True")
            print("Verbose mode enabled. Tracing information will be printed.")
            return {
                "user_input": "",          # Don't send "verbose" to LLM
                "should_exit": False,
                "verbose": True
            }
        elif user_input.lower() == "quiet":
            if verbose:
                print("[TRACE] Setting verbose mode to False")
            print("Quiet mode enabled. Tracing information will not be printed.")
            return {
                "user_input": "",          # Don't send "quiet" to LLM
                "should_exit": False,
                "verbose": False
            }

        if verbose:
            print(f"[TRACE] Exiting node: get_user_input, returning user_input: '{user_input}'")

        # Any other input - continue to LLM (or loop back if empty)
        return {
            "user_input": user_input,
            "should_exit": False,           # Signal to proceed to router
            "verbose": verbose              # Preserve verbose setting
        }

    # =========================================================================
    # NODE 2: call_llm (Llama)
    # =========================================================================
    # This node takes the user input from state, sends it to the Llama LLM,
    # and stores the response back in state.
    # State changes:
    #   - user_input: Unchanged (read only)
    #   - should_exit: Unchanged (read only)
    #   - verbose: Unchanged (read only)
    #   - llm_response: Set to the Llama LLM's generated response
    def call_llm(state: AgentState) -> dict:
        """
        Node that invokes the Llama LLM with formatted conversation history.

        Reads state:
            - messages: The conversation history
            - verbose: Whether to print tracing information
        Updates state:
            - messages: Adds AIMessage with "Llama: " prefix
            - last_model: Set to "llama"
        
        History formatting for Llama:
        - Human messages: remain as user role with "Human: " prefix
        - Qwen messages: converted to assistant role with "Qwen: " prefix
        - Llama messages: converted to assistant role with "Llama: " prefix
        """
        verbose = state.get("verbose", False)
        messages = state.get("messages", [])
        
        if verbose:
            print("[TRACE] Entering node: call_llm (Llama)")
            print(f"[TRACE] Message history length: {len(messages)}")
        
        if not messages:
            if verbose:
                print("[TRACE] No messages in history, skipping LLM call")
            return {}

        # System prompt for Llama
        system_prompt = "You are Llama, participating in a conversation with a Human and Qwen. When Qwen speaks, it appears as an assistant message. Respond naturally and helpfully."
        
        # Format history for Llama
        # Rules: Human stays as user, Qwen becomes assistant, Llama becomes assistant
        formatted_messages = [SystemMessage(content=system_prompt)]
        
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            
            if isinstance(msg, HumanMessage):
                # Human messages stay as user role
                formatted_messages.append(HumanMessage(content=content))
            elif isinstance(msg, AIMessage):
                # Check if it's from Qwen or Llama by content prefix
                if content.startswith("Qwen: "):
                    # Qwen messages become assistant role
                    formatted_messages.append(AIMessage(content=content))
                elif content.startswith("Llama: "):
                    # Previous Llama messages become assistant role
                    formatted_messages.append(AIMessage(content=content))
                else:
                    # Fallback: treat as assistant
                    formatted_messages.append(AIMessage(content=content))
        
        if verbose:
            print(f"[TRACE] Formatted {len(formatted_messages)} messages for Llama")
            for i, msg in enumerate(formatted_messages):
                msg_type = type(msg).__name__
                content_preview = msg.content[:50] if hasattr(msg, 'content') else str(msg)[:50]
                print(f"[TRACE] Message {i}: {msg_type} - {content_preview}...")

        if verbose:
            print("[TRACE] Invoking Llama LLM with formatted history...")

        # Invoke the Llama LLM with formatted message history
        response = llama_llm.invoke(formatted_messages)
        
        # Extract response content
        if isinstance(response, BaseMessage):
            response_content = response.content
        elif isinstance(response, str):
            response_content = response
        else:
            response_content = str(response)
        
        if verbose:
            print(f"[TRACE] Llama LLM response received: '{response_content[:100]}...'")
            print("[TRACE] Exiting node: call_llm (Llama)")

        # Return only the field we're updating
        return {"llm_response": response}

    # =========================================================================
    # NODE 3: call_qwen
    # =========================================================================
    # This node takes the user input from state, sends it to the Qwen LLM,
    # and stores the response back in state.
    # State changes:
    #   - user_input: Unchanged (read only)
    #   - should_exit: Unchanged (read only)
    #   - verbose: Unchanged (read only)
    #   - qwen_response: Set to the Qwen LLM's generated response
    def call_qwen(state: AgentState) -> dict:
        """
        Node that invokes the Qwen LLM with the user's input.

        Reads state:
            - user_input: The text to send to the LLM
            - verbose: Whether to print tracing information
        Updates state:
            - qwen_response: The text generated by the Qwen LLM
        """
        verbose = state.get("verbose", False)
        user_input = state["user_input"]
        
        if verbose:
            print("[TRACE] Entering node: call_qwen (Qwen)")
            print(f"[TRACE] Current state - user_input: '{user_input}', verbose: {verbose}")

        # Format the prompt for the instruction-tuned model
        # Note: Empty input should never reach this node due to router logic
        prompt = f"User: {user_input}\nAssistant:"
        
        if verbose:
            print(f"[TRACE] Formatted prompt for Qwen: '{prompt}'")

        if verbose:
            print("[TRACE] Invoking Qwen LLM...")

        # Invoke the Qwen LLM and get the response
        response = qwen_llm.invoke(prompt)
        
        if verbose:
            print(f"[TRACE] Qwen LLM response received: '{response}'")
            print("[TRACE] Exiting node: call_qwen (Qwen)")

        # Return only the field we're updating
        return {"qwen_response": response}

    # =========================================================================
    # NODE 4: print_both_responses
    # =========================================================================
    # This node reads both LLM responses from state and prints them to stdout.
    # State changes:
    #   - No changes (this node only reads state, doesn't modify it)
    def print_both_responses(state: AgentState) -> dict:
        """
        Node that prints both LLM responses to stdout.

        Reads state:
            - llm_response: The Llama response to print
            - qwen_response: The Qwen response to print
            - verbose: Whether to print tracing information
        Updates state:
            - Nothing (returns empty dict, state unchanged)
        """
        verbose = state.get("verbose", False)
        
        if verbose:
            print("[TRACE] Entering node: print_both_responses")
            print(f"[TRACE] Current state - llm_response length: {len(state.get('llm_response', ''))}, qwen_response length: {len(state.get('qwen_response', ''))}")

        print("\n" + "=" * 50)
        print("MODEL RESPONSE")
        print("=" * 50)
        
        # Print the response from whichever model was used
        llama_response = state.get("llm_response", "")
        qwen_response = state.get("qwen_response", "")
        
        # Determine which model was used and print its response
        # Check Qwen first (if it has a response and is not the flag, Qwen was used)
        # The flag "__USE_QWEN__" is set by route_to_models but gets overwritten by call_qwen
        # So if qwen_response exists and is not the flag, Qwen was just used
        if qwen_response and qwen_response.strip() and qwen_response != "__USE_QWEN__":
            # Qwen was used (has actual response, not the flag)
            print("\n" + "-" * 50)
            print("Qwen Response:")
            print("-" * 50)
            print(qwen_response)
        elif llama_response and llama_response.strip():
            # Llama was used (has response)
            print("\n" + "-" * 50)
            print("Llama Response:")
            print("-" * 50)
            print(llama_response)
        else:
            # No response (shouldn't happen, but handle gracefully)
            print("\n(No response)")
            if verbose:
                print(f"[TRACE] Debug - llama_response: '{llama_response[:50] if llama_response else 'empty'}...'")
                print(f"[TRACE] Debug - qwen_response: '{qwen_response[:50] if qwen_response else 'empty'}...'")
        
        print("\n" + "=" * 50)

        if verbose:
            print("[TRACE] Exiting node: print_both_responses")

        # Return empty dict - no state updates from this node
        return {}

    # =========================================================================
    # NODE 5: route_to_models
    # =========================================================================
    # This node processes the input and removes the "Hey Qwen" prefix if present.
    # It stores which model to use in the qwen_response field temporarily (as a flag).
    # This is a workaround since we don't want to modify the state definition.
    def route_to_models(state: AgentState) -> dict:
        """
        Node that processes the input and removes the "Hey Qwen" prefix if present.
        It also stores which model to use (by setting qwen_response to a special value
        if Qwen should be used, empty string otherwise).

        Reads state:
            - user_input: The text entered by the user
            - verbose: Whether to print tracing information
        Updates state:
            - user_input: The text with "Hey Qwen" prefix removed if present
            - qwen_response: Set to "__USE_QWEN__" if input started with "Hey Qwen", "" otherwise
        """
        verbose = state.get("verbose", False)
        user_input = state.get("user_input", "")
        
        if verbose:
            print("[TRACE] Entering node: route_to_models")
            print(f"[TRACE] Current user_input: '{user_input}'")
        
        # Check if input starts with "Hey Qwen" (case-insensitive) BEFORE processing
      # Handle both "Hey Qwen" and "Hey Q wen" (with space) variations
        import re
        user_input_stripped = user_input.strip()
        user_input_lower = user_input_stripped.lower()
        
        # Check for "Hey Qwen" or "Hey Q wen" (with optional space between Q and wen)
        use_qwen = bool(re.match(r'^hey\s+q\s?wen', user_input_lower))
        
        # Process the input: remove "Hey Qwen" prefix if present
        processed_input = user_input_stripped
        if use_qwen:
            # Remove "Hey Qwen" or "Hey Q wen" prefix (case-insensitive, preserve rest of input)
            # Match "Hey" followed by optional space, "Q", optional space, "wen", then optional whitespace
            processed_input = re.sub(r'^hey\s+q\s?wen\s*', '', user_input_stripped, flags=re.IGNORECASE).strip()
            
            if verbose:
                print(f"[TRACE] Detected 'Hey Qwen' prefix")
                print(f"[TRACE] Original input: '{user_input_stripped}'")
                print(f"[TRACE] Processed input: '{processed_input}'")
        else:
            if verbose:
                print("[TRACE] No 'Hey Qwen' prefix detected, using input as-is")
        
        if verbose:
            print("[TRACE] Exiting node: route_to_models")
        
        # Debug output to show which model will be used
        if use_qwen:
            print(f"\n[INFO] Routing to Qwen model (input started with 'Hey Qwen')")
        else:
            print(f"\n[INFO] Routing to Llama model (default)")
        
        # Return updated user_input and flag indicating which model to use
        # We use qwen_response as a temporary flag (will be overwritten by actual response)
        # Also clear the unused model's response to avoid showing stale responses
        if use_qwen:
            # Clear Llama response since we're using Qwen
            return {
                "user_input": processed_input,
                "qwen_response": "__USE_QWEN__",
                "llm_response": ""  # Clear previous Llama response
            }
        else:
            # Clear Qwen response since we're using Llama
            return {
                "user_input": processed_input,
                "qwen_response": "",  # Clear previous Qwen response
                "llm_response": ""  # Will be set by call_llm, but clear any stale data
            }

    # =========================================================================
    # ROUTER FUNCTION: route_to_model
    # =========================================================================
    # This router determines which model to use based on the flag set by route_to_models.
    def route_to_model(state: AgentState) -> str:
        """
        Router function that determines which model to use based on the flag
        set by route_to_models.
        
        Examines state:
            - qwen_response: Checks if set to "__USE_QWEN__" (flag set by route_to_models)
            - verbose: Whether to print tracing information
            
        Returns:
            - "call_qwen": If qwen_response is "__USE_QWEN__" (input started with "Hey Qwen")
            - "call_llm": Otherwise (default to Llama)
        """
        verbose = state.get("verbose", False)
        qwen_flag = state.get("qwen_response", "")
        
        if verbose:
            print("[TRACE] Entering router: route_to_model")
            print(f"[TRACE] qwen_response flag: '{qwen_flag}'")
        
        # Check the flag set by route_to_models
        if qwen_flag == "__USE_QWEN__":
            if verbose:
                print("[TRACE] Router decision: call_qwen (input started with 'Hey Qwen')")
            print("[INFO] Router: Selected Qwen model")
            return "call_qwen"
        else:
            if verbose:
                print("[TRACE] Router decision: call_llm (default to Llama)")
            print("[INFO] Router: Selected Llama model")
            return "call_llm"

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    # This function examines the state and determines which node to go to next.
    # It's used for conditional edges after get_user_input.
    # Three possible routes:
    #   1. User wants to quit -> END
    #   2. User entered empty input -> loop back to get_user_input
    #   3. User entered valid input -> proceed to call_llm
    def route_after_input(state: AgentState) -> str:
        """
        Routing function that determines the next node based on state.

        Examines state:
            - should_exit: If True, terminate the graph
            - user_input: If empty, loop back to get_user_input
            - verbose: Whether to print tracing information

        Returns:
            - "__end__": If user wants to quit
            - "get_user_input": If user input is empty (loop back)
            - "call_llm": If user provided valid non-empty input
        """
        verbose = state.get("verbose", False)
        
        if verbose:
            print("[TRACE] Entering router: route_after_input")
            print(f"[TRACE] Current state - should_exit: {state.get('should_exit', False)}, user_input: '{state.get('user_input', '')}'")
        
        # Check if user wants to exit
        if state.get("should_exit", False):
            if verbose:
                print("[TRACE] Router decision: END (user wants to exit)")
            return END

        # Check if input is empty - loop back to get_user_input
        user_input = state.get("user_input", "")
        if not user_input.strip():  # Empty or whitespace-only input
            if verbose:
                print("[TRACE] Router decision: get_user_input (empty input, looping back)")
            print("Empty input detected. Please enter some text.")
            return "get_user_input"

        # Valid non-empty input - proceed to route_to_models node
        if verbose:
            print("[TRACE] Router decision: route_to_models (valid input)")
        return "route_to_models"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    # Create a StateGraph with our defined state structure
    graph_builder = StateGraph(AgentState)

    # Add all nodes to the graph
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("route_to_models", route_to_models)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_both_responses", print_both_responses)

    # Define edges:
    # 1. START -> get_user_input (always start by getting user input)
    graph_builder.add_edge(START, "get_user_input")

    # 2. get_user_input -> [conditional] -> route_to_models OR get_user_input OR END
    #    Uses route_after_input to decide based on state.should_exit and user_input
    #    Three-way branch:
    #    - END: User wants to quit
    #    - get_user_input: Empty input (loop back to prompt again)
    #    - route_to_models: Valid non-empty input (will route to both models)
    graph_builder.add_conditional_edges(
        "get_user_input",      # Source node
        route_after_input,      # Routing function that examines state
        {
            "route_to_models": "route_to_models",  # Valid input -> route to both models
            "get_user_input": "get_user_input",    # Empty input -> loop back
            END: END                                # Quit command -> terminate graph
        }
    )

    # 3. route_to_models -> [conditional] -> call_llm OR call_qwen
    #    Uses route_to_model to decide which model to use based on input prefix
    #    - call_qwen: If input started with "Hey Qwen"
    #    - call_llm: Otherwise (default to Llama)
    graph_builder.add_conditional_edges(
        "route_to_models",      # Source node
        route_to_model,          # Routing function that examines state
        {
            "call_llm": "call_llm",      # Default -> use Llama
            "call_qwen": "call_qwen"     # "Hey Qwen" prefix -> use Qwen
        }
    )

    # 4. Both call_llm and call_qwen -> print_both_responses
    #    The print node will execute after the selected model completes
    #    and will print the response from state
    graph_builder.add_edge("call_llm", "print_both_responses")
    graph_builder.add_edge("call_qwen", "print_both_responses")

    # 5. print_both_responses -> get_user_input (loop back for next input)
    #    This creates the continuous loop - after printing, go back to get more input
    graph_builder.add_edge("print_both_responses", "get_user_input")

    # Compile the graph into an executable form
    graph = graph_builder.compile()

    return graph

def save_graph_image(graph, filename="lg_graph.png"):
    """
    Generate a Mermaid diagram of the graph and save it as a PNG image.
    Uses the graph's built-in Mermaid export functionality.
    """
    try:
        # Get the Mermaid PNG representation of the graph
        # This requires the 'grandalf' package for rendering
        png_data = graph.get_graph(xray=True).draw_mermaid_png()

        # Write the PNG data to file
        with open(filename, "wb") as f:
            f.write(png_data)

        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")

def main():
    """
    Main function that orchestrates the simple agent workflow:
    1. Initialize both LLMs (Llama and Qwen)
    2. Create the LangGraph with both models
    3. Save the graph visualization
    4. Run the graph once (it loops internally until user quits)

    The graph handles all looping internally through its edge structure:
    - get_user_input: Prompts and reads from stdin
    - route_to_models: Routes input to both models
    - call_llm and call_qwen: Process input through both LLMs in parallel
    - print_both_responses: Outputs both responses, then loops back to get_user_input

    The graph only terminates when the user types 'quit', 'exit', or 'q'.
    """
    print("=" * 50)
    print("LangGraph Agent with Llama-3.2-1B-Instruct and Qwen-0.5B-Instruct")
    print("=" * 50)
    print()

    # Step 1: Create and configure both LLMs
    print("Loading models...")
    llama_llm = create_llama_llm()
    qwen_llm = create_qwen_llm()

    # Step 2: Build the LangGraph with both LLMs
    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created successfully!")

    # Step 3: Save a visual representation of the graph before execution
    # This happens BEFORE any graph execution, showing the graph structure
    print("\nSaving graph visualization...")
    save_graph_image(graph)

    # Step 4: Run the graph - it will loop internally until user quits
    # Create initial state with empty/default values
    # The graph will loop continuously, updating state as it goes:
    #   - get_user_input displays banner, populates user_input and should_exit
    #   - call_llm and call_qwen populate llm_response and qwen_response (in parallel)
    #   - print_both_responses displays both outputs, then loops back to get_user_input
    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "llm_response": "",
        "qwen_response": "",
        "verbose": False  # Start with verbose disabled
    }

    # Single invocation - the graph loops internally via print_both_responses -> get_user_input
    # The graph only exits when route_after_input returns END (user typed quit/exit/q)
    graph.invoke(initial_state)

# Entry point - only run main() if this script is executed directly
if __name__ == "__main__":
    main()