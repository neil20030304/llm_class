"""
Bare-Bones Chat Agent for Llama 3.2-1B-Instruct

This is a minimal chat interface that demonstrates:
1. How to load a model without quantization
2. How chat history is maintained and fed back to the model
3. The difference between plain text history and tokenized input

No classes, no fancy features - just the essentials.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# CONFIGURATION - Change these settings as needed
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# System prompt - This sets the chatbot's behavior and personality
# Change this to customize how the chatbot responds
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and friendly."

# Context management settings
USE_CONTEXT_HISTORY = True  # Set to False to disable conversation memory (stateless mode)
MAX_RECENT_MESSAGES = 10    # Number of recent messages to keep in full detail
MAX_CONTEXT_TOKENS = 4000   # Maximum tokens for context (reserves space for generation)

# ============================================================================
# LOAD MODEL (NO QUANTIZATION)
# ============================================================================

print("Loading model (this takes 1-2 minutes)...")

# Load tokenizer (converts text to numbers and vice versa)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model in half precision (float16) for efficiency
# Use float16 on GPU, or float32 on CPU if needed
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,                        # Use FP16 for efficiency
    device_map="auto",                          # Automatically choose GPU/CPU
    low_cpu_mem_usage=True
)

model.eval()  # Set to evaluation mode (no training)
print(f"✓ Model loaded! Using device: {model.device}")
print(f"✓ Memory usage: ~2.5 GB (FP16)")
print(f"✓ Context history: {'ENABLED' if USE_CONTEXT_HISTORY else 'DISABLED (stateless mode)'}\n")

# ============================================================================
# CONTEXT MANAGER - Hybrid approach for managing conversation history
# ============================================================================

class ContextManager:
    """
    Manages conversation context using a hybrid approach:
    - Keeps recent messages in full detail
    - Summarizes older messages to preserve long-term context
    - Ensures total context stays within token budget
    """
    
    def __init__(self, max_recent_messages=MAX_RECENT_MESSAGES, max_tokens=MAX_CONTEXT_TOKENS):
        self.max_recent = max_recent_messages
        self.max_tokens = max_tokens
    
    def manage_context(self, chat_history, tokenizer):
        """
        Manage context using hybrid strategy:
        1. Keep system prompt
        2. Keep most recent N messages in full
        3. Summarize older messages
        4. Ensure total fits in token budget
        """
        if not chat_history:
            return chat_history
        
        # Extract system message
        system_msg = chat_history[0] if chat_history[0]["role"] == "system" else None
        messages = chat_history[1:] if system_msg else chat_history
        
        # If we have few messages, no need to manage
        if len(messages) <= self.max_recent:
            return chat_history
        
        # Split into old and recent messages
        old_messages = messages[:-self.max_recent]
        recent_messages = messages[-self.max_recent:]
        
        # Create compact summary of old messages
        summary = self._create_compact_summary(old_messages)
        summary_msg = {
            "role": "system",
            "content": f"[Previous conversation summary: {summary}]"
        }
        
        # Build test history: system + summary + recent
        test_history = []
        if system_msg:
            test_history.append(system_msg)
        test_history.append(summary_msg)
        test_history.extend(recent_messages)
        
        # Check if it fits in token budget
        try:
            tokens = tokenizer.apply_chat_template(test_history, return_tensors="pt")
            token_count = tokens.shape[1]
            
            if token_count > self.max_tokens:
                # Still too long - truncate recent messages
                return self._truncate_to_fit(system_msg, summary_msg, recent_messages, tokenizer)
            
            return test_history
        except Exception as e:
            # Fallback: just use recent messages
            if system_msg:
                return [system_msg] + recent_messages
            return recent_messages
    
    def _create_compact_summary(self, messages):
        """Create a brief summary of old messages"""
        user_topics = []
        key_points = []
        
        for msg in messages:
            if msg["role"] == "user":
                # Extract first part of user questions/statements
                content = msg["content"][:50].strip()
                if content:
                    user_topics.append(content)
            elif msg["role"] == "assistant":
                # Extract key information from assistant responses
                content = msg["content"][:40].strip()
                if content:
                    key_points.append(content)
        
        # Build summary
        summary_parts = []
        if user_topics:
            summary_parts.append(f"Topics discussed: {', '.join(user_topics[:5])}")
        if key_points:
            summary_parts.append(f"Key points: {', '.join(key_points[:3])}")
        
        return " | ".join(summary_parts) if summary_parts else "General conversation"
    
    def _truncate_to_fit(self, system_msg, summary_msg, recent_messages, tokenizer):
        """Truncate messages to fit within token budget"""
        # Start with system + summary
        base_history = []
        if system_msg:
            base_history.append(system_msg)
        base_history.append(summary_msg)
        
        # Add recent messages one by one until we hit the limit
        kept_messages = []
        for msg in reversed(recent_messages):
            test = base_history + kept_messages + [msg]
            try:
                tokens = tokenizer.apply_chat_template(test, return_tensors="pt")
                if tokens.shape[1] <= self.max_tokens:
                    kept_messages.insert(0, msg)
                else:
                    break
            except:
                break
        
        return base_history + kept_messages

# Initialize context manager
context_manager = ContextManager(
    max_recent_messages=MAX_RECENT_MESSAGES,
    max_tokens=MAX_CONTEXT_TOKENS
)

# ============================================================================
# CHAT HISTORY - This is stored as PLAIN TEXT (list of dictionaries)
# ============================================================================
# The chat history is a list of messages in this format:
# [
#   {"role": "system", "content": "You are a helpful assistant"},
#   {"role": "user", "content": "Hello!"},
#   {"role": "assistant", "content": "Hi! How can I help?"},
#   {"role": "user", "content": "What's 2+2?"},
#   {"role": "assistant", "content": "2+2 equals 4."}
# ]
#
# This is PLAIN TEXT - humans can read it
# The model CANNOT use this directly - it needs to be tokenized first

chat_history = []

# Add system prompt to history (this persists across the entire conversation)
chat_history.append({
    "role": "system",
    "content": SYSTEM_PROMPT
})

# ============================================================================
# CHAT LOOP
# ============================================================================

print("="*70)
print("Chat started! Type 'quit' or 'exit' to end the conversation.")
if USE_CONTEXT_HISTORY:
    print(f"Context management: ENABLED (max {MAX_RECENT_MESSAGES} recent messages, {MAX_CONTEXT_TOKENS} tokens)")
else:
    print("Context management: DISABLED (stateless mode - no conversation memory)")
print("="*70 + "\n")

while True:
    # ========================================================================
    # STEP 1: Get user input (PLAIN TEXT)
    # ========================================================================
    user_input = input("You: ").strip()
    
    # Check for exit commands
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break
    
    # Skip empty inputs
    if not user_input:
        continue
    
    # ========================================================================
    # STEP 2: Add user message to chat history (PLAIN TEXT)
    # ========================================================================
    # The chat history grows with each exchange
    # Handle context history based on USE_CONTEXT_HISTORY flag
    if USE_CONTEXT_HISTORY:
        # Add user message to chat history (maintains conversation memory)
        chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Manage context using hybrid approach (truncate/summarize if needed)
        managed_history = context_manager.manage_context(chat_history, tokenizer)
    else:
        # Stateless mode: Only use current message (no history)
        managed_history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    
    # At this point, managed_history contains:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},      â† Just added
    # ]
    # This is still PLAIN TEXT
    
    # ========================================================================
    # STEP 3: Convert chat history to model input (TOKENIZATION)
    # ========================================================================
    # The model needs numbers (tokens), not text
    # apply_chat_template() does two things:
    #   1. Formats the chat history with special tokens (like <|start|>, <|end|>)
    #   2. Converts the formatted text into token IDs (numbers)
    
    # Apply chat template to managed history and convert to tokens
    input_ids = tokenizer.apply_chat_template(
        managed_history,                 # Managed PLAIN TEXT history
        add_generation_prompt=True,      # Add prompt for assistant's response
        return_tensors="pt"              # Return as PyTorch tensor (numbers)
    ).to(model.device)

    # Create attention mask (1 for all tokens since we have no padding)
    attention_mask = torch.ones_like(input_ids)

    # Now input_ids is TOKENIZED - it's a tensor of numbers like:
    # tensor([[128000, 128006, 9125, 128007, 271, 2675, 527, 264, ...]])
    # These numbers represent our entire conversation history

    # ========================================================================
    # STEP 4: Generate assistant response (MODEL INFERENCE)
    # ========================================================================
    # The model looks at the ENTIRE chat history (in tokenized form)
    # and generates a response

    print("Assistant: ", end="", flush=True)

    with torch.no_grad():  # Don't calculate gradients (we're not training)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,   # Explicitly pass attention mask
            max_new_tokens=512,              # Maximum length of response
            do_sample=True,                  # Use sampling for variety
            temperature=0.7,                 # Lower = more focused, higher = more random
            top_p=0.9,                       # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id
        )
    
    # outputs contains: [original input tokens + new generated tokens]
    # We only want the NEW tokens (the assistant's response)
    
    # ========================================================================
    # STEP 5: Decode the response (DETOKENIZATION)
    # ========================================================================
    # Extract only the newly generated tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    
    # Convert tokens (numbers) back to text (PLAIN TEXT)
    assistant_response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True  # Remove special tokens like <|end|>
    )
    
    print(assistant_response)  # Display the response
    
    # ========================================================================
    # STEP 6: Add assistant response to chat history (PLAIN TEXT)
    # ========================================================================
    # Only add to history if context management is enabled
    # In stateless mode, we don't store the response
    
    if USE_CONTEXT_HISTORY:
        chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })
    
    # Now chat_history has grown again:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},
    #   {"role": "assistant", "content": "4"}              â† Just added
    # ]
    
    # When the loop repeats:
    # - User enters new message
    # - We add it to chat_history
    # - We tokenize the ENTIRE history (including all previous exchanges)
    # - Model sees everything and generates response
    # - We add response to history
    # - Repeat...
    
    # This is how the chatbot "remembers" the conversation!
    # Each turn, we feed it the ENTIRE conversation history
    
    print()  # Blank line for readability

# ============================================================================
# SUMMARY OF HOW CHAT HISTORY WORKS
# ============================================================================
"""
PLAIN TEXT vs TOKENIZED:

1. PLAIN TEXT (chat_history):
   - Human-readable format
   - List of dictionaries: [{"role": "user", "content": "Hi"}, ...]
   - Stored in memory between turns
   - Gets longer with each message

2. TOKENIZED (input_ids):
   - Numbers (token IDs)
   - Created fresh each turn from chat_history
   - This is what the model actually "reads"
   - Example: [128000, 128006, 9125, 128007, ...]

PROCESS EACH TURN:
   User input (text)
   â†“
   Add to chat_history (text)
   â†“
   Tokenize entire chat_history (text â†’ numbers)
   â†“
   Model generates response (numbers)
   â†“
   Decode response (numbers â†’ text)
   â†“
   Add response to chat_history (text)
   â†“
   Loop back to start

WHY FEED ENTIRE HISTORY?
- The model has no memory between calls
- Each generation is independent
- To "remember" previous turns, we must include them in the input
- This is why context length matters - longer conversations = more tokens

WHAT HAPPENS AS CONVERSATION GROWS?
- chat_history gets longer (more messages)
- Tokenized input gets longer (more tokens)
- Eventually hits model's max context length (for Llama 3.2: 128K tokens)
- Then you need context management (truncation, summarization, etc.)
- But for this simple demo, we let it grow without limit
"""