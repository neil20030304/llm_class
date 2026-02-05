# Topic 3: Sequential vs Parallel LLM Execution with Ollama

Task1
## Results
| Execution Mode | Astronomy Duration | Business Ethics Duration | Total Wall Clock Time |
|---------------|-------------------|--------------------------|----------------------|
| Individual Run | 34.7s | 21.5s | N/A |
| Sequential | 32.9s | 21.7s | **~54.6s** |
| Parallel | 38.1s | 27.6s | **~38.1s** |


## Observations

Key Finding: Parallel Execution Provides Meaningful Speedup

Despite both programs competing for the same Ollama server resources, parallel execution reduced total wall clock time by approximately **30%** (from ~55 seconds to ~38 seconds).

Task2:
The line client = OpenAI() initializes an OpenAI client instance that handles authentication and communication with the OpenAI API using the API key stored in the environment. The call to client.chat.completions.create(...) sends a chat-style request to the GPT-4o Mini model, including a user message that instructs the model to respond with a short phrase. The messages parameter defines the conversation input, while max_tokens=5 limits the length of the model’s generated output. This test verifies that the OpenAI library is correctly installed and that the API connection is functioning as expected.