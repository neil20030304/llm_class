"""
MMLU Evaluation Script using Ollama - Program 1 (Astronomy)

This script evaluates Llama 3.2-1B on the MMLU benchmark using Ollama.
This program specifically evaluates the 'astronomy' subject.

Usage:
1. Install: pip install ollama datasets tqdm
2. Make sure Ollama server is running: ollama serve
3. Pull the model: ollama pull llama3.2:latest
4. Run: python program1.py
"""

import ollama
from datasets import load_dataset
import json
from tqdm.auto import tqdm
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "llama3.2:latest"  # Ollama model name
SUBJECT = "astronomy"  # Single subject for this program


def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer with only the letter (A, B, C, or D):"
    return prompt


def get_model_prediction(prompt):
    """Get model's prediction for multiple-choice question using Ollama"""
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "num_predict": 5,  # Only need a few tokens for the answer
                "temperature": 0.0,  # Deterministic output
            }
        )
        
        generated_text = response['response'].strip()
        
        # Extract the answer letter
        answer = generated_text[:1].upper()
        
        if answer not in ["A", "B", "C", "D"]:
            for char in generated_text.upper():
                if char in ["A", "B", "C", "D"]:
                    answer = char
                    break
            else:
                answer = "A"  # Default if no valid answer found
        
        return answer
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "A"  # Default on error


def evaluate_subject(subject):
    """Evaluate model on a specific MMLU subject"""
    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'='*70}")
    
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"❌ Error loading subject {subject}: {e}")
        return None
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]
        
        prompt = format_mmlu_prompt(question, choices)
        predicted_answer = get_model_prediction(prompt)
        
        if predicted_answer == correct_answer:
            correct += 1
        total += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✓ Result: {correct}/{total} correct = {accuracy:.2f}%")
    
    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }


def main():
    """Main evaluation function"""
    print("\n" + "="*70)
    print(f"MMLU Evaluation using Ollama - {SUBJECT}")
    print(f"Model: {MODEL_NAME}")
    print("="*70 + "\n")
    
    # Check Ollama connection
    try:
        ollama.list()
        print("✓ Connected to Ollama server")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama server: {e}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    
    start_time = datetime.now()
    
    # Evaluate single subject
    result = evaluate_subject(SUBJECT)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if result:
        # Print summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"Model: {MODEL_NAME}")
        print(f"Subject: {SUBJECT}")
        print(f"Total Questions: {result['total']}")
        print(f"Total Correct: {result['correct']}")
        print(f"Accuracy: {result['accuracy']:.2f}%")
        print(f"Duration: {duration:.1f} seconds")
        print("="*70)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"ollama_mmlu_{SUBJECT}_{timestamp}.json"
        
        output_data = {
            "model": MODEL_NAME,
            "subject": SUBJECT,
            "timestamp": timestamp,
            "duration_seconds": duration,
            "accuracy": result['accuracy'],
            "correct": result['correct'],
            "total": result['total']
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    print("\n✅ Evaluation complete!")
    return result


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

