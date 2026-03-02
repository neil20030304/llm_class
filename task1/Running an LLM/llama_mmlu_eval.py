"""
Llama 3.2-1B MMLU Evaluation Script (Laptop Optimized with Quantization)

This script evaluates Llama 3.2-1B on the MMLU benchmark.
Optimized for laptops with 4-bit or 8-bit quantization to reduce memory usage.

Quantization options:
- 4-bit: ~1.5 GB VRAM/RAM (default for laptop)
- 8-bit: ~2.5 GB VRAM/RAM
- No quantization: ~5 GB VRAM/RAM

Usage:
1. Install: pip install transformers torch datasets accelerate tqdm bitsandbytes
2. Login: huggingface-cli login
3. Run: python llama_mmlu_eval_quantized.py

Set QUANTIZATION_BITS below to choose quantization level.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import json
from tqdm.auto import tqdm
import os
from datetime import datetime
import sys
import platform

import argparse
import time
import psutil
# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

MODEL_NAME = ["meta-llama/Llama-3.2-1B-Instruct",
                "Qwen/Qwen2.5-0.5B",  # Qwen 2.5 0.5B model
        "google/gemma-2b"] # OLMo-2-1B model


# GPU settings
# If True, will attempt to use the best available GPU (CUDA for NVIDIA, MPS for Apple Silicon)
# If False, will always use CPU regardless of available hardware
USE_GPU = True # Set to False to force CPU-only execution

MAX_NEW_TOKENS = 1

# Quantization settings
# Options: 4, 8, or None (default is None for full precision)
#
# To enable quantization, change QUANTIZATION_BITS to one of the following:
#   QUANTIZATION_BITS = 4   # 4-bit quantization: ~1.5 GB memory (most memory efficient)
#   QUANTIZATION_BITS = 8   # 8-bit quantization: ~2.5 GB memory (balanced quality/memory)
#   QUANTIZATION_BITS = None  # No quantization: ~5 GB memory (full precision, best quality)
#
# Notes:
# - Quantization requires the 'bitsandbytes' package: pip install bitsandbytes
# - Quantization only works with CUDA (NVIDIA GPUs), not with Apple Metal (MPS)
# - If using Apple Silicon, quantization will be automatically disabled

QUANTIZATION_BITS = None  # Change to 4 or 8 to enable quantization

VERBOSE = False

# For quick testing, you can reduce this list
MMLU_SUBJECTS = [
    # "abstract_algebra", "anatomy", 
    "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", 
    # "conceptual_physics",
    # "econometrics", "electrical_engineering", "elementary_mathematics",
    # "formal_logic", "global_facts", "high_school_biology",
    # "high_school_chemistry", "high_school_computer_science",
    # "high_school_european_history", "high_school_geography",
    # "high_school_government_and_politics", "high_school_macroeconomics",
    # "high_school_mathematics", "high_school_microeconomics",
    # "high_school_physics", "high_school_psychology", "high_school_statistics",
    # "high_school_us_history", "high_school_world_history", "human_aging",
    # "human_sexuality", "international_law", "jurisprudence",
    # "logical_fallacies", "machine_learning", "management", "marketing",
    # "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    # "nutrition", "philosophy", "prehistory", "professional_accounting",
    # "professional_law", "professional_medicine", "professional_psychology",
    # "public_relations", "security_studies", "sociology", "us_foreign_policy",
    # "virology", "world_religions"
]

class TimingTracker:
    """Track real, CPU, and GPU time"""
    def __init__(self, device):
        self.device = device
        self.real_time = 0.0
        self.cpu_time = 0.0
        self.gpu_time = 0.0
        self.process = psutil.Process()
    
    def start(self):
        self.start_real = time.time()
        cpu_times = self.process.cpu_times()
        self.start_cpu = cpu_times.user + cpu_times.system
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_gpu = torch.cuda.Event(enable_timing=True)
            self.end_gpu = torch.cuda.Event(enable_timing=True)
            self.start_gpu.record()
        elif self.device == "mps":
            torch.mps.synchronize()
            self.start_gpu = time.time()
    
    def stop(self):
        self.end_real = time.time()
        cpu_times = self.process.cpu_times()
        self.end_cpu = cpu_times.user + cpu_times.system
        
        if self.device == "cuda" and torch.cuda.is_available():
            self.end_gpu.record()
            torch.cuda.synchronize()
            self.gpu_time += self.start_gpu.elapsed_time(self.end_gpu) / 1000.0
        elif self.device == "mps":
            torch.mps.synchronize()
            self.end_gpu = time.time()
            self.gpu_time += self.end_gpu - self.start_gpu
        
        self.real_time += self.end_real - self.start_real
        self.cpu_time += self.end_cpu - self.start_cpu
    
    def get_timings(self):
        return {
            "real_time": self.real_time,
            "cpu_time": self.cpu_time,
            "gpu_time": self.gpu_time if self.device in ["cuda", "mps"] else 0.0
        }

def detect_device():
    """Detect the best available device (CUDA, MPS, or CPU)"""

    # If GPU is disabled, always use CPU
    if not USE_GPU:
        return "cpu"

    # Check for CUDA
    if torch.cuda.is_available():
        return "cuda"

    # Check for Apple Silicon with Metal
    if torch.backends.mps.is_available():
        # Check if we're actually on Apple ARM
        is_apple_arm = platform.system() == "Darwin" and platform.processor() == "arm"

        if is_apple_arm:
            # Metal is available but incompatible with quantization
            if QUANTIZATION_BITS is not None:
                print("\n" + "="*70)
                print("ERROR: Metal and Quantization Conflict")
                print("="*70)
                print("Metal Performance Shaders (MPS) is incompatible with quantization.")
                print(f"You have USE_GPU = True and QUANTIZATION_BITS = {QUANTIZATION_BITS}")
                print("")
                print("Please choose one of the following options:")
                print("  1. Set USE_GPU = False to use CPU with quantization")
                print("  2. Set QUANTIZATION_BITS = None to use Metal without quantization")
                print("="*70 + "\n")
                sys.exit(1)
            return "mps"

    # Default to CPU
    return "cpu"




def check_environment():
    global QUANTIZATION_BITS
    """Check environment and dependencies"""
    print("="*70)
    print("Environment Check")
    print("="*70)

    # Check if in Colab
    try:
        import google.colab
        print("âœ“ Running in Google Colab")
        in_colab = True
    except:
        print("âœ“ Running locally (not in Colab)")
        in_colab = False

    # Check system info
    print(f"âœ“ Platform: {platform.system()} ({platform.machine()})")
    if platform.system() == "Darwin":
        print(f"âœ“ Processor: {platform.processor()}")

    # Detect and set device
    device = detect_device()

    # Check device
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"âœ“ GPU Available: {gpu_name}")
        print(f"âœ“ GPU Memory: {gpu_memory:.2f} GB")
    elif device == "mps":
        print("âœ“ Apple Metal (MPS) Available")
        print("âœ“ Using Metal Performance Shaders for GPU acceleration")
    else:
        print("âš ï¸  No GPU detected - running on CPU")
       
    # Check quantization support

    if QUANTIZATION_BITS is not None:
        try:
            import bitsandbytes
            print(f"âœ“ bitsandbytes installed - {QUANTIZATION_BITS}-bit quantization available")
        except ImportError:
            print(f"âŒ bitsandbytes NOT installed - cannot use quantization")
            sys.exit(1)
        if device == 'mps':
            print(f"âŒ Apple METAL is incompatible with quantization")
            print("âœ“ Quantization disabled - loading full precision model")
            QUANTIZATION_BITS = None
            sys.exit(1)
    else:
        print("âœ“ Quantization disabled - loading full precision model")
    
    # Check HF authentication
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("âœ“ Hugging Face authenticated")
        else:
            print("âš ï¸  No Hugging Face token found")
            print("Run: huggingface-cli login")
    except:
        print("âš ï¸  Could not check Hugging Face authentication")
    
    # Print configuration
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(f"Models: {len(MODEL_NAME)}")
    for i, model in enumerate(MODEL_NAME, 1):
        print(f"  {i}. {model}")
    print(f"Device: {device}")
    if QUANTIZATION_BITS is not None:
        print(f"Quantization: {QUANTIZATION_BITS}-bit")
        if QUANTIZATION_BITS == 4:
            print(f"Expected memory: ~1.5 GB")
        elif QUANTIZATION_BITS == 8:
            print(f"Expected memory: ~2.5 GB")
    else:
        print(f"Quantization: None (full precision)")
        if device == "cuda":
            print(f"Expected memory: ~2.5 GB (FP16)")
        elif device == "mps":
            print(f"Expected memory: ~2.5 GB (FP16)")
        else:
            print(f"Expected memory: ~5 GB (FP32)")
    print(f"Number of subjects: {len(MMLU_SUBJECTS)}")

    print("="*70 + "\n")
    return in_colab, device


def get_quantization_config():
    """Create quantization config based on settings"""
    if QUANTIZATION_BITS is None:
        return None
    
    if QUANTIZATION_BITS == 4:
        # 4-bit quantization (most memory efficient)
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
            bnb_4bit_quant_type="nf4"  # NormalFloat4 - better for LLMs
        )
        print("Using 4-bit quantization (NF4 + double quant)")
        print("Memory usage: ~1.5 GB")
    elif QUANTIZATION_BITS == 8:
        # 8-bit quantization (balanced)
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        print("Using 8-bit quantization")
        print("Memory usage: ~2.5 GB")
    else:
        raise ValueError(f"Invalid QUANTIZATION_BITS: {QUANTIZATION_BITS}. Use 4, 8, or None")
    
    return config


def load_model_and_tokenizer(model_name, device):
    """Load model with optional quantization"""
    print(f"\nLoading model {model_name}...")
    print(f"Device: {device}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded")

        # Get quantization config
        quant_config = get_quantization_config()

        # Load model
        print("Loading model (this may take 2-3 minutes)...")

        if quant_config is not None:
            # Quantized model loading (only works with CUDA)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Non-quantized model loading
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            elif device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            else:  # CPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)

        model.eval()

        # Print model info
        print("âœ“ Model loaded successfully!")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")

        # Check memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

            # Check if using quantization
            if quant_config is not None:
                print(f"  Quantization: {QUANTIZATION_BITS}-bit active")
        elif device == "mps":
            print(f"  Running on Apple Metal (MPS)")

        return model, tokenizer
        
    except Exception as e:
        print(f"\nâŒ Error loading model: {e}")
        print("\nPossible causes:")
        print("1. No Hugging Face token - Run: huggingface-cli login")
        print("2. Llama license not accepted - Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        print("3. bitsandbytes not installed - Run: pip install bitsandbytes")
        print("4. Out of memory - Try 4-bit quantization or smaller model")
        raise


def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def get_model_prediction(model, tokenizer, prompt):
    """Get model's prediction for multiple-choice question"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    answer = generated_text.strip()[:1].upper()
    
    if answer not in ["A", "B", "C", "D"]:
        for char in generated_text.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break
        else:
            answer = "A"
    
    return answer


def evaluate_subject(model, tokenizer, subject, model_name, timing_tracker):
    """Evaluate model on a specific MMLU subject"""
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name} on subject: {subject}")
    print(f"{'='*70}")
    
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"âŒ Error loading subject {subject}: {e}")
        return None
    
    correct = 0
    total = 0
    question_details = []
    
    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]
        
        prompt = format_mmlu_prompt(question, choices)
        timing_tracker.start()
        predicted_answer = get_model_prediction(model, tokenizer, prompt)
        timing_tracker.stop()

        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
        total += 1
        
        # Always save question details for analysis
        question_details.append({
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        })
        
        if VERBOSE:
            status = "✓" if is_correct else "✗"
            print(f"\n{status} Question {total}:")
            print(f"  Q: {question}")
            print(f"  Choices: {choices}")
            print(f"  Correct: {correct_answer}, Predicted: {predicted_answer}")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✓ Result: {correct}/{total} correct = {accuracy:.2f}%")
    
    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "question_details": question_details
    }


def main():
    """Main evaluation function"""
    global VERBOSE
    
    parser = argparse.ArgumentParser(description="Multi-Model MMLU Evaluation")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each question, model answer, and correctness")
    args = parser.parse_args()
    VERBOSE = args.verbose
    
    print("\n" + "="*70)
    print("Multi-Model MMLU Evaluation")
    print("="*70 + "\n")

    in_colab, device = check_environment()
    
    all_results = []
    overall_start = time.time()
    
    for model_name in MODEL_NAME:
        print(f"\n{'#'*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'#'*70}")
        
        try:
            model, tokenizer = load_model_and_tokenizer(model_name, device)
            timing_tracker = TimingTracker(device)
            
            results = []
            total_correct = 0
            total_questions = 0
            
            for i, subject in enumerate(MMLU_SUBJECTS, 1):
                print(f"\nProgress: {i}/{len(MMLU_SUBJECTS)} subjects")
                result = evaluate_subject(model, tokenizer, subject, model_name, timing_tracker)
                if result:
                    results.append(result)
                    total_correct += result["correct"]
                    total_questions += result["total"]
            
            timings = timing_tracker.get_timings()
            overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
            
            all_results.append({
                "model_name": model_name,
                "results": results,
                "total_correct": total_correct,
                "total_questions": total_questions,
                "overall_accuracy": overall_accuracy,
                "timings": timings
            })
            
            # Clean up
            del model, tokenizer
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
                
        except Exception as e:
            print(f"\n✗ Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    overall_duration = time.time() - overall_start
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    for result in all_results:
        print(f"\n{'-'*70}")
        print(f"Model: {result['model_name']}")
        print(f"Accuracy: {result['overall_accuracy']:.2f}% ({result['total_correct']}/{result['total_questions']})")
        print(f"Timing:")
        print(f"  Real Time: {result['timings']['real_time']:.2f}s ({result['timings']['real_time']/60:.2f}m)")
        print(f"  CPU Time: {result['timings']['cpu_time']:.2f}s ({result['timings']['cpu_time']/60:.2f}m)")
        if device in ["cuda", "mps"]:
            print(f"  GPU Time: {result['timings']['gpu_time']:.2f}s ({result['timings']['gpu_time']/60:.2f}m)")
        else:
            print(f"  GPU Time: N/A")
    
    print(f"\n{'-'*70}")
    print(f"Total Duration: {overall_duration:.2f}s ({overall_duration/60:.2f}m)")
    print("="*70)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"multi_model_mmlu_results_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "device": str(device),
            "quantization_bits": QUANTIZATION_BITS,
            "subjects": MMLU_SUBJECTS,
            "overall_duration_seconds": overall_duration,
            "model_results": all_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("\n✓ Evaluation complete!")
    return output_file


if __name__ == "__main__":
    try:
        output_file = main()
    except KeyboardInterrupt:
        print("\n\nâš ï¸  Evaluation interrupted by user")
    except Exception as e:
        print(f"\nâŒ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()