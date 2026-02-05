"""
Analyze MMLU evaluation results and create visualizations
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import seaborn as sns
from pathlib import Path
import sys

def load_results(json_file):
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def plot_accuracy_by_subject(results_data):
    """Plot accuracy for each model by subject"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = []
    subjects = []
    accuracies = []
    
    for model_result in results_data['model_results']:
        model_name = model_result['model_name'].split('/')[-1]
        models.append(model_name)
        
        for subject_result in model_result['results']:
            subjects.append(subject_result['subject'])
            accuracies.append(subject_result['accuracy'])
    
    # Create DataFrame-like structure for plotting
    unique_subjects = sorted(set(subjects))
    unique_models = [m.split('/')[-1] for m in [mr['model_name'] for mr in results_data['model_results']]]
    
    x = np.arange(len(unique_subjects))
    width = 0.25
    
    for i, model_name in enumerate(unique_models):
        model_accuracies = []
        for subject in unique_subjects:
            for model_result in results_data['model_results']:
                if model_result['model_name'].split('/')[-1] == model_name:
                    for sr in model_result['results']:
                        if sr['subject'] == subject:
                            model_accuracies.append(sr['accuracy'])
                            break
                    break
        
        ax.bar(x + i*width, model_accuracies, width, label=model_name[:20])
    
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy by Subject', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(unique_subjects, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    return fig

def plot_overall_accuracy(results_data):
    """Plot overall accuracy comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = []
    accuracies = []
    
    for model_result in results_data['model_results']:
        model_name = model_result['model_name'].split('/')[-1]
        models.append(model_name)
        accuracies.append(model_result['overall_accuracy'])
    
    bars = ax.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Overall Accuracy (%)', fontsize=12)
    ax.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def analyze_mistake_patterns(results_data):
    """Analyze which questions models get wrong"""
    # Create a mapping of question text to correctness across models
    question_correctness = defaultdict(lambda: {'correct': [], 'wrong': []})
    all_questions = []
    
    for model_result in results_data['model_results']:
        model_name = model_result['model_name'].split('/')[-1]
        
        for subject_result in model_result['results']:
            if 'question_details' not in subject_result:
                continue
                
            for q_detail in subject_result['question_details']:
                question_text = q_detail['question']
                question_key = f"{subject_result['subject']}|||{question_text}"
                
                if question_key not in all_questions:
                    all_questions.append(question_key)
                
                if q_detail['is_correct']:
                    question_correctness[question_key]['correct'].append(model_name)
                else:
                    question_correctness[question_key]['wrong'].append(model_name)
    
    # Analyze patterns
    all_wrong = []  # Questions all models got wrong
    all_correct = []  # Questions all models got correct
    mixed = []  # Questions some got right, some got wrong
    
    num_models = len(results_data['model_results'])
    
    for q_key, correctness in question_correctness.items():
        num_wrong = len(correctness['wrong'])
        num_correct = len(correctness['correct'])
        
        if num_wrong == num_models:
            all_wrong.append(q_key)
        elif num_correct == num_models:
            all_correct.append(q_key)
        else:
            mixed.append(q_key)
    
    return {
        'all_wrong': all_wrong,
        'all_correct': all_correct,
        'mixed': mixed,
        'question_correctness': question_correctness,
        'num_models': num_models
    }

def plot_mistake_overlap(results_data):
    """Plot overlap of mistakes between models"""
    patterns = analyze_mistake_patterns(results_data)
    
    if not patterns['question_correctness']:
        print("No question-level data available. Run evaluation with question details saved.")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart of question categories
    labels = ['All Correct', 'All Wrong', 'Mixed Results']
    sizes = [
        len(patterns['all_correct']),
        len(patterns['all_wrong']),
        len(patterns['mixed'])
    ]
    colors = ['#2ca02c', '#d62728', '#ff7f0e']
    
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Question Difficulty Distribution', fontsize=12, fontweight='bold')
    
    # Bar chart showing overlap
    models = [mr['model_name'].split('/')[-1] for mr in results_data['model_results']]
    
    # Count mistakes per model
    model_mistakes = defaultdict(int)
    for q_key, correctness in patterns['question_correctness'].items():
        for model_name in correctness['wrong']:
            model_mistakes[model_name] += 1
    
    ax2.bar(models, [model_mistakes[m] for m in models], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Number of Mistakes', fontsize=12)
    ax2.set_title('Total Mistakes per Model', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_mistake_overlap_matrix(results_data):
    """Create a heatmap showing mistake overlap between models"""
    patterns = analyze_mistake_patterns(results_data)
    
    if not patterns['question_correctness']:
        return None
    
    models = [mr['model_name'].split('/')[-1] for mr in results_data['model_results']]
    n = len(models)
    overlap_matrix = np.zeros((n, n))
    
    # Count questions where both models got wrong
    for q_key, correctness in patterns['question_correctness'].items():
        wrong_models = correctness['wrong']
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if model1 in wrong_models and model2 in wrong_models:
                    overlap_matrix[i][j] += 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(overlap_matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
                xticklabels=models, yticklabels=models, ax=ax)
    ax.set_title('Mistake Overlap Matrix\n(Number of questions both models got wrong)', 
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_subject_difficulty(results_data):
    """Plot average accuracy per subject across all models"""
    subject_accuracies = defaultdict(list)
    
    for model_result in results_data['model_results']:
        for subject_result in model_result['results']:
            subject_accuracies[subject_result['subject']].append(subject_result['accuracy'])
    
    subjects = sorted(subject_accuracies.keys())
    avg_accuracies = [np.mean(subject_accuracies[s]) for s in subjects]
    std_accuracies = [np.std(subject_accuracies[s]) for s in subjects]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(subjects))
    bars = ax.bar(x, avg_accuracies, yerr=std_accuracies, capsize=5, 
                  color='steelblue', alpha=0.7)
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax.set_title('Subject Difficulty (Average Across All Models)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, avg, std) in enumerate(zip(bars, avg_accuracies, std_accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 2,
                f'{avg:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def analyze_subject_correlation(results_data):
    """Analyze if models make mistakes on the same subjects"""
    # Create subject accuracy correlation
    subject_accuracies = {}
    
    for model_result in results_data['model_results']:
        model_name = model_result['model_name'].split('/')[-1]
        subject_accuracies[model_name] = {}
        
        for subject_result in model_result['results']:
            subject_accuracies[model_name][subject_result['subject']] = subject_result['accuracy']
    
    # Calculate correlation between models for each subject
    models = list(subject_accuracies.keys())
    subjects = sorted(set([s for m in subject_accuracies.values() for s in m.keys()]))
    
    # Find subjects where all models perform poorly (potential systematic difficulty)
    difficult_subjects = []
    easy_subjects = []
    
    for subject in subjects:
        accs = [subject_accuracies[m][subject] for m in models]
        avg_acc = np.mean(accs)
        
        if avg_acc < 30:  # All models struggle
            difficult_subjects.append((subject, avg_acc))
        elif avg_acc > 50:  # All models do relatively well
            easy_subjects.append((subject, avg_acc))
    
    return {
        'difficult_subjects': sorted(difficult_subjects, key=lambda x: x[1]),
        'easy_subjects': sorted(easy_subjects, key=lambda x: x[1], reverse=True),
        'subject_accuracies': subject_accuracies
    }

def print_analysis_summary(results_data, patterns):
    """Print summary of analysis"""
    print("\n" + "="*70)
    print("MISTAKE PATTERN ANALYSIS")
    print("="*70)
    
    if not patterns['question_correctness']:
        print("\n⚠️  No question-level data available for detailed analysis.")
        print("   The current results file only contains subject-level summaries.")
        print("   To analyze specific question patterns, re-run evaluation with updated code.")
        print("\n   However, we can analyze patterns from subject-level data:")
        
        # Analyze subject-level patterns
        subject_analysis = analyze_subject_correlation(results_data)
        
        print(f"\n   Subjects where ALL models struggle (<30% avg accuracy):")
        for subject, avg_acc in subject_analysis['difficult_subjects']:
            print(f"     - {subject}: {avg_acc:.1f}% average")
        
        print(f"\n   Subjects where models perform relatively well (>50% avg accuracy):")
        for subject, avg_acc in subject_analysis['easy_subjects']:
            print(f"     - {subject}: {avg_acc:.1f}% average")
        
        # Check if models make mistakes on same subjects
        print(f"\n   Subject-level correlation analysis:")
        models = [mr['model_name'].split('/')[-1] for mr in results_data['model_results']]
        
        # Find subjects where models have similar performance
        subject_accuracies = subject_analysis['subject_accuracies']
        subjects = sorted(set([s for m in subject_accuracies.values() for s in m.keys()]))
        
        high_correlation_subjects = []
        for subject in subjects:
            accs = [subject_accuracies[m][subject] for m in models]
            std_acc = np.std(accs)
            if std_acc < 5:  # Low variance = high correlation
                high_correlation_subjects.append((subject, np.mean(accs), std_acc))
        
        if high_correlation_subjects:
            print(f"     Subjects where models have similar performance (std < 5%):")
            for subject, avg, std in high_correlation_subjects:
                print(f"       - {subject}: {avg:.1f}% ± {std:.1f}%")
            print(f"     → This suggests systematic difficulty, not random mistakes")
        else:
            print(f"     Models show varying performance across subjects")
            print(f"     → Suggests different strengths/weaknesses, not purely random")
        
        return
    
    total_questions = len(patterns['question_correctness'])
    
    print(f"\nTotal questions analyzed: {total_questions}")
    print(f"Questions all models got correct: {len(patterns['all_correct'])} ({100*len(patterns['all_correct'])/total_questions:.1f}%)")
    print(f"Questions all models got wrong: {len(patterns['all_wrong'])} ({100*len(patterns['all_wrong'])/total_questions:.1f}%)")
    print(f"Questions with mixed results: {len(patterns['mixed'])} ({100*len(patterns['mixed'])/total_questions:.1f}%)")
    
    # Count mistakes per model
    model_mistakes = defaultdict(int)
    for q_key, correctness in patterns['question_correctness'].items():
        for model_name in correctness['wrong']:
            model_mistakes[model_name] += 1
    
    print(f"\nMistakes per model:")
    for model_result in results_data['model_results']:
        model_name = model_result['model_name'].split('/')[-1]
        print(f"  {model_name}: {model_mistakes[model_name]} mistakes")
    
    # Analyze if mistakes are random or systematic
    if len(patterns['all_wrong']) > 0:
        print(f"\n{'='*70}")
        print("SYSTEMATIC PATTERNS DETECTED:")
        print(f"  - {len(patterns['all_wrong'])} questions were difficult for ALL models")
        print(f"    This suggests these questions are genuinely difficult or ambiguous")
    
    if len(patterns['mixed']) > 0:
        overlap_ratio = len(patterns['mixed']) / total_questions
        print(f"\n  - {len(patterns['mixed'])} questions ({100*overlap_ratio:.1f}%) had mixed results")
        print(f"    Models disagree on these questions - suggests different strengths/weaknesses")
    
    # Check correlation of mistakes
    models = [mr['model_name'].split('/')[-1] for mr in results_data['model_results']]
    if len(models) >= 2:
        # Count questions where both models got wrong
        both_wrong = 0
        for q_key, correctness in patterns['question_correctness'].items():
            if len(correctness['wrong']) >= 2:
                both_wrong += 1
        
        if both_wrong > 0:
            correlation = both_wrong / total_questions
            print(f"\n  - Mistake correlation: {100*correlation:.1f}% of questions had multiple models failing")
            if correlation > 0.3:
                print(f"    HIGH correlation - models make similar mistakes (not random)")
            elif correlation > 0.1:
                print(f"    MODERATE correlation - some systematic patterns")
            else:
                print(f"    LOW correlation - mistakes appear more random")

def main():
    # Find the most recent results file
    result_files = sorted(Path('.').glob('multi_model_mmlu_results_*.json'), reverse=True)
    
    if not result_files:
        print("No results file found. Please run the evaluation first.")
        sys.exit(1)
    
    results_file = result_files[0]
    print(f"Loading results from: {results_file}")
    
    results_data = load_results(results_file)
    
    # Analyze mistake patterns
    patterns = analyze_mistake_patterns(results_data)
    
    # Print summary
    print_analysis_summary(results_data, patterns)
    
    # Create visualizations
    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)
    
    output_dir = Path('results_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Accuracy by subject
    fig1 = plot_accuracy_by_subject(results_data)
    fig1.savefig(output_dir / 'accuracy_by_subject.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'accuracy_by_subject.png'}")
    
    # Plot 2: Overall accuracy
    fig2 = plot_overall_accuracy(results_data)
    fig2.savefig(output_dir / 'overall_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'overall_accuracy.png'}")
    
    # Plot 3: Mistake overlap (if question details available)
    fig3 = plot_mistake_overlap(results_data)
    if fig3:
        fig3.savefig(output_dir / 'mistake_overlap.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / 'mistake_overlap.png'}")
    
    # Plot 4: Mistake overlap matrix (if question details available)
    fig4 = plot_mistake_overlap_matrix(results_data)
    if fig4:
        fig4.savefig(output_dir / 'mistake_overlap_matrix.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir / 'mistake_overlap_matrix.png'}")
    
    # Plot 5: Subject difficulty
    fig5 = plot_subject_difficulty(results_data)
    fig5.savefig(output_dir / 'subject_difficulty.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'subject_difficulty.png'}")
    
    print(f"\n✓ All visualizations saved to: {output_dir}/")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
