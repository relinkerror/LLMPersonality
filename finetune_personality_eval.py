import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_loader import load_model
from evaluate_BigFive import EvaluateBigFive
import gc
import torch

def plot_personality_boxplot(model_label, evaluation_result, output_dir):
    """
    Generate a box plot for a single model's evaluation result.
    
    evaluation_result: A dictionary returned by EvaluateBigFive.evaluate(), where keys represent the Big Five traits 
    (e.g., "neuroticism", "extraversion", etc.), and each value contains a "metrics" list with scores for each round.
    """
    data_entries = []
    for personality, data in evaluation_result.items():
        scores = data.get("metrics", [])
        for s in scores:
            data_entries.append({
                "personality": personality,
                "score": s,
            })
    if not data_entries:
        print(f"No data available to plot for: {model_label}")
        return

    df = pd.DataFrame(data_entries)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    personality_order = ["neuroticism", "extraversion", "openness", "agreeableness", "conscientiousness"]
    ax = sns.boxplot(x="personality", y="score", data=df, order=personality_order, showmeans=True, meanline=True)
    # Overlay a strip plot to display individual data points
    sns.stripplot(x="personality", y="score", data=df, order=personality_order, color="black", alpha=0.6)
    plt.xlabel("Personality Trait")
    plt.ylabel("Score")
    plt.title(f"Big Five Personality Test - {model_label}")
    file_name = f"big_five_boxplot_{model_label}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {file_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Big Five personality questionnaire using LoRA fine-tuned models, comparing the base model with multiple fine-tuned models."
    )
    parser.add_argument("--model_path", type=str, default="./models/DeepSeek-R1-Distill-Qwen-7B/",
                        help="Path to the base model, e.g., './models/DeepSeek-R1-Distill-Qwen-7B/'")
    parser.add_argument("--adapter_paths", type=str, nargs="+", default=[],
                        help="Paths to multiple LoRA adapters (fine-tuned models), e.g., './adapters/model1/', './adapters/model2/'")
    parser.add_argument("--num_rounds", type=int, default=2, help="Number of evaluation rounds")
    parser.add_argument("--output_dir", type=str, default="plot",
                        help="Directory to save output plots. All outputs will be stored in this folder (default: 'plot').")
    args = parser.parse_args()

    # Use a unified system prompt for all evaluations
    system_prompt = "***You are participating in a survey. Please answer the following questions honestly.*** " 

    evaluation_results = []

    # 1. Evaluate the base model (no adapter) and label it as "Base"
    print("===== Evaluating Base Model (No Fine-tuning) =====")
    tokenizer, model = load_model(args.model_path, adapter_path=None)
    evaluator = EvaluateBigFive(tokenizer, model, num_rounds=args.num_rounds, system_prompt=system_prompt)
    result = evaluator.evaluate()
    evaluation_results.append(("Base", result))

    # 在评估完当前模型后清理内存
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Evaluate each fine-tuned model using its corresponding LoRA adapter
    for adapter_path in args.adapter_paths:
        print(f"===== Evaluating Fine-tuned Model, Adapter Path: {adapter_path} =====")
        tokenizer, model = load_model(args.model_path, adapter_path=adapter_path)
        evaluator = EvaluateBigFive(tokenizer, model, num_rounds=args.num_rounds, system_prompt=system_prompt)
        result = evaluator.evaluate()
        # Use the last directory name of the adapter path as the model label
        model_label = os.path.basename(adapter_path.rstrip("/"))
        evaluation_results.append((model_label, result))
        # 在评估完当前模型后清理内存
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Plot the evaluation results for each model as a box plot (showing all five personality traits)
    for label, res in evaluation_results:
        plot_personality_boxplot(label, res, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
