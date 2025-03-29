import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_loader import load_model
from evaluate_BigFive import EvaluateBigFive
import openai
import aiohttp
import asyncio

def extract_variant_and_personality(system_prompt):
    """
    从系统提示中提取变体（低/中/高）和目标人格特征
    假设格式类似 "Please play someone with low neuroticism"
    返回：(variant, prompt_personality)
    """
    prompt_clean = system_prompt.strip().lower()
    words = prompt_clean.split()
    if len(words) >= 6:
        variant = words[-2]  # 例如 "low", "high", "moderate"
        prompt_personality = words[-1].strip(" ,.")
    else:
        variant = "unknown"
        prompt_personality = "unknown"
    return variant, prompt_personality

def plot_grouped_results(evaluation_data, output_dir="plot"):
    """
    evaluation_data: list of元组，每个元组为 (system_prompt, results)
    每个 results 为 EvaluateBigFive.evaluate() 返回的字典，
    字典的 key 为各人格维度（如 "neuroticism", "extroversion", 等），
    每个 value 中包含 "metrics" 字段（即各轮次得分列表）。

    此函数将所有评估数据合并，构造包含以下字段的 DataFrame：
       - score_personality: 得分对应的人格维度（来自结果字典的 key）
       - prompt_personality: 系统提示中指定的人格特征（通过解析 system_prompt 得到）
       - variant: 系统提示中指定的变体（如 low、moderate、high）
       - score: 得分值

    绘图时，针对每个得分人格维度（score_personality）绘制一张图，
    横轴为系统提示中的目标人格（prompt_personality，固定顺序），
    并以 variant 作为 hue，保证相同变体使用同一颜色。
    """
    data_entries = []
    for system_prompt, results in evaluation_data:
        variant, prompt_personality = extract_variant_and_personality(system_prompt)
        # 对结果字典中所有人格维度进行遍历
        for personality, data in results.items():
            scores = data["metrics"]
            for s in scores:
                data_entries.append({
                    "score_personality": personality,      # 评价得到的得分对应的人格维度
                    "prompt_personality": prompt_personality,  # 系统提示中指定的人格
                    "variant": variant,                      # 低/中/高
                    "score": s,
                    "system_prompt": system_prompt           # 可选：原始提示文本
                })
    
    if not data_entries:
        print("没有数据可绘图。")
        return

    df = pd.DataFrame(data_entries)
    os.makedirs(output_dir, exist_ok=True)

    # 固定变体顺序及对应的颜色映射（同一变体使用相同颜色）
    variant_order = ["low", "moderate", "high"]
    palette = {"low": "skyblue", "moderate": "lightgreen", "high": "salmon"}

    # 固定横轴组顺序（系统提示中的目标人格）
    prompt_order = ["neuroticism", "extraversion", "openness", "agreeableness", "conscientiousness"]

    # 针对每个得分人格（score_personality）绘图
    for sp in df["score_personality"].unique():
        df_sp = df[df["score_personality"] == sp]
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        # 使用 boxplot，横轴为 prompt_personality，hue 为 variant
        ax = sns.boxplot(
            x="prompt_personality",
            y="score",
            hue="variant",
            data=df_sp,
            order=prompt_order,
            hue_order=variant_order,
            palette=palette,
            showmeans=True,
            meanline=True,
            width=0.5
        )
        # 叠加散点图，显示每个数据点位置
        sns.stripplot(
            x="prompt_personality",
            y="score",
            hue="variant",
            data=df_sp,
            order=prompt_order,
            hue_order=variant_order,
            dodge=True,
            palette=palette,
            color="black",
            alpha=0.6
        )
        # 为避免重复图例，取 boxplot 的图例项
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[0:len(variant_order)], labels[0:len(variant_order)], title="Variant")
        plt.xlabel("System Prompt Target Personality")
        plt.ylabel("Normalized Score")
        plt.title(f"Big Five Scores Distribution for {sp.capitalize()} (All System Prompts)")
        file_name = f"big_five_boxplot_all_{sp}.png"
        file_path = os.path.join(output_dir, file_name)
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"保存图像至: {file_path}")

'''def main():
    parser = argparse.ArgumentParser(
        description="Batch Evaluate Big Five Personality Questionnaire with different append_str"
    )
    parser.add_argument("--model_path", type=str, default="./models/deepseek-llm-7b-chat/",
                        help="模型路径，例如 './models/deepseek-llm-7b-chat/'")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="适配器权重目录路径，例如 'deepseek-llm-7b-chat/'，目录中应包含 adapter_config.json 和 adapter_model.bin")
    parser.add_argument("--num_rounds", type=int, default=2,
                        help="测试轮数")
    args = parser.parse_args()

    # 定义你要测试的系统提示列表
    system_prompt_list = [
        " Please play someone with low neuroticism ",
        " Please play someone with moderate neuroticism ",
        " Please play someone with high neuroticism ",

        " Please play someone with low extroversion ",
        " Please play someone with moderate extroversion ",
        " Please play someone with high extroversion ",

        " Please play someone with low openness ",
        " Please play someone with moderate openness ",
        " Please play someone with high openness ",

        " Please play someone with low agreeableness ",
        " Please play someone with moderate agreeableness ",
        " Please play someone with high agreeableness ",

        " Please play someone with low conscientiousness ",
        " Please play someone with moderate conscientiousness ",
        " Please play someone with high conscientiousness ",
    ]

    # 加载模型和分词器
    tokenizer, model = load_model(args.model_path, adapter_path=args.adapter_path)
    
    # 存储所有评估结果，便于后续按人格类型分组绘图
    evaluation_data = []
    
    for prompt in system_prompt_list:
        print(f"===== Running evaluation with prompt: {prompt} =====")
        evaluator = EvaluateBigFive(tokenizer, model, num_rounds=args.num_rounds, system_prompt=prompt)
        results = evaluator.evaluate()
        evaluation_data.append((prompt, results))
    
    # 将相同人格类型下的评估结果绘制在一起
    plot_grouped_results(evaluation_data, output_dir="plot")

if __name__ == "__main__":
    main()'''
def main():
    parser = argparse.ArgumentParser(
        description="通过 API 批量评估 Big Five 人格问卷"
    )
    parser.add_argument("--num_rounds", type=int, default=20, help="测试轮数")
    args = parser.parse_args()

    client = openai.OpenAI(
        api_key="",  # 请替换为实际的 API 密钥
        base_url="https://api.deepseek.com"
    )

    system_prompt_list = [
        " Please play someone with low neuroticism ",
        " Please play someone with moderate neuroticism ",
        " Please play someone with high neuroticism ",

        " Please play someone with low extraversion ",
        " Please play someone with moderate extraversion ",
        " Please play someone with high extraversion ",

        " Please play someone with low openness ",
        " Please play someone with moderate openness ",
        " Please play someone with high openness ",

        " Please play someone with low agreeableness ",
        " Please play someone with moderate agreeableness ",
        " Please play someone with high agreeableness ",

        " Please play someone with low conscientiousness ",
        " Please play someone with moderate conscientiousness ",
        " Please play someone with high conscientiousness ",
    ]

    evaluation_data = []
    for prompt in system_prompt_list:
        print(f"===== Running evaluation with prompt: {prompt} =====")
        evaluator = EvaluateBigFive(
            client=client,
            system_prompt=prompt,
            num_rounds=args.num_rounds
        )
        results = asyncio.run(evaluator.evaluate())
        evaluation_data.append((prompt, results))

    plot_grouped_results(evaluation_data, output_dir="plot")

if __name__ == "__main__":
    main()

