import argparse
import matplotlib.pyplot as plt
from config.questionnaire_config import QUESTIONNAIRE_CONFIG
from questionnaire_evaluator import QuestionnaireEvaluator, simple_metric
from model_loader import load_model
import os
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import aiohttp
import asyncio

'''class EvaluateBigFive:
    def __init__(self, tokenizer, model, num_rounds=10, system_prompt=""):
        """
        初始化评估类

        参数:
         - tokenizer: 分词器
         - model: 加载好的模型
         - num_rounds: 测试轮数
         - system_prompt: 系统提示词，用于微调生成风格
        """
        self.tokenizer = tokenizer
        self.model = model
        self.evaluator = QuestionnaireEvaluator(tokenizer, model)
        self.num_rounds = num_rounds
        self.system_prompt = system_prompt

    def evaluate(self):
        """
        对每个问卷维度执行多轮评估，返回一个字典，
        每个维度包含各轮次得分列表和平均得分
        """
        all_results = {}
        for dimension, config in QUESTIONNAIRE_CONFIG.items():
            print(f"Processing dimension: {dimension}")
            standardize = config["standardize"]
            inputs = config["inputs"]
            reversed_indices = config["reversed_indices"]
            metrics = []
            # 重复测试 num_rounds 次
            for i in range(self.num_rounds):
                predictions = self.evaluator.generate_answers(standardize, inputs, system_prompt=self.system_prompt)
                print(predictions)
                metric = simple_metric(predictions, reversed_indices, fixed_n=len(inputs))
                print(metric)
                metrics.append(metric)
                print(f"Round {i+1} - {dimension} metric: {metric}")
            average_metric = sum(metrics) / len(metrics)
            all_results[dimension] = {"metrics": metrics, "average_metric": average_metric}
            print(f"{dimension.capitalize()} average metric: {average_metric}\n")
        return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Big Five Personality Questionnaire")
    parser.add_argument("--model_path", type=str, default="./models/deepseek-llm-7b-chat/",
                        help="模型路径，例如 './models/deepseek-llm-7b-chat/'")
    parser.add_argument("--system_prompt", type=str, default=" Please play someone with high extroversion ",
                        help="追加在提示词后的字符串")
    parser.add_argument("--num_rounds", type=int, default=2,
                        help="测试轮数")
    # 新增适配器权重参数，默认为 None（不加载适配器）
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="适配器权重目录路径，例如 'deepseek-llm-7b-chat/'，目录中应包含 adapter_config.json 和 adapter_model.bin")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path, adapter_path=args.adapter_path)
    evaluator = EvaluateBigFive(tokenizer, model, num_rounds=args.num_rounds, system_prompt=args.system_prompt)
    results = evaluator.evaluate()
    evaluator.plot_results(results,"deepseek-llm-7b-chat/",args.system_prompt)'''
class EvaluateBigFive:
    def __init__(self, client, num_rounds=20, system_prompt=""):
        """
        参数:
         - client: API 客户端实例
         - num_rounds: 测试轮数
         - system_prompt: 系统提示词
        """
        self.client = client
        self.evaluator = QuestionnaireEvaluator(client)
        self.num_rounds = num_rounds
        self.system_prompt = system_prompt

    '''def evaluate(self):
        """
        针对每个问卷维度执行多轮评估，返回字典，键为维度，值包含各轮得分列表及平均分
        """
        all_results = {}
        for dimension, config in QUESTIONNAIRE_CONFIG.items():
            print(f"Processing dimension: {dimension}")
            standardize = config["standardize"]
            inputs = config["inputs"]
            reversed_indices = config["reversed_indices"]
            metrics = []
            for i in range(self.num_rounds):
                predictions = self.evaluator.generate_answers(standardize, inputs, system_prompt=self.system_prompt)
                metric = simple_metric(predictions, reversed_indices, fixed_n=len(inputs))
                metrics.append(metric)
                print(f"Round {i+1} - {dimension} metric: {metric}")
            average_metric = sum(metrics) / len(metrics)
            all_results[dimension] = {"metrics": metrics, "average_metric": average_metric}
            print(f"{dimension.capitalize()} average metric: {average_metric}\n")
        return all_results'''
    
    async def evaluate(self):
        all_results = {}
        for dimension, config in QUESTIONNAIRE_CONFIG.items():
            print(f"Processing dimension: {dimension}")
            standardize = config["standardize"]
            inputs = config["inputs"]
            reversed_indices = config["reversed_indices"]
            
            # 创建一个任务列表，任务数量为 num_rounds
            tasks = [
                self.evaluator.generate_answers(standardize, inputs, system_prompt=self.system_prompt)
                for _ in range(self.num_rounds)
            ]
            # 并发执行所有任务
            predictions_list = await asyncio.gather(*tasks)
            
            metrics = []
            # 如果你希望按任务在任务列表中的顺序计算指标，
            # 那么 predictions_list 的顺序就与任务列表一致
            for predictions in predictions_list:
                print(predictions)
                metric = simple_metric(predictions, reversed_indices, fixed_n=len(inputs))
                metrics.append(metric)
                print(f"Metric: {metric}")
                
            average_metric = sum(metrics) / len(metrics)
            all_results[dimension] = {"metrics": metrics, "average_metric": average_metric}
            print(f"{dimension.capitalize()} average metric: {average_metric}\n")
        return all_results

