# questionnaire_evaluator.py
import re
import torch
import aiohttp
import asyncio

class QuestionnaireEvaluator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def generate_answer_for_input(self, prompt: str, max_new_tokens: int = 256, max_attempts: int = 5) -> int:
        """
        根据给定 prompt 生成答案，返回 1-5 之间的整数。
        多次尝试后仍未生成有效答案，则返回 3。
        """
        valid_answer = None
        attempt = 0
        while attempt < max_attempts and valid_answer is None:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            raw_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Raw answer: {raw_answer}")
            
            # 如果存在链式思考部分，则仅取</think>之后的文本进行处理
            if "</think>" in raw_answer:
                relevant_text = raw_answer.split("</think>")[-1]
            else:
                relevant_text = raw_answer

            # 按行查找，逆序选择最后一行中只含单个数字的候选答案
            lines = relevant_text.splitlines()
            candidate = None
            for line in reversed(lines):
                line_stripped = line.strip()
                if re.fullmatch(r'[1-5]', line_stripped):
                    candidate = line_stripped
                    break

            if candidate:
                valid_answer = int(candidate)
            else:
                # 如果按行提取未果，再使用正则在整个相关文本中查找最后一个符合条件的数字
                matches = re.findall(r'\b([1-5])\b', relevant_text)
                if matches:
                    # 这里选择最后一个匹配项，确保尽可能提取最终的答案数字
                    valid_answer = int(matches[-1])
                else:
                    attempt += 1
            
            print(f"Answer :{valid_answer}")

        return valid_answer if valid_answer is not None else 3

    def generate_answers(self, standardize: str, inputs: list, system_prompt: str = "") -> list:
        """
        针对一组问卷题目生成答案列表。

        参数:
         - instruction: 指令提示词
         - inputs: 问题列表
         - system_prompt: 系统提示（可选）
         - append_str: 追加在提示词后的字符串（可选）
        """
        full_system_prompt = system_prompt + standardize
        answers = []
        for inp in inputs:
            prompt = (
                f"User: {full_system_prompt}\n"
                f"Human: {inp}\n\nAssistant:"
            )
            answer = self.generate_answer_for_input(prompt)
            answers.append(answer)
        return answers

def simple_metric(predictions: list, reversed_indices: list, fixed_n: int) -> float:
    """
    计算问卷得分：
      - 正向题目得分直接使用预测值；
      - 反向题目的得分为：6 - 预测值；
      - 最后计算所有题目的平均得分，并归一化到0-1之间（满分为1）。
    """
    if len(predictions) == 1 and isinstance(predictions[0], list):
        predictions = predictions[0]

    total_score = 0
    for i, pred in enumerate(predictions):
        if isinstance(pred, list):
            pred = pred[0]
        score = (6 - pred) if i in reversed_indices else pred
        total_score += score

    average_score = total_score / fixed_n
    normalized_metric = average_score / 5
    return normalized_metric
'''class QuestionnaireEvaluator:
    def __init__(self, client, max_concurrency=479):
        self.client = client
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def async_generate_answer(self, session: aiohttp.ClientSession, messages: list) -> int:
        """异步生成单个答案"""
        async with self.semaphore:
            for attempt in range(5):
                try:
                    async with session.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.client.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "deepseek-chat",
                            "messages": messages,
                            "temperature": 0.7
                        }
                    ) as response:
                        data = await response.json()
                        raw_answer = data['choices'][0]['message']['content']
                        if re.match(r'^[1-5]$', raw_answer.strip()):
                            return int(raw_answer.strip())
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {str(e)}")
                    await asyncio.sleep(2 ** attempt)
        return 3

    async def generate_answers(self, standardize: str, inputs: list,system_prompt) -> list:
        """批量异步生成答案"""
        full_system_prompt = system_prompt + standardize
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for inp in inputs:
                messages = [
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": f"{inp}（请用1-5的整数回答）"}
                ]
                tasks.append(self.async_generate_answer(session, messages))
            
            return await asyncio.gather(*tasks)

def simple_metric(predictions: list, reversed_indices: list, fixed_n: int) -> float:
    """
    计算问卷得分，归一化到 0-1
    """
    if len(predictions) == 1 and isinstance(predictions[0], list):
        predictions = predictions[0]

    total_score = 0
    for i, pred in enumerate(predictions):
        if isinstance(pred, list):
            pred = pred[0]
        score = (6 - pred) if i in reversed_indices else pred
        total_score += score

    average_score = total_score / fixed_n
    normalized_metric = average_score / 5
    return normalized_metric'''
