import pandas as pd
import torch
from datasets import Dataset
from evaluate import load
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import matplotlib.pyplot as plt

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用 4-bit 量化
    bnb_4bit_compute_dtype=torch.float16,  # 计算时使用 FP16
    bnb_4bit_quant_type="nf4",  # 量化类型（nf4 比 fp4 误差更小）
    bnb_4bit_use_double_quant=True  # 双重量化（节省显存）
)

test_ds = Dataset.from_parquet("test_data-10K.parquet")

# 加载基础模型（需提前下载或本地存在）
base_model = AutoModelForCausalLM.from_pretrained(
    'deepseek-llm-7b-chat/',  
    trust_remote_code=True,  
    quantization_config=bnb_config,  # ✅ 正确的 4-bit 量化配置
    device_map="auto",
    low_cpu_mem_usage=True  # 降低 CPU 内存占用
)

# 加载适配器权重
model = PeftModel.from_pretrained(
    base_model, 
    "./Psychology-10k"  # 包含 adapter_config.json 和 adapter_model.bin 的目录
)
#model = model.merge_and_unload()  # 合并适配器到基础模型（可选）

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("./Psychology-10k")

def generate_answer(example, max_new_tokens=256):
    """生成模型回答"""
    prompt = f"Human: {example['instruction']}\n{example['input']}\n\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Assistant:")[-1].strip()

# 随机采样100条进行评估（可根据显存调整）
eval_samples = test_ds.shuffle(seed=42).select(range(100))

# 生成预测结果
predictions = [generate_answer(sample) for sample in eval_samples]
references = [[sample["output"].split("Assistant:")[-1].strip()] for sample in eval_samples]

# 计算指标
"""rouge = load("rouge")"""
bleu = load("bleu")
# bertscore = load("bertscore")

"""rouge_results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)"""
bleu_results = bleu.compute(predictions=predictions, references=references)
# bertscore_results = bertscore.compute(
    # predictions=predictions, references=references, lang="zh", model_type="bert-base-chinese"
# )

# 指标汇总
metrics = {
    """"ROUGE-1": round(rouge_results["rouge1"], 4),
    "ROUGE-L": round(rouge_results["rougeL"], 4),"""
    "BLEU": round(bleu_results["bleu"], 4),
    # "BERTScore": round(sum(bertscore_results["f1"])/len(bertscore_results["f1"]), 4),
}

print("\n自动评估结果:")
for k, v in metrics.items():
    print(f"{k}: {v}")