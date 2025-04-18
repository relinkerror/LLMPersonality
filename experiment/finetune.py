'''import argparse
import gc
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    GenerationConfig,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model

def process_func(example, tokenizer, max_length=384):
    """
    将 messages 转换为用于指令微调的训练样本
    """
    # 拼接完整对话作为上下文
    dialogue = ""
    for message in example["messages"]:
        role = message["role"]
        content = message["content"]
        if role == "system":
            dialogue += f"[SYSTEM]: {content}\n"
        elif role == "user":
            dialogue += f"User: {content}\n"
        elif role == "assistant":
            dialogue += f"Assistant: {content}\n"

    # 查找最后一个 user -> assistant 的配对
    last_user_input = ""
    target_output = ""
    for i in range(len(example["messages"]) - 2, -1, -1):
        if (example["messages"][i]["role"] == "user" and
            example["messages"][i + 1]["role"] == "assistant"):
            last_user_input = example["messages"][i]["content"]
            target_output = example["messages"][i + 1]["content"]
            break

    if last_user_input and target_output:
        instruction = tokenizer(f"User: {last_user_input}\n\n", add_special_tokens=False)
        response = tokenizer(f"Assistant: {target_output}<|endofsentence|>", add_special_tokens=False)
    else:
        return {"input_ids": [], "attention_mask": [], "labels": []}  # 跳过无效对话

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    # 设置环境变量，指定使用的 GPU（例如，使用索引为 0 的 GPU）
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.cuda.empty_cache()
    gc.collect()
    print("程序开始时 - allocated:", torch.cuda.memory_allocated())
    print("程序开始时 - reserved:", torch.cuda.memory_reserved())

    parser = argparse.ArgumentParser(description="微调模型参数设置")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="JSONL 数据集路径")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="模型及分词器目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="训练后保存模型的目录")
    args = parser.parse_args()

    ds = Dataset.from_json(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    # 数据集预处理后：
    tokenized_dataset = ds.map(lambda ex: process_func(ex, tokenizer), remove_columns=ds.column_names)
    # 设置格式，确保转换后的张量为整数类型
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.half,
        low_cpu_mem_usage=True,
    )

    # 使用 LoRA 包装模型，目标是只微调 "q_proj" 和 "v_proj" 层
    model = get_peft_model(model, lora_config)


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=10,
        num_train_epochs=3,
        gradient_checkpointing=True,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )


    trainer.train()

    test_text = "你好！"
    inputs = tokenizer(f"User: {test_text}\n\n", return_tensors="pt")
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
    print("程序结束时 - allocated:", torch.cuda.memory_allocated())
    print("程序结束时 - reserved:", torch.cuda.memory_reserved())
'''
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch
from peft import LoraConfig, TaskType, get_peft_model
import argparse

def process_func(examples: dict[str, list[any]]) -> dict[str, list[any]]:
    input_ids_batch, attention_mask_batch, labels_batch = [], [], []

    for messages in examples["messages"]:
        # 拆分历史对话与回复
        prompt_messages, reply = messages[:-1], messages[-1]["content"]

        # 构造 prompt 串（不做编码）
        prompt_str = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 拼接完整生成序列
        full_str = prompt_str + reply + tokenizer.eos_token

        # 使用同一配置编码完整序列
        tokenized_full = tokenizer(
            full_str,
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        ids  = tokenized_full["input_ids"]
        mask = tokenized_full["attention_mask"]

        # 用相同配置编码 prompt，准确计算长度
        prompt_ids = tokenizer(
            prompt_str,
            truncation=True,
            max_length=512,
            padding="max_length"
        )["input_ids"]
        prompt_len = len(prompt_ids)

        # 构造 labels：前 prompt_len 个置 -100，后续保留真实 id
        labels = ids.copy()
        labels[:prompt_len] = [-100] * prompt_len

        # 聚合到 batch 列表
        input_ids_batch.append(ids)
        attention_mask_batch.append(mask)
        labels_batch.append(labels)

    return {
        "input_ids":      input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels":         labels_batch,
    }

# 应用方式：
# ds = ds.map(process_func, batched=True, remove_columns=["messages"])




# lora 配置参数
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "v_proj"],  # 不同模型可能需要设置不同的参数，需要看模型中的attention层
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=16,  # Lora alpha，具体作用参见 Lora 原理
    lora_dropout=0.2  # Dropout 比例
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="微调模型参数设置")
    parser.add_argument("--dataset_path", type=str, required=True, help="CSV 数据集路径")
    parser.add_argument("--model_dir", type=str, required=True, help="模型及分词器目录")
    parser.add_argument("--output_dir", type=str, required=True, help="训练后保存模型的目录")
    #parser.add_argument("--personality", type=str, required=True, help="人格特质")
    args = parser.parse_args()

    
    df = pd.read_json(args.dataset_path, lines=True, orient='records')
    ds = Dataset.from_pandas(df)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id  # 将eos_token_id设为pad_token_id

    # 将数据集转换为 token 形式，并传入 personality 参数
    tokenized_ds = ds = ds.map(process_func, batched=True, remove_columns=["messages"])

    # 加载模型，以半精度形式加载
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, 
        trust_remote_code=True, 
        torch_dtype=torch.half, 
        device_map="auto"
    )
    model.enable_input_require_grads()  # 开启梯度检查点时需要调用该方法
    # 加载 LoRA 参数
    model = get_peft_model(model, config)

    # 设置训练参数
    trainer_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=10,
        num_train_epochs=1,
        gradient_checkpointing=True,
        save_steps=1000,
        learning_rate=1e-5,
        save_on_each_node=True
    )
    
    # 实例化 Trainer
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    # 开始训练
    trainer.train()
