from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch
from peft import LoraConfig, TaskType, get_peft_model
import argparse

def process_func(examples: dict[str, list[any]]) -> dict[str, list[any]]:
    input_ids_batch, attention_mask_batch, labels_batch = [], [], []

    for idx, messages in enumerate(examples["messages"]):
        # 初始调试：打印原始 messages
        if idx < 5:
            print(f"[DEBUG idx={idx}] raw messages: {messages}")

        # 拆分历史对话与回复
        prompt_messages, reply = messages[:-1], messages[-1]["content"]

        # 调试：打印 reply
        if idx < 5:
            print(f"[DEBUG idx={idx}] reply: {reply}")

        # 构造 prompt 串（不做编码）
        prompt_str = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 调试：打印 prompt_str
        if idx < 5:
            print(f"[DEBUG idx={idx}] prompt_str: {prompt_str}")

        # 拼接完整生成序列
        full_str = prompt_str + reply + tokenizer.eos_token
        # 调试：打印 full_str 预览和长度
        if idx < 5:
            print(f"[DEBUG idx={idx}] full_str preview: {full_str[:100]}...")
            print(f"[DEBUG idx={idx}] full_str length: {len(full_str)}")

        # 编码完整序列，保证 fixed_length
        tokenized_full = tokenizer(
            full_str,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_attention_mask=True
        )
        ids = tokenized_full["input_ids"]
        mask = tokenized_full["attention_mask"]
        # 调试：打印 tokenized_full 关键信息
        if idx < 5:
            print(f"[DEBUG idx={idx}] input_ids[:10]: {ids[:10]}")
            print(f"[DEBUG idx={idx}] attention_mask sum: {sum(mask)}")

        # 编码 prompt 串，不做 padding，以准确计算长度
        tokenized_prompt = tokenizer(
            prompt_str,
            truncation=True,
            max_length=512,
            padding=False,
            return_attention_mask=True
        )
        # 真实 prompt 长度等于其 attention_mask 的有效 token 数
        prompt_len = int(sum(tokenized_prompt["attention_mask"]))
        # 调试：打印 prompt_len
        if idx < 5:
            print(f"[DEBUG idx={idx}] prompt_len: {prompt_len}")

        # 构造 labels：前 prompt_len 个置 -100，后续保留真实 id
        labels = ids.copy()
        labels[:prompt_len] = [-100] * prompt_len
        # 调试：打印 labels 前 10 个
        if idx < 5:
            print(f"[DEBUG idx={idx}] labels[:10]: {labels[:10]}")

        # 聚合到 batch 列表
        input_ids_batch.append(ids)
        attention_mask_batch.append(mask)
        labels_batch.append(labels)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch,
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
