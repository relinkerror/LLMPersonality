import os
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, 
                          DataCollatorForSeq2Seq, TrainingArguments, 
                          Trainer, GenerationConfig)
from peft import LoraConfig, TaskType, get_peft_model

# 全局变量 tokenizer 便于 process_func 中使用
tokenizer = None

def process_func(example):
    MAX_LENGTH = 384
    # 这里将 instruction 和 input 拼接起来
    instruction = tokenizer(
        f"User: {example['instruction']}{example['input']}\n\n", 
        add_special_tokens=False
    )
    response = tokenizer(
        f"Assistant: {example['output']}<｜end▁of▁sentence｜>", 
        add_special_tokens=False
    )
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 若超过最大长度，则截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    # 定义各目录路径（注意根据实际情况调整相对路径）
    data_path = os.path.join("..", "datax", "train_data-10K.parquet")
    model_path = os.path.join("..", "models", "deepseek-llm-7b-chat")
    output_dir = os.path.join("..", "output", "DeepSeek")
    final_model_dir = os.path.join("..", "models", "Psychology-10k")
    tokenizer_save_dir = os.path.join("..", "models", "Psychology-10K")
    
    # 加载数据集
    ds = Dataset.from_parquet(data_path)
    
    # 加载分词器
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    
    # 处理数据集
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.half,
        device_map="auto",
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.half,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # 配置生成参数
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    # 开启梯度检查点
    model.enable_input_require_grads()
    
    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit"
    )
    
    # 构建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    
    # 开始训练
    trainer.train()
    
    # 手动保存模型和分词器
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(tokenizer_save_dir)
    
    # 测试生成结果
    test_text = "I loved."
    inputs = tokenizer(f"User: {test_text}\n\n", return_tensors="pt")
    outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("生成结果：", result)

if __name__ == "__main__":
    main()
