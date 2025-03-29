import argparse
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
    处理单个样本，将 CSV 中的 Input 和 Output 拼接成训练所需的 token id 序列
    """
    # 构造输入文本：使用 CSV 中的 "Input" 作为用户发言
    instruction = tokenizer(f"User: {example['Input']}\n\n", add_special_tokens=False)
    # 构造输出文本：使用 CSV 中的 "Output" 作为助手回复，并添加结束标识
    response = tokenizer(f"Assistant: {example['Output']}<|endofsentence|>", add_special_tokens=False)
    
    # 拼接得到整个样本的 token id 序列，并在最后添加 pad_token_id
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # 拼接 attention mask（对 eos token 同样标记为 1）
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # 构造 labels：对输入部分不计算 loss，用 -100 表示，输出部分使用实际 token id
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 如果整体序列超过最大长度，则进行截断
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
    parser = argparse.ArgumentParser(description="微调模型参数设置")
    parser.add_argument("--dataset_path", type=str, default="./datax/CPED/extraversion_low_pairs.csv",
                        help="CSV 数据集路径")
    parser.add_argument("--model_dir", type=str, default="./models/QwQ-32B",
                        help="模型及分词器目录")
    parser.add_argument("--output_dir", type=str, default="./models/Extraversion_low",
                        help="训练后保存模型的目录")
    args = parser.parse_args()
    
    # 1. 加载 CSV 数据集
    ds = Dataset.from_csv(args.dataset_path)
    
    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    
    # 3. 对数据集进行预处理（这里使用 map 传入额外的 tokenizer 参数）
    tokenized_dataset = ds.map(lambda ex: process_func(ex, tokenizer), remove_columns=ds.column_names)
    
    # 4. 加载预训练模型（使用 8 位量化、低 CPU 内存模式等配置）
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, 
        trust_remote_code=True, 
        torch_dtype=torch.half, 
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )
    
    # 5. 配置生成参数，确保 pad_token 与 eos_token 一致
    model.generation_config = GenerationConfig.from_pretrained(args.model_dir)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # 6. 开启梯度检查点（必要时）
    model.enable_input_require_grads()
    
    # 7. 配置 LoRA 参数并对模型进行改造
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
    
    # 8. 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
    
    # 9. 构建 Trainer（这里使用 DataCollatorForSeq2Seq 进行动态 padding）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    # 10. 开始训练
    trainer.train()
    
    # 11. 测试生成效果
    test_text = "I loved."
    inputs = tokenizer(f"User: {test_text}\n\n", return_tensors="pt")
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("生成结果：", result)
    
    # 12. 保存最终模型与分词器
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
