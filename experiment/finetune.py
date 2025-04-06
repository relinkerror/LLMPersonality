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
    
    print(f"加载 CSV 数据集：{args.dataset_path}")
    ds = Dataset.from_csv(args.dataset_path)
    print(f"成功加载数据集，共 {len(ds)} 条数据")
    
    print(f"加载分词器：{args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    print("分词器加载成功")
    
    print("开始对数据集进行预处理...")
    tokenized_dataset = ds.map(lambda ex: process_func(ex, tokenizer), remove_columns=ds.column_names)
    print("数据集预处理完成")
    
    print(f"加载预训练模型：{args.model_dir}")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, 
        trust_remote_code=True, 
        torch_dtype=torch.half, 
        device_map="auto",
        low_cpu_mem_usage=True,
        #quantization_config=bnb_config,
    )
    print("预训练模型加载成功")
    
    print("配置生成参数...")
    model.generation_config = GenerationConfig.from_pretrained(args.model_dir)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    print("生成参数配置完成")
    
    print("开启梯度检查点...")
    model.enable_input_require_grads()
    
    print("配置 LoRA 参数并改造模型...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("LoRA 配置完成")
    
    print("配置训练参数...")
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
    print("训练参数配置完成")
    
    print("构建 Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    print("Trainer 构建成功")
    
    print("开始训练...")
    trainer.train()
    print("训练完成")
    
    print("测试生成效果...")
    test_text = "I loved."
    inputs = tokenizer(f"User: {test_text}\n\n", return_tensors="pt")
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("生成结果：", result)
    
    print(f"保存模型到 {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("模型和分词器保存成功")

if __name__ == "__main__":
    main()
