import argparse
import gc
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

# 自定义 Trainer，确保计算 loss 时 labels 在 logits 同一设备上
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, **kwargs):
        # 提取 labels 并确保在与 logits 同一设备上
        if "labels" in inputs:
            labels = inputs["labels"]
        else:
            labels = None
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if labels is not None:
            inputs["labels"] = labels.to(logits.device)
        loss = outputs.loss
        return loss

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
    tokenized_dataset = ds.map(lambda ex: process_func(ex, tokenizer), remove_columns=ds.column_names)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.half,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
    )
    print("模型加载后 - allocated:", torch.cuda.memory_allocated())
    print("模型加载后 - reserved:", torch.cuda.memory_reserved())

    model.generation_config = GenerationConfig.from_pretrained(args.model_dir)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=50,
        num_train_epochs=3,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch",
        eval_steps=500,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    print("训练结束后 - allocated:", torch.cuda.memory_allocated())
    print("训练结束后 - reserved:", torch.cuda.memory_reserved())

    test_text = "你好！"
    inputs = tokenizer(f"User: {test_text}\n\n", return_tensors="pt")
    inputs = inputs.to(model.device)
    print("生成前 - allocated:", torch.cuda.memory_allocated())
    print("生成前 - reserved:", torch.cuda.memory_reserved())

    outputs = model.generate(**inputs, max_new_tokens=100)
    print("生成后 - allocated:", torch.cuda.memory_allocated())
    print("生成后 - reserved:", torch.cuda.memory_reserved())

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("生成结果：", result)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
    print("程序结束时 - allocated:", torch.cuda.memory_allocated())
    print("程序结束时 - reserved:", torch.cuda.memory_reserved())
 
