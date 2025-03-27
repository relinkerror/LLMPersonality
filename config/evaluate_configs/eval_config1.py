import os

CONFIG = {
    "test_data_path": os.path.join("..", "datax", "test_data-10K.parquet"),
    "base_model_path": os.path.join("..", "models", "deepseek-llm-7b-chat"),
    "adapter_path": os.path.join("..", "models", "Psychology-10k"),
    "tokenizer_path": os.path.join("..", "models", "Psychology-10k"),
    "eval_sample_count": 100,
    "shuffle_seed": 42,
    "generation_config": {
         "max_new_tokens": 256,
         "temperature": 0.7,
         "do_sample": True,
    },
    # 增加评估指标的配置，其中键为指标名称，值为传递给 compute 的额外参数
    "metrics": {
         """"rouge": {"use_stemmer": True},  # 对于 rouge 指标，启用词干处理
         "bleu": {},                     # bleu 无额外参数
         # 未来可以继续添加其他指标，如 "bertscore": {"lang": "zh", "model_type": "bert-base-chinese"}"""
         "BigFive":{
               "system_prompt": "这是新的系统提示词，请根据以下问题回答：",
               "questions": [
                    "问题1：请解释……",
                    "问题2：如何看待……",
                    # 更多固定问题...
               ],
               # 其他评估相关参数...
               }
    }
}
