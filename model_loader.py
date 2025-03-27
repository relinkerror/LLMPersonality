import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model(model_path: str, use_4bit: bool = True, adapter_path: str = None):
    """
    加载指定路径的模型和分词器，配置4-bit量化。
    如果提供了 adapter_path，则加载适配器权重。

    参数:
      - model_path: 模型目录路径
      - use_4bit: 是否使用4-bit量化（默认 True）
      - adapter_path: 适配器权重目录路径（可选）
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # 如果提供了适配器权重路径，则加载适配器
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        # 如有需要，可以选择合并适配器到基础模型
        # model = model.merge_and_unload()
        
    return tokenizer, model


