import torch
print(torch.cuda.is_available())  # 应输出True
print(torch.version.cuda)  # 应显示与nvidia-smi匹配的CUDA版本