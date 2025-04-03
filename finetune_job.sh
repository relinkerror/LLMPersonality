#!/bin/bash
#SBATCH --job-name=ai_job           # 作业名称
#SBATCH --output=output.log         # 标准输出文件
#SBATCH --error=error.log           # 错误输出文件
#SBATCH --ntasks=1                  # 任务数
#SBATCH --time=02:00:00             # 运行时间上限
#SBATCH --gres=gpu:1                # 请求 GPU 数量

# 加载必要模块
module load python/3.12
pip install torch --no-index

# 执行你的 AI 程序
python experiment/finetune.py