#!/bin/bash
#SBATCH --job-name=finetune_eval
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --ntasks=1
#SBATCH --time=7:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

# 加载必要模块
module load python/3.12
module load gcc
module load arrow/19
source /project/6078835/gn533549/LLMPersonality/ENV/bin/activate
pip install pyarrow --no-index
pip install torch --no-index

python finetune_personality_eval.py --model_path ./models/Qwen2.5-7B-Instruct/ --adapter_paths ./models/Qwen_7B_Conscientiousness_high/ ./models/Qwen_7B_Extraversion_low/ ./models/Qwen_7B_Neuroticism_high/ --num_rounds 20