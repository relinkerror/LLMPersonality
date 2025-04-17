#!/bin/bash
#SBATCH --job-name=prompt_eval
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2

# 加载必要模块
module load python/3.12
module load gcc
module load arrow/19
source /project/6078835/gn533549/LLMPersonality/ENV/bin/activate
pip install pyarrow --no-index
pip install torch --no-index

python prompt_personality_eval.py --model_path ./models/QwQ-32B/ --num_rounds 20