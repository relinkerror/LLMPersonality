#!/bin/bash
#SBATCH --job-name=finetune_job_array
#SBATCH --output=output_%a.log
#SBATCH --error=error_%a.log
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --array=0-2

# 加载必要模块
module load python/3.12
module load gcc
module load arrow/19
source /project/6078835/gn533549/LLMPersonality/ENV/bin/activate
pip install pyarrow --no-index
pip install torch --no-index

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1


# 根据 SLURM_ARRAY_TASK_ID 选择不同参数
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    DATASET_PATH="./datax/CPED/Extraversion_low.jsonl"
    MODEL_DIR="./models/Qwen2.5-7B-Instruct"
    OUTPUT_DIR="./models/Qwen_7B_Extraversion_low"
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    DATASET_PATH="./datax/CPED/Conscientiousness_high.jsonl"
    MODEL_DIR="./models/Qwen2.5-7B-Instruct"
    OUTPUT_DIR="./models/Qwen_7B_Conscientiousness_high"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    DATASET_PATH="./datax/CPED/Neuroticism_high.jsonl"
    MODEL_DIR="./models/Qwen2.5-7B-Instruct"
    OUTPUT_DIR="./models/Qwen_7B_Neuroticism_high"
fi

python experiment/finetune.py \
    --dataset_path "$DATASET_PATH" \
    --model_dir "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR"

