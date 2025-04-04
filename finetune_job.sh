#!/bin/bash
#SBATCH --job-name=finetune_job_array
#SBATCH --output=output_%a.log
#SBATCH --error=error_%a.log
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --array=0-2

# 根据 SLURM_ARRAY_TASK_ID 选择不同参数
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    DATASET_PATH="./datax/CPED/extraversion_low_pairs.csv"
    MODEL_DIR="./models/QwQ-32B"
    OUTPUT_DIR="./models/Extraversion_low"
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    DATASET_PATH="./datax/CPED/conscientiousness_high_pairs.csv"
    MODEL_DIR="./models/QwQ-32B"
    OUTPUT_DIR="./models/Conscientiousness_high"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    DATASET_PATH="./datax/CPED/neuroticism_high_pairs.csv"
    MODEL_DIR="./models/QwQ-32B"
    OUTPUT_DIR="./models/Neuroticism_high"
fi

# 运行微调程序
python experiment/finetune.py --dataset_path $DATASET_PATH --model_dir $MODEL_DIR --output_dir $OUTPUT_DIR
