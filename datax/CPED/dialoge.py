import pandas as pd
import json
import argparse
from tqdm import tqdm

def process_dialogues(input_file, output_file, trait, level):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 筛选出指定人格特质为指定水平的行
    filtered_df = df[df[trait] == level]

    # 获取满足条件的发言者名单
    target_speakers = filtered_df['Speaker'].unique()

    # 获取包含这些发言者的对话 ID
    dialogue_ids = df[df['Speaker'].isin(target_speakers)]['Dialogue_ID'].unique()

    # 初始化结果列表
    results = []

    # 遍历每个对话 ID
    for dialogue_id in tqdm(dialogue_ids, desc="Processing Dialogues"):
        # 获取当前对话的所有发言
        dialogue = df[df['Dialogue_ID'] == dialogue_id]

        # 获取当前对话中目标发言者
        current_target_speakers = dialogue[dialogue[trait] == level]['Speaker'].unique()

        # 如果当前对话中没有目标发言者，则跳过
        if len(current_target_speakers) == 0:
            continue

        # 遍历当前对话中的每个目标发言者
        for target_speaker in current_target_speakers:
            messages = [{"role": "system", "content": f"你是一个{trait}程度为{level}的人"}]

            # 遍历对话中的每一行
            for _, row in dialogue.iterrows():
                role = "assistant" if row['Speaker'] == target_speaker else "user"
                messages.append({"role": role, "content": row['Utterance']})

            # 将结果添加到列表中
            results.append({"messages": messages})

    # 将结果写入 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"处理完成，结果已保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据指定的人格特质筛选对话并转换格式")
    parser.add_argument("input_file", type=str, help="输入的 CSV 文件路径")
    parser.add_argument("output_file", type=str, help="输出的 JSONL 文件路径")
    parser.add_argument("trait", type=str, help="人格特质名称（例如：Neuroticism）")
    parser.add_argument("level", type=str, help="人格特质水平（例如：high 或 low）")

    args = parser.parse_args()

    process_dialogues(args.input_file, args.output_file, args.trait, args.level)
