import pandas as pd

def store_low_extraversion_pairs(csv_file, output_file):
    """
    从 csv_file 中读取数据，筛选出 Extraversion == 'low' 的行，
    并将上一行的 Utterance 作为 Input，该行的 Utterance 作为 Output，
    最后将结果保存到 output_file。
    """
    # 读取原始 CSV 文件
    df = pd.read_csv(csv_file)
    
    # 用于存储 (Input, Output) 对
    pairs = []
    
    # 从第2行开始遍历，这样能保证可以拿到上一行
    for i in range(1, len(df)):
        if df.loc[i, 'Extraversion'] == 'low':
            input_utt = df.loc[i - 1, 'Utterance']
            output_utt = df.loc[i, 'Utterance']
            pairs.append((input_utt, output_utt))
    
    # 将结果转换为 DataFrame 并写入 CSV 文件
    result_df = pd.DataFrame(pairs, columns=['Input', 'Output'])
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"处理完成，共筛选到 {len(pairs)} 条记录，已写入 {output_file}。")

if __name__ == "__main__":
    # 示例用法：将结果保存到 extraversion_low_pairs.csv
    store_low_extraversion_pairs("train_split.csv", "extraversion_low_pairs.csv")
