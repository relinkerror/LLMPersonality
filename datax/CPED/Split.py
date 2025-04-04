import pandas as pd

class CSVPairExtractor:
    """
    通用 CSV 数据对提取器。
    
    从 CSV 文件中读取数据，根据指定列的过滤条件，
    按照设定的行偏移量构成 (Input, Output) 对，并将结果保存到指定的输出文件中。

    参数:
        csv_file (str): 输入 CSV 文件路径。
        output_file (str): 输出 CSV 文件路径，用于保存结果。
        filter_column (str): 用于过滤的列名。
        filter_value (Any): 过滤时需要匹配的值。
        input_column (str): 用作 Input 的列名（提取条件行之前的行）。
        output_column (str): 用作 Output 的列名（提取条件行本身）。
        offset (int): 行偏移量，默认为 1，表示使用上一行作为 Input。
    """
    def __init__(self, csv_file, output_file, filter_column, filter_value, input_column, output_column, offset=1):
        self.csv_file = csv_file
        self.output_file = output_file
        self.filter_column = filter_column
        self.filter_value = filter_value
        self.input_column = input_column
        self.output_column = output_column
        self.offset = offset

    def process(self):
        # 读取 CSV 文件
        df = pd.read_csv(self.csv_file)
        pairs = []
        
        # 从指定的偏移量开始遍历，确保可以取到上一行的数据
        for i in range(self.offset, len(df)):
            if df.loc[i, self.filter_column] == self.filter_value:
                input_val = df.loc[i - self.offset, self.input_column]
                output_val = df.loc[i, self.output_column]
                pairs.append((input_val, output_val))
        
        # 将结果转换为 DataFrame 并写入 CSV 文件
        result_df = pd.DataFrame(pairs, columns=['Input', 'Output'])
        result_df.to_csv(self.output_file, index=False, encoding='utf-8')
        
        print(f"处理完成，共筛选到 {len(pairs)} 条记录，已写入 {self.output_file}。")

if __name__ == "__main__":
    # 示例用法：
    # 针对 Extraversion 列为 'low' 的记录，
    # 从上一行的 Utterance 作为 Input，当前行的 Utterance 作为 Output，
    # 并保存到 extraversion_low_pairs.csv 文件中。
    extractor = CSVPairExtractor(
        csv_file="train_split.csv",
        output_file="neuroticism_high_pairs.csv",
        filter_column="Neuroticism",
        filter_value="high",
        input_column="Utterance",
        output_column="Utterance",
        offset=1
    )
    extractor.process()

