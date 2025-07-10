import pandas as pd

# 加载原始数据（含新闻和Label）
df = pd.read_csv('dataset/DJIA_merged_data.csv')

# 重命名列（统一小写，与 Dataset_Custom 要求匹配）
df.rename(columns={
    'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
    'Close': 'close', 'Volume': 'volume', 'Label': 'label'
}, inplace=True)

# 保留的列：结构化价格 + label
columns_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume', 'label']
df = df[columns_to_keep].copy()

# 添加 OT 列（目标值为“下一天的 close”，用于单步预测）
df['OT'] = df['close'].shift(-1)

# 删除最后一行（OT为空）
df.dropna(inplace=True)

# 保存为新数据文件
df.to_csv('dataset/djia_processed.csv', index=False)

print("数据转化完成，保存为 dataset/djia_processed.csv")
