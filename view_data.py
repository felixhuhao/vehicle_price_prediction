import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)

df = pd.read_csv('used_car_train.csv', sep=' ')

print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print("\n前5行数据:")
print(df.head())
