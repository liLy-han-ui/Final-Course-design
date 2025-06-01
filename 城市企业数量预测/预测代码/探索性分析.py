import pandas as pd
import matplotlib.pyplot as plt
import os


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# 读取数据（你可以将下面的字符串替换为实际文件路径）
data_str = "../数据文件/合并后的城市经济与企业数据.csv"

# 将字符串转换为DataFrame
from io import StringIO

df = pd.read_csv(data_str)



# 指定要绘图的字段列表
fields = [
    'GRP（单位：亿元）',
    '科学技术支出（单位：万元）',
    '教育支出（单位：万元）',
    '全部企业'
]

# 对每个字段分别绘图
for field in fields:
    plt.figure(figsize=(10, 6))

    # 按城市分组绘制
    for city, group in df.groupby('城市'):
        plt.plot(group['年份'], group[field], marker='o', label=city)

    plt.title(f'{field} 随年份变化')
    plt.xlabel('年份')
    plt.ylabel(field)
    plt.grid(True)

    # 将图例移到图表右边外侧
    plt.legend(title='城市', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整布局防止被截断
    plt.show()