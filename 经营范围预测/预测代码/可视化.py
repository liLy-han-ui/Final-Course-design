import pandas as pd
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示及负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取CSV文件（请替换为你的实际路径）
file_path = '../../企业经营范围/企业统计结果.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 2. 数据预览
print(df.head())

# 对年份列进行转换，如果是以整数形式存储的话
df['年份'] = df['年份'].astype(int)

# 3. 筛选出累计存活企业数量全为0的领域并删除这些记录
df['is_all_zero'] = df.groupby('领域')['累计存活企业数量'].transform(lambda x: (x == 0).all())
df = df[~df['is_all_zero']].drop(columns='is_all_zero')

# 对剩余领域的0值替换为1，以避免log scale等操作时出现问题
df['累计存活企业数量'] = df.groupby('领域')['累计存活企业数量'].transform(
    lambda x: x.replace(0, 1)
)

# 4. 数据透视：将“领域”转为列，便于绘图
df_pivot = df.pivot(index='年份', columns='领域', values='累计存活企业数量')

# 5. 绘图设置
plt.figure(figsize=(14, 8))

# 6. 对每个领域分别画折线
for column in df_pivot.columns:
    plt.plot(df_pivot.index, df_pivot[column], marker='o', label=column)

# 7. 添加标题和标签
plt.title('2000-2024年不同领域累计存活企业数量趋势')
plt.xlabel('年份')
plt.ylabel('累计存活企业数量')
plt.legend(title='领域', bbox_to_anchor=(1.05, 1), loc='upper left')  # 调整图例位置避免重叠
plt.grid(True)
plt.tight_layout()

# 8. 显示图形或保存图片
plt.show()
