import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
df = pd.read_csv('城市风险评级结果.csv')

# 2. 数据预处理
required_columns = ['城市', '风险等级', '动态预警']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"缺少必需的列：{', '.join(required_columns)}")

# 3. 统计每个风险等级下各预警等级的数量
grouped = df.groupby(['风险等级', '动态预警']).size().unstack(fill_value=0)

# 4. 保证风险等级的顺序为：低风险、中风险、高风险
risk_order = ['低风险', '中风险', '高风险']
grouped = grouped.reindex(risk_order)

# 5. 定义预警等级顺序及对应颜色
alert_order = ['无预警', '黄色预警', '橙色预警', '红色预警']
alert_colors = {
    '无预警': 'blue',
    '黄色预警': 'yellow',
    '橙色预警': 'orange',
    '红色预警': 'red'
}

# 6. 准备绘图数据
bar_width = 0.15
indices = range(len(grouped.index))

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制每个预警类型的柱子
for i, alert in enumerate(alert_order):
    if alert in grouped.columns:
        values = grouped[alert].values
        plt.bar([x + bar_width * i for x in indices], values, width=bar_width, label=alert, color=alert_colors[alert])

# 设置坐标轴标签和标题
plt.title('各地区风险等级中各预警数量分布')
plt.xlabel('地区风险等级')
plt.ylabel('预警数量')

# 设置X轴刻度
plt.xticks([x + bar_width * (len(alert_order) - 1) / 2 for x in indices], grouped.index)

# 添加图例
plt.legend(title='预警等级')

# 保存图像
save_path = '../预测结果/预警数量分布.png'
plt.tight_layout()
plt.savefig(save_path)
plt.close()

print(f"图像已保存至: {save_path}")