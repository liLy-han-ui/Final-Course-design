import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv("../数据文件/合并后的企业流失数据.csv")

# 列名统一
df = df.rename(columns={'注销': '注销'})

# 缺失值处理
df = df.sort_values(by=['城市', '年份'])
df = df.groupby('城市', group_keys=False).apply(lambda x: x.ffill()).reset_index(drop=True)

# 计算核心指标 - 注销率
df['注销率'] = df['注销'] / df['全部企业']


# ============== 辅助函数：四舍五入到两位小数 ==============
def round_to_two_decimals(value):
    """将值四舍五入到两位小数，处理各种数据类型"""
    if isinstance(value, (int, float)):
        return round(value, 2)
    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
        return round(float(value), 2)
    else:
        return value


# ============== 风险评估系统 ==============
def calculate_risk_score(group):
    """
    基于注销率变化和波动的风险评分
    评分逻辑：
    1. 当前注销率水平（60%）
    2. 近期变化趋势（30%）
    3. 历史波动性（10%）

    返回包含所有计算指标的字典，所有数值保留两位小数
    """
    # 获取最新年份和注销率
    latest_year = group['年份'].max()
    current_rate = group.loc[group['年份'] == latest_year, '注销率'].values[0]

    # 当前注销率得分（标准化到0-60分）
    rate_score = 60 * min(current_rate / 0.2, 1)  # 假设20%为最高风险

    # 初始化变量
    trend_score = 0
    slope = 0
    volatility_score = 0
    volatility = 0
    recent_years = []
    recent_rates = []

    # 最近3年变化趋势（标准化到0-30分）
    if len(group) >= 3:
        # 获取最近3年数据
        recent_data = group.sort_values('年份').tail(3)
        recent_years = recent_data['年份'].tolist()
        recent_rates = recent_data['注销率'].tolist()

        # 计算趋势斜率
        slope = np.polyfit(range(3), recent_rates, 1)[0]  # 线性趋势斜率
        trend_score = 30 * min(max(slope / 0.05, 0), 1)  # 每年增长5%为最高风险

    # 历史波动性（标准化到0-10分）
    if len(group) >= 3:
        volatility = group['注销率'].std()
        volatility_score = 10 * min(volatility / 0.1, 1)  # 标准差10%为最高风险

    # 总风险评分
    total_score = rate_score + trend_score + volatility_score

    # 返回所有计算指标，所有数值保留两位小数
    return {
        '城市': group['城市'].iloc[0],
        '年份': latest_year,
        '注销数量': round_to_two_decimals(group.loc[group['年份'] == latest_year, '注销'].values[0]),
        '最新注销率': round_to_two_decimals(current_rate),
        '注销率得分': round_to_two_decimals(rate_score),
        '趋势斜率': round_to_two_decimals(slope),
        '趋势得分': round_to_two_decimals(trend_score),
        '历史波动性': round_to_two_decimals(volatility),
        '波动性得分': round_to_two_decimals(volatility_score),
        '风险评分': round_to_two_decimals(total_score),
        '最近3年年份': ','.join(map(str, recent_years)) if recent_years else '',
        '最近3年注销率': ','.join([str(round_to_two_decimals(rate)) for rate in recent_rates]) if recent_rates else ''
    }


# 应用评分计算并收集所有指标
risk_data = []
for city, group in df.groupby('城市'):
    # 为每个城市每年计算风险评分（历史数据）
    city_years = sorted(group['年份'].unique())
    for year in city_years:
        year_data = group[group['年份'] <= year]
        # 确保有足够数据计算评分（至少3年）
        if len(year_data) >= 3:
            city_data = calculate_risk_score(year_data)
            risk_data.append(city_data)

# 创建包含所有指标的数据框
risk_history_df = pd.DataFrame(risk_data)

# 只保留最新一年的结果用于最终评级
if not risk_history_df.empty:
    latest_year = risk_history_df['年份'].max()
    risk_df = risk_history_df[risk_history_df['年份'] == latest_year].copy()
else:
    print("没有足够数据计算风险评分")
    exit()


# ============== 风险等级划分 ==============
def classify_risk(score):
    if score >= 70:
        return '高风险'
    elif score >= 50:
        return '中风险'
    else:
        return '低风险'


risk_df['风险等级'] = risk_df['风险评分'].apply(classify_risk)


# ============== 动态预警系统（新规则） ==============
def calculate_alert(row, history_df):
    city_data = history_df[history_df['城市'] == row['城市']].sort_values('年份')

    # 检查是否有足够历史数据（至少3年）
    if len(city_data) < 3:
        return '无预警', '', ''

    # 获取最近三年的数据
    latest_years = city_data.tail(3)

    # 规则1：风险评分趋势（连续2年增长）
    rule1_triggered = False
    score_growth_rates = []

    # 检查最近两年是否有增长
    for i in range(1, 3):
        prev_row = city_data[city_data['年份'] == row['年份'] - i]
        if not prev_row.empty:
            prev_score = prev_row['风险评分'].values[0]
            current_score = row['风险评分'] if i == 1 else latest_years.iloc[-2]['风险评分']

            if prev_score > 0:  # 避免除以零
                growth_rate = (current_score - prev_score) / prev_score
                score_growth_rates.append(round_to_two_decimals(growth_rate))

    # 检查是否有连续两年增长
    if len(score_growth_rates) == 2:
        # 检查是否连续两年增长
        if score_growth_rates[0] > 0 and score_growth_rates[1] > 0:
            rule1_triggered = True

    # 规则2：指标异常波动（注销率年度增幅超过1）- 修改为注销率
    rule2_triggered = False
    rate_growth_rates = []

    # 检查最近两年注销率增长
    for i in range(1, 3):
        prev_row = city_data[city_data['年份'] == row['年份'] - i]
        if not prev_row.empty:
            prev_rate = prev_row['最新注销率'].values[0]
            current_rate = row['最新注销率'] if i == 1 else latest_years.iloc[-2]['最新注销率']

            if prev_rate > 0:  # 避免除以零
                growth_rate = (current_rate - prev_rate) / prev_rate
                rounded_growth = round_to_two_decimals(growth_rate)
                rate_growth_rates.append(rounded_growth)
                if rounded_growth > 1:  # 增长
                    rule2_triggered = True

    # 规则3：阈值突破（当前风险评分高于70分）
    rule3_triggered = row['风险评分'] >= 70

    # 预警等级划分
    if rule1_triggered and rule3_triggered:
        alert_details = f"风险评分连续增长: "
        if len(score_growth_rates) >= 1:
            alert_details += f"去年→今年: {score_growth_rates[0]:.2%}"
        if len(score_growth_rates) >= 2:
            alert_details += f", 前年→去年: {score_growth_rates[1]:.2%}"
        alert_details += f"; 当前评分:{row['风险评分']}"
        return '红色预警', alert_details, "高风险且趋势恶化"
    elif rule2_triggered or rule3_triggered:
        if rule2_triggered:
            alert_details = f"注销率异常增长: "
            if len(rate_growth_rates) >= 1:
                alert_details += f"去年→今年: {rate_growth_rates[0]:.2%}"
            if len(rate_growth_rates) >= 2:
                alert_details += f", 前年→去年: {rate_growth_rates[1]:.2%}"
            return '橙色预警', alert_details, "注销率异常增长"
        else:
            alert_details = f"当前风险评分:{round_to_two_decimals(row['风险评分'])} (超过70分阈值)"
            return '橙色预警', alert_details, "风险评分超阈值"
    elif rule1_triggered:
        alert_details = f"风险评分连续增长: "
        if len(score_growth_rates) >= 1:
            alert_details += f"去年→今年: {score_growth_rates[0]:.2%}"
        if len(score_growth_rates) >= 2:
            alert_details += f", 前年→去年: {score_growth_rates[1]:.2%}"
        return '黄色预警', alert_details, "风险趋势恶化"
    else:
        return '无预警', '', ''


# 应用预警系统并收集详细信息
alert_results = risk_df.apply(lambda row: calculate_alert(row, risk_history_df), axis=1, result_type='expand')
alert_results.columns = ['动态预警', '预警详情', '预警原因']
risk_df = pd.concat([risk_df, alert_results], axis=1)

# ============== 结果输出 ==============
# 最终结果排序
result_df = risk_df.sort_values('风险评分', ascending=False)
result_df = result_df[[
    '城市', '年份', '注销数量', '最新注销率', '风险评分', '风险等级', '动态预警',
    '注销率得分', '趋势得分', '波动性得分',
    '趋势斜率', '历史波动性',
    '最近3年年份', '最近3年注销率',
    '预警详情', '预警原因'
]]

# 确保所有数值数据保留两位小数
numeric_cols = ['注销数量', '最新注销率', '风险评分', '注销率得分', '趋势得分',
                '波动性得分', '趋势斜率', '历史波动性']
result_df[numeric_cols] = result_df[numeric_cols].applymap(round_to_two_decimals)

# 特殊处理：最近3年注销率已经是字符串格式的两位小数
# 预警详情中的增长率已经是百分比格式的两位小数

# 保存所有指标到CSV
result_df.to_csv('../预测结果/城市风险评级结果.csv', index=False)

# ============== 可视化 ==============
# 风险评分分布
plt.figure(figsize=(10, 6))
sns.histplot(result_df['风险评分'], bins=20, kde=True)
plt.axvline(x=50, color='orange', linestyle='--', label='中风险阈值')
plt.axvline(x=70, color='red', linestyle='--', label='高风险阈值')
plt.title('城市风险评分分布')
plt.xlabel('风险评分')
plt.ylabel('城市数量')
plt.legend()
plt.savefig('../预测结果/风险评分分布.png')
plt.close()

# 注销率与风险评分关系
plt.figure(figsize=(10, 6))
sns.scatterplot(data=result_df, x='最新注销率', y='风险评分', hue='风险等级',
                palette={'高风险': 'red', '中风险': 'orange', '低风险': 'green'})
plt.title('注销率与风险评分关系')
plt.xlabel('注销率')
plt.ylabel('风险评分')
plt.grid(True)
plt.savefig('../预测结果/注销率与风险评分.png')
plt.close()
# 预警城市分布
alert_counts = result_df[result_df['动态预警'] != '无预警']['动态预警'].value_counts()

if not alert_counts.empty:
    # 1. 定义预警等级到颜色的固定映射
    alert_colors = {
        '红色预警': 'red',
        '橙色预警': 'orange',
        '黄色预警': 'yellow',
        # 可扩展其他预警等级
    }

    # 2. 按预警等级重要性排序（红>橙>黄）
    alert_order = ['红色预警', '橙色预警', '黄色预警']
    ordered_alerts = alert_counts.reindex(alert_order).dropna()

    # 3. 生成对应的颜色列表
    colors = [alert_colors[level] for level in ordered_alerts.index]

    plt.figure(figsize=(8, 8))
    plt.pie(
        ordered_alerts,
        labels=ordered_alerts.index,
        autopct='%1.1f%%',
        colors=colors,  # 使用映射的颜色列表
        startangle=90
    )
    plt.title('预警城市分布')
    plt.savefig('../预测结果/预警城市分布.png')
    plt.close()
else:
    print("没有触发预警的城市")

# 风险评分构成分析
score_components = result_df[['注销率得分', '趋势得分', '波动性得分']].mean()
plt.figure(figsize=(8, 6))
score_components.plot(kind='bar', color=['#4CAF50', '#2196F3', '#9C27B0'])
plt.title('风险评分构成分析')
plt.ylabel('平均得分')
plt.xticks(rotation=0)
plt.savefig('../预测结果/风险评分构成.png')
plt.close()

# 新增：预警类型与风险等级关系
if not result_df[result_df['动态预警'] != '无预警'].empty:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=result_df, x='动态预警', y='风险评分', order=['黄色预警', '橙色预警', '红色预警'],
                palette={'黄色预警': 'yellow', '橙色预警': 'orange', '红色预警': 'red'})
    plt.title('预警类型与风险评分分布')
    plt.xlabel('预警类型')
    plt.ylabel('风险评分')
    plt.savefig('../预测结果/预警类型与风险评分.png')
    plt.close()

print("分析完成！所有数值数据已保留两位小数，详细结果已保存至 ../预测结果/城市风险评级结果.csv")