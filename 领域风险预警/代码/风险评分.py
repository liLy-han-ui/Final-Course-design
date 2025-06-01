import pandas as pd
import numpy as np

# 1. 数据加载与预处理
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['年份'] = df['年份'].astype(int)
    df['领域'] = df['领域'].str.strip().str.lower()

    # 计算增长率
    df = df.sort_values(['领域', '年份'])
    df['增长率'] = df.groupby('领域')['累计存活企业数量'].pct_change().fillna(0)
    df['增长率'] = df['增长率'].replace([np.inf, -np.inf], 0)

    historical = df[df['年份'] <= 2024]
    forecast = df[df['年份'] >= 2025]
    return df, historical, forecast


# 2. 计算历史基准增长率（抗异常值版）
def calculate_robust_historical_growth(hist_group):
    if hist_group.empty:
        return 0.0

    # 使用中位数代替加权平均，避免异常值影响
    growth_rates = hist_group['增长率'].values
    if len(growth_rates) >= 3:
        return np.median(growth_rates)
    elif len(growth_rates) > 0:
        return np.mean(growth_rates)
    return 0.0


# 3. 计算动态波动率阈值
def calculate_dynamic_volatility_threshold(hist_growth_rates):
    if len(hist_growth_rates) < 2:
        return 0.0

    sigma_hist = np.std(hist_growth_rates)
    q1, q3 = np.percentile(hist_growth_rates, [25, 75])
    iqr = q3 - q1
    return max(0.05, sigma_hist + 1.5 * iqr)  # 设置最小阈值


# 4. 连续风险评分函数
def calculate_continuous_risk_scores(forecast_rates, hist_benchmark, hist_volatility):
    """计算连续风险评分"""
    scores = {
        '衰退强度': 0.0,
        '预期偏离度': 0.0,
        '波动风险': 0.0
    }

    # 1. 衰退强度评分 = min(1, 累计负增长幅度 / 30%)
    negative_growth = sum([abs(r) for r in forecast_rates if r < 0])
    scores['衰退强度'] = min(1.0, negative_growth / 0.3)

    # 2. 预期偏离度评分 = min(1, |(预测平均 - 历史基准)| / |历史基准|)
    if abs(hist_benchmark) > 1e-5:  # 避免除零
        deviation = (np.mean(forecast_rates) - hist_benchmark) / abs(hist_benchmark)
        # 只关注负向偏离（预测增长低于历史基准）
        if deviation < 0:
            scores['预期偏离度'] = min(1.0, abs(deviation))
    else:
        if np.mean(forecast_rates) < -0.05:  # 预测负增长超过5%
            scores['预期偏离度'] = min(1.0, abs(np.mean(forecast_rates)) * 10)

    # 3. 波动风险评分 = min(1, 波动率 /  动态阈值)
    volatility = np.std(forecast_rates) if len(forecast_rates) > 1 else 0.0
    dynamic_threshold = calculate_dynamic_volatility_threshold(hist_volatility)

    if dynamic_threshold > 1e-5:
        scores['波动风险'] = min(1.0, volatility / dynamic_threshold)

    return scores


# 5. 风险等级判定（连续版）
def assess_continuous_risk_level(total_score):
    if total_score >= 0.7:
        return '高风险'
    elif total_score >= 0.4:
        return '中风险'
    else:
        return '低风险'


# 6. 主风险计算函数
def calculate_risk_scores_continuous(df, historical_df, forecast_df, forecast_period='短期'):
    results = []

    for domain, group in df.groupby('领域'):
        domain = domain.strip().lower()

        # 获取历史数据（2019-2024）
        hist_group = historical_df[
            (historical_df['领域'].str.strip().str.lower() == domain) &
            (historical_df['年份'] >= 2019)
            ]
        hist_rates = hist_group['增长率'].values

        # 获取预测数据
        fc_group = forecast_df[forecast_df['领域'].str.strip().str.lower() == domain]
        fc_rates = fc_group['增长率'].values

        # 截取预测期
        if forecast_period == '短期' and len(fc_rates) >= 3:
            fc_rates = fc_rates[:3]

        # 计算历史基准（抗异常值）
        hist_benchmark = calculate_robust_historical_growth(hist_group)

        # 计算连续风险评分
        risk_scores = calculate_continuous_risk_scores(fc_rates, hist_benchmark, hist_rates)

        # 加权总分（可调整权重）
        weights = {'衰退强度': 0.4, '预期偏离度': 0.3, '波动风险': 0.3}
        total_score = sum(risk_scores[k] * weights[k] for k in risk_scores)

        # 保存结果
        result = {
            '领域': domain,
            '预测期平均增长率': np.mean(fc_rates) if fc_rates.size > 0 else 0.0,
            '历史基准增长率': hist_benchmark,
            '预测期波动率': np.std(fc_rates) if len(fc_rates) > 1 else 0.0
        }
        result.update(risk_scores)
        result['综合风险分'] = total_score
        result['风险等级'] = assess_continuous_risk_level(total_score)
        results.append(result)

    return pd.DataFrame(results)


# 7. 主流程
if __name__ == "__main__":
    # 加载数据
    df, historical, forecast = load_data("../../经营范围预测/预测结果/企业技术领域发展数据_2000_2030.csv")

    # 计算风险评分
    short_risk_df = calculate_risk_scores_continuous(df, historical, forecast, '短期')
    long_risk_df = calculate_risk_scores_continuous(df, historical, forecast, '长期')

    # 重命名列用于对比
    short_risk_df = short_risk_df.rename(columns={
        col: f'短期_{col}' for col in short_risk_df.columns if col != '领域'
    })
    long_risk_df = long_risk_df.rename(columns={
        col: f'长期_{col}' for col in long_risk_df.columns if col != '领域'
    })

    # 合并结果
    merged_df = pd.merge(short_risk_df, long_risk_df, on='领域', how='outer')

    # 保存结果
    output_file = '../结果/风险评分结果_连续优化版.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        short_risk_df.to_excel(writer, sheet_name='短期风险', index=False)
        long_risk_df.to_excel(writer, sheet_name='长期风险', index=False)
        merged_df.to_excel(writer, sheet_name='对比分析', index=False)

    print(f"优化版风险评估结果已保存至 {output_file}")