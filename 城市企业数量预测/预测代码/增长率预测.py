import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 计算增长率并保存结果
# ======================
print("\n开始计算2024-2030年增长率...")
plt.rcParams['font.sans-serif']=['Arial Unicode MS','Arial'] # 指定默认字体
plt.rcParams['axes.unicode_minus']=False # 解决负号'-'显示为方块的问题
model_save_dir = '../预测模型'
result_save_dir = '../预测结果'
best_n_clusters=3
# 读取预测结果文件
prediction_file = os.path.join(result_save_dir, "各城市企业数量预测_历史与未来.csv")
if not os.path.exists(prediction_file):
    print(f"预测结果文件不存在: {prediction_file}")
else:
    # 读取预测数据
    predictions_df = pd.read_csv(prediction_file)

    # 获取所有年份列
    year_columns = sorted([col for col in predictions_df.columns if col.isdigit()], key=int)

    # 创建新的DataFrame存储增长率结果
    growth_rate_df = predictions_df[['城市', '聚类']].copy()
    avg_cluster_growth = []

    # 计算每个城市2024-2030年的增长率
    for year in range(2024, 2031):
        prev_year = str(year - 1)
        curr_year = str(year)

        # 确保存在所需年份的数据
        if prev_year in predictions_df.columns and curr_year in predictions_df.columns:
            # 计算单个城市的增长率
            growth_rate_df[f'{year}_增长率'] = (
                    (predictions_df[curr_year] - predictions_df[prev_year]) /
                    predictions_df[prev_year]
            ).round(4)

            # 计算每个聚类在该年份的平均增长率
            for cluster_id in range(best_n_clusters):
                cluster_mask = (predictions_df['聚类'] == cluster_id)
                cluster_growth = growth_rate_df.loc[cluster_mask, f'{year}_增长率']

                # 计算平均增长率（忽略NaN值）
                if not cluster_growth.empty:
                    avg_growth = cluster_growth.mean(skipna=True)
                    avg_cluster_growth.append({
                        '聚类': cluster_id,
                        '年份': year,
                        '平均增长率': avg_growth
                    })

    # 保存城市增长率结果
    city_growth_file = os.path.join(result_save_dir, "城市企业数量增长率_2024-2030.csv")
    growth_rate_df.to_csv(city_growth_file, index=False, encoding='utf-8-sig')
    print(f"✅ 城市增长率已保存至: {city_growth_file}")

    # 保存聚类平均增长率结果
    if avg_cluster_growth:
        cluster_growth_df = pd.DataFrame(avg_cluster_growth)
        cluster_growth_file = os.path.join(result_save_dir, "聚类平均增长率_2024-2030.csv")
        cluster_growth_df.to_csv(cluster_growth_file, index=False, encoding='utf-8-sig')
        print(f"✅ 聚类平均增长率已保存至: {cluster_growth_file}")

        # 可视化聚类平均增长率
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=cluster_growth_df,
            x='年份',
            y='平均增长率',
            hue='聚类',
            marker='o',
            markersize=10,
            linewidth=3
        )
        plt.title('聚类平均增长率趋势 (2024-2030)', fontsize=16)
        plt.ylabel('平均增长率', fontsize=14)
        plt.xlabel('年份', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title='聚类', fontsize=12, title_fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(result_save_dir, "聚类平均增长率趋势图.png"), dpi=300)
        plt.close()
        print("✅ 聚类平均增长率趋势图已生成")
    else:
        print("⚠️ 未计算到聚类平均增长率")