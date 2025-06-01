import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')

# ======================
# 设置路径
# ======================
model_save_dir = '../预测模型'
result_save_dir = '../预测结果'
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(result_save_dir, exist_ok=True)

# ======================
# 数据读取与预处理
# ======================
print("读取数据...")
data = pd.read_csv("../数据文件/合并后的城市经济与企业数据.csv")


def filter_by_year_range(df, start_year, end_year):
    return df[(df['年份'] >= start_year) & (df['年份'] <= end_year)]


start_year = 2018
end_year = 2023
data = filter_by_year_range(data, start_year, end_year)


# 创建特征（如滞后项、增长率等）
def create_features(df):
    df = df.sort_values(['城市', '年份'])
    df['企业数量_Lag1'] = df.groupby('城市')['全部企业'].shift(1)
    df['GRP增长率'] = df.groupby('城市')['GRP（单位：亿元）'].pct_change()
    df['累计政策'] = df.groupby('城市')['政策数量'].cumsum()
    return df.dropna()


print("创建特征...")
data = create_features(data)

# ======================
# 聚类分析
# ======================
print("开始聚类分析...")

# 特征选择
features_for_clustering = ['GRP（单位：亿元）', 'AI百度指数', '政策数量',
                           '科学技术支出（单位：万元）', '教育支出（单位：万元）']

# 按城市聚合特征 - 使用中位数减少异常值影响
city_features = data.groupby('城市')[features_for_clustering].median().reset_index()

# 全局归一化（确保跨年可比性）
scaler = StandardScaler()
scaled_features = scaler.fit_transform(city_features[features_for_clustering])
city_features_scaled = pd.DataFrame(scaled_features, columns=features_for_clustering)
city_features_scaled['城市'] = city_features['城市']

# 确定最佳聚类数
print("确定最佳聚类数...")
range_n_clusters = [3, 4, 5, 6, 7, 8]
silhouette_scores = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(city_features_scaled[features_for_clustering])
    silhouette_avg = silhouette_score(city_features_scaled[features_for_clustering], cluster_labels)
    silhouette_scores.append(silhouette_avg)

# 选择最佳聚类数（轮廓系数最大）
# best_n_clusters=3
best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
print(f"最佳聚类数: {best_n_clusters} (轮廓系数: {max(silhouette_scores):.3f})")

# 使用最佳聚类数进行最终聚类
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
city_features_scaled['cluster'] = kmeans.fit_predict(city_features_scaled[features_for_clustering])

# 将聚类结果映射回原始数据
cluster_mapping = city_features_scaled[['城市', 'cluster']].set_index('城市')['cluster'].to_dict()
data['cluster'] = data['城市'].map(cluster_mapping)

# 聚类结果评估
print("\n聚类结果评估:")
silhouette_avg = silhouette_score(city_features_scaled[features_for_clustering], city_features_scaled['cluster'])
print(f"轮廓系数: {silhouette_avg:.4f}")

ch_score = calinski_harabasz_score(city_features_scaled[features_for_clustering], city_features_scaled['cluster'])
print(f"Calinski-Harabasz指数: {ch_score:.2f}")

db_score = davies_bouldin_score(city_features_scaled[features_for_clustering], city_features_scaled['cluster'])
print(f"Davies-Bouldin指数: {db_score:.4f}")

# 聚类大小分布
cluster_sizes = city_features_scaled['cluster'].value_counts().sort_index()
print("\n聚类大小分布:")
print(cluster_sizes)

# 聚类中心分析
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features_for_clustering
)
print("\n聚类中心特征值:")
print(cluster_centers)

# ======================
# 聚类结果可视化
# ======================
print("\n生成可视化图表...")

# 1. PCA降维可视化
pca = PCA(n_components=2)
pca_result = pca.fit_transform(city_features_scaled[features_for_clustering])
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['Cluster'] = city_features_scaled['cluster']
pca_df['城市'] = city_features_scaled['城市']

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=pca_df,
    x='PC1', y='PC2',
    hue='Cluster',
    palette='viridis',
    s=100,
    alpha=0.7
)

# 标注特殊城市
special_cities = ['北京市', '上海市', '深圳市', '广州市', '杭州市', '成都市']
for _, row in pca_df[pca_df['城市'].isin(special_cities)].iterrows():
    plt.annotate(
        row['城市'],
        (row['PC1'], row['PC2']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
    )

plt.title("城市聚类可视化 (PCA降维)", fontsize=16)
plt.xlabel(f"主成分1 (方差解释率: {pca.explained_variance_ratio_[0] * 100:.1f}%)")
plt.ylabel(f"主成分2 (方差解释率: {pca.explained_variance_ratio_[1] * 100:.1f}%)")
plt.legend(title='聚类')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(result_save_dir, "城市聚类_PCA可视化.png"), dpi=300, bbox_inches='tight')
plt.close()


# 2. 雷达图可视化聚类特征
def plot_radar_chart(cluster_centers, features):
    # 标准化中心点用于雷达图
    scaler_radar = StandardScaler()
    centers_scaled = scaler_radar.fit_transform(cluster_centers[features])

    # 设置雷达图角度
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    for i in range(len(cluster_centers)):
        values = centers_scaled[i].tolist()
        values += values[:1]  # 闭合

        ax.plot(angles, values, linewidth=2, label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.25)

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=10)
    ax.set_yticklabels([])
    plt.title('聚类特征雷达图', size=16, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(result_save_dir, "聚类特征雷达图.png"), dpi=300, bbox_inches='tight')
    plt.close()


plot_radar_chart(cluster_centers, features_for_clustering)

# 3. 特征分布箱线图
city_features_with_cluster = city_features.copy()
city_features_with_cluster['cluster'] = city_features_scaled['cluster']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_for_clustering, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='cluster', y=feature, data=city_features_with_cluster)
    plt.title(f'{feature}分布')
    plt.xlabel('聚类')
plt.tight_layout()
plt.suptitle('各聚类特征分布箱线图', fontsize=16, y=1.02)
plt.savefig(os.path.join(result_save_dir, "聚类特征箱线图.png"), dpi=300, bbox_inches='tight')
plt.close()

# ======================
# 分析聚类趋势和波动率
# ======================
print("\n分析聚类趋势和波动率...")


def analyze_cluster_trend_and_volatility(cluster_data):
    trend_strengths = []
    volatilities = []

    for city, group in cluster_data.groupby('城市'):
        y = group['全部企业'].values
        if len(y) < 2:  # 至少需要2个点计算增长率
            continue

        # 计算增长率
        growth = np.diff(y) / y[:-1]

        # 计算趋势强度和波动性
        trend_strength = abs(np.mean(growth)) * len(growth)
        volatility = np.std(growth)

        trend_strengths.append(trend_strength)
        volatilities.append(volatility)

    # 计算聚类的平均趋势强度和波动性
    avg_trend = np.mean(trend_strengths) if trend_strengths else 0
    avg_volatility = np.mean(volatilities) if volatilities else 0

    return {
        'trend': avg_trend,
        'volatility': avg_volatility,
        'n_cities': len(trend_strengths)
    }


cluster_profiles = {}
for cluster_id in range(best_n_clusters):
    cluster_data = data[data['cluster'] == cluster_id]
    profile = analyze_cluster_trend_and_volatility(cluster_data)
    cluster_profiles[cluster_id] = profile
    print(
        f"聚类 {cluster_id}: 趋势强度={profile['trend']:.4f}, 波动率={profile['volatility']:.4f}, 城市数={profile['n_cities']}")

# ======================
# 模型类型映射
# ======================
model_mapping = {}
print("\n模型类型映射:")

for cluster_id, profile in cluster_profiles.items():
    trend = profile['trend']
    volatility = profile['volatility']

    # 根据趋势和波动性选择模型
    if trend > 3 and volatility > 0.18:
        model_type = 'exponential_smoothing'
    elif trend < 1.7 and volatility < 0.1:
        model_type = 'linear_regression'
    else:
        model_type = 'holt'

    model_mapping[cluster_id] = model_type
    print(f"聚类 {cluster_id} -> {model_type}模型")

# ======================
# 模型训练与评估
# ======================
print("\n开始模型训练与评估...")
trained_models = {}  # 存放所有 cluster 的训练模型
evaluation_results = []  # 存放评估结果

# 划分训练集和测试集（最后两年作为测试集）
train_data = data[data['年份'] < end_year - 1]
test_data = data[data['年份'] >= end_year - 1]

# 为每个聚类训练模型
for cluster_id in range(best_n_clusters):
    print(f"\n训练聚类 {cluster_id} 的模型 ({model_mapping[cluster_id]})...")

    # 获取该聚类在训练集和测试集的数据
    train_mask = (train_data['cluster'] == cluster_id)
    test_mask = (test_data['cluster'] == cluster_id)
    train_cluster = train_data[train_mask]
    test_cluster = test_data[test_mask]

    # 初始化模型
    if model_mapping[cluster_id] == 'linear_regression':
        model = LinearRegression()
    else:
        model = None  # 时间序列模型需要为每个城市单独训练

    # 存储每个城市的模型
    city_models = {}

    # 训练每个城市的模型
    city_mae = []
    city_rmse = []
    city_mape=[]

    for city, group in train_cluster.groupby('城市'):
        # 准备训练数据
        X_train = group['年份'].values.reshape(-1, 1)
        y_train = group['全部企业'].values

        if len(y_train) < 2:  # 至少需要2个点训练模型
            continue

        # 根据模型类型训练
        if model_mapping[cluster_id] == 'linear_regression':
            model.fit(X_train, y_train)
            city_model = model
        elif model_mapping[cluster_id] == 'exponential_smoothing':
            city_model = ExponentialSmoothing(y_train, trend='add', seasonal=None).fit()
        elif model_mapping[cluster_id] == 'holt':
            city_model = Holt(y_train, exponential=True).fit()
        else:
            city_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(0, 1, 0, 12)).fit(disp=False)

        city_models[city] = city_model

        # 在测试集上评估（如果该城市有测试数据）
        test_points = test_cluster[test_cluster['城市'] == city]
        if not test_points.empty:
            X_test = test_points['年份'].values.reshape(-1, 1)
            y_test = test_points['全部企业'].values

            if model_mapping[cluster_id] == 'linear_regression':
                y_pred = model.predict(X_test)
            elif model_mapping[cluster_id] == 'exponential_smoothing':
                y_pred = city_model.forecast(len(y_test))
            elif model_mapping[cluster_id] == 'holt':
                y_pred = city_model.forecast(len(y_test))
            else:
                y_pred = city_model.get_forecast(steps=len(y_test)).predicted_mean

            # 计算误差
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            mape= np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            city_mae.append(mae)
            city_rmse.append(rmse)
            city_mape.append(mape)

    # 保存聚类模型
    trained_models[cluster_id] = {
        'model_type': model_mapping[cluster_id],
        'city_models': city_models
    }

    # 计算聚类的平均误差
    if city_mae:
        avg_mae = np.mean(city_mae)
        avg_rmse = np.mean(city_rmse)
        avg_mape=np.mean(city_mape)
        evaluation_results.append({
            'Cluster': cluster_id,
            'Model Type': model_mapping[cluster_id],
            'MAE': avg_mae,
            'RMSE': avg_rmse,
            'MAPE':avg_mape,
            'Cities': len(city_models)
        })
        print(f"聚类 {cluster_id} 平均评估: MAE={avg_mae:.2f}, RMSE={avg_rmse:.2f}")

# 保存评估结果
eval_df = pd.DataFrame(evaluation_results)
print("\n模型评估结果:")
print(eval_df)
eval_df.to_csv(os.path.join(result_save_dir, "模型评估结果.csv"), index=False, encoding='utf-8-sig')


# ======================
# 预测函数
# ======================
def predict_for_city(city_name, data_all, future_years=7):
    city_data = data_all[data_all['城市'] == city_name].sort_values('年份')
    if city_data.empty:
        print(f"未找到城市 {city_name} 的数据")
        return None

    # 获取城市聚类
    cluster_id = city_data['cluster'].iloc[0]

    # 获取模型
    if cluster_id not in trained_models:
        print(f"未找到聚类 {cluster_id} 的模型")
        return None

    model_info = trained_models[cluster_id]
    model_type = model_info['model_type']

    # 获取该城市的模型
    if city_name in model_info['city_models']:
        model = model_info['city_models'][city_name]
    else:
        print(f"未找到城市 {city_name} 的模型")
        return None

    # 准备数据
    y_city = city_data['全部企业'].values
    last_year = city_data['年份'].max()

    # 根据模型类型进行预测
    if model_type == 'linear_regression':
        # 线性回归使用年份作为特征
        X_train = city_data['年份'].values.reshape(-1, 1)
        future_years_arr = np.arange(last_year + 1, last_year + 1 + future_years).reshape(-1, 1)
        forecast = model.predict(future_years_arr)
    elif model_type == 'exponential_smoothing':
        forecast = model.forecast(steps=future_years)
    elif model_type == 'holt':
        holt_model = Holt(y_city, exponential=False).fit()
        forecast = holt_model.forecast(steps=future_years)
    else:  # SARIMAX
        forecast = model.get_forecast(steps=future_years).predicted_mean.values

    # 四舍五入为整数
    forecast = np.round(forecast).astype(int)

    # 创建预测结果字典
    prediction = {}
    for i, year in enumerate(range(last_year + 1, last_year + 1 + future_years)):
        prediction[year] = forecast[i]

    return prediction


# ======================
# 进行预测并保存结果
# ======================
print("\n开始预测未来企业数量...")
all_cities = data['城市'].unique()
future_years = 7  # 预测 2024 - 2030
all_predictions = []

# 使用进度条
for city_name in tqdm(all_cities, desc="预测各城市"):
    try:
        prediction = predict_for_city(
            city_name=city_name,
            data_all=data,
            future_years=future_years
        )

        if prediction is not None:
            row = {'城市': city_name, '聚类': data[data['城市'] == city_name]['cluster'].iloc[0]}
            # 添加历史数据
            city_data = data[data['城市'] == city_name].sort_values('年份')
            for _, row_data in city_data.iterrows():
                row[str(row_data['年份'])] = row_data['全部企业']
            # 添加预测数据
            for year in range(2024, 2024 + future_years):
                row[str(year)] = prediction.get(year, None)
            all_predictions.append(row)
    except Exception as e:
        print(f"预测失败: {city_name}, 错误: {e}")
        continue

# 保存预测结果
output_df = pd.DataFrame(all_predictions)
csv_file_path = os.path.join(result_save_dir, "各城市企业数量预测_历史与未来.csv")
output_df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 预测完成，已保存至: {csv_file_path}")

# ======================
# 保存聚类模型和映射关系
# ======================
joblib.dump(kmeans, os.path.join(model_save_dir, "kmeans_cluster_model.pkl"))
joblib.dump(scaler, os.path.join(model_save_dir, "feature_scaler.pkl"))
joblib.dump(cluster_mapping, os.path.join(model_save_dir, "city_cluster_mapping.pkl"))
joblib.dump(model_mapping, os.path.join(model_save_dir, "cluster_model_mapping.pkl"))
print("聚类模型和映射关系已保存")

print("\n所有处理完成!")