import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 数据读取与预处理
# ======================
data = pd.read_csv("../数据文件/合并后的城市经济与企业数据.csv")


def filter_by_year_range(df, start_year, end_year):
    return df[(df['年份'] >= start_year) & (df['年份'] <= end_year)]


start_year = 2015
end_year = 2023
data = filter_by_year_range(data, start_year, end_year)


# 创建特征（如滞后项、增长率等）
def create_features(df):
    df = df.sort_values(['城市', '年份'])
    df['企业数量_Lag1'] = df.groupby('城市')['全部企业'].shift(1)
    df['GRP增长率'] = df.groupby('城市')['GRP（单位：亿元）'].pct_change()
    df['累计政策'] = df.groupby('城市')['政策数量'].cumsum()
    return df.dropna()


data = create_features(data)

# ======================
# 特征选择 + 按城市聚合特征（确保每个城市一个聚类）
# ======================
features_for_clustering = ['GRP（单位：亿元）', 'AI百度指数', '政策数量',
                           '科学技术支出（单位：万元）', '教育支出（单位：万元）']

# 按城市聚合特征 - 使用中位数减少异常值影响
city_features = data.groupby('城市')[features_for_clustering].median().reset_index()

# 全局归一化（确保跨年可比性）
scaler = StandardScaler()
scaled_features = scaler.fit_transform(city_features[features_for_clustering])
city_features_scaled = pd.DataFrame(scaled_features, columns=features_for_clustering)
city_features_scaled['城市'] = city_features['城市']

# ======================
# 聚类分析（优化版）
# ======================
# 1. 确定最佳聚类数
range_n_clusters = [3, 4, 5, 6, 7, 8]
silhouette_scores = []
ch_scores = []
db_scores = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(city_features_scaled[features_for_clustering])

    # 计算评估指标
    silhouette_avg = silhouette_score(city_features_scaled[features_for_clustering], cluster_labels)
    ch_score = calinski_harabasz_score(city_features_scaled[features_for_clustering], cluster_labels)
    db_score = davies_bouldin_score(city_features_scaled[features_for_clustering], cluster_labels)

    silhouette_scores.append(silhouette_avg)
    ch_scores.append(ch_score)
    db_scores.append(db_score)

# 可视化聚类数评估
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(range_n_clusters, silhouette_scores, 'bo-')
plt.xlabel('聚类数')
plt.ylabel('轮廓系数')
plt.title('轮廓系数评估')

plt.subplot(1, 3, 2)
plt.plot(range_n_clusters, ch_scores, 'go-')
plt.xlabel('聚类数')
plt.ylabel('Calinski-Harabasz 指数')
plt.title('CH指数评估')

plt.subplot(1, 3, 3)
plt.plot(range_n_clusters, db_scores, 'ro-')
plt.xlabel('聚类数')
plt.ylabel('Davies-Bouldin 指数')
plt.title('DB指数评估')
plt.tight_layout()
plt.show()

# 选择最佳聚类数（轮廓系数最大）
#best_n_clusters=3
best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
print(f"最佳聚类数: {best_n_clusters} (轮廓系数: {max(silhouette_scores):.3f})")

# 2. 使用最佳聚类数进行最终聚类
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
city_features_scaled['cluster'] = kmeans.fit_predict(city_features_scaled[features_for_clustering])

# 将聚类结果映射回原始数据
cluster_mapping = city_features_scaled[['城市', 'cluster']].set_index('城市')['cluster'].to_dict()
data['cluster'] = data['城市'].map(cluster_mapping)

# ======================
# 聚类结果评估
# ======================
# 1. 轮廓系数
silhouette_avg = silhouette_score(city_features_scaled[features_for_clustering], city_features_scaled['cluster'])
print(f"轮廓系数: {silhouette_avg:.4f}")

# 2. Calinski-Harabasz指数
ch_score = calinski_harabasz_score(city_features_scaled[features_for_clustering], city_features_scaled['cluster'])
print(f"Calinski-Harabasz指数: {ch_score:.2f}")

# 3. Davies-Bouldin指数
db_score = davies_bouldin_score(city_features_scaled[features_for_clustering], city_features_scaled['cluster'])
print(f"Davies-Bouldin指数: {db_score:.4f}")

# 4. 聚类大小分布
cluster_sizes = city_features_scaled['cluster'].value_counts().sort_index()
print("\n聚类大小分布:")
print(cluster_sizes)

# 5. 聚类中心分析
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features_for_clustering
)
print("\n聚类中心特征值:")
print(cluster_centers)

# ======================
# 可视化聚类结果
# ======================
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
plt.show()


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
    plt.show()


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
plt.show()

# ======================
# 查看每个组别包含哪些城市
# ======================
print("\n聚类结果:")
for cluster_id in range(best_n_clusters):
    cities_in_cluster = city_features_scaled[city_features_scaled['cluster'] == cluster_id]['城市'].unique()
    print(f"\nCluster {cluster_id} 包含 {len(cities_in_cluster)} 个城市:")
    # 按城市分组打印
    for i in range(0, len(cities_in_cluster), 10):
        print(", ".join(cities_in_cluster[i:i + 10]))