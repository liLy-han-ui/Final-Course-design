import pandas as pd
import os

# 设置文件夹路径
all_enterprise_folder = "../../城市企业统计/全部企业"  # 替换为实际路径
net_increase_folder = "../../城市企业统计/净增加" # 替换为实际路径
output_file = "../数据文件/合并后的企业流失数据.csv"

# 初始化结果DataFrame
merged_df = pd.DataFrame()

# 处理2020-2024年的数据
for year in range(2020, 2025):
    # 构建文件路径
    all_file = os.path.join(all_enterprise_folder, f"全部企业_{year}.csv")
    net_file = os.path.join(net_increase_folder, f"净增加企业_{year}.csv")

    # 检查文件是否存在
    if not os.path.exists(all_file):
        print(f"警告: 全部企业文件不存在 - {all_file}")
        continue
    if not os.path.exists(net_file):
        print(f"警告: 净增加企业文件不存在 - {net_file}")
        continue

    try:
        # 读取全部企业文件
        all_df = pd.read_csv(all_file)


        # 读取净增加企业文件
        net_df = pd.read_csv(net_file)


        # 合并两个数据集
        year_df = pd.merge(
            all_df[['城市', '全部企业']],
            net_df[['城市', '注销']],
            on='城市',
            how='inner'  # 只保留两个文件中都存在的城市
        )

        # 添加年份列
        year_df['年份'] = year

        # 添加到结果DataFrame
        merged_df = pd.concat([merged_df, year_df], ignore_index=True)

        print(f"成功合并 {year} 年数据")

    except Exception as e:
        print(f"处理 {year} 年数据时出错: {str(e)}")
        continue

# 检查是否成功合并了数据
if merged_df.empty:
    print("错误: 没有成功合并任何数据")
    exit()

# 保存结果
merged_df.to_csv(output_file, index=False)
print(f"合并完成! 结果已保存至: {output_file}")

# 显示前几行数据预览
print("\n数据预览:")
print(merged_df.head())