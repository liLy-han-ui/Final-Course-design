import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from prophet import Prophet
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from scipy.optimize import curve_fit  # 新增导入

# 设置中文字体和输出路径
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

output_dir = '../预测结果/'
os.makedirs(output_dir, exist_ok=True)


# --- 辅助函数 ---
def calculate_metrics(y_true, y_pred):
    """计算 RMSE、MAE、MAPE 和 SMAPE"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # 计算 MAPE，注意要避开真实值为0的情况（但我们在前面已经替换了0）
    mape = 100 * np.mean(np.abs((y_true - y_pred) / y_true))

    # SMAPE（对称平均绝对百分比误差）
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'SMAPE': smape
    }


def plot_forecast(title, history_years, history_values, forecast_years, forecast_values, conf_int=None):
    """可视化历史 + 预测曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(history_years, history_values, 'bo-', label='历史数据')
    plt.plot(forecast_years, forecast_values, 'ro--', label='未来预测')
    if conf_int is not None:
        plt.fill_between(forecast_years,
                         conf_int[:, 0], conf_int[:, 1],
                         color='pink', alpha=0.3, label='置信区间')
    plt.title(title)
    plt.xlabel('年份')
    plt.ylabel('企业数量')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title.replace(' ', '_')}.png")  # 保存图表
    plt.close()  # 关闭图表避免内存泄漏


# --- 主程序开始 ---
df = pd.read_csv('../../企业经营范围/企业统计结果.csv')

# --- 数据预处理：过滤掉全年数量都为0的领域，并替换0为1 ---
# 筛选出每个领域是否所有年份都为0
df['is_all_zero'] = df.groupby('领域')['累计存活企业数量'].transform(lambda x: (x == 0).all())
df = df[~df['is_all_zero']].drop(columns='is_all_zero')

# 对剩余领域的0值替换为1
df['累计存活企业数量'] = df.groupby('领域')['累计存活企业数量'].transform(
    lambda x: x.replace(0, 1)
)
print("✅ 数据预处理完成：已删除全零领域并替换0值为1")

forecast_results = []
model_metrics = []

# 定义不同领域的建模策略
good_domains = ['半导体', '物联网', '数据服务', '机器人']  # 表现好的，用ARIMA
auto_domains = ['大数据分析']  # 手动设置参数
ses_domains = ['自然语言处理', '深度学习']  # 数据稀疏，用指数平滑
prophet_domains = ['软件开发']  # 使用prophet模型或S曲线

# 对每个有效领域进行建模
for domain in df['领域'].unique():
    try:
        print(f"\n🚀 正在处理领域：{domain}")
        domain_df = df[df['领域'] == domain][['年份', '累计存活企业数量']].copy()
        domain_df['年份'] = pd.to_datetime(domain_df['年份'], format='%Y')
        ts_data = domain_df.set_index('年份')['累计存活企业数量']

        if ts_data.sum() == 0:
            print(f"⚠️ {domain} 所有年份企业数量均为0，跳过该领域")
            continue

        # --- 根据领域类型选择不同模型 ---
        model_type = ''
        forecast_mean_values = []
        manual_forecast_years = []
        train_pred = []  # 用于保存拟合值

        # --- 特别处理：大数据分析领域（使用训练/测试划分+未来预测）---
        if domain in auto_domains:
            model_type = "ARIMA"
            p, d, q = 1, 0, 4
            print(f"{domain}: 使用 ARIMA({p},{d},{q})")

            train_size = int(len(ts_data) * 0.8)
            train = ts_data.iloc[:train_size]
            test = ts_data.iloc[train_size:]

            if len(test) == 0:
                print(f"⚠️ {domain} 数据量太少，无法划分测试集，跳过评估")
                continue

            model = ARIMA(train, order=(p, d, q))
            results = model.fit()

            # 测试集预测与评估
            pred_values_test = results.forecast(steps=len(test))
            pred_values_test = np.round(pred_values_test).astype(int)

            # 绘图：测试集 vs 预测
            plot_forecast(
                f'{output_dir}/{domain}领域预测 - {model_type} (测试集)',
                train.index.year, train.values,
                test.index.year, pred_values_test
            )

            # 评估指标
            metrics_test = calculate_metrics(test.values.astype(int), pred_values_test)
            metrics_test.update({
                'Domain': domain,
                'Model': model_type,
                'RMSE': metrics_test['RMSE'],
                'MAE': metrics_test['MAE'],
                'MAPE': metrics_test['MAPE'],
                'SMAPE': metrics_test['SMAPE']
            })
            model_metrics.append(metrics_test)

            # 存储测试集预测结果
            for year, pred in zip(test.index.year, pred_values_test):
                forecast_results.append({
                    '领域': domain,
                    '年份': year,
                    '预测值': pred,
                    '预测下限': None,
                    '预测上限': None
                })

            # 进入通用流程：未来5年预测
            forecast_steps = 6
            forecast_future = results.get_forecast(steps=forecast_steps)
            forecast_mean_future = np.round(forecast_future.predicted_mean).astype(int)
            last_year = ts_data.index.year.max()
            forecast_years_future = range(last_year + 1, last_year + 1 + forecast_steps)

            # 将未来预测合并到通用变量中，以便统一绘图和存储
            forecast_mean_values = forecast_mean_future
            manual_forecast_years = forecast_years_future

        elif domain in good_domains:
            # Auto ARIMA 模型
            model_type = "Auto ARIMA"
            auto_model = auto_arima(ts_data, start_p=0, max_p=3, start_q=0, max_q=3,
                                    seasonal=False, trace=False, error_action='ignore',
                                    suppress_warnings=True)
            p, d, q = auto_model.order
            print(f"使用 ARIMA({p},{d},{q}) 建模")

            model = ARIMA(ts_data, order=(p, d, q))
            results = model.fit()
            train_pred = results.get_prediction(start=ts_data.index.min(),
                                                end=ts_data.index.max()).predicted_mean

            # 未来5年预测
            forecast_steps = 6
            forecast = results.get_forecast(steps=forecast_steps)
            forecast_mean_values = np.round(forecast.predicted_mean).astype(int)
            conf_int_values = np.round(forecast.conf_int()).astype(int)
            last_year = ts_data.index.year.max()
            manual_forecast_years = range(last_year + 1, last_year + 1 + forecast_steps)

        elif domain in ses_domains:
            # 简单指数平滑（适合零值多、变化少）
            model_type = "SimpleExponentialSmoothing"
            model = SimpleExpSmoothing(ts_data)
            fit = model.fit()
            alpha_value = fit.params['smoothing_level']

            print(f"平滑参数α: {alpha_value:.4f}")
            forecast_steps = 6
            forecast_mean_values = np.round(fit.forecast(steps=forecast_steps)).astype(int)
            train_pred = fit.fittedvalues
            last_year = ts_data.index.year.max()
            manual_forecast_years = range(last_year + 1, last_year + 1 + forecast_steps)

        elif domain in prophet_domains:
            # 如果是"软件开发"领域，使用S曲线预测逻辑
            if domain == '软件开发':
                model_type = "S-Curve (Logistic)"
                print(f"{domain}: 使用S曲线预测逻辑")

                # 提取历史数据
                history_years = domain_df['年份'].dt.year.values
                history_values = domain_df['累计存活企业数量'].values

                # 数据分段：拐点(2000-2019)和(2020-2024)
                pre_covid_mask = (domain_df['年份'].dt.year <= 2019)
                post_covid_mask = (domain_df['年份'].dt.year >= 2020)

                train_pre = domain_df[pre_covid_mask]
                train_post = domain_df[post_covid_mask]

                # 确保有足够的数据点
                if len(train_post) < 3:
                    raise ValueError("拐点后数据不足，需要至少3个数据点进行S曲线拟合")


                # S曲线回归模型 (Logistic函数)
                def logistic_func(x, a, b, c):
                    return c / (1 + np.exp(-a * (x - b)))


                # 准备拐点后数据
                x_post = np.arange(len(train_post))
                y_post = train_post['累计存活企业数量'].values

                # 拟合S曲线
                popt, _ = curve_fit(logistic_func, x_post, y_post,
                                    p0=[0.5, 3, 1500000],
                                    bounds=([0, 0, 1000000], [2, 10, 3000000]),
                                    maxfev=5000)

                # 预测未来6年 (2025-2030)
                future_years = list(range(2025, 2031))
                x_future = np.arange(len(train_post), len(train_post) + 6)
                post_pred = logistic_func(x_future, *popt)
                future_pred = np.round(post_pred).astype(int)

                # 计算拟合值（拐点后阶段）
                fitted_values = logistic_func(x_post, *popt)

                # 评估指标（拐点后阶段）
                metrics = calculate_metrics(y_post, fitted_values)
                metrics.update({
                    'Domain': domain,
                    'Model': model_type,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'MAPE': metrics['MAPE'],
                    'SMAPE': metrics['SMAPE']
                })
                model_metrics.append(metrics)

                # 存储预测结果
                for year, pred in zip(future_years, future_pred):
                    forecast_results.append({
                        '领域': domain,
                        '年份': year,
                        '预测值': pred,
                        '预测下限': None,
                        '预测上限': None
                    })

                # 可视化
                plt.figure(figsize=(12, 8))

                # 历史数据
                plt.plot(history_years, history_values, 'bo-',
                         label='实际值', markersize=6, linewidth=1.5)

                # 拟合值（拐点后阶段）
                post_years = train_post['年份'].dt.year
                plt.plot(post_years, fitted_values, 'g--',
                         label='S曲线拟合值', linewidth=1.5)

                # 未来预测
                plt.plot(future_years, future_pred, 'ro--',
                         label='预测值 (2025-2030)', markersize=5, linewidth=1.5)

                # 分段标记
                plt.axvline(x=2019.5, color='gray', linestyle='--', alpha=0.7)
                plt.text(2017, max(history_values) * 0.8, '拐点前数据', fontsize=12, ha='right')
                plt.text(2021, max(history_values) * 0.8, 'S曲线拟合', fontsize=12)

                # 图表装饰
                plt.title(f'{domain}领域预测 - {model_type}', fontsize=16)
                plt.xlabel('年份', fontsize=12)
                plt.ylabel('累计企业数量', fontsize=12)
                plt.legend(loc='upper left', fontsize=10)
                plt.grid(True, linestyle='--', linewidth=0.5)
                plt.xticks(range(min(history_years), max(future_years) + 1, 5))
                plt.tight_layout()

                # 保存图表
                plt.savefig(f'{output_dir}/{domain}_s_curve_forecast.png', dpi=300)
                plt.close()

                # 输出关键参数
                print("=" * 50)
                print("S曲线关键参数:")
                print(f"增长速率参数(a): {popt[0]:.4f}")
                print(f"拐点年份参数(b): {2020 + popt[1]:.1f}")
                print(f"市场饱和值(c): {popt[2]:,.0f}家企业")
                print(f"预计达到饱和值95%的年份: {2020 + np.log(19) / popt[0]:.1f}")

                # 跳过后续处理
                continue

            # 其他prophet_domains中的领域仍然使用Prophet
            model_type = "Prophet"
            prophet_df = domain_df.rename(columns={'年份': 'ds', '累计存活企业数量': 'y'})
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            model = Prophet(
                yearly_seasonality=True,
                changepoints=['2020'],
                changepoint_prior_scale=0.4,
                interval_width=0.8
            ).add_country_holidays(country_name='CN')
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=5, freq='Y')
            forecast = model.predict(future)
            prophet_forecast = forecast[forecast['ds'].isin(prophet_df['ds'])]
            train_pred = prophet_forecast['yhat'].values

            future_values = forecast.iloc[-5:]
            forecast_mean_values = np.round(future_values['yhat']).astype(int)
            lower_bounds = np.round(future_values['yhat_lower']).astype(int)
            upper_bounds = np.round(future_values['yhat_upper']).astype(int)
            manual_forecast_years = future_values['ds'].dt.year.values

            # 存储预测时保留上下限
            for year, pred, lo, hi in zip(manual_forecast_years, forecast_mean_values, lower_bounds, upper_bounds):
                forecast_results.append({
                    '领域': domain, '年份': year, '预测值': pred, '预测下限': lo, '预测上限': hi
                })

            # 绘图（带置信区间）
            plot_forecast(f'{domain}领域预测 - {model_type}',
                          ts_data.index.year, ts_data.values,
                          manual_forecast_years, forecast_mean_values,
                          np.column_stack([lower_bounds, upper_bounds]))

            # 评估指标
            metrics = calculate_metrics(ts_data.values, train_pred)
            metrics.update({
                'Domain': domain,
                'Model': model_type,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'SMAPE': metrics['SMAPE']
            })
            model_metrics.append(metrics)

            continue

        else:
            print(f"⚠️ 领域 {domain} 未定义建模方式，跳过")
            continue

        # --- 如果是 Prophet 已经处理过，则跳过以下步骤 ---
        if domain in prophet_domains:
            continue

        # --- 可视化（复用 plot_forecast）---
        plot_forecast(f'{domain}领域预测 - {model_type}',
                      ts_data.index.year, ts_data.values,
                      manual_forecast_years, forecast_mean_values)

        # --- 评估指标（基于训练集）---
        if domain not in auto_domains:
            metrics = calculate_metrics(ts_data.values, train_pred)
            metrics.update({
                'Domain': domain,
                'Model': model_type,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'SMAPE': metrics['SMAPE']
            })
            model_metrics.append(metrics)

        # --- 存储预测结果（未来5年）---
        for year, pred in zip(manual_forecast_years, forecast_mean_values):
            forecast_results.append({
                '领域': domain,
                '年份': year,
                '预测值': pred,
                '预测下限': None,
                '预测上限': None
            })

    except Exception as e:
        print(f"❌ 领域 {domain} 预测失败: {str(e)}")
        continue

# --- 保存结果 ---
if forecast_results:
    final_forecast = pd.DataFrame(forecast_results)
    forecast_file = os.path.join(output_dir, '综合预测结果.csv')
    final_forecast.to_csv(forecast_file, index=False)

    metrics_df = pd.DataFrame(model_metrics)[['Domain', 'Model', 'RMSE', 'MAE', 'MAPE', 'SMAPE']]
    metrics_file = os.path.join(output_dir, '综合模型评估指标.csv')
    metrics_df.to_csv(metrics_file, index=False)

    print(f"\n✅ 预测结果已保存至 {forecast_file}")
    print(f"✅ 模型评估指标已保存至 {metrics_file}")
else:
    print("⚠️ 未生成任何预测结果，请检查输入数据或模型训练过程。")