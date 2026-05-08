import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import os
import re
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 忽略警告并设置中文字体（根据系统可能需要调整字体名称，如 'SimHei' 或 'Arial Unicode MS'）
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 确保输出目录存在
os.makedirs('outputs', exist_ok=True)


# ==========================================
# M1: 数据处理模块
# ==========================================
def m1_data_processing(filepath):
    print(">>> [M1] 开始加载数据...")
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        print(f"数据加载失败，请确保文件存放在 {filepath}。错误: {e}")
        return None

    # 1. 数据质量报告
    missing_rate = df.isnull().sum() / len(df) * 100
    print("\n--- 数据质量报告：缺失率 (%) ---")
    print(missing_rate[missing_rate > 0])

    # 2. 数据清洗 (附带策略理由)
    print("\n--- 执行数据清洗 ---")
    initial_len = len(df)

    # 策略1：去除车费金额 <= 0 的记录。理由：免费或倒贴的行程属于异常交易记录。
    df = df[df['fare_amount'] > 0]

    # 策略2：去除行程距离 <= 0 的记录。理由：距离为0但产生车费通常是GPS故障或取消的订单。
    df = df[df['trip_distance'] > 0]

    # 策略3：去除乘客数为 0 的记录。理由：空车行驶不应计入有效的出行需求统计。
    df = df[df['passenger_count'] > 0]

    # 策略4：限制极端的行程距离(如 > 100英里)。理由：排除可能是输入错误或非典型纽约市内出行的离群值。
    df = df[df['trip_distance'] < 100]

    print(f"清洗完毕，过滤了 {initial_len - len(df)} 条异常/无效记录。")

    # 3. 特征工程 (提取与衍生)
    print("\n--- 执行特征提取与衍生 ---")
    # 提取时间特征
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['weekday'] = df['tpep_pickup_datetime'].dt.weekday  # 0=周一, 6=周日
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # 高峰期判断：工作日的早高峰(7-9)和晚高峰(17-19)
    df['is_peak'] = ((df['is_weekend'] == 0) &
                     ((df['pickup_hour'].between(7, 9)) | (df['pickup_hour'].between(17, 19)))).astype(int)

    # 衍生特征1：平均行驶速度 (英里/小时)。理由：可以反映路况拥堵程度。
    # 注意：需将时间差转换为小时，避免除以0，加上一个极小值
    trip_time_hours = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 3600
    df['trip_speed_mph'] = df['trip_distance'] / (trip_time_hours + 0.0001)
    # 过滤掉不合理的超高速（如大于 100 mph）
    df = df[df['trip_speed_mph'] < 100]

    # 衍生特征2：每英里平均车费 (美元/英里)。理由：反映不同路线或时段的计费效率。
    df['fare_per_mile'] = df['fare_amount'] / df['trip_distance']

    return df

# ==========================================
# M2: 分析可视化模块
# ==========================================
def m2_visualization(df):
    print("\n>>> [M2] 开始生成分析可视化图表...")

    # 1. 出行需求时间规律 (分小时平均订单量)
    plt.figure(figsize=(10, 5))
    hourly_demand = df.groupby('pickup_hour').size()
    plt.plot(hourly_demand.index, hourly_demand.values, marker='o', linestyle='-', color='b')
    plt.title('24小时出行需求平均订单量')
    plt.xlabel('小时 (0-23)')
    plt.ylabel('订单量')
    plt.grid(True)
    plt.savefig('outputs/1_hourly_demand.png')
    plt.close()

    # 2. 区域热度分析 (上客量最高的 TOP 10 区域)
    plt.figure(figsize=(10, 5))
    top_10_zones = df['PULocationID'].value_counts().head(10)
    top_10_zones.plot(kind='bar', color='orange')
    plt.title('上客量最高的 TOP 10 区域')
    plt.xlabel('区域 ID (PULocationID)')
    plt.ylabel('订单量')
    plt.xticks(rotation=45)
    plt.savefig('outputs/2_top_10_pickup_zones.png')
    plt.close()

    # 3. 车费影响因素分析 (行程距离-车费散点图，采样以提高绘图速度)
    plt.figure(figsize=(8, 6))
    sample_df = df.sample(n=10000, random_state=42)  # 采样1万条避免点叠在一起
    plt.scatter(sample_df['trip_distance'], sample_df['fare_amount'], alpha=0.3, s=5)
    plt.title('行程距离与车费关系散点图 (1万条抽样)')
    plt.xlabel('行程距离 (英里)')
    plt.ylabel('车费金额 (美元)')
    plt.savefig('outputs/3_distance_vs_fare.png')
    plt.close()

    # 4. 自选分析：不同星期几的高峰期与非高峰期平均车速对比
    plt.figure(figsize=(10, 5))
    speed_analysis = df.groupby(['weekday', 'is_peak'])['trip_speed_mph'].mean().unstack()
    speed_analysis.columns = ['非高峰期', '高峰期']
    speed_analysis.index = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    speed_analysis.plot(kind='bar', figsize=(10, 5))
    plt.title('星期与高峰状态对平均车速的影响')
    plt.ylabel('平均车速 (mph)')
    plt.xticks(rotation=0)
    plt.savefig('outputs/4_speed_analysis.png')
    plt.close()
    print("图表已全部保存至 outputs/ 目录。")