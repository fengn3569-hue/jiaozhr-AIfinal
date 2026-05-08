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