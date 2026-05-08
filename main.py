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

# ==========================================
# M3: 预测模型模块
# ==========================================
# 定义简单的神经网络
class DemandPredictor(nn.Module):
    def __init__(self):
        super(DemandPredictor, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def m3_train_models(df):
    print("\n>>> [M3] 开始构建并训练预测模型 (神经网络 vs 随机森林)...")

    # 准备数据集：预测某区域(PULocationID)某时段(hour, weekday)的出行需求量(订单数)
    # 按天、小时、区域聚合数据
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
    demand_df = df.groupby(['pickup_date', 'pickup_hour', 'weekday', 'PULocationID']).size().reset_index(name='demand')

    # 为了避免新手电脑内存溢出，我们选取需求量最大的Top 5个区域进行预测训练
    top_zones = demand_df['PULocationID'].value_counts().head(5).index
    demand_df = demand_df[demand_df['PULocationID'].isin(top_zones)]

    X = demand_df[['pickup_hour', 'weekday', 'PULocationID']].values
    y = demand_df['demand'].values

    # 划分训练集和测试集 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. 训练随机森林 (作为对比基准)
    print("训练随机森林模型...")
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    print(f"[随机森林] 测试集 MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")

    # 2. 训练 PyTorch 神经网络
    print("训练 PyTorch 神经网络...")
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    nn_model = DemandPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.01)

    epochs = 20
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = nn_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))

    # 绘制 Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, marker='o')
    plt.title('神经网络训练 Loss 曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig('outputs/5_nn_loss_curve.png')
    plt.close()

    # 评估神经网络
    nn_model.eval()
    with torch.no_grad():
        nn_preds = nn_model(X_test_tensor).numpy()

    nn_mae = mean_absolute_error(y_test, nn_preds)
    nn_rmse = np.sqrt(mean_squared_error(y_test, nn_preds))
    print(f"[神经网络] 测试集 MAE: {nn_mae:.2f}, RMSE: {nn_rmse:.2f}")

    print("模型评估完成！Loss 曲线已保存至 outputs/5_nn_loss_curve.png")
    return rf_model, nn_model

# ==========================================
# M4: 问答接口模块
# ==========================================
def m4_qa_system(df):
    print("\n" + "=" * 50)
    print(" 欢迎使用城市出租车智能问答系统 ")
    print("支持的问题类型例如：")
    print("1. 几点最容易打车？ (时段查询)")
    print("2. 哪个区域打车人最多？ (区域排名)")
    print("3. 车费一般和什么有关？ (影响因素)")
    print("4. 高峰期和非高峰期车速差多少？ (数据洞察)")
    print("5. 周末的订单量大概是多少？ (统计查询)")
    print("输入 '退出' 结束问答。")
    print("=" * 50)

    while True:
        question = input("\n请提出你的出行问题：")
        if question.lower() in ['退出', 'exit', 'quit']:
            print("感谢使用，再见！")
            break

        # 匹配规则 1：时段查询
        if re.search(r'(几点|时间|时段).*(打车|订单|多)', question):
            peak_hour = df['pickup_hour'].value_counts().idxmax()
            print(f" 结论：根据数据，每天的 {peak_hour}:00 是打车需求量最大的时间段。")
            print(" 相关图表路径： outputs/1_hourly_demand.png")

        # 匹配规则 2：区域排名
        elif re.search(r'(区域|哪里|地点).*(最多|排)', question):
            top_zone = df['PULocationID'].value_counts().idxmax()
            print(f" 结论：上客量最高的区域 ID 是 {top_zone}。")
            print(" 相关图表路径： outputs/2_top_10_pickup_zones.png")

        # 匹配规则 3：费用关系
        elif re.search(r'(车费|钱|费用).*(关系|因素|有关)', question):
            corr = df['trip_distance'].corr(df['fare_amount'])
            print(f" 结论：车费与行程距离呈高度正相关（相关系数: {corr:.2f}）。")
            print(" 相关图表路径： outputs/3_distance_vs_fare.png")

        # 匹配规则 4：速度洞察
        elif re.search(r'(速度|车速|高峰期)', question):
            peak_speed = df[df['is_peak'] == 1]['trip_speed_mph'].mean()
            non_peak_speed = df[df['is_peak'] == 0]['trip_speed_mph'].mean()
            print(f" 结论：高峰期平均车速为 {peak_speed:.1f} mph，非高峰期为 {non_peak_speed:.1f} mph。")
            print(" 相关图表路径： outputs/4_speed_analysis.png")

        # 匹配规则 5：统计查询
        elif re.search(r'(周末|工作日).*(订单|打车)', question):
            weekend_count = len(df[df['is_weekend'] == 1])
            weekday_count = len(df[df['is_weekend'] == 0])
            print(f" 结论：数据集中周末订单量为 {weekend_count} 条，工作日订单量为 {weekday_count} 条。")

        else:
            print("抱歉，目前的系统还无法理解这个问题，请尝试换一种包含核心关键词的问法。")
