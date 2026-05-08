import streamlit as st
import pandas as pd
import os
from PIL import Image
# 导入你 main.py 中的核心功能函数
from main import m1_data_processing, m2_visualization, m3_train_models

# 设置页面配置
st.set_page_config(page_title="纽约出租车数据分析系统", layout="wide")

st.title("🚕 城市出租车出行数据分析与智能问答系统")
st.markdown("---")

# 侧边栏导航
menu = st.sidebar.selectbox("功能导航", ["数据概览与处理", "分析可视化展示", "模型训练预测", "智能问答交互"])

# 全局变量：数据路径
DATA_PATH = 'data/yellow_tripdata_2023-01.parquet'

if menu == "数据概览与处理":
    st.header("📊 模块 M1: 数据处理与质量报告")
    if st.button("开始加载并清洗数据"):
        with st.spinner("数据处理中，请稍候..."):
            df = m1_data_processing(DATA_PATH)
            if df is not None:
                st.success("数据处理成功！")
                st.subheader("清洗后数据预览 (前100行)")
                st.dataframe(df.head(100))

                # 展示特征衍生结果
                st.info(f"当前数据集条数：{len(df)}")
                st.write("已提取特征：小时、星期、是否高峰、平均时速、每英里费用等。")
                # 缓存数据供其他模块使用
                st.session_state['df'] = df

elif menu == "分析可视化展示":
    st.header("📈 模块 M2: 统计分析可视化")
    if st.button("生成/更新分析图表"):
        if 'df' in st.session_state:
            with st.spinner("正在绘图..."):
                m2_visualization(st.session_state['df'])
                st.success("图表已更新！")
        else:
            st.error("请先在‘数据概览’页面加载数据。")

    # 展示已生成的图片
    if os.path.exists('outputs'):
        cols = st.columns(2)
        images = [f for f in os.listdir('outputs') if f.endswith('.png') and f.startswith(('1', '2', '3', '4'))]
        for i, img_name in enumerate(sorted(images)):
            with cols[i % 2]:
                img = Image.open(f'outputs/{img_name}')
                st.image(img, caption=img_name, use_container_width=True)
    else:
        st.warning("outputs 文件夹暂无图表，请点击上方按钮生成。")

elif menu == "模型训练预测":
    st.header("🤖 模块 M3: 需求预测模型 (NN vs RF)")
    if st.button("开始模型对比训练"):
        if 'df' in st.session_state:
            with st.spinner("神经网络训练中...请观察控制台进度"):
                rf_model, nn_model = m3_train_models(st.session_state['df'])
                st.success("模型训练完成！")

                # 展示 Loss 曲线
                if os.path.exists('outputs/5_nn_loss_curve.png'):
                    st.image('outputs/5_nn_loss_curve.png', caption="神经网络训练 Loss 曲线", width=600)
        else:
            st.error("请先加载数据。")

elif menu == "智能问答交互":
    st.header("💬 模块 M4: 智能问答接口")
    st.write("请输入关于纽约出租车数据的自然语言问题，例如：'哪里的打车人最多？'")

    user_input = st.text_input("你的问题：")

    if user_input:
        if 'df' in st.session_state:
            df = st.session_state['df']
            import re

            # 这里复用 main.py 中的问答逻辑，但改为 st.write 输出
            if re.search(r'(几点|时间|时段).*(打车|订单|多)', user_input):
                peak_hour = df['pickup_hour'].value_counts().idxmax()
                st.write(f"🤖 **回答：** 根据数据，每天的 **{peak_hour}:00** 是打车需求量最大的时间段。")
                if os.path.exists('outputs/1_hourly_demand.png'):
                    st.image('outputs/1_hourly_demand.png', width=500)

            elif re.search(r'(区域|哪里|地点).*(最多|排)', user_input):
                top_zone = df['PULocationID'].value_counts().idxmax()
                st.write(f"🤖 **回答：** 上客量最高的区域 ID 是 **{top_zone}**。")
                if os.path.exists('outputs/2_top_10_pickup_zones.png'):
                    st.image('outputs/2_top_10_pickup_zones.png', width=500)

            elif re.search(r'(费用|车费|钱)', user_input):
                corr = df['trip_distance'].corr(df['fare_amount'])
                st.write(f"🤖 **回答：** 车费与行程距离高度正相关，相关系数达 **{corr:.2f}**。")

            elif re.search(r'(速度|车速|高峰期)', user_input):
                peak_speed = df[df['is_peak'] == 1]['trip_speed_mph'].mean()
                st.write(f"🤖 **回答：** 高峰期平均车速约为 **{peak_speed:.1f} mph**。")
                if os.path.exists('outputs/4_speed_analysis.png'):
                    st.image('outputs/4_speed_analysis.png', width=500)
            else:
                st.write("🤖 抱歉，我还在学习中，请尝试换一种问法（包含关键词如：时间、区域、费用、速度）。")
        else:
            st.error("请先在侧边栏导航到‘数据概览’并加载数据。")