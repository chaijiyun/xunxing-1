import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 验证逻辑：找出所有新高点并返回日期 ---
def get_new_high_info(nav_series):
    if nav_series.empty: return None, 0
    peak_series = nav_series.cummax()
    # 提高容差到 0.1%，看看是不是因为差那一点点没回去
    new_high_mask = nav_series >= (peak_series * 0.999) 
    new_high_dates = nav_series[new_high_mask].index
    return new_high_dates, peak_series

# --- 身份验证 (281699) ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if not st.session_state["authenticated"]:
    st.set_page_config(page_title="验证")
    pwd = st.sidebar.text_input("授权码", type="password")
    if pwd == "281699": st.session_state["authenticated"] = True
    else: st.stop()

st.set_page_config(layout="wide")
st.title("🏛️ 寻星 2.4 - 净值新高穿透诊断")

uploaded_file = st.sidebar.file_uploader("上传数据", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all').sort_index()
    
    # 日期筛选
    s_date = st.sidebar.date_input("开始", df.index.min())
    e_date = st.sidebar.date_input("结束", df.index.max())
    data = df.loc[s_date:e_date]
    
    selected_fund = st.selectbox("选择要诊断的产品", data.columns)
    nav = data[selected_fund].dropna()
    
    high_dates, peaks = get_new_high_info(nav)
    
    # 计算间隔
    if len(high_dates) >= 2:
        max_gap = pd.Series(high_dates).diff().dt.days.max()
    else:
        max_gap = (nav.index[-1] - nav.index[0]).days

    # --- 核心可视化：直接画出“坑”在哪里 ---
    st.subheader(f"📈 {selected_fund} 创新高路径分析")
    fig = go.Figure()
    
    # 1. 实际净值线
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name="实际净值", line=dict(color="#1e3a8a", width=2)))
    
    # 2. 累计最高值线 (天花板)
    fig.add_trace(go.Scatter(x=peaks.index, y=peaks, name="历史最高线", line=dict(color="rgba(255,0,0,0.3)", dash="dash")))
    
    # 3. 标记系统认定的“新高点” (红点)
    fig.add_trace(go.Scatter(x=high_dates, y=nav[high_dates], mode='markers', 
                             marker=dict(color='red', size=8), name="系统认定的新高点"))

    fig.update_layout(height=600, hovermode="x unified", title=f"历史最长无新高间隔：{max_gap} 天")
    st.plotly_chart(fig, use_container_width=True)

    # --- 原始数据透视 ---
    with st.expander("查看最近 10 条新高记录日期"):
        st.write(high_dates[-10:].tolist())

    st.warning(f"💡 观察红点：如果 2025 年期间没有红点出现，说明净值始终在‘历史最高线’下方。请看图中蓝色线是否真正触碰了那条红色的虚线。")

else:
    st.info("请上传数据开始诊断。")
