import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 身份验证
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.set_page_config(page_title="身份验证", page_icon="🔐")
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center; background-color: #f0f2f6; padding: 30px; border-radius: 10px; border: 1px solid #dcdfe6;'>
                <h2 style='color: #1e3a8a;'>🏛️ 寻星投研系统 2.2</h2>
                <p style='color: #666;'>数据穿透诊断版 | 揪出294天元凶</p>
            </div>
        """, unsafe_allow_html=True)
        pwd = st.text_input("", type="password", placeholder="请输入授权码...")
        if st.button("进入诊断模式", use_container_width=True):
            if pwd == "281699":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("密码错误")
    st.stop()

# ==========================================
# 2. 核心逻辑 - 带诊断信息的计算
# ==========================================
st.set_page_config(layout="wide", page_title="寻星诊断版2.2")

if st.sidebar.button("🔒 退出"):
    st.session_state["authenticated"] = False
    st.rerun()

st.title("🏛️ 寻星配置分析系统 2.2 (诊断模式)")
st.caption("🔴 当前版本会显示最高点日期和数值，请核对是否与你的认知一致。")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    # 强制清洗：删除全空行
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all')
    raw_df = raw_df.sort_index()
    returns_df = raw_df.pct_change()

    st.sidebar.subheader("2. 区间设置")
    min_date = raw_df.index.min().to_pydatetime()
    max_date = raw_df.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("开始日期", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date, min_value=min_date, max_value=max_date)
    
    funds = raw_df.columns.tolist()
    
    # 简单的权重设置
    target_weights = {f: 1.0/len(funds) for f in funds}
    
    # 切片
    mask = (returns_df.index >= pd.Timestamp(start_date)) & (returns_df.index <= pd.Timestamp(end_date))
    period_returns = returns_df.loc[mask]

    # --- 组合计算 (简化版) ---
    weights_series = pd.Series(target_weights)
    daily_contributions = period_returns.fillna(0).multiply(weights_series)
    fof_cum_nav = (1 + daily_contributions.sum(axis=1)).cumprod()

    # --- 3. 核心诊断表 (关键修改) ---
    st.subheader("🔍 294天根源排查表")
    st.markdown("请仔细对比下方 **【最高点净值】** 和 **【最新净值】**。")
    
    analysis_data = []
    
    for fund in funds:
        # 获取该产品在所选时间段内的净值序列 (归一化从1开始)
        f_ret = period_returns[fund].dropna()
        if f_ret.empty: continue
        
        # 重新构建净值曲线 (起点设为1)
        f_cum_inner = (1 + f_ret).cumprod()
        
        # 1. 找到系统眼中的“最高点”
        peak_val = f_cum_inner.max()
        peak_idx = f_cum_inner.idxmax() # 最高点发生的日期
        
        # 2. 找到“最新点”
        curr_val = f_cum_inner.iloc[-1]
        
        # 3. 计算回撤状态
        # 容差 0.05%
        is_recovered = curr_val >= (peak_val * 0.9995) 
        
        # 4. 重新计算天数逻辑 (复用之前的逻辑)
        f_peak_series = f_cum_inner.cummax()
        f_dd_series = (f_cum_inner - f_peak_series) / f_peak_series
        
        max_rec_days = 0
        tmp_start = None
        last_date = f_dd_series.index[-1]
        
        for date, val in f_dd_series.items():
            if val < -0.0005 and tmp_start is None:
                tmp_start = date
            elif val >= -0.0005 and tmp_start is not None:
                duration = (date - tmp_start).days
                max_rec_days = max(max_rec_days, duration)
                tmp_start = None
        
        if tmp_start is not None:
            ongoing_duration = (last_date - tmp_start).days
            display_days = f"⚠️ 持续 {ongoing_duration} 天"
        else:
            display_days = f"✅ 最大修复 {max_rec_days} 天"

        analysis_data.append({
            "产品名称": fund,
            "判定最高点日期": peak_idx.strftime('%Y-%m-%d'),
            "最高点净值": f"{peak_val:.4f}",
            "最新净值": f"{curr_val:.4f}",
            "当前状态": display_days,
            "恢复缺口": f"{(curr_val/peak_val - 1)*100:.2f}%"
        })
        
    st.table(pd.DataFrame(analysis_data).style.applymap(
        lambda x: 'color: red; font-weight: bold' if '持续' in str(x) else '', subset=['当前状态']
    ))

    # 绘图
    st.subheader("📈 归一化净值走势 (验证诊断结果)")
    fig = go.Figure()
    for fund in funds:
        f_ret = period_returns[fund].dropna()
        if not f_ret.empty:
            f_cum = (1 + f_ret).cumprod()
            fig.add_trace(go.Scatter(x=f_cum.index, y=f_cum, name=fund))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("请上传数据进行诊断。")
