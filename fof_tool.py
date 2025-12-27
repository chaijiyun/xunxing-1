import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 醒目的版本标志 (用于确认部署成功)
# ==========================================
VERSION = "1.6-FINAL-PRO" 

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.set_page_config(page_title="🔐 身份验证", page_icon="🏛️")
    st.markdown(f"<div style='text-align:center;'><h2>🏛️ 寻星投研系统 {VERSION}</h2><p>内部专用授权版本</p></div>", unsafe_allow_html=True)
    pwd = st.text_input("授权码", type="password")
    if st.button("进入系统", use_container_width=True):
        if pwd == "281699":
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("授权码错误")
    st.stop()

# ==========================================
# 2. 核心穿透算法：彻底解决空值与 294 天死锁
# ==========================================
def robust_recovery_calc(series):
    """
    暴力穿透算法：
    1. 强制数值化，解决 Excel 格式问题。
    2. 线性插值，填补 Excel 空值(NaN)坑位。
    3. 0.1% 容差，解决浮点数精度导致的不回正。
    """
    # 处理空值：先插值补齐中间，再补齐两头
    s = pd.to_numeric(series, errors='coerce').interpolate(limit_direction='both').ffill().bfill()
    if s.empty: return 0, 0
    
    max_rec, ongoing = 0, 0
    peak_val, peak_dt = -np.inf, None
    in_dd = False
    
    for dt, val in s.items():
        # 只要回到最高点的 99.9% 就算修复
        if val >= peak_val or (peak_val > 0 and (val / peak_val) >= 0.999):
            if in_dd:
                max_rec = max(max_rec, (dt - peak_dt).days)
                in_dd = False
            peak_val, peak_dt = val, dt
        else:
            in_dd = True
            
    if in_dd and peak_dt:
        ongoing = (s.index[-1] - peak_dt).days
    return max_rec, ongoing

# ==========================================
# 3. 主界面布局
# ==========================================
st.set_page_config(layout="wide", page_title=f"寻星系统 {VERSION}")
st.title(f"🏛️ 寻星配置分析系统 {VERSION}")
st.caption("核心更新：空值线性修复逻辑 | 全局最高点对撞算法 | 业绩基准对比")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("1. 上传净值 Excel", type=["xlsx"])

if uploaded_file:
    # A. 加载原始全量数据
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index()
    benchmarks = [c for c in raw_df.columns if any(x in str(c) for x in ["300", "500"])]
    funds = [c for c in raw_df.columns if c not in benchmarks]

    # B. 策略参数
    st.sidebar.subheader("2. 策略参数")
    start_date = st.sidebar.date_input("开始日期", value=raw_df.index.min())
    end_date = st.sidebar.date_input("结束日期", value=raw_df.index.max())
    target_weights = {f: st.sidebar.slider(f, 0.0, 1.0, 1.0/len(funds)) for f in funds}

    # C. 指标计算 (关键：基于原始全量数据 raw_df)
    st.markdown("### 🔍 深度画像排查 (空值修复版)")
    analysis = []
    
    # 为了组合计算，先制作一个平滑的 period_df
    smooth_df = raw_df.interpolate().ffill().bfill()
    mask = (smooth_df.index >= pd.Timestamp(start_date)) & (smooth_df.index <= pd.Timestamp(end_date))
    period_df = smooth_df.loc[mask]
    
    for item in (funds + benchmarks):
        # 调用暴力穿透算法
        max_h, ongoing = robust_recovery_calc(raw_df[item])
        
        # 计算所选区间收益
        p_sub = period_df[item]
        p_ret = (p_sub.iloc[-1] / p_sub.iloc[0] - 1) if len(p_sub) > 0 else 0
        
        analysis.append({
            "名称": item,
            "历史最长修复": f"{max_h} 天",
            "当前持续时长": f"{ongoing} 天" if ongoing > 0 else "✅ 已创新高",
            "状态判定": "⚠️ 正在回撤" if ongoing > 0 else "✅ 正常",
            "区间累计收益": f"{p_ret*100:.2f}%"
        })
    st.table(pd.DataFrame(analysis))

    # D. 组合业绩看板
    returns_df = period_df.pct_change().fillna(0)
    w_sum = sum(target_weights.values()) or 1
    w_series = pd.Series({k: v/w_sum for k, v in target_weights.items()})
    fof_ret = (returns_df[funds] * w_series).sum(axis=1)
    fof_cum = (1 + fof_ret).cumprod()

    c1, c2, c3 = st.columns(3)
    total_fof_ret = fof_cum.iloc[-1] - 1
    mdd_fof = ((fof_cum / fof_cum.expanding().max()) - 1).min()
    c1.metric("组合累计收益", f"{total_fof_ret*100:.2f}%")
    c2.metric("组合最大回撤", f"{mdd_fof*100:.2f}%")
    c3.metric("成分股数量", len(funds))

    # E. 净值曲线图
    fig = go.Figure()
    for b in benchmarks:
        b_nav = period_df[b] / period_df[b].iloc[0]
        fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav, name=f'基准-{b}', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=fof_cum.index, y=fof_cum, name='寻星组合', line=dict(color='red', width=4)))
    st.plotly_chart(fig, use_container_width=True)

    # F. 相关性矩阵
    st.subheader("📊 资产相关性")
    st.dataframe(returns_df[funds].corr().style.background_gradient(cmap='RdYlGn').format("{:.2f}"))

else:
    st.info("👋 请上传包含净值数据的 Excel 文件，系统将自动穿透处理空值。")
