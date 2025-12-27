import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 版本与身份验证
# ==========================================
VERSION = "1.7-OFFICIAL" 

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.set_page_config(page_title="寻星投研系统", page_icon="🏛️")
    st.markdown(f"<div style='text-align:center; margin-top:50px;'><h2>🏛️ 寻星配置分析系统 {VERSION}</h2><p>专业资产配置与深度回撤穿透工具</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pwd = st.text_input("授权码", type="password", placeholder="请输入内部授权码...")
        if st.button("立即进入系统", use_container_width=True):
            if pwd == "281699":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("授权码错误，请联系系统管理员")
    st.stop()

# ==========================================
# 2. 核心金融算法：严谨修复时长计算
# ==========================================
def calculate_recovery_days(series):
    """
    专门解决空值导致的天数膨胀问题：
    1. 剔除无效点
    2. 记录真实高点日期
    3. 自然日相减
    """
    # 强制数值转换并剔除空值，不进行任何“补全”操作，只认真实数据
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) < 2: return 0, 0
    
    max_rec_days = 0
    current_ongoing = 0
    
    # 计算滚动最高点
    roll_max = s.cummax()
    # 计算回撤，并给定 0.05% 的容差（解决精度误差）
    drawdown_series = (s / roll_max) - 1
    
    last_peak_dt = s.index[0]
    is_in_pit = False
    
    for i in range(len(s)):
        current_dt = s.index[i]
        dd_val = drawdown_series.iloc[i]
        
        # 判定：只要回撤大于 -0.0005（即回升到99.95%以上），视为修复
        if dd_val >= -0.0005:
            if is_in_pit:
                # 刚从坑里爬出来，计算从掉下去前的最高点到今天的天数
                duration = (current_dt - last_peak_dt).days
                max_rec_days = max(max_rec_days, duration)
                is_in_pit = False
            last_peak_dt = current_dt # 刷新最高点日期
        else:
            is_in_pit = True
            
    # 如果数据最后一天还在坑里
    if is_in_pit:
        current_ongoing = (s.index[-1] - last_peak_dt).days
        
    return max_rec_days, current_ongoing

# ==========================================
# 3. 主界面布局
# ==========================================
st.set_page_config(layout="wide", page_title=f"寻星系统 {VERSION}")

if st.sidebar.button("🔒 退出并锁定"):
    st.session_state["authenticated"] = False
    st.rerun()

st.title(f"🏛️ 寻星配置分析系统 {VERSION}")
st.caption("2025 内部投研版 | 已修复空值干扰及天数计算溢出问题")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    # A. 原始数据加载与初步清洗
    df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index()
    
    # 自动识别指数与基金产品
    all_cols = df_raw.columns.tolist()
    benchmarks = [c for c in all_cols if any(x in str(c) for x in ["300", "500"])]
    funds = [c for c in all_cols if c not in benchmarks]

    # B. 侧边栏交互设置
    st.sidebar.subheader("2. 策略配置")
    min_date, max_date = df_raw.index.min().to_pydatetime(), df_raw.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("分析开始日期", value=min_date)
    end_date = st.sidebar.date_input("分析结束日期", value=max_date)
    
    # 权重滑块
    target_weights = {f: st.sidebar.slider(f, 0.0, 1.0, 1.0/len(funds)) for f in funds}

    # C. 数据切片处理
    # 对于组合计算，需要对缺失数据进行前向填充
    df_filled = df_raw.ffill()
    mask = (df_filled.index >= pd.Timestamp(start_date)) & (df_filled.index <= pd.Timestamp(end_date))
    period_df = df_filled.loc[mask]
    returns_df = period_df.pct_change().fillna(0)

    # D. 深度画像分析表 (核心修正点)
    st.markdown("### 🔍 深度指标排查 (已剔除空值干扰)")
    analysis_data = []
    for item in (funds + benchmarks):
        # 传入原始全量序列 raw_df[item]，让算法识别全局最高点
        max_h, ongoing = calculate_recovery_days(df_raw[item])
        
        # 计算特定区间的收益
        sub_nav = df_raw[item].loc[mask].dropna()
        p_ret = (sub_nav.iloc[-1] / sub_nav.iloc[0] - 1) if len(sub_nav) > 1 else 0

        analysis_data.append({
            "名称": item,
            "类型": "底层产品" if item in funds else "业绩基准",
            "历史最长修复": f"{max_h} 天",
            "当前回撤持续": f"{ongoing} 天" if ongoing > 0 else "✅ 已创新高",
            "区间累计收益": f"{p_ret*100:.2f}%",
            "状态状态": "⚠️ 回撤中" if ongoing > 0 else "✅ 正常"
        })
    st.table(pd.DataFrame(analysis_data))

    # E. 组合业绩计算
    w_sum = sum(target_weights.values()) or 1
    w_series = pd.Series({k: v/w_sum for k, v in target_weights.items()})
    fof_ret = (returns_df[funds] * w_series).sum(axis=1)
    fof_cum = (1 + fof_ret).cumprod()

    # 指标看板
    c1, c2, c3, c4 = st.columns(4)
    total_fof_ret = fof_cum.iloc[-1] - 1
    peak = fof_cum.expanding().max()
    mdd_fof = ((fof_cum / peak) - 1).min()
    days_span = max((fof_cum.index[-1] - fof_cum.index[0]).days, 1)
    ann_ret = (1 + total_fof_ret)**(365.25/days_span) - 1
    
    c1.metric("组合累计收益", f"{total_fof_ret*100:.2f}%")
    c2.metric("组合年化收益", f"{ann_ret*100:.2f}%")
    c3.metric("组合最大回撤", f"{mdd_fof*100:.2f}%")
    c4.metric("成分产品数量", len(funds))

    # F. 净值曲线图
    fig = go.Figure()
    for b in benchmarks:
        b_nav = df_raw[b].loc[mask].dropna()
        if not b_nav.empty:
            fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav/b_nav.iloc[0], name=f'基准-{b}', line=dict(dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=fof_cum.index, y=fof_cum, name='寻星组合', line=dict(color='red', width=4)))
    fig.update_layout(title="组合净值 vs 业绩基准", hovermode="x unified", height=600)
    st.plotly_chart(fig, use_container_width=True)

    # G. 相关性矩阵
    st.subheader("📊 底层资产相关性矩阵")
    st.dataframe(returns_df[funds].corr().style.background_gradient(cmap='RdYlGn').format("{:.2f}"))

else:
    st.info("👋 欢迎使用寻星投研系统。请在左侧上传 Excel 净值表开始分析。")
