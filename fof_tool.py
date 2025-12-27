import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 身份验证逻辑 (保持 281699)
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.set_page_config(page_title="身份验证", page_icon="🔐")
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center;'><h2>🏛️ 寻星投研系统 2.4</h2><p>创新高最大间隔天数分析版</p></div>", unsafe_allow_html=True)
        pwd = st.text_input("", type="password", placeholder="请输入授权码...")
        if st.button("进入系统", use_container_width=True):
            if pwd == "281699":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("密码错误")
    st.stop()

# ==========================================
# 2. 核心算法：创新高间隔分析
# ==========================================
def analyze_new_high_gap(nav_series):
    """
    计算历史上创新高的最大间隔天数，以及当前距离上次新高的天数
    """
    if nav_series.empty: return 0, 0, "N/A"
    
    # 计算累计最高点
    peak = nav_series.cummax()
    
    # 找到所有“创新高”的日期 (使用 0.05% 容差)
    new_high_mask = nav_series >= (peak * 0.9995)
    new_high_dates = nav_series[new_high_mask].index
    
    # 1. 计算历史最长间隔
    if len(new_high_dates) >= 2:
        # 相邻新高日期的差值
        gaps = pd.Series(new_high_dates).diff().dt.days
        max_historical_gap = int(gaps.max())
    else:
        # 如果从未创新高，则为整个区间长度
        max_historical_gap = (nav_series.index[-1] - nav_series.index[0]).days
    
    # 2. 计算当前距离上次新高的天数
    last_high_date = new_high_dates[-1]
    current_gap = (nav_series.index[-1] - last_high_date).days
    
    # 3. 状态判定
    if current_gap > 7: # 超过一周没新高才显示警告
        status = f"⚠️ 已持续 {current_gap} 天"
    else:
        status = "✅ 处于新高附近"
        
    return max_historical_gap, current_gap, status

# ==========================================
# 3. 界面展示
# ==========================================
st.set_page_config(layout="wide", page_title="寻星 2.4")
st.title("🏛️ 寻星配置分析系统 2.4")
st.caption("核心指标：创新高最大间隔天数（衡量产品“磨人”程度与修复弹性）")

uploaded_file = st.sidebar.file_uploader("上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    # 加载数据
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all').sort_index()
    returns_df = raw_df.pct_change()

    # 侧边栏设置
    min_date, max_date = raw_df.index.min().to_pydatetime(), raw_df.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("开始日期", value=min_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date)
    
    # 数据切片
    mask = (raw_df.index >= pd.Timestamp(start_date)) & (raw_df.index <= pd.Timestamp(end_date))
    period_nav = raw_df.loc[mask]
    period_returns = returns_df.loc[mask]
    
    funds = period_nav.columns.tolist()
    
    # 权重与贡献计算 (用于面板)
    weights = {f: 1.0/len(funds) for f in funds} # 默认平权，也可改为 slider
    weights_series = pd.Series(weights)
    fof_returns = period_returns.fillna(0).multiply(weights_series).sum(axis=1)
    fof_cum_nav = (1 + fof_returns).cumprod()

    # --- 顶层看板 ---
    c1, c2, c3, c4 = st.columns(4)
    total_ret = fof_cum_nav.iloc[-1] - 1
    mdd = (fof_cum_nav / fof_cum_nav.cummax() - 1).min()
    
    c1.metric("累计收益率", f"{total_ret*100:.2f}%")
    c2.metric("最大回撤", f"{mdd*100:.2f}%")
    
    # --- 核心深度画像表 ---
    st.subheader("🔍 底层资产“无新高周期”深度排查")
    analysis_results = []

    for fund in funds:
        f_nav = period_nav[fund].dropna()
        if f_nav.empty: continue
        
        max_gap, curr_gap, status = analyze_new_high_gap(f_nav)
        
        # 综合对比：如果当前间隔超过了历史最长，则历史最长取当前值
        true_max_gap = max(max_gap, curr_gap)
        
        analysis_results.append({
            "产品名称": fund,
            "最长不创新高周期 (历史)": f"{true_max_gap} 天",
            "当前无新高状态": status,
            "区间累计收益": f"{(f_nav.iloc[-1]/f_nav.iloc[0]-1)*100:.2f}%",
            "区间最大回撤": f"{(f_nav/f_nav.cummax()-1).min()*100:.2f}%"
        })

    st.table(pd.DataFrame(analysis_results))

    # --- 绘图辅助 ---
    st.subheader("📈 净值创新高路径验证")
    
    fig = go.Figure()
    for fund in funds:
        # 归一化显示
        f_norm = period_nav[fund] / period_nav[fund].iloc[0]
        fig.add_trace(go.Scatter(x=f_norm.index, y=f_norm, name=fund))
    fig.update_layout(hovermode="x unified", height=500, title="各产品净值走势 (基准=1.0)")
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
        💡 **指标解释**：
        - **最长不创新高周期**：指历史上任意两次刷新净值高点之间，经历的最长自然天数。
        - **当前无新高状态**：指从最近一次历史高点至今，已经有多少天没能创出新高。
    """)
else:
    st.info("👋 请上传净值数据 Excel。系统将为您深度剖析每只底层基金的‘持有人等待成本’。")
