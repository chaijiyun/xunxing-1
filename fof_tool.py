import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 身份验证 (保持 281699)
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.set_page_config(page_title="身份验证", page_icon="🔐")
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center;'><h2>🏛️ 寻星投研系统 2.3</h2><p>最大回撤修复分析专项版</p></div>", unsafe_allow_html=True)
        pwd = st.text_input("", type="password", placeholder="请输入授权码...")
        if st.button("进入系统", use_container_width=True):
            if pwd == "281699":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("密码错误")
    st.stop()

# ==========================================
# 2. 核心算法：最大回撤修复天数计算
# ==========================================
def analyze_mdd_repair(nav_series):
    """
    计算最大回撤及其对应的修复天数
    """
    if nav_series.empty: return None
    
    # 1. 计算回撤序列
    peak = nav_series.cummax()
    drawdown = (nav_series - peak) / peak
    
    # 2. 找到最大回撤发生的时刻和数值
    mdd_val = drawdown.min()
    if mdd_val >= 0: return 0, "无回撤", "N/A"
    
    t_bottom = drawdown.idxmin() # 坑底日期
    
    # 3. 找到导致这次最大回撤的“前高”点 (起点)
    # 在坑底之前的序列里，最后一个净值等于最高点的日期
    before_bottom = nav_series[:t_bottom]
    t_start = before_bottom[before_bottom == before_bottom.max()].index[-1]
    
    # 4. 找到从坑底爬出来、回到或超过前高的时刻 (终点)
    peak_val = nav_series[t_start]
    # 坑底之后的序列
    after_bottom = nav_series[t_bottom:]
    # 容差 0.05%
    recovered_points = after_bottom[after_bottom >= peak_val * 0.9995]
    
    if not recovered_points.empty:
        t_recover = recovered_points.index[0]
        repair_days = (t_recover - t_start).days
        status = f"✅ 已修复 (历时{repair_days}天)"
        return mdd_val, status, repair_days
    else:
        # 至今未修复
        ongoing_days = (nav_series.index[-1] - t_start).days
        status = f"⚠️ 尚未修复 (已持续{ongoing_days}天)"
        return mdd_val, status, ongoing_days

# ==========================================
# 3. 界面展示
# ==========================================
st.set_page_config(layout="wide", page_title="寻星 2.3")
st.title("🏛️ 寻星配置分析系统 2.3")
st.caption("针对底层资产“最大回撤坑”的爬坑能力专项分析")

uploaded_file = st.sidebar.file_uploader("上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all').sort_index()
    returns_df = raw_df.pct_change()

    # 区间选择
    min_date, max_date = raw_df.index.min().to_pydatetime(), raw_df.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("开始日期", value=min_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date)
    
    mask = (raw_df.index >= pd.Timestamp(start_date)) & (raw_df.index <= pd.Timestamp(end_date))
    period_nav = raw_df.loc[mask]
    
    funds = period_nav.columns.tolist()
    analysis_results = []

    for fund in funds:
        f_nav = period_nav[fund].dropna()
        if f_nav.empty: continue
        
        mdd_val, status, days = analyze_mdd_repair(f_nav)
        
        # 计算区间表现
        total_ret = (f_nav.iloc[-1] / f_nav.iloc[0]) - 1
        
        analysis_results.append({
            "产品名称": fund,
            "区间最大回撤 (坑深)": f"{mdd_val*100:.2f}%",
            "最大回撤修复状态": status,
            "修复总天数 (从前高到回正)": days,
            "区间累计收益": f"{total_ret*100:.2f}%"
        })

    # --- 数据呈现 ---
    st.subheader("📊 最大回撤修复能力排查表")
    res_df = pd.DataFrame(analysis_results)
    st.table(res_df)

    # --- 绘图辅助验证 ---
    st.subheader("📈 净值走势对照 (验证“坑”的位置)")
    fig = go.Figure()
    for fund in funds:
        f_nav_norm = period_nav[fund] / period_nav[fund].iloc[0]
        fig.add_trace(go.Scatter(x=f_nav_norm.index, y=f_nav_norm, name=fund))
    fig.update_layout(hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 逻辑说明：系统先锁定区间内跌幅最深的一次‘最大回撤’，随后计算从该次跌破前高开始，到重新站上该高度的总天数。")
else:
    st.info("请上传数据，系统将分析每一只底层产品最深的那个‘坑’是怎么爬出来的。")
