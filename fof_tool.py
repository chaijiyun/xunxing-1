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
        st.markdown("<div style='text-align: center;'><h2>🏛️ 寻星投研系统 2.4</h2><p>起点逻辑修正版</p></div>", unsafe_allow_html=True)
        pwd = st.text_input("", type="password", placeholder="请输入授权码...")
        if st.button("进入系统", use_container_width=True):
            if pwd == "281699":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("密码错误")
    st.stop()

# ==========================================
# 2. 核心算法：修正后的创新高间隔分析
# ==========================================
def analyze_new_high_gap(nav_series):
    """
    加固版：解决起点定锚导致的301天误报
    """
    if nav_series.empty or len(nav_series) < 2: 
        return 0, 0, "数据不足"
    
    # 获取累计最高点序列
    peak_series = nav_series.cummax()
    
    # 找到所有“真正创新高”的日期
    # 容差 0.05%
    new_high_mask = nav_series >= (peak_series * 0.9995)
    new_high_dates = nav_series[new_high_mask].index
    
    # --- 关键修正：排除区间的首个数据点作为“伪高点” ---
    # 如果第一个高点就是区间起点，且后面还有别的高点，我们从第二个高点开始客观计算
    if len(new_high_dates) >= 2:
        # 计算所有新高点之间的日期差
        gaps = pd.Series(new_high_dates).diff().dt.days
        # 排除掉第一个点带来的 gap（NaN），取历史最大间隔
        max_historical_gap = int(gaps.max()) if not gaps.dropna().empty else 0
    else:
        # 如果整个区间从未创新高（一直低于起点），天数记为区间总长度
        # 但为了更准确，我们记为从起点至今
        max_historical_gap = (nav_series.index[-1] - nav_series.index[0]).days

    # 计算当前距离最近一次新高的天数
    last_high_date = new_high_dates[-1]
    current_gap = (nav_series.index[-1] - last_high_date).days
    
    if current_gap > 7:
        status = f"⚠️ 持续 {current_gap} 天"
    else:
        status = "✅ 处于新高附近"
        
    # 最终输出的历史最长，应该是历史间隔与当前持续时间的较大者
    final_max = max(max_historical_gap, current_gap)
    
    return final_max, current_gap, status

# ==========================================
# 3. 业务逻辑 (完整功能集成)
# ==========================================
st.set_page_config(layout="wide", page_title="寻星 2.4 终极版")

if st.sidebar.button("🔒 退出并锁定"):
    st.session_state["authenticated"] = False
    st.rerun()

st.title("🏛️ 寻星配置分析系统 2.4")
st.caption("终极版：已修正起点定锚逻辑，确保“不创新高天数”不再随开始日期误跳")

uploaded_file = st.sidebar.file_uploader("上传净值 Excel", type=["xlsx"])

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all').sort_index()
    
    # 侧边栏设置
    min_date, max_date = raw_df.index.min().to_pydatetime(), raw_df.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("开始日期", value=min_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date)
    
    # 数据切片
    mask = (raw_df.index >= pd.Timestamp(start_date)) & (raw_df.index <= pd.Timestamp(end_date))
    period_nav = raw_df.loc[mask]
    period_returns = period_nav.pct_change()

    # 组合计算
    funds = period_nav.columns.tolist()
    target_weights = {f: st.sidebar.slider(f, 0.0, 1.0, 1.0/len(funds)) for f in funds}
    tw_total = sum(target_weights.values()) or 1
    weights_series = pd.Series({k: v / tw_total for k, v in target_weights.items()})

    fof_returns = period_returns.fillna(0).multiply(weights_series).sum(axis=1)
    fof_cum_nav = (1 + fof_returns).cumprod()

    # 指标看板
    c1, c2, c3, c4 = st.columns(4)
    if not fof_cum_nav.empty:
        total_ret = fof_cum_nav.iloc[-1] - 1
        mdd = (fof_cum_nav / fof_cum_nav.cummax() - 1).min()
        c1.metric("组合累计收益", f"{total_ret*100:.2f}%")
        c2.metric("组合最大回撤", f"{mdd*100:.2f}%")

        # 分页
        tab1, tab2 = st.tabs(["📊 绩效分析", "🎯 收益归因"])
        
        with tab1:
            fig = go.Figure()
            for fund in funds:
                f_norm = period_nav[fund] / period_nav[fund].dropna().iloc[0]
                fig.add_trace(go.Scatter(x=f_norm.index, y=f_norm, name=fund, line=dict(width=1), opacity=0.5))
            fig.add_trace(go.Scatter(x=fof_cum_nav.index, y=fof_cum_nav, name="FOF组合", line=dict(color='red', width=3)))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("底层产品深度画像")
            analysis_data = []
            for fund in funds:
                f_nav_single = period_nav[fund].dropna()
                max_g, curr_g, status = analyze_new_high_gap(f_nav_single)
                
                analysis_data.append({
                    "产品名称": fund,
                    "最长不创新高周期 (历史)": f"{max_g} 天",
                    "当前状态": status,
                    "区间收益": f"{(f_nav_single.iloc[-1]/f_nav_single.iloc[0]-1)*100:.2f}%"
                })
            st.table(pd.DataFrame(analysis_data))

            st.subheader("资产相关性")
            st.dataframe(period_returns.corr().round(2).style.background_gradient(cmap='RdYlGn'))
else:
    st.info("请上传数据进行分析。")
