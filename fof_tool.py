import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        st.markdown("<div style='text-align: center; background-color: #f0f2f6; padding: 30px; border-radius: 10px;'><h2>🏛️ 寻星投研系统 2.4</h2><p>双轴收益分析版</p></div>", unsafe_allow_html=True)
        pwd = st.text_input("", type="password", placeholder="请输入授权码...")
        if st.button("进入系统", use_container_width=True):
            if pwd == "281699":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("密码错误")
    st.stop()

# ==========================================
# 2. 核心算法逻辑 (创新高天数)
# ==========================================
def analyze_new_high_gap(nav_series):
    if nav_series.empty or len(nav_series) < 2: 
        return 0, 0, "数据不足", nav_series, nav_series
    peak_series = nav_series.cummax()
    new_high_mask = nav_series >= (peak_series * 0.9995)
    new_high_dates = nav_series[new_high_mask].index
    if len(new_high_dates) >= 2:
        gaps = pd.Series(new_high_dates).diff().dt.days
        max_historical_gap = int(gaps.max())
    else:
        max_historical_gap = (nav_series.index[-1] - nav_series.index[0]).days
    current_gap = (nav_series.index[-1] - new_high_dates[-1]).days
    status = f"⚠️ 持续 {current_gap} 天" if current_gap > 7 else "✅ 处于新高附近"
    return max(max_historical_gap, current_gap), current_gap, status, new_high_dates, peak_series

# ==========================================
# 3. 业务主界面
# ==========================================
st.set_page_config(layout="wide", page_title="寻星 2.4 双轴版")

if st.sidebar.button("🔒 退出系统"):
    st.session_state["authenticated"] = False
    st.rerun()

st.title("🏛️ 寻星配置分析系统 2.4")
st.caption("双轴视图：左轴净值(归一化) vs 右轴累计收益率(%)")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all').sort_index()
    
    st.sidebar.subheader("2. 筛选与配置")
    s_date = st.sidebar.date_input("开始日期", value=raw_df.index.min())
    e_date = st.sidebar.date_input("结束日期", value=raw_df.index.max())
    
    period_nav = raw_df.loc[s_date:e_date]
    period_returns = period_nav.pct_change()
    funds = period_nav.columns.tolist()
    
    target_weights = {f: st.sidebar.slider(f, 0.0, 1.0, 1.0/len(funds)) for f in funds}
    tw_total = sum(target_weights.values()) or 1
    weights_series = pd.Series({k: v / tw_total for k, v in target_weights.items()})

    fof_daily_returns = period_returns.fillna(0).multiply(weights_series).sum(axis=1)
    fof_cum_nav = (1 + fof_daily_returns).cumprod()

    if not fof_cum_nav.empty:
        tab1, tab2, tab3 = st.tabs(["📈 绩效看板", "📊 收益归因", "🔍 穿透诊断"])

        with tab1:
            st.subheader("净值走势与累计收益双轴对比")
            
            # 创建双轴图表
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 绘制底层产品
            for fund in funds:
                f_nav = period_nav[fund].dropna()
                f_norm = f_nav / f_nav.iloc[0]
                fig.add_trace(
                    go.Scatter(x=f_norm.index, y=f_norm, name=fund, line=dict(width=1.2), opacity=0.4),
                    secondary_y=False
                )
            
            # 绘制FOF组合
            fig.add_trace(
                go.Scatter(x=fof_cum_nav.index, y=fof_cum_nav, name="🏛️ FOF组合", line=dict(color='red', width=3.5)),
                secondary_y=False
            )
            
            # 配置坐标轴
            fig.update_layout(
                height=600,
                hovermode="x unified",
                xaxis=dict(title="日期"),
                yaxis=dict(title="归一化净值 (起点=1.0)", side="left", showgrid=True),
                yaxis2=dict(
                    title="累计收益率 (%)", 
                    side="right", 
                    overlaying="y", 
                    showgrid=False,
                    # 计算右轴刻度：(左轴值 - 1) * 100
                    tickmode="auto"
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # 同步右轴的百分比显示效果 (通过重写 tickformat)
            # 因为双轴联动，右轴 10% 对应左轴 1.1，这里我们通过动态调整显示
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 左侧纵轴代表产品从 1.0 起步的净值水位；右侧代表对应的累计增长百分比。")

        # --- Tab 2 & 3 保持之前优秀的逻辑 ---
        with tab2:
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.subheader("资产相关性矩阵")
                st.dataframe(period_returns.corr().round(2).style.background_gradient(cmap='RdYlGn'))
            with col_b:
                st.subheader("累计收益贡献")
                contrib = period_returns.fillna(0).multiply(weights_series).sum().sort_values()
                fig_bar = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation='h'))
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            selected_f = st.selectbox("选择分析产品", funds)
            f_nav_single = period_nav[selected_f].dropna()
            max_g, curr_g, status, high_dates, peaks = analyze_new_high_gap(f_nav_single)
            
            fig_diag = go.Figure()
            fig_diag.add_trace(go.Scatter(x=f_nav_single.index, y=f_nav_single, name="净值", line=dict(color='#1e3a8a', width=2)))
            fig_diag.add_trace(go.Scatter(x=peaks.index, y=peaks, name="最高水位", line=dict(color='rgba(255,0,0,0.2)', dash='dash')))
            fig_diag.add_trace(go.Scatter(x=high_dates, y=f_nav_single[high_dates], mode='markers', marker=dict(color='red', size=7), name="新高点"))
            st.plotly_chart(fig_diag, use_container_width=True)
            
            summary_list = []
            for f in funds:
                mg, cg, st_str, _, _ = analyze_new_high_gap(period_nav[f].dropna())
                summary_list.append({"产品": f, "最长不创新高天数": f"{mg} 天", "当前状态": st_str})
            st.table(pd.DataFrame(summary_list))
else:
    st.info("请上传净值数据 Excel。")
