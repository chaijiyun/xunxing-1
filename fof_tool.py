import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 核心算法逻辑
# ==========================================
def analyze_new_high_gap(nav_series):
    """计算创新高间隔及当前状态"""
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
    status = f"⚠️ 已持续 {current_gap} 天" if current_gap > 7 else "✅ 处于新高附近"
    return max(max_historical_gap, current_gap), current_gap, status, new_high_dates, peak_series

# ==========================================
# 2. 界面配置
# ==========================================
st.set_page_config(layout="wide", page_title="寻星 2.4.2", page_icon="🏛️")

st.title("🏛️ 寻星配置分析系统 2.4.2")
st.caption("2025-12-28 更新：新增底层产品全集成看板 | 修复总收益率指标")

uploaded_file = st.sidebar.file_uploader("1. 上传清洗后的数据库", type=["xlsx"])

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all').sort_index()
    
    st.sidebar.subheader("2. 配置参数")
    s_date = st.sidebar.date_input("分析起点", value=raw_df.index.min())
    e_date = st.sidebar.date_input("分析终点", value=raw_df.index.max())
    
    period_nav = raw_df.loc[s_date:e_date]
    period_returns = period_nav.pct_change()
    funds = period_nav.columns.tolist()
    
    # 权重配置
    target_weights = {f: st.sidebar.slider(f, 0.0, 1.0, 1.0/len(funds)) for f in funds}
    tw_total = sum(target_weights.values()) or 1
    weights_series = pd.Series({k: v / tw_total for k, v in target_weights.items()})

    # 计算FOF组合
    fof_daily_returns = period_returns.fillna(0).multiply(weights_series).sum(axis=1)
    fof_cum_nav = (1 + fof_daily_returns).cumprod()

    if not fof_cum_nav.empty:
        # --- 核心数据准备 ---
        total_ret = fof_cum_nav.iloc[-1] - 1  # 改进1：总收益率
        days_diff = (fof_cum_nav.index[-1] - fof_cum_nav.index[0]).days
        ann_ret = (1 + total_ret)**(365.25/max(days_diff, 1)) - 1
        mdd = (fof_cum_nav / fof_cum_nav.cummax() - 1).min()
        vol = fof_daily_returns.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / vol if vol != 0 else 0

        tab1, tab2, tab3 = st.tabs(["📈 FOF绩效看板", "🔍 底层产品全集成分析", "📊 资产相关性"])

        # --- TAB 1: FOF绩效看板 ---
        with tab1:
            st.markdown("##### 🏛️ FOF组合核心表现")
            c0, c1, c2, c3, c4 = st.columns(5)
            c0.metric("累计总收益", f"{total_ret*100:.2f}%", help="分析期内总回报")
            c1.metric("年化收益率", f"{ann_ret*100:.2f}%")
            c2.metric("最大回撤", f"{mdd*100:.2f}%")
            c3.metric("夏普比率", f"{sharpe:.2f}")
            c4.metric("年化波动率", f"{vol*100:.2f}%")
            
            st.divider()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            y1_all = [fof_cum_nav.max(), fof_cum_nav.min()]
            
            for fund in funds:
                f_norm = period_nav[fund].dropna() / period_nav[fund].dropna().iloc[0]
                y1_all.extend([f_norm.max(), f_norm.min()])
                fig.add_trace(go.Scatter(x=f_norm.index, y=f_norm, name=fund, line=dict(width=1), opacity=0.3), secondary_y=False)
            
            fig.add_trace(go.Scatter(x=fof_cum_nav.index, y=fof_cum_nav, name="🏛️ FOF组合", line=dict(color='red', width=4)), secondary_y=False)
            
            y1_max = max(y1_all) * 1.05
            y1_min = min(y1_all) * 0.98
            fig.update_layout(height=600, hovermode="x unified",
                              yaxis=dict(title="净值水位", range=[y1_min, y1_max]),
                              yaxis2=dict(title="累计涨幅", range=[(y1_min-1)*100, (y1_max-1)*100], ticksuffix="%"),
                              legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig, use_container_width=True)

        # --- TAB 2: 底层产品集成分析 (改进2) ---
        with tab2:
            st.subheader("🔍 底层产品深度穿透")
            selected_f = st.selectbox("🎯 选择要穿透分析的底层产品", funds)
            
            # 数据切片
            f_nav_raw = period_nav[selected_f].dropna()
            f_norm = f_nav_raw / f_nav_raw.iloc[0]
            f_ret = f_nav_raw.pct_change()
            
            # 指标计算
            f_total_ret = f_norm.iloc[-1] - 1
            f_mdd = (f_norm / f_norm.cummax() - 1).min()
            f_vol = f_ret.std() * np.sqrt(252)
            f_contrib = (f_ret.fillna(0) * weights_series[selected_f]).sum()
            
            # 第一行：基础指标
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("累计总收益", f"{f_total_ret*100:.2f}%")
            m2.metric("最大回撤", f"{f_mdd*100:.2f}%")
            m3.metric("年化波动率", f"{f_vol*100:.2f}%")
            m4.metric("对组合总收益贡献", f"{f_contrib*100:.2f}%", help="该产品在持仓期间为FOF带来的点数贡献")
            
            # 第二行：走势与路径诊断
            st.markdown("---")
            max_g, curr_g, status, high_dates, peaks = analyze_new_high_gap(f_nav_raw)
            
            fig_diag = go.Figure()
            fig_diag.add_trace(go.Scatter(x=f_nav_raw.index, y=f_nav_raw, name="产品原值走势", line=dict(color='#1e3a8a', width=2)))
            fig_diag.add_trace(go.Scatter(x=peaks.index, y=peaks, name="水位线", line=dict(color='rgba(200,200,200,0.5)', dash='dash')))
            fig_diag.add_trace(go.Scatter(x=high_dates, y=f_nav_raw[high_dates], mode='markers', marker=dict(color='red', size=7), name="创新高时刻"))
            
            fig_diag.update_layout(title=f"路径分析：历史最长无新高间隔 {max_g} 天 | 当前状态：{status}", height=450)
            st.plotly_chart(fig_diag, use_container_width=True)
            
            # 第三行：年度/季度分析 (额外赠送)
            st.markdown("##### 📅 年度收益表现")
            yearly_ret = f_ret.resample('YE').apply(lambda x: (1+x).prod()-1)
            y_cols = st.columns(len(yearly_ret))
            for i, (year, val) in enumerate(yearly_ret.items()):
                y_cols[i].metric(f"{year.year}年", f"{val*100:.2f}%")

        # --- TAB 3: 相关性分析 ---
        with tab3:
            st.subheader("📊 资产相关性矩阵")
            st.dataframe(period_returns.corr().round(2).style.background_gradient(cmap='RdYlGn'), use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🎯 各产品对FOF组合的贡献排行")
            contrib = period_returns.fillna(0).multiply(weights_series).sum().sort_values()
            fig_bar = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation='h', marker_color='#1e3a8a'))
            fig_bar.update_layout(xaxis_tickformat=".2%", height=500)
            st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("👋 请上传由脚本生成的 '寻星底层数据库.xlsx' 开始分析。")
