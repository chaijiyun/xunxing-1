import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 绝对优先的身份验证逻辑
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
                <h2 style='color: #1e3a8a;'>🏛️ 寻星投研系统 2.4</h2>
                <p style='color: #666;'>内部专用版 | 终极修复与归因版</p>
            </div>
        """, unsafe_allow_html=True)
        pwd = st.text_input("", type="password", placeholder="请输入授权码并按回车...")
        if st.button("进入系统", use_container_width=True):
            if pwd == "281699":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("密码错误")
    st.stop()

# ==========================================
# 2. 核心金融算法函数
# ==========================================
def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative/peak) - 1
    return drawdown.min()

def analyze_new_high_gap(nav_series):
    """核心算法：创新高最大间隔天数"""
    if nav_series.empty: return 0, 0, "N/A"
    peak = nav_series.cummax()
    new_high_mask = nav_series >= (peak * 0.9995)
    new_high_dates = nav_series[new_high_mask].index
    if len(new_high_dates) >= 2:
        gaps = pd.Series(new_high_dates).diff().dt.days
        max_historical_gap = int(gaps.max())
    else:
        max_historical_gap = (nav_series.index[-1] - nav_series.index[0]).days
    last_high_date = new_high_dates[-1]
    current_gap = (nav_series.index[-1] - last_high_date).days
    status = f"⚠️ 已持续 {current_gap} 天" if current_gap > 7 else "✅ 处于新高附近"
    return max_historical_gap, current_gap, status

# ==========================================
# 3. 业务逻辑代码
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统 2.4")

if st.sidebar.button("🔒 退出系统并锁定"):
    st.session_state["authenticated"] = False
    st.rerun()

st.title("🏛️ 寻星配置分析系统 2.4")
st.caption("专业的私募FOF资产配置与收益归因工具 | 创新高周期分析专项")
st.markdown("---")

st.sidebar.header("🛠️ 系统控制面板")
uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all').sort_index()
    returns_df = raw_df.pct_change()

    st.sidebar.subheader("2. 回测区间设置")
    min_date, max_date = raw_df.index.min().to_pydatetime(), raw_df.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("开始日期", value=min_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date)
    
    funds = raw_df.columns.tolist()
    st.sidebar.subheader("3. 目标配置比例")
    target_weights = {fund: st.sidebar.slider(f"{fund}", 0.0, 1.0, 1.0/len(funds)) for fund in funds}
    
    freq_option = st.sidebar.selectbox("横轴日期频率", ["月度展示", "季度展示"])
    dtick_val = "M1" if freq_option == "月度展示" else "M3"

    # 数据切片
    mask = (returns_df.index >= pd.Timestamp(start_date)) & (returns_df.index <= pd.Timestamp(end_date))
    period_returns = returns_df.loc[mask]
    period_nav = raw_df.loc[mask]

    # 权重处理
    total_tw = sum(target_weights.values()) or 1
    weights_series = pd.Series({k: v / total_tw for k, v in target_weights.items()})

    # FOF 表现计算
    daily_contributions = period_returns.fillna(0).multiply(weights_series)
    fof_daily_returns = daily_contributions.sum(axis=1)
    fof_cum_nav = (1 + fof_daily_returns).cumprod()

    if not fof_cum_nav.empty:
        # 指标看板
        c1, c2, c3, c4 = st.columns(4)
        total_ret = fof_cum_nav.iloc[-1] - 1
        mdd = calculate_max_drawdown(fof_daily_returns)
        days_diff = max((fof_cum_nav.index[-1] - fof_cum_nav.index[0]).days, 1)
        ann_ret = (1 + total_ret)**(365.25/days_diff)-1
        vol = fof_daily_returns.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / vol if vol != 0 else 0

        c1.metric("累计收益率", f"{total_ret*100:.2f}%")
        c2.metric("年化收益率", f"{ann_ret*100:.2f}%")
        c3.metric("最大回撤", f"{mdd*100:.2f}%")
        c4.metric("夏普比率", f"{sharpe:.2f}")

        tab1, tab2 = st.tabs(["📈 净值与回撤", "📊 收益贡献归因"])

        with tab1:
            fig = go.Figure()
            for fund in funds:
                f_nav = (1 + period_returns[fund].dropna()).cumprod()
                fig.add_trace(go.Scatter(x=f_nav.index, y=f_nav, name=f'底层-{fund}', line=dict(dash='dot', width=1.2), opacity=0.4))
            fig.add_trace(go.Scatter(x=fof_cum_nav.index, y=fof_cum_nav, name='寻星组合净值', line=dict(color='red', width=3.5)))
            dd_series = (fof_cum_nav - fof_cum_nav.cummax()) / fof_cum_nav.cummax()
            fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series, name='组合回撤(右轴)', fill='tozeroy', line=dict(color='rgba(255,0,0,0.1)'), yaxis='y2'))
            fig.update_layout(height=600, xaxis=dict(dtick=dtick_val, tickformat="%Y-%m"), yaxis2=dict(overlaying='y', side='right', range=[-0.6, 0], tickformat=".0%"), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("🎯 组合收益贡献度拆解")
            cum_contrib = daily_contributions.sum().sort_values(ascending=True)
            fig_contrib = go.Figure(go.Bar(x=cum_contrib.values, y=cum_contrib.index, orientation='h', marker_color=['#d62728' if x > 0 else '#2ca02c' for x in cum_contrib.values]))
            fig_contrib.update_layout(xaxis_tickformat=".2%", height=max(400, len(funds)*40))
            st.plotly_chart(fig_contrib, use_container_width=True)

        # --- 底层资产“无新高周期”深度画像 ---
        st.markdown("### 🔍 底层产品深度画像 (创新高周期分析)")
        analysis_data = []
        for fund in funds:
            f_ret = period_returns[fund].dropna()
            if f_ret.empty: continue
            f_nav_inner = (1 + f_ret).cumprod()
            
            max_gap, curr_gap, status = analyze_new_high_gap(f_nav_inner)
            
            analysis_data.append({
                "产品名称": fund,
                "配置比例": f"{weights_series[fund]*100:.1f}%",
                "本期贡献": f"{daily_contributions[fund].sum()*100:.2f}%",
                "最长无新高天数 (历史)": f"{max(max_gap, curr_gap)} 天",
                "当前无新高状态": status,
                "区间最大回撤": f"{(f_nav_inner/f_nav_inner.cummax()-1).min()*100:.2f}%"
            })
        st.table(pd.DataFrame(analysis_data))
        
        st.subheader("📊 底层资产相关性矩阵")
        st.dataframe(period_returns.corr().round(2).style.background_gradient(cmap='RdYlGn'))
else:
    st.info("👋 欢迎使用寻星配置分析系统 2.4！请上传数据开始深度分析。")
