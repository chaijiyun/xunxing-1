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
                <h2 style='color: #1e3a8a;'>🏛️ 寻星投研系统</h2>
                <p style='color: #666;'>内部专用版 | 请输入授权码访问</p>
            </div>
        """, unsafe_allow_html=True)
        pwd = st.text_input("", type="password", placeholder="请输入授权码并按回车...")
        if st.button("进入系统", use_container_width=True):
            if pwd == "281699":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("密码错误，请联系管理员")
    st.stop()

# ==========================================
# 2. 自定义金融计算函数
# ==========================================
def calculate_sharpe(returns):
    if returns.std() == 0: return 0
    return (returns.mean() / returns.std()) * (252 ** 0.5)

def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative/peak) - 1
    return drawdown.min()

# ==========================================
# 3. 业务逻辑代码 - 2.0 归因修复版
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统2.0")

if st.sidebar.button("🔒 退出系统并锁定"):
    st.session_state["authenticated"] = False
    st.rerun()

st.title("🏛️ 寻星配置分析系统 2.0")
st.caption("专业的私募FOF资产配置与收益归因工具 | 内部专用版")
st.markdown("---")

st.sidebar.header("🛠️ 系统控制面板")
uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
    raw_df = raw_df.sort_index()
    returns_df = raw_df.pct_change()

    st.sidebar.subheader("2. 回测区间设置")
    min_date = raw_df.index.min().to_pydatetime()
    max_date = raw_df.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("开始日期", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date, min_value=min_date, max_value=max_date)
    
    funds = raw_df.columns.tolist()
    st.sidebar.subheader("3. 目标配置比例")
    target_weights = {}
    for fund in funds:
        target_weights[fund] = st.sidebar.slider(f"{fund}", 0.0, 1.0, 1.0/len(funds))
    
    st.sidebar.subheader("4. 图表显示设置")
    freq_option = st.sidebar.selectbox("横轴日期频率", ["月度展示", "季度展示"])
    dtick_val = "M1" if freq_option == "月度展示" else "M3"

    mask = (returns_df.index >= pd.Timestamp(start_date)) & (returns_df.index <= pd.Timestamp(end_date))
    period_returns = returns_df.loc[mask]

    total_tw = sum(target_weights.values()) if sum(target_weights.values()) != 0 else 1
    weights_series = pd.Series({k: v / total_tw for k, v in target_weights.items()})

    daily_contributions = period_returns.fillna(0).multiply(weights_series)
    fof_daily_returns = daily_contributions.sum(axis=1)
    fof_cum_nav = (1 + fof_daily_returns).cumprod()

    if not fof_cum_nav.empty:
        c1, c2, c3, c4 = st.columns(4)
        total_ret = fof_cum_nav[-1] - 1
        mdd = calculate_max_drawdown(fof_daily_returns)
        vol = fof_daily_returns.std() * np.sqrt(252)
        ann_ret = (1 + total_ret)**(365.25/max((fof_cum_nav.index[-1] - fof_cum_nav.index[0]).days, 1))-1
        sharpe = (ann_ret - 0.02) / vol if vol != 0 else 0

        c1.metric("累计收益率", f"{total_ret*100:.2f}%")
        c2.metric("年化收益率", f"{ann_ret*100:.2f}%")
        c3.metric("最大回撤", f"{mdd*100:.2f}%")
        c4.metric("夏普比率", f"{sharpe:.2f}")

        tab1, tab2 = st.tabs(["📈 净值曲线与回撤", "📊 收益贡献归因"])

        with tab1:
            fig = go.Figure()
            for fund in funds:
                f_ret = period_returns[fund].dropna()
                if not f_ret.empty:
                    f_cum = (1 + f_ret).cumprod()
                    fig.add_trace(go.Scatter(x=f_cum.index, y=f_cum, name=f'底层-{fund}', 
                                             line=dict(dash='dot', width=1.2), opacity=0.4, yaxis='y1'))
            fig.add_trace(go.Scatter(x=fof_cum_nav.index, y=fof_cum_nav, name='寻星组合净值', 
                                     line=dict(color='red', width=3.5), yaxis='y1'))
            dd_series = (fof_cum_nav - fof_cum_nav.cummax()) / fof_cum_nav.cummax()
            fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series, name='组合回撤(右轴)', 
                                     fill='tozeroy', line=dict(color='rgba(255,0,0,0.1)'), yaxis='y2'))
            fig.update_layout(height=600, xaxis=dict(dtick=dtick_val, tickformat="%Y-%m"), yaxis2=dict(overlaying='y', side='right', range=[-0.6, 0], tickformat=".0%"), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("🎯 组合收益贡献度拆解")
            cum_contrib = daily_contributions.sum().sort_values(ascending=True)
            fig_contrib = go.Figure(go.Bar(
                x=cum_contrib.values, y=cum_contrib.index, orientation='h',
                marker_color=['red' if x > 0 else 'green' for x in cum_contrib.values]
            ))
            fig_contrib.update_layout(title="底层基金对组合总收益的贡献 (绝对点数)", xaxis_tickformat=".2%", height=max(400, len(funds)*40))
            st.plotly_chart(fig_contrib, use_container_width=True)

        # --- 3. 深度分析表与相关性 ---
        st.markdown("### 🔍 底层产品深度画像")
        analysis_data = []
        for fund in funds:
            f_ret = period_returns[fund].dropna()
            if f_ret.empty: continue
            
            # 正收益概率
            pos_prob = (f_ret > 0).sum() / len(f_ret)
            # 累计贡献
            fund_contrib = daily_contributions[fund].sum()

            # ---【修正：最长回撤修复天数算法】---
            f_cum_inner = (1 + f_ret).cumprod()
            f_peak_inner = f_cum_inner.cummax()
            f_dd_inner = (f_cum_inner - f_peak_inner) / f_peak_inner
            
            max_rec_days = 0
            tmp_start = None
            for date, val in f_dd_inner.items():
                if val < 0 and tmp_start is None:
                    tmp_start = date # 记录回撤开始日期
                elif val == 0 and tmp_start is not None:
                    # 修复成功，计算跨度
                    duration = (date - tmp_start).days
                    if duration > max_rec_days:
                        max_rec_days = duration
                    tmp_start = None
            # -----------------------------------

            analysis_data.append({
                "产品": fund,
                "配置权重": f"{weights_series[fund]*100:.1f}%",
                "收益贡献": f"{fund_contrib*100:.2f}%",
                "胜率": f"{pos_prob*100:.1f}%",
                "最长回撤修复时长": f"{max_rec_days} 天"
            })
        st.table(pd.DataFrame(analysis_data))
        
        st.subheader("📊 底层资产相关性")
        # 此处已删除 background_gradient 以避免 matplotlib 缺失报错
        st.dataframe(period_returns.corr().round(2))
else:
    st.info("👋 欢迎使用寻星配置分析系统2.0！请上传数据开始深度分析。")
