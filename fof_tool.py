import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from empyrical import max_drawdown, sharpe_ratio

# --- 系统配置 ---
st.set_page_config(layout="wide", page_title="寻星配置分析系统1.0")

# --- 网页标题 ---
st.title("🏛️ 寻星配置分析系统 1.0")
st.caption("专业的私募FOF资产配置与深度产品画像工具 | 内部专用版")
st.markdown("---")

# --- 侧边栏：数据与参数 ---
st.sidebar.header("🛠️ 系统控制面板")
uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    # 加载原始数据
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
    raw_df = raw_df.sort_index()
    
    # 提取收益率
    returns_df = raw_df.pct_change()

    # 2. 日期筛选
    st.sidebar.subheader("2. 回测区间设置")
    min_date = raw_df.index.min().to_pydatetime()
    max_date = raw_df.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("开始日期", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date, min_value=min_date, max_value=max_date)
    
    # 3. 权重设置
    funds = raw_df.columns.tolist()
    st.sidebar.subheader("3. 目标配置比例")
    target_weights = {}
    for fund in funds:
        target_weights[fund] = st.sidebar.slider(f"{fund}", 0.0, 1.0, 1.0/len(funds))
    
    # 4. 刻度频率选择
    st.sidebar.subheader("4. 图表显示设置")
    freq_option = st.sidebar.selectbox("横轴日期频率", ["月度展示", "季度展示"])
    dtick_val = "M1" if freq_option == "月度展示" else "M3"

    # --- 核心计算逻辑 ---
    mask = (returns_df.index >= pd.Timestamp(start_date)) & (returns_df.index <= pd.Timestamp(end_date))
    period_returns = returns_df.loc[mask]

    # 权重归一化
    total_tw = sum(target_weights.values()) if sum(target_weights.values()) != 0 else 1
    weights_series = pd.Series({k: v / total_tw for k, v in target_weights.items()})

    def calculate_dynamic_fof(daily_ret):
        available = daily_ret.notna() 
        if not available.any(): return 0.0
        curr_w = weights_series[available]
        if curr_w.sum() == 0: return 0.0
        actual_w = curr_w / curr_w.sum()
        return (daily_ret[available] * actual_w).sum()

    fof_daily_returns = period_returns.apply(calculate_dynamic_fof, axis=1)
    fof_cum_nav = (1 + fof_daily_returns).cumprod()

    # --- 1. 指标展示 ---
    if not fof_cum_nav.empty:
        c1, c2, c3, c4 = st.columns(4)
        days_span = (fof_cum_nav.index[-1] - fof_cum_nav.index[0]).days
        years_span = max(days_span / 365.25, 0.01)
        total_ret = fof_cum_nav[-1] - 1
        ann_ret = (1 + total_ret)**(1/years_span)-1
        mdd = max_drawdown(fof_daily_returns)
        vol = fof_daily_returns.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / vol if vol != 0 else 0

        c1.metric("累计收益率", f"{total_ret*100:.2f}%")
        c2.metric("年化收益率", f"{ann_ret*100:.2f}%")
        c3.metric("最大回撤", f"{mdd*100:.2f}%")
        c4.metric("夏普比率", f"{sharpe:.2f}")

        # --- 2. 绘图逻辑 ---
        fig = go.Figure()

        for fund in funds:
            f_ret = period_returns[fund].dropna()
            if not f_ret.empty:
                f_cum = (1 + f_ret).cumprod()
                fig.add_trace(go.Scatter(x=f_cum.index, y=f_cum, name=f'底层-{fund}', 
                                         line=dict(dash='dot', width=1.2), opacity=0.4, yaxis='y1'))

        fig.add_trace(go.Scatter(x=fof_cum_nav.index, y=fof_cum_nav, name='寻星组合净值', 
                                 line=dict(color='red', width=3.5), yaxis='y1'))
        
        rolling_max = fof_cum_nav.cummax()
        dd_series = (fof_cum_nav - rolling_max) / rolling_max
        fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series, name='组合回撤(右轴)', 
                                 fill='tozeroy', line=dict(color='rgba(255,0,0,0.1)'), yaxis='y2'))

        fig.update_layout(
            title=f"寻星组合分析图 (当前频率: {freq_option})",
            xaxis=dict(
                title="日期", 
                tickformat="%Y-%m", 
                dtick=dtick_val, 
                tickangle=-45, 
                showgrid=True
            ),
            yaxis=dict(title="累计净值", side='left'),
            yaxis2=dict(
                title="回撤幅度", 
                overlaying='y', 
                side='right', 
                range=[-0.6, 0], 
                tickformat=".0%"
            ),
            hovermode="x unified", 
            height=600, 
            margin=dict(b=100),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 3. 深度分析表 ---
        st.markdown("### 🔍 底层产品深度指标分析")
        analysis_data = []
        for fund in funds:
            f_ret = period_returns[fund].dropna()
            if f_ret.empty: continue
            f_cum = (1 + f_ret).cumprod()
            pos_prob = (f_ret > 0).sum() / len(f_ret)
            
            # 窗口逻辑
            window = 52 if len(f_ret) > 60 else 12
            rolling_ret = f_cum.pct_change(periods=window)
            win_rate = (rolling_ret > 0).sum() / len(rolling_ret.dropna()) if not rolling_ret.dropna().empty else 0
            
            f_rolling_max = f_cum.cummax()
            f_dd = (f_cum - f_rolling_max) / f_rolling_max
            max_rec, tmp_start = 0, None
            for date, val in f_dd.items():
                if val < 0 and tmp_start is None: tmp_start = date
                elif val == 0 and tmp_start is not None:
                    max_rec = max(max_rec, (date - tmp_start).days)
                    tmp_start = None
            
            analysis_data.append({
                "产品": fund,
                "正收益概率(胜率)": f"{pos_prob*100:.1f}%",
                "持有1年盈利概率": f"{win_rate*100:.1f}%",
                "最长回撤修复天数": f"{max_rec} 天"
            })
        st.table(pd.DataFrame(analysis_data))

        # --- 4. 相关性 ---
        st.subheader("📊 底层资产相关性")
        st.dataframe(period_returns.corr().style.background_gradient(cmap='RdYlGn').format("{:.2f}"))
    else:
        st.warning("所选日期范围内没有足够数据，请调整开始日期。")
else:
    st.info("👋 欢迎使用寻星配置分析系统1.0！请上传Excel文件开始。")