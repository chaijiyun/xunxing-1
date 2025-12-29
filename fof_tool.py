import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==========================================
# 1. 核心计算引擎 (保持 2.9.0 兼容性)
# ==========================================
def calculate_metrics(nav, bench=None):
    """计算全套量化指标（增强了对 NaN 的防护）"""
    res = {}
    nav = nav.dropna().ffill()
    if len(nav) < 2:
        return {k: 0.0 for k in ["总收益率", "年化收益", "最大回撤", "夏普比率", "索提诺", "卡玛比率", "波动率", "信息比率"]}
    
    returns = nav.pct_change().fillna(0)
    days = (nav.index[-1] - nav.index[0]).days
    
    total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / max(days, 1)) - 1
    mdd = (nav / nav.cummax() - 1).min()
    vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / vol if vol > 0 else 0
    
    downside_vol = returns[returns < 0].std() * np.sqrt(252)
    sortino = (ann_ret - 0.02) / downside_vol if downside_vol > 0 else 0
    calmar = ann_ret / abs(mdd) if abs(mdd) > 0 else 0
    
    res = {
        "总收益率": total_ret, "年化收益": ann_ret, "最大回撤": mdd, 
        "夏普比率": sharpe, "索提诺": sortino, "卡玛比率": calmar, "波动率": vol
    }
    
    if bench is not None:
        bench = bench.reindex(nav.index).ffill()
        b_ret = bench.pct_change().fillna(0)
        active_ret = returns - b_ret
        te = active_ret.std() * np.sqrt(252)
        ir = (active_ret.mean() * 252) / te if te > 0 else 0
        res["信息比率"] = ir
    return res

def analyze_new_high_gap(nav_series):
    """计算创新高间隔及路径诊断"""
    nav_series = nav_series.dropna()
    if nav_series.empty:
        return 0, "无数据", nav_series.index
    peak_series = nav_series.cummax()
    new_high_mask = nav_series >= (peak_series * 0.9995)
    new_high_dates = nav_series[new_high_mask].index
    
    if len(new_high_dates) > 0:
        current_gap = (nav_series.index[-1] - new_high_dates[-1]).days
        status = f"已持续 {current_gap} 天" if current_gap > 7 else "✅ 处于新高附近"
        gaps = pd.Series(new_high_dates).diff().dt.days
        m_gap = int(gaps.max()) if not gaps.empty and not pd.isna(gaps.max()) else current_gap
    else:
        status = "无新高记录"
        m_gap = 0
    return m_gap, status, new_high_dates

# ==========================================
# 2. 系统 UI 配置
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统 2.10.0", page_icon="📈")

st.sidebar.header("🏛️ 寻星投研控制台")
uploaded_file = st.sidebar.file_uploader("1. 上传底层数据库 (xlsx)", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index()
    all_cols = df_raw.columns.tolist()
    
    sel_bench = st.sidebar.selectbox("选择对标基准", all_cols)
    fund_pool = [c for c in all_cols if c != sel_bench]
    sel_funds = st.sidebar.multiselect("挑选拟配置产品", fund_pool, default=fund_pool[:min(3, len(fund_pool))])
    
    if not sel_funds:
        st.warning("👈 请先勾选底层产品进行配置。")
        st.stop()
    
    st.sidebar.markdown("---")
    weights = {f: st.sidebar.number_input(f"权重: {f}", 0.0, 1.0, 1.0/len(sel_funds), step=0.05) for f in sel_funds}
    total_w = sum(weights.values())
    st.sidebar.markdown(f"**当前总权重: {total_w:.2%}**")
    
    analysis_start = st.sidebar.date_input("分析起点", value=df_raw.index.min())
    analysis_end = st.sidebar.date_input("分析终点", value=df_raw.index.max())

    # --- 数据对齐与归一化逻辑 (底座逻辑) ---
    period_data = df_raw.loc[analysis_start:analysis_end].ffill()
    norm_data = period_data.copy()
    for col in norm_data.columns:
        first_valid = norm_data[col].first_valid_index()
        if first_valid is not None:
            norm_data[col] = norm_data[col] / norm_data.loc[first_valid, col]
    
    w_series = pd.Series(weights) / (total_w if total_w > 0 else 1)
    fof_daily_ret = (norm_data[sel_funds].pct_change().fillna(0) * w_series).sum(axis=1)
    fof_nav = (1 + fof_daily_ret).cumprod()
    bench_nav = norm_data[sel_bench].ffill()
    
    stats = calculate_metrics(fof_nav, bench_nav)

    # 导航栏：前5个保持不变，新增第6个实验视图
    tabs = st.tabs(["🚀 配置驾驶舱", "🛡️ 风险压力测试", "🔍 底层穿透诊断", "🧩 资产配置逻辑", "📝 投研报告生成", "🧪 模拟测试(Beta)"])

    # --- Tab 1: 配置驾驶舱 (保持不变) ---
    with tabs[0]:
        st.markdown("### 🏛️ 寻星配置核心表现")
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("总收益率", f"{stats['总收益率']:.2%}")
        c2.metric("年化收益", f"{stats['年化收益']:.2%}")
        c3.metric("最大回撤", f"{stats['最大回撤']:.2%}", delta_color="inverse")
        c4.metric("夏普比率", f"{stats['夏普比率']:.2f}")
        c5.metric("索提诺比率", f"{stats['索提诺']:.2f}")
        c6.metric("卡玛比率", f"{stats['卡玛比率']:.2f}")
        c7.metric("信息比率", f"{stats['信息比率']:.2f}")

        st.markdown("---")
        fig_top = go.Figure()
        fig_top.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name=f"基准:{sel_bench}", line=dict(color="#BDC3C7", dash="dot", width=2)))
        fig_top.add_trace(go.Scatter(x=fof_nav.index, y=fof_nav, name="🏛️ FOF 组合", line=dict(color="#1E3A8A", width=4)))
        fig_top.update_layout(height=450, title="图1：FOF 组合 vs 业绩基准", hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig_top, use_container_width=True)

        fig_bot = go.Figure()
        cp = ['#16A085', '#2980B9', '#8E44AD', '#D35400', '#2C3E50', '#C0392B', '#27AE60']
        for i, f in enumerate(sel_funds):
            f_plot = norm_data[f].dropna()
            fig_bot.add_trace(go.Scatter(x=f_plot.index, y=f_plot, name=f"底层:{f}", line=dict(width=1.8, color=cp[i % len(cp)]), opacity=0.7))
        fig_bot.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name=f"基准:{sel_bench}", line=dict(color="#BDC3C7", dash="dot", width=2)))
        fig_bot.add_trace(go.Scatter(x=fof_nav.index, y=fof_nav, name="🏛️ FOF 组合", line=dict(color="#1E3A8A", width=4.5)))
        fig_bot.update_layout(height=550, title="图2：全资产穿透对比", hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig_bot, use_container_width=True)

    # --- Tab 2: 风险压力测试 (保持不变) ---
    with tabs[1]:
        st.subheader("🛡️ 风险路径分析")
        mdd_curve = (fof_nav / fof_nav.cummax() - 1)
        fig_mdd = go.Figure(go.Scatter(x=mdd_curve.index, y=mdd_curve, fill='tozeroy', line=dict(color="#E74C3C")))
        fig_mdd.update_layout(height=400, title="组合动态回撤路径", yaxis_tickformat=".1%", template="plotly_white")
        st.plotly_chart(fig_mdd, use_container_width=True)

    # --- Tab 3: 底层穿透诊断 (保持不变) ---
    with tabs[2]:
        target_f = st.selectbox("🎯 选择诊断目标", sel_funds)
        tn = norm_data[target_f].dropna(); tr = period_data[target_f].dropna()
        ts = calculate_metrics(tn, bench_nav)
        
        ca, cb, cc = st.columns(3)
        ca.metric("该资产累计收益", f"{ts['总收益率']:.2%}"); cb.metric("最大历史回撤", f"{ts['最大回撤']:.2%}"); cc.metric("配置权重", f"{w_series[target_f]:.1%}")

        max_g, status_str, high_dates = analyze_new_high_gap(tr)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=tn.index, y=tn, name="实际净值", line=dict(color='#1e3a8a', width=2.5)))
        fig_f.add_trace(go.Scatter(x=high_dates, y=tn[high_dates], mode='markers', name="新高时刻", marker=dict(color='red', size=7)))
        fig_f.update_layout(title=f"{target_f} 路径分析 (最长新高间隔: {max_g}天 | 当前: {status_str})", height=450, template="plotly_white")
        st.plotly_chart(fig_f, use_container_width=True)

    # --- Tab 4: 资产配置逻辑 (更新：数字标注 + 上下布局) ---
    with tabs[3]:
        st.subheader("🧩 资产配置穿透逻辑")
        
        # 1. 相关性矩阵 (增加数字标注)
        st.markdown("#### 1. 底层资产相关性系数 (数值视图)")
        corr = period_data[sel_funds].pct_change().corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}", # 核心更新：显示数字
            hoverinfo="z"
        ))
        fig_corr.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        
        # 2. 贡献度排行 (纵向排列，解决拥挤)
        st.markdown("#### 2. 产品贡献度分析 (绝对贡献)")
        contrib = (period_data[sel_funds].pct_change().fillna(0) * w_series).sum().sort_values()
        fig_contrib = go.Figure(go.Bar(
            x=contrib.values, y=contrib.index, 
            orientation='h', 
            marker_color='#1E3A8A',
            text=[f"{v:.2%}" for v in contrib.values], textposition='auto'
        ))
        fig_contrib.update_layout(height=400 + (len(sel_funds) * 20), xaxis_tickformat=".2%", template="plotly_white")
        st.plotly_chart(fig_contrib, use_container_width=True)

    # --- Tab 5: 投研报告生成 (保持不变) ---
    with tabs[4]:
        st.subheader("📝 投研报告生成预览")
        report_html = f"""<div style="border: 2px solid #1E3A8A; padding: 30px; border-radius: 15px; font-family: sans-serif;">
            <h2 style="color: #1E3A8A; text-align: center;">🏛️ 寻星配置分析系统 投研报告</h2>
            <p style="text-align: right;">日期: {datetime.now().strftime('%Y-%m-%d')}</p><hr>
            <h4>1. 核心表现 (FOF组合)</h4><ul>
                <li>年化收益: {stats['年化收益']:.2%}</li><li>最大回撤: {stats['最大回撤']:.2%}</li>
                <li>夏普比率: {stats['夏普比率']:.2f}</li><li>卡玛比率: {stats['卡玛比率']:.2f}</li>
            </ul></div>"""
        st.markdown(report_html, unsafe_allow_html=True)
        st.download_button("💾 下载报告 (HTML)", report_html, "寻星投研报告.html", "text/html")

    # --- Tab 6: 模拟测试 (Beta 实验模块) ---
    with tabs[5]:
        st.header("🧪 策略模拟实验室 (Beta)")
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.subheader("🗠 蒙特卡洛收益路径预测")
            n_sim = st.slider("模拟路径次数", 100, 1000, 500)
            t_days = st.number_input("未来预测天数 (交易日)", 20, 252, 126)
            
            if st.button("运行蒙特卡洛模拟"):
                mu = fof_daily_ret.mean()
                sigma = fof_daily_ret.std()
                sim_results = np.zeros((t_days, n_sim))
                for i in range(n_sim):
                    daily_sim = np.random.normal(mu, sigma, t_days)
                    sim_results[:, i] = fof_nav.iloc[-1] * (1 + daily_sim).cumprod()
                
                fig_sim = go.Figure()
                for i in range(min(50, n_sim)): # 展示50条样本
                    fig_sim.add_trace(go.Scatter(y=sim_results[:, i], mode='lines', line=dict(width=0.6), opacity=0.3, showlegend=False))
                fig_sim.update_layout(title=f"未来 {t_days} 天净值演化路径", yaxis_title="预期净值", template="plotly_white")
                st.plotly_chart(fig_sim, use_container_width=True)
                st.success(f"模拟完成！持有期末净值中位数预测: {np.median(sim_results[-1, :]):.4f}")

        with col_s2:
            st.subheader("📉 极端情景压力测试")
            st.write("模拟当前组合在历史极端行情下的即时冲击：")
            scene_data = {
                "2015 股灾流动性冲击": -0.15,
                "2018 中美贸易战慢熊": -0.08,
                "2022 权益市场深度回调": -0.12,
                "自定义黑天鹅事件": -0.20
            }
            sel_scene = st.selectbox("选择压力测试场景", list(scene_data.keys()))
            impact = scene_data[sel_scene]
            
            stress_nav = fof_nav.iloc[-1] * (1 + impact)
            st.metric("情景后预估净值", f"{stress_nav:.4f}", delta=f"{impact:.1%}", delta_color="inverse")
            st.info("注：压力测试基于静态权重，未考虑风险平价调仓的防御效应。")

else:
    st.info("👋 欢迎使用寻星配置分析系统 2.10.0。请在左侧上传经脚本清洗后的 Excel 总库以开启底座。")
