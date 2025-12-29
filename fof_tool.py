import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==========================================
# 1. 桥水级量化分析引擎
# ==========================================
def calculate_metrics(nav, bench=None):
    """计算全套量化指标"""
    res = {}
    returns = nav.pct_change().fillna(0)
    days = (nav.index[-1] - nav.index[0]).days
    
    # 核心指标计算
    total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / max(days, 1)) - 1
    mdd = (nav / nav.cummax() - 1).min()
    vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / vol if vol > 0 else 0
    
    # 索提诺比率 (Sortino)
    downside_vol = returns[returns < 0].std() * np.sqrt(252)
    sortino = (ann_ret - 0.02) / downside_vol if downside_vol > 0 else 0
    
    # 卡玛比率 (Calmar)
    calmar = ann_ret / abs(mdd) if abs(mdd) > 0 else 0
    
    res = {
        "总收益率": total_ret, "年化收益": ann_ret, "最大回撤": mdd, 
        "夏普比率": sharpe, "索提诺": sortino, "卡玛比率": calmar, "波动率": vol
    }
    
    if bench is not None:
        b_ret = bench.pct_change().fillna(0)
        active_ret = returns - b_ret
        te = active_ret.std() * np.sqrt(252)
        ir = (active_ret.mean() * 252) / te if te > 0 else 0
        res["信息比率"] = ir
    return res

# ==========================================
# 2. 系统 UI 配置
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统 2.9.0", page_icon="📈")

st.sidebar.header("🏛️ 寻星投研控制台")
uploaded_file = st.sidebar.file_uploader("1. 上传底层数据库 (xlsx)", type=["xlsx"])

if uploaded_file:
    # 加载数据并强制对齐日期，使用 ffill 解决断点问题
    df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
    all_cols = df_raw.columns.tolist()
    
    # 自动识别基准
    bench_keywords = ["300", "500", "1000", "指数", "基准"]
    def_bench = [c for c in all_cols if any(k in c for k in bench_keywords)]
    
    st.sidebar.subheader("2. 组合策略配置")
    sel_bench = st.sidebar.selectbox("选择对标基准", def_bench if def_bench else all_cols)
    fund_pool = [c for c in all_cols if c != sel_bench]
    sel_funds = st.sidebar.multiselect("挑选拟配置产品", fund_pool, default=fund_pool[:min(3, len(fund_pool))])
    
    if not sel_funds:
        st.warning("👈 请先勾选底层产品进行配置。")
        st.stop()
    
    # 权重配置
    st.sidebar.markdown("---")
    weights = {}
    for f in sel_funds:
        weights[f] = st.sidebar.number_input(f"权重: {f}", 0.0, 1.0, 1.0/len(sel_funds), step=0.05)
    
    total_w = sum(weights.values())
    st.sidebar.markdown(f"**当前总权重: {total_w:.2%}**")
    
    analysis_start = st.sidebar.date_input("分析起点", value=df_raw.index.min())
    analysis_end = st.sidebar.date_input("分析终点", value=df_raw.index.max())

    # 数据归一化处理
    period_data = df_raw.loc[analysis_start:analysis_end].ffill().dropna(how='all')
    norm_data = period_data / period_data.iloc[0]
    
    # 计算组合净值
    w_series = pd.Series(weights) / (total_w if total_w > 0 else 1)
    fof_daily_ret = (norm_data[sel_funds].pct_change().fillna(0) * w_series).sum(axis=1)
    fof_nav = (1 + fof_daily_ret).cumprod()
    bench_nav = norm_data[sel_bench]
    
    # 计算指标
    stats = calculate_metrics(fof_nav, bench_nav)

    # 看板导航
    tabs = st.tabs(["🚀 配置驾驶舱", "🛡️ 风险压力测试", "🔍 底层穿透诊断", "🧩 资产配置逻辑", "📝 投研报告生成"])

    # --- Tab 1: FOF 驾驶舱 (核心优化区) ---
    with tabs[0]:
        st.markdown("### 🏛️ 寻星配置核心表现")
        
        # 1. 核心指标区
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("总收益率", f"{stats['总收益率']:.2%}")
        c2.metric("年化收益", f"{stats['年化收益']:.2%}")
        c3.metric("最大回撤", f"{stats['最大回撤']:.2%}", delta_color="inverse")
        c4.metric("夏普比率", f"{stats['夏普比率']:.2f}")
        c5.metric("索提诺比率", f"{stats['索提诺']:.2f}")
        c6.metric("卡玛比率", f"{stats['卡玛比率']:.2f}")
        c7.metric("信息比率", f"{stats['信息比率']:.2f}")

        st.markdown("---")
        
        # 2. 上图：FOF vs 基准 (纯净对标)
        fig_top = go.Figure()
        fig_top.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name=f"基准:{sel_bench}", 
                                     line=dict(color="#BDC3C7", dash="dot", width=2)))
        fig_top.add_trace(go.Scatter(x=fof_nav.index, y=fof_nav, name="🏛️ FOF 组合", 
                                     line=dict(color="#1E3A8A", width=4)))
        
        fig_top.update_layout(height=450, title="图1：FOF 组合 vs 业绩基准 (核心收益曲线)", 
                              hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig_top, use_container_width=True)

        # 3. 下图：全资产穿透 (包含底层产品，颜色加深，线条连续)
        fig_bot = go.Figure()
        
        # 丰富的深色调调色盘，确保底层产品清晰
        color_palette = ['#16A085', '#2980B9', '#8E44AD', '#D35400', '#2C3E50', '#C0392B', '#27AE60']
        
        for i, f in enumerate(sel_funds):
            fig_bot.add_trace(go.Scatter(
                x=norm_data.index, y=norm_data[f], 
                name=f"底层:{f}", 
                line=dict(width=1.8, color=color_palette[i % len(color_palette)]),
                opacity=0.7  # 提高透明度饱和度，确保看得清
            ))
        
        fig_bot.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name=f"基准:{sel_bench}", 
                                     line=dict(color="#BDC3C7", dash="dot", width=2)))
        fig_bot.add_trace(go.Scatter(x=fof_nav.index, y=fof_nav, name="🏛️ FOF 组合", 
                                     line=dict(color="#1E3A8A", width=4.5)))
        
        fig_bot.update_layout(height=550, title="图2：全资产穿透对比 (组合归因与底层贡献)", 
                              hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig_bot, use_container_width=True)

    # --- 其他看板保持专业水准 ---
    with tabs[1]:
        st.subheader("🛡️ 风险路径分析")
        mdd_curve = (fof_nav / fof_nav.cummax() - 1)
        fig_mdd = go.Figure(go.Scatter(x=mdd_curve.index, y=mdd_curve, fill='tozeroy', line=dict(color="#E74C3C")))
        fig_mdd.update_layout(height=400, title="组合动态回撤路径", yaxis_tickformat=".1%")
        st.plotly_chart(fig_mdd, use_container_width=True)

    with tabs[2]:
        st.subheader("🔍 底层产品深度诊断")
        target_f = st.selectbox("🎯 选择诊断目标", sel_funds)
        tn = norm_data[target_f]
        fig_diag = go.Figure(go.Scatter(x=tn.index, y=tn, name=target_f, line=dict(color="#1E3A8A", width=2)))
        fig_diag.update_layout(title=f"{target_f} 净值走势")
        st.plotly_chart(fig_diag, use_container_width=True)

    with tabs[3]:
        st.subheader("🧩 相关性分析逻辑")
        corr = period_data[sel_funds].pct_change().corr()
        fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu_r'))
        st.plotly_chart(fig_corr, use_container_width=True)

    with tabs[4]:
        st.subheader("📝 投研报告生成预览")
        report_html = f"""
        <div style="border: 2px solid #1E3A8A; padding: 20px; border-radius: 10px;">
            <h2 style="color: #1E3A8A;">寻星投研简报 2.9.0</h2>
            <p>分析区间: {analysis_start} 至 {analysis_end}</p>
            <ul>
                <li>年化收益: {stats['年化收益']:.2%}</li>
                <li>最大回撤: {stats['最大回撤']:.2%}</li>
                <li>夏普比率: {stats['夏普比率']:.2f}</li>
            </ul>
        </div>
        """
        st.markdown(report_html, unsafe_allow_html=True)
        st.download_button("💾 下载报告 (HTML)", report_html, "寻星投研报告.html", "text/html")

else:
    st.info("👋 欢迎使用寻星配置分析系统 2.9.0。请在左侧上传经脚本清洗后的 Excel 总库。")

