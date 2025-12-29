import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==========================================
# 1. 核心计算引擎
# ==========================================
def calculate_metrics(nav, bench=None):
    """计算全套量化指标"""
    res = {}
    returns = nav.pct_change().fillna(0)
    days = (nav.index[-1] - nav.index[0]).days
    
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

def analyze_new_high_gap(nav_series):
    """计算创新高间隔及路径诊断 (复刻 2.5.1)"""
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
st.set_page_config(layout="wide", page_title="寻星配置分析系统 2.9.0", page_icon="📈")

st.sidebar.header("🏛️ 寻星投研控制台")
uploaded_file = st.sidebar.file_uploader("1. 上传底层数据库 (xlsx)", type=["xlsx"])

if uploaded_file:
    # 加载数据并使用 ffill 解决断点问题
    df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
    all_cols = df_raw.columns.tolist()
    
    bench_keywords = ["300", "500", "1000", "指数", "基准"]
    def_bench = [c for c in all_cols if any(k in c for k in bench_keywords)]
    
    st.sidebar.subheader("2. 组合策略配置")
    sel_bench = st.sidebar.selectbox("选择对标基准", def_bench if def_bench else all_cols)
    fund_pool = [c for c in all_cols if c != sel_bench]
    sel_funds = st.sidebar.multiselect("挑选拟配置产品", fund_pool, default=fund_pool[:min(3, len(fund_pool))])
    
    if not sel_funds:
        st.warning("👈 请先勾选底层产品进行配置。")
        st.stop()
    
    st.sidebar.markdown("---")
    weights = {}
    for f in sel_funds:
        weights[f] = st.sidebar.number_input(f"权重: {f}", 0.0, 1.0, 1.0/len(sel_funds), step=0.05)
    
    total_w = sum(weights.values())
    st.sidebar.markdown(f"**当前总权重: {total_w:.2%}**")
    
    analysis_start = st.sidebar.date_input("分析起点", value=df_raw.index.min())
    analysis_end = st.sidebar.date_input("分析终点", value=df_raw.index.max())

    period_data = df_raw.loc[analysis_start:analysis_end].ffill().dropna(how='all')
    norm_data = period_data / period_data.iloc[0]
    
    w_series = pd.Series(weights) / (total_w if total_w > 0 else 1)
    fof_daily_ret = (norm_data[sel_funds].pct_change().fillna(0) * w_series).sum(axis=1)
    fof_nav = (1 + fof_daily_ret).cumprod()
    bench_nav = norm_data[sel_bench]
    
    stats = calculate_metrics(fof_nav, bench_nav)

    tabs = st.tabs(["🚀 配置驾驶舱", "🛡️ 风险压力测试", "🔍 底层穿透诊断", "🧩 资产配置逻辑", "📝 投研报告生成"])

    # --- Tab 1: 配置驾驶舱 ---
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
            fig_bot.add_trace(go.Scatter(x=norm_data.index, y=norm_data[f], name=f"底层:{f}", line=dict(width=1.8, color=cp[i % len(cp)]), opacity=0.7))
        fig_bot.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name=f"基准:{sel_bench}", line=dict(color="#BDC3C7", dash="dot", width=2)))
        fig_bot.add_trace(go.Scatter(x=fof_nav.index, y=fof_nav, name="🏛️ FOF 组合", line=dict(color="#1E3A8A", width=4.5)))
        fig_bot.update_layout(height=550, title="图2：全资产穿透对比", hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig_bot, use_container_width=True)

    # --- Tab 2: 底层穿透诊断 (修正变量名错误) ---
    with tabs[2]:
        mode = st.radio("选择诊断模式", ["单产品深度诊断", "多产品对比分析"], horizontal=True)
        
        if mode == "单产品深度诊断":
            target_f = st.selectbox("🎯 选择诊断目标", sel_funds)
            tn = norm_data[target_f]
            tr = period_data[target_f]
            ts = calculate_metrics(tn, bench_nav)
            
            ca, cb, cc = st.columns(3)
            ca.metric("该资产累计收益", f"{ts['总收益率']:.2%}")
            cb.metric("最大历史回撤", f"{ts['最大回撤']:.2%}")
            cc.metric("配置权重", f"{w_series[target_f]:.1%}")

            # 修复点：确保变量名为 max_g
            max_g, status_str, high_dates = analyze_new_high_gap(tr)
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=tn.index, y=tn, name="实际净值", line=dict(color='#1e3a8a', width=2.5)))
            fig_f.add_trace(go.Scatter(x=high_dates, y=tn[high_dates], mode='markers', name="新高时刻", marker=dict(color='red', size=7)))
            fig_f.update_layout(title=f"{target_f} 路径分析 (最长新高间隔: {max_g}天 | 当前: {status_str})", 
                              height=450, template="plotly_white")
            st.plotly_chart(fig_f, use_container_width=True)

            st.markdown("##### 📅 年度收益对照")
            y_ret = tr.pct_change().fillna(0).resample('YE').apply(lambda x: (1+x).prod()-1)
            y_df = pd.DataFrame(y_ret).T
            y_df.index = ["收益率"]
            y_df.columns = [d.year for d in y_df.columns]
            st.dataframe(y_df.style.format("{:.2%}"), use_container_width=True)

        else:
            st.markdown("### 📐 底层产品多维度对比分析")
            compare_funds = st.multiselect("选择对比产品", sel_funds, default=sel_funds[:min(2, len(sel_funds))])
            if compare_funds:
                fig_comp = go.Figure()
                for f in compare_funds:
                    fig_comp.add_trace(go.Scatter(x=norm_data.index, y=norm_data[f], name=f, line=dict(width=2)))
                fig_comp.update_layout(height=500, title="对比净值走势 (起点归一化)", template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig_comp, use_container_width=True)
                
                comp_metrics = []
                for f in compare_funds:
                    f_m = calculate_metrics(norm_data[f], bench_nav)
                    comp_metrics.append({
                        "产品": f, "总收益率": f"{f_m['总收益率']:.2%}", "年化收益": f"{f_m['年化收益']:.2%}",
                        "最大回撤": f"{f_m['最大回撤']:.2%}", "夏普比率": f"{f_m['夏普比率']:.2f}",
                        "卡玛比率": f"{f_m['卡玛比率']:.2f}"
                    })
                st.table(pd.DataFrame(comp_metrics).set_index("产品"))

    # --- Tab 1, 3, 4, 5 保持功能稳定 ---
    with tabs[1]:
        st.subheader("🛡️ 风险压力测试")
        mdd_curve = (fof_nav / fof_nav.cummax() - 1)
        fig_mdd = go.Figure(go.Scatter(x=mdd_curve.index, y=mdd_curve, fill='tozeroy', line=dict(color="#E74C3C")))
        fig_mdd.update_layout(height=400, title="组合动态回撤路径", yaxis_tickformat=".1%", template="plotly_white")
        st.plotly_chart(fig_mdd, use_container_width=True)

    with tabs[3]:
        st.subheader("🧩 资产配置逻辑")
        col_l, col_r = st.columns(2)
        with col_l:
            st.write("相关性矩阵")
            corr = period_data[sel_funds].pct_change().corr()
            st.plotly_chart(go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu_r')), use_container_width=True)
        with col_r:
            st.write("产品贡献度排行")
            contrib = (period_data[sel_funds].pct_change().fillna(0) * w_series).sum().sort_values()
            fig_contrib = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation='h', marker_color='#1E3A8A'))
            fig_contrib.update_layout(xaxis_tickformat=".2%", height=400)
            st.plotly_chart(fig_contrib, use_container_width=True)

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

else:
    st.info("👋 欢迎使用寻星配置分析系统 2.9.0。请在左侧上传经脚本清洗后的 Excel 总库。")
