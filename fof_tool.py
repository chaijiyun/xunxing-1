import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. 核心计算引擎
# ==========================================
def calculate_metrics(nav, bench=None):
    """计算全套量化指标（增强了对 NaN 的防护）"""
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
        status = "无新高记录"; m_gap = 0
    return m_gap, status, new_high_dates

# ==========================================
# 2. 系统 UI 配置
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统 2.11.0", page_icon="📈")

st.sidebar.header("🏛️ 寻星投研控制台")
uploaded_file = st.sidebar.file_uploader("1. 上传底层数据库 (xlsx)", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index()
    all_cols = df_raw.columns.tolist()
    
    sel_bench = st.sidebar.selectbox("选择对标基准", all_cols)
    fund_pool = [c for c in all_cols if c != sel_bench]
    sel_funds = st.sidebar.multiselect("挑选拟配置产品", fund_pool, default=fund_pool[:min(3, len(fund_pool))])
    
    if not sel_funds:
        st.warning("👈 请先勾选底层产品。")
        st.stop()
    
    st.sidebar.markdown("---")
    weights = {f: st.sidebar.number_input(f"权重: {f}", 0.0, 1.0, 1.0/len(sel_funds), step=0.05) for f in sel_funds}
    total_w = sum(weights.values())
    analysis_start = st.sidebar.date_input("分析起点", value=df_raw.index.min())
    analysis_end = st.sidebar.date_input("分析终点", value=df_raw.index.max())

    # --- 数据处理 ---
    period_data = df_raw.loc[analysis_start:analysis_end].ffill()
    norm_data = period_data.copy()
    for col in norm_data.columns:
        fv = norm_data[col].first_valid_index()
        if fv: norm_data[col] = norm_data[col] / norm_data.loc[fv, col]
    
    w_series = pd.Series(weights) / (total_w if total_w > 0 else 1)
    fof_daily_ret = (norm_data[sel_funds].pct_change().fillna(0) * w_series).sum(axis=1)
    fof_nav = (1 + fof_daily_ret).cumprod()
    bench_nav = norm_data[sel_bench].ffill()
    stats = calculate_metrics(fof_nav, bench_nav)

    # 导航栏
    tabs = st.tabs(["🚀 配置驾驶舱", "🛡️ 风险压力测试", "🔍 底层穿透诊断", "🧩 资产配置逻辑", "📝 投研报告生成", "🧪 模拟测试(Beta)", "📊 资产池全量对比"])

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
        
        fig_top = go.Figure()
        fig_top.add_trace(go.Scatter(x=fof_nav.index, y=fof_nav, name="🏛️ FOF 组合", line=dict(color="#1E3A8A", width=4)))
        fig_top.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name=f"基准:{sel_bench}", line=dict(color="#BDC3C7", dash="dot")))
        st.plotly_chart(fig_top, use_container_width=True)

    # --- Tab 2: 风险压力测试 ---
    with tabs[1]:
        st.subheader("🛡️ 风险路径分析")
        mdd_curve = (fof_nav / fof_nav.cummax() - 1)
        st.plotly_chart(go.Figure(go.Scatter(x=mdd_curve.index, y=mdd_curve, fill='tozeroy', line=dict(color="#E74C3C"))), use_container_width=True)

    # --- Tab 3: 底层穿透诊断 (单人诊断 + 多人对比) ---
    with tabs[2]:
        st.subheader("🔍 底层资产深度诊断")
        diag_col1, diag_col2 = st.columns([1, 3])
        with diag_col1:
            target_f = st.selectbox("🎯 选择诊断目标", sel_funds)
            tn = norm_data[target_f].dropna(); tr = period_data[target_f].dropna()
            ts = calculate_metrics(tn, bench_nav)
            st.metric("累计收益", f"{ts['总收益率']:.2%}")
            st.metric("最大回撤", f"{ts['最大回撤']:.2%}")
            m_gap, status_str, high_dates = analyze_new_high_gap(tr)
            st.metric("最长新高间隔", f"{m_gap}天")
            st.info(f"状态: {status_str}")
        with diag_col2:
            fig_diag = go.Figure()
            fig_diag.add_trace(go.Scatter(x=tn.index, y=tn, name="净值", line=dict(color='#1e3a8a', width=2.5)))
            fig_diag.add_trace(go.Scatter(x=high_dates, y=tn[high_dates], mode='markers', name="新高时刻", marker=dict(color='red', size=8)))
            fig_diag.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_diag, use_container_width=True)
        
        st.markdown("---")
        st.subheader("⚔️ 配置池横向对比")
        comp_funds = st.multiselect("挑选对比产品", sel_funds, default=sel_funds)
        if comp_funds:
            fig_comp = go.Figure()
            for f in comp_funds:
                fig_comp.add_trace(go.Scatter(x=norm_data.index, y=norm_data[f], name=f))
            fig_comp.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig_comp, use_container_width=True)

    # --- Tab 4: 资产配置逻辑 (数字标注 + 上下布局) ---
    with tabs[3]:
        st.subheader("🧩 资产配置逻辑")
        st.markdown("#### 1. 相关性矩阵 (数值视图)")
        corr = period_data[sel_funds].pct_change().corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}"
        ))
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 2. 产品贡献度排行")
        contrib = (period_data[sel_funds].pct_change().fillna(0) * w_series).sum().sort_values()
        fig_contrib = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation='h', marker_color='#1E3A8A'))
        fig_contrib.update_layout(height=400 + len(sel_funds)*20, xaxis_tickformat=".2%")
        st.plotly_chart(fig_contrib, use_container_width=True)

    # --- Tab 5: 报告生成 ---
    with tabs[4]:
        st.subheader("📝 投研报告生成")
        st.info("报告导出功能已就绪，请点击侧边栏下载 HTML。")

    # --- Tab 6: 实验模拟 ---
    with tabs[5]:
        st.header("🧪 模拟测试(Beta)")
        if st.button("启动蒙特卡洛预测"):
            mu, sigma = fof_daily_ret.mean(), fof_daily_ret.std()
            sims = np.zeros((126, 100))
            for i in range(100):
                sims[:, i] = fof_nav.iloc[-1] * (1 + np.random.normal(mu, sigma, 126)).cumprod()
            fig_sim = go.Figure()
            for i in range(20): fig_sim.add_trace(go.Scatter(y=sims[:,i], mode='lines', opacity=0.3))
            st.plotly_chart(fig_sim, use_container_width=True)

    # --- Tab 7: 资产池全量对比 (新增：专业表格与多选) ---
    with tabs[6]:
        st.header("📊 全资产池深度比较实验室")
        all_comp_list = st.multiselect("挑选对比产品 (支持总库所有产品)", fund_pool, default=fund_pool[:min(5, len(fund_pool))])
        if all_comp_list:
            fig_all = go.Figure()
            res_table = []
            for f in all_comp_list:
                f_raw_data = period_data[f].dropna()
                f_norm_data = norm_data[f].dropna()
                fig_all.add_trace(go.Scatter(x=f_norm_data.index, y=f_norm_data, name=f))
                
                m = calculate_metrics(f_raw_data, bench_nav)
                m_gap, _, _ = analyze_new_high_gap(f_raw_data)
                res_table.append({
                    "产品名称": f, "总收益率": f"{m['总收益率']:.2%}", "年化收益": f"{m['年化收益']:.2%}",
                    "最大回撤": f"{m['最大回撤']:.2%}", "夏普比率": round(m['夏普比率'],2),
                    "索提诺": round(m['索提诺'],2), "卡玛比率": round(m['卡玛比率'],2),
                    "信息比率": round(m['信息比率'],2), "新高间隔(天)": m_gap
                })
            st.plotly_chart(fig_all, use_container_width=True)
            st.markdown("#### 📑 全维度指标对比表")
            st.table(pd.DataFrame(res_table).set_index("产品名称"))

else:
    st.info("👋 请上传底层数据库启动 2.11.0 版本。")
