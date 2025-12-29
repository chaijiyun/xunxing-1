import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. 核心计算引擎
# ==========================================
def calculate_metrics(nav, bench=None):
    """计算全套量化指标"""
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
    
    res = {"总收益率": total_ret, "年化收益": ann_ret, "最大回撤": mdd, 
           "夏普比率": sharpe, "索提诺": sortino, "卡玛比率": calmar, "波动率": vol}
    
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
    if nav_series.empty: return 0, "无数据", nav_series.index
    peak_series = nav_series.cummax()
    new_high_mask = nav_series >= (peak_series * 0.9995)
    new_high_dates = nav_series[new_high_mask].index
    if len(new_high_dates) > 0:
        current_gap = (nav_series.index[-1] - new_high_dates[-1]).days
        status = f"已持续 {current_gap} 天" if current_gap > 7 else "✅ 处于新高附近"
        gaps = pd.Series(new_high_dates).diff().dt.days
        m_gap = int(gaps.max()) if not gaps.empty and not pd.isna(gaps.max()) else current_gap
    else:
        status = "无新高"; m_gap = 0
    return m_gap, status, new_high_dates

# ==========================================
# 2. 系统 UI 配置
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统 2.11.2", page_icon="📈")

st.sidebar.header("🏛️ 寻星投研控制台")
uploaded_file = st.sidebar.file_uploader("1. 上传底层数据库 (xlsx)", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index()
    all_cols = df_raw.columns.tolist()
    sel_bench = st.sidebar.selectbox("选择对标基准", all_cols)
    fund_pool = [c for c in all_cols if c != sel_bench]
    sel_funds = st.sidebar.multiselect("挑选拟配置产品", fund_pool, default=fund_pool[:min(3, len(fund_pool))])
    
    if not sel_funds: st.stop()
    
    weights = {f: st.sidebar.number_input(f"权重: {f}", 0.0, 1.0, 1.0/len(sel_funds), step=0.05) for f in sel_funds}
    total_w = sum(weights.values())
    analysis_start = st.sidebar.date_input("分析起点", value=df_raw.index.min())
    analysis_end = st.sidebar.date_input("分析终点", value=df_raw.index.max())

    # 数据归一化处理
    period_data = df_raw.loc[analysis_start:analysis_end].ffill()
    norm_data = period_data.copy()
    for col in norm_data.columns:
        fv = norm_data[col].first_valid_index()
        if fv: norm_data[col] = norm_data[col] / norm_data.loc[fv, col]
    
    w_series = pd.Series(weights) / (total_w if total_w > 0 else 1)
    # 计算寻星配置组合表现
    star_daily_ret = (norm_data[sel_funds].pct_change().fillna(0) * w_series).sum(axis=1)
    star_nav = (1 + star_daily_ret).cumprod()
    bench_nav = norm_data[sel_bench].ffill()
    stats = calculate_metrics(star_nav, bench_nav)

    tabs = st.tabs(["🚀 配置驾驶舱", "🛡️ 风险压力测试", "🔍 底层产品分析", "🧩 资产配置逻辑", "📝 投研报告生成", "🧪 模拟测试(Beta)", "📊 资产池全量对比"])

    # --- Tab 1: 配置驾驶舱 ---
    with tabs[0]:
        st.markdown("### 🏛️ 寻星配置组合表现总览")
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("总收益率", f"{stats['总收益率']:.2%}")
        c2.metric("年化收益", f"{stats['年化收益']:.2%}")
        c3.metric("最大回撤", f"{stats['最大回撤']:.2%}", delta_color="inverse")
        c4.metric("夏普比率", f"{stats['夏普比率']:.2f}")
        c5.metric("索提诺比率", f"{stats['索提诺']:.2f}")
        c6.metric("卡玛比率", f"{stats['卡玛比率']:.2f}")
        c7.metric("信息比率", f"{stats['信息比率']:.2f}")
        
        fig_top = go.Figure()
        fig_top.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="🏛️ 寻星配置组合", line=dict(color="#1E3A8A", width=4)))
        fig_top.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name=f"基准:{sel_bench}", line=dict(color="#BDC3C7", dash="dot")))
        fig_top.update_layout(height=450, title="净值曲线：寻星配置组合 vs 业绩基准", template="plotly_white")
        st.plotly_chart(fig_top, use_container_width=True)

    # --- Tab 2: 风险压力测试 ---
    with tabs[1]:
        st.subheader("🛡️ 寻星配置组合风险分析")
        mdd_curve = (star_nav / star_nav.cummax() - 1)
        st.plotly_chart(go.Figure(go.Scatter(x=mdd_curve.index, y=mdd_curve, fill='tozeroy', line=dict(color="#E74C3C"))), use_container_width=True)

    # --- Tab 3: 底层产品分析 (重点重构模块) ---
    with tabs[2]:
        st.subheader("⚔️ 配置池底层产品横向对比")
        
        # 1. 列表对比
        st.markdown("#### 1. 核心量化指标对比表")
        comp_results = []
        for f in sel_funds:
            f_nav_single = period_data[f].dropna()
            m = calculate_metrics(f_nav_single, bench_nav)
            comp_results.append({
                "产品名称": f,
                "总收益率": f"{m['总收益率']:.2%}",
                "年化收益": f"{m['年化收益']:.2%}",
                "最大回撤": f"{m['最大回撤']:.2%}",
                "夏普比率": round(m['夏普比率'], 2),
                "索提诺": round(m['索提诺'], 2),
                "卡玛比率": round(m['卡玛比率'], 2),
                "信息比率": round(m['信息比率'], 2)
            })
        st.table(pd.DataFrame(comp_results).set_index("产品名称"))

        # 2. 走势图对比 (含寻星配置组合)
        st.markdown("#### 2. 净值走势对比 (含寻星配置组合)")
        sel_plot = st.multiselect("筛选曲线", ["🏛️ 寻星配置组合"] + sel_funds, default=["🏛️ 寻星配置组合"] + sel_funds)
        
        fig_multi = go.Figure()
        if "🏛️ 寻星配置组合" in sel_plot:
            fig_multi.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="🏛️ 寻星配置组合", line=dict(color="#1E3A8A", width=4)))
        for f in sel_funds:
            if f in sel_plot:
                fig_multi.add_trace(go.Scatter(x=norm_data.index, y=norm_data[f], name=f, opacity=0.7))
        fig_multi.update_layout(height=500, template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_multi, use_container_width=True)

        st.markdown("---")
        
        # 3. 单产品深度诊断
        st.subheader("🔍 单产品深度路径诊断")
        target_f = st.selectbox("🎯 切换剖析目标", sel_funds)
        diag_l, diag_r = st.columns([3, 1])
        
        tn = norm_data[target_f].dropna(); tr = period_data[target_f].dropna()
        ts = calculate_metrics(tn, bench_nav)
        m_gap, status_str, high_dates = analyze_new_high_gap(tr)

        with diag_l:
            fig_diag = go.Figure()
            fig_diag.add_trace(go.Scatter(x=tn.index, y=tn, name="归一化净值", line=dict(color='#1e3a8a', width=2.5)))
            fig_diag.add_trace(go.Scatter(x=high_dates, y=tn[high_dates], mode='markers', name="新高点", marker=dict(color='red', size=8)))
            fig_diag.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_diag, use_container_width=True)
            
        with diag_r:
            st.markdown("#### 📊 诊断量化指标")
            st.metric("区间累计收益", f"{ts['总收益率']:.2%}")
            st.metric("区间最大回撤", f"{ts['最大回撤']:.2%}")
            st.metric("最长新高间隔", f"{m_gap}天")
            st.metric("年化波动率", f"{ts['波动率']:.2%}")
            st.info(f"**新高状态**: \n{status_str}")

    # --- Tab 4: 资产配置逻辑 (数字标注 + 上下布局) ---
    with tabs[3]:
        st.subheader("🧩 资产配置逻辑穿透")
        st.markdown("#### 1. 相关性矩阵 (数值视图)")
        corr = period_data[sel_funds].pct_change().corr()
        st.plotly_chart(go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu_r', zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}"
        )), use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 2. 产品收益贡献排行")
        contrib = (period_data[sel_funds].pct_change().fillna(0) * w_series).sum().sort_values()
        st.plotly_chart(go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation='h', marker_color='#1E3A8A')), use_container_width=True)

    # --- Tab 6: 实验模块 ---
    with tabs[5]:
        st.header("🧪 模拟实验室 (Beta)")
        if st.button("生成寻星路径预测"):
            mu, sigma = star_daily_ret.mean(), star_daily_ret.std()
            sims = np.zeros((126, 50))
            for i in range(50):
                sims[:, i] = star_nav.iloc[-1] * (1 + np.random.normal(mu, sigma, 126)).cumprod()
            fig_sim = go.Figure()
            for i in range(50): fig_sim.add_trace(go.Scatter(y=sims[:,i], mode='lines', opacity=0.2, showlegend=False))
            fig_sim.update_layout(title="寻星配置组合未来半年路径预测", template="plotly_white")
            st.plotly_chart(fig_sim, use_container_width=True)

    # --- Tab 7: 资产池全量对比 ---
    with tabs[6]:
        st.header("📊 全资产池深度比较")
        all_comp = st.multiselect("挑选对比产品 (全库)", fund_pool, default=fund_pool[:min(5, len(fund_pool))])
        if all_comp:
            res_table = []
            fig_all = go.Figure()
            for f in all_comp:
                f_data = norm_data[f].dropna()
                fig_all.add_trace(go.Scatter(x=f_data.index, y=f_data, name=f))
                m = calculate_metrics(period_data[f].dropna(), bench_nav)
                res_table.append({
                    "产品名称": f, "年化收益": f"{m['年化收益']:.2%}", "最大回撤": f"{m['最大回撤']:.2%}",
                    "夏普比率": round(m['夏普比率'],2), "卡玛比率": round(m['卡玛比率'],2)
                })
            st.plotly_chart(fig_all, use_container_width=True)
            st.table(pd.DataFrame(res_table).set_index("产品名称"))

else:
    st.info("👋 欢迎使用寻星配置分析系统 2.11.2。请上传 Excel 数据库启动系统底座。")
