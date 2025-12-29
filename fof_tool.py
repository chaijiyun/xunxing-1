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
    """计算核心指标组：年化、回撤、夏普、索提诺、卡玛、信息比率"""
    res = {}
    returns = nav.pct_change().fillna(0)
    days = (nav.index[-1] - nav.index[0]).days
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / max(days, 1)) - 1
    mdd = (nav / nav.cummax() - 1).min()
    vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / vol if vol > 0 else 0
    # 索提诺比率 (Sortino)
    downside_vol = returns[returns < 0].std() * np.sqrt(252)
    sortino = (ann_ret - 0.02) / downside_vol if downside_vol > 0 else 0
    # 卡玛比率 (Calmar)
    calmar = ann_ret / abs(mdd) if abs(mdd) > 0 else 0
    
    res = {"年化收益": ann_ret, "最大回撤": mdd, "夏普比率": sharpe, 
           "索提诺": sortino, "卡玛比率": calmar, "波动率": vol}
    
    if bench is not None:
        b_ret = bench.pct_change().fillna(0)
        active_ret = returns - b_ret
        te = active_ret.std() * np.sqrt(252) # 跟踪误差
        ir = (active_ret.mean() * 252) / te if te > 0 else 0
        res["信息比率"] = ir
        res["跟踪误差"] = te
    return res

# ==========================================
# 2. 系统 UI & 交互配置
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统 2.9.0", page_icon="📈")

# 侧边栏：核心配置区
st.sidebar.header("🏛️ 寻星投研控制台")
uploaded_file = st.sidebar.file_uploader("1. 数据源上传 (底层数据库.xlsx)", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all').sort_index()
    all_cols = df_raw.columns.tolist()
    
    # 自动识别指数/基准
    bench_keywords = ["300", "500", "1000", "指数", "基准"]
    def_bench = [c for c in all_cols if any(k in c for k in bench_keywords)]
    
    st.sidebar.subheader("2. 策略组合配置")
    sel_bench = st.sidebar.selectbox("选择基准 (Benchmark)", def_bench if def_bench else all_cols)
    fund_pool = [c for c in all_cols if c != sel_bench]
    
    # 自选产品功能
    sel_funds = st.sidebar.multiselect("挑选拟持仓产品", fund_pool, default=fund_pool[:min(3, len(fund_pool))])
    
    if not sel_funds:
        st.warning("👈 请在左侧勾选拟分析的底层产品。")
        st.stop()
    
    # 动态权重分配
    st.sidebar.markdown("---")
    st.sidebar.caption("权重分配 (合计需为 1.0)")
    weights = {}
    for f in sel_funds:
        weights[f] = st.sidebar.number_input(f"权重: {f}", 0.0, 1.0, 1.0/len(sel_funds), step=0.05)
    
    total_w = sum(weights.values())
    w_color = "#27AE60" if abs(total_w-1.0) < 0.01 else "#E74C3C"
    st.sidebar.markdown(f"**权重合计: <span style='color:{w_color}'>{total_w:.2%}</span>**", unsafe_allow_html=True)
    
    # 时间范围
    analysis_start = st.sidebar.date_input("分析起点", value=df_raw.index.min())
    analysis_end = st.sidebar.date_input("分析终点", value=df_raw.index.max())

    # 数据处理
    period_data = df_raw.loc[analysis_start:analysis_end].ffill()
    norm_data = period_data / period_data.iloc[0]
    
    # 组合净值计算
    w_series = pd.Series(weights) / (total_w if total_w > 0 else 1)
    fof_daily_ret = (norm_data[sel_funds].pct_change().fillna(0) * w_series).sum(axis=1)
    fof_nav = (1 + fof_daily_ret).cumprod()
    bench_nav = norm_data[sel_bench]
    
    # ==========================================
    # 3. 五大功能看板渲染
    # ==========================================
    tabs = st.tabs(["🚀 FOF 驾驶舱", "🛡️ 风险压力测试", "🔍 底层穿透诊断", "🧩 资产配置逻辑", "📝 投研报告生成"])

    # --- Tab 1: FOF 驾驶舱 ---
    with tabs[0]:
        st.subheader("🏛️ FOF 组合核心表现 (对标: %s)" % sel_bench)
        stats = calculate_metrics(fof_nav, bench_nav)
        b_stats = calculate_metrics(bench_nav)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("年化收益率", f"{stats['年化收益']:.2%}")
        c2.metric("最大回撤", f"{stats['最大回撤']:.2%}", f"基准 {b_stats['最大回撤']:.1%}", delta_color="inverse")
        c3.metric("夏普比率", f"{stats['夏普比率']:.2f}")
        c4.metric("卡玛比率", f"{stats['卡玛比率']:.2f}", help="收益回撤比")
        c5.metric("信息比率 (IR)", f"{stats['信息比率']:.2f}", help="超越基准的稳定性")

        # 核心多线走势图
        fig_main = go.Figure()
        for f in sel_funds:
            fig_main.add_trace(go.Scatter(x=norm_data.index, y=norm_data[f], name=f"底层:{f}", line=dict(width=1, color="rgba(100,100,100,0.2)")))
        fig_main.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name=f"基准:{sel_bench}", line=dict(color="#BDC3C7", dash="dot", width=2)))
        fig_main.add_trace(go.Scatter(x=fof_nav.index, y=fof_nav, name="🏛️ FOF 组合", line=dict(color="#1E3A8A", width=4)))
        
        fig_main.update_layout(height=600, hovermode="x unified", title="组合 vs 单资产累计表现", template="plotly_white")
        st.plotly_chart(fig_main, use_container_width=True)

    # --- Tab 2: 风险压力测试 ---
    with tabs[1]:
        st.subheader("🛡️ 风险路径与暴露分析")
        mdd_curve = (fof_nav / fof_nav.cummax() - 1)
        
        cola, colb = st.columns([2, 1])
        with cola:
            fig_mdd = go.Figure()
            fig_mdd.add_trace(go.Scatter(x=mdd_curve.index, y=mdd_curve, fill='tozeroy', name="FOF 回撤", line=dict(color="#E74C3C")))
            fig_mdd.update_layout(height=450, title="组合动态回撤路径", yaxis_tickformat=".1%")
            st.plotly_chart(fig_mdd, use_container_width=True)
        
        with colb:
            st.write("**风险体检表**")
            risk_table = pd.DataFrame({
                "分析维度": ["年化波动率", "最大回撤", "下行标准差", "跟踪误差 (TE)"],
                "组合数值": [f"{stats['波动率']:.2%}", f"{stats['最大回撤']:.2%}", 
                           f"{(fof_daily_ret[fof_daily_ret<0].std()*np.sqrt(252)):.2%}", f"{stats['跟踪误差']:.2%}"]
            })
            st.table(risk_table)

    # --- Tab 3: 底层穿透诊断 ---
    with tabs[2]:
        st.subheader("🔍 单一底层资产穿透分析")
        target_f = st.selectbox("🎯 选择诊断目标", sel_funds)
        tn = norm_data[target_f]
        ts = calculate_metrics(tn, bench_nav)
        
        # 计算潜伏期（无新高天数）
        peak_t = period_data[target_f].cummax()
        high_dates = period_data[target_f][period_data[target_f] >= (peak_t * 0.9995)].index
        max_gap = pd.Series(high_dates).diff().dt.days.max()

        ca, cb, cc, cd = st.columns(4)
        ca.metric("年化收益率", f"{ts['年化收益']:.2%}")
        cb.metric("最大回撤", f"{ts['最大回撤']:.2%}")
        cc.metric("夏普比率", f"{ts['夏普比率']:.2f}")
        cd.metric("最长无新高周期", f"{max_gap} 天")

        fig_diag = go.Figure()
        fig_diag.add_trace(go.Scatter(x=tn.index, y=tn, name=target_f, line=dict(color="#1E3A8A", width=2)))
        fig_diag.add_trace(go.Scatter(x=high_dates, y=tn[high_dates], mode='markers', name="创新高时刻", marker=dict(color="red", size=6)))
        fig_diag.update_layout(height=450, title=f"{target_f} 净值与创新高时刻诊断")
        st.plotly_chart(fig_diag, use_container_width=True)

    # --- Tab 4: 资产配置逻辑 ---
    with tabs[3]:
        st.subheader("🧩 组合配置与相关性逻辑")
        la, lb = st.columns(2)
        with la:
            st.write("**1. 资产相关性矩阵 (低相关是组合的灵魂)**")
            corr = period_data[sel_funds].pct_change().corr().round(2)
            fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu_r', zmin=-1, zmax=1))
            st.plotly_chart(fig_corr, use_container_width=True)
        with lb:
            st.write("**2. 累计收益贡献度分析**")
            contrib = (period_data[sel_funds].pct_change().fillna(0) * w_series).sum().sort_values()
            fig_bar = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation='h', marker_color='#34495E'))
            fig_bar.update_layout(xaxis_tickformat=".2%", title="各资产对FOF总收益的绝对贡献")
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- Tab 5: 投研报告生成 ---
    with tabs[4]:
        st.subheader("📝 投研分析报告预览")
        curr_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        report_body = f"""
        <div style="font-family: 'Microsoft YaHei', sans-serif; border: 3px solid #1E3A8A; padding: 40px; border-radius: 20px;">
            <h1 style="color: #1E3A8A; text-align: center;">🏛️ 寻星配置分析系统 2.9.0 投研简报</h1>
            <p style="text-align: right; color: gray;">生成日期: {curr_time}</p>
            <hr>
            <h3 style="color: #2C3E50;">一、组合绩效总结</h3>
            <p>在指定的分析周期内，组合表现优异：</p>
            <ul>
                <li><b>年化回报:</b> {stats['年化收益']:.2%}</li>
                <li><b>风险控制:</b> 最大回撤 {stats['最大回撤']:.2%}，卡玛比率 {stats['卡玛比率']:.2f}</li>
                <li><b>超额效率:</b> 信息比率(IR)为 {stats['信息比率']:.2f}，表明配置具有极强的阿尔法获取能力。</li>
            </ul>
            <h3 style="color: #2C3E50;">二、持仓构成表</h3>
            <p>{weights}</p>
            <hr>
            <p style="color: #95A5A6; font-size: 13px; text-align: center;">由寻星自动化数据中心驱动 | 严禁用于非法募资展示</p>
        </div>
        """
        st.markdown(report_body, unsafe_allow_html=True)
        st.download_button("💾 下载投研报告 (HTML)", report_body, f"寻星投研报告_{datetime.now().strftime('%m%d')}.html", "text/html")

else:
    st.info("👋 寻星系统 2.9.0 已启动。请在左侧侧边栏上传清洗后的数据库开始投研分析。")
