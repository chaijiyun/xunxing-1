import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==========================================
# 1. 深度风险分析引擎 (桥水投研风格)
# ==========================================
def analyze_advanced_stats(nav_series, benchmark_nav=None):
    """计算包含索提诺、信息比率等高级指标"""
    ret_daily = nav_series.pct_change().fillna(0)
    days = (nav_series.index[-1] - nav_series.index[0]).days
    ann_ret = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (365.25 / max(days, 1)) - 1
    
    # 最大回撤
    mdd = (nav_series / nav_series.cummax() - 1).min()
    # 年化波动率
    vol = ret_daily.std() * np.sqrt(252)
    # 夏普比率 (无风险利率设为 2%)
    sharpe = (ann_ret - 0.02) / vol if vol > 0 else 0
    # 索提诺比率 (仅针对下行波动)
    downside_ret = ret_daily[ret_daily < 0]
    downside_vol = downside_ret.std() * np.sqrt(252)
    sortino = (ann_ret - 0.02) / downside_vol if downside_vol > 0 else 0
    # 卡玛比率
    calmar = ann_ret / abs(mdd) if abs(mdd) > 0 else 0
    
    info_ratio = 0
    active_risk = 0
    if benchmark_nav is not None:
        bench_ret_daily = benchmark_nav.pct_change().fillna(0)
        active_ret_daily = ret_daily - bench_ret_daily
        active_ret_ann = (1 + active_ret_daily).prod() ** (365.25/max(days, 1)) - 1
        active_risk = active_ret_daily.std() * np.sqrt(252) # 跟踪误差
        info_ratio = active_ret_ann / active_risk if active_risk > 0 else 0

    return {
        "ann_ret": ann_ret, "mdd": mdd, "vol": vol, "sharpe": sharpe,
        "sortino": sortino, "calmar": calmar, "info_ratio": info_ratio,
        "active_risk": active_risk
    }

# ==========================================
# 2. 系统界面
# ==========================================
st.set_page_config(layout="wide", page_title="寻星 2.7.0", page_icon="🏛️")
st.title("🏛️ 寻星配置分析系统 2.7.0")
st.caption("向桥水分析系统致敬：全维度风险对标 | 索提诺/信息比率穿透 | 多产品对比走势图")

uploaded_file = st.sidebar.file_uploader("1. 上传底层数据库", type=["xlsx"])

if uploaded_file:
    # 加载与自选
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all').sort_index()
    all_cols = raw_df.columns.tolist()
    
    st.sidebar.subheader("2. 策略与基准配置")
    bench_candidates = [c for c in all_cols if any(x in c for x in ["300", "500", "指数", "基准"])]
    selected_bench = st.sidebar.selectbox("选择基准 (Benchmark)", bench_candidates if bench_candidates else all_cols)
    
    other_funds = [c for c in all_cols if c != selected_bench]
    selected_funds = st.sidebar.multiselect("勾选配置产品", other_funds, default=other_funds[:min(3, len(other_funds))])
    
    if not selected_funds:
        st.warning("👈 请勾选需要组合的产品。")
        st.stop()

    weights_dict = {}
    for f in selected_funds:
        weights_dict[f] = st.sidebar.number_input(f"权重: {f}", 0.0, 1.0, 1.0/len(selected_funds))
    
    s_date = st.sidebar.date_input("起点", value=raw_df.index.min())
    e_date = st.sidebar.date_input("终点", value=raw_df.index.max())
    
    # 数据计算
    p_nav = raw_df.loc[s_date:e_date].ffill()
    p_nav_norm = p_nav / p_nav.iloc[0]
    
    # FOF 计算
    w_series = pd.Series(weights_dict)
    w_series = w_series / w_series.sum()
    fof_ret = (p_nav_norm[selected_funds].pct_change() * w_series).sum(axis=1)
    fof_nav = (1 + fof_ret).cumprod()
    bench_nav = p_nav_norm[selected_bench]
    
    # 统计核心指标
    f_stats = analyze_advanced_stats(fof_nav, bench_nav)
    b_stats = analyze_advanced_stats(bench_nav)

    t1, t2, t3 = st.tabs(["📊 FOF 看板 (全维度对比)", "🔍 桥水风险诊断", "📄 导出深度投研报告"])

    with t1:
        st.markdown("### 🏛️ FOF 综合配置绩效")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("年化收益", f"{f_stats['ann_ret']:.2%}")
        m2.metric("最大回撤", f"{f_stats['mdd']:.2%}", f"基准: {b_stats['mdd']:.1%}", delta_color="inverse")
        m3.metric("索提诺比率", f"{f_stats['sortino']:.2f}", help="针对下行波动衡量，越高越稳")
        m4.metric("信息比率", f"{f_stats['info_ratio']:.2f}", help="每单位主动风险带来的超额收益")
        m5.metric("卡玛比率", f"{f_stats['calmar']:.2f}", help="收益/回撤比，衡量性价比")

        # 增强走势图：增加所有底层产品曲线
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        
        # 1. 细线展示底层产品
        for fund in selected_funds:
            fig.add_trace(go.Scatter(x=p_nav_norm.index, y=p_nav_norm[fund], name=f"底层:{fund}", 
                                     line=dict(width=1), opacity=0.4), row=1, col=1)
        
        # 2. 粗线展示 FOF 和 基准
        fig.add_trace(go.Scatter(x=fof_nav.index, y=fof_nav, name="🏛️ FOF组合", line=dict(color='red', width=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name=f"基准:{selected_bench}", line=dict(color='gray', width=2, dash='dot')), row=1, col=1)
        
        # 3. 回撤填充
        mdd_curve = (fof_nav / fof_nav.cummax() - 1)
        fig.add_trace(go.Scatter(x=mdd_curve.index, y=mdd_curve, name="FOF回撤", fill='tozeroy', line=dict(color='rgba(255,0,0,0.15)')), row=2, col=1)
        
        fig.update_layout(height=700, hovermode="x unified", legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown("### 🧬 桥水式风险诊断")
        col_la, col_lb = st.columns(2)
        
        with col_la:
            st.write("**风险调整后回报对比 (Risk-Adjusted)**")
            comparison_df = pd.DataFrame({
                "指标": ["年化收益", "夏普比率", "索提诺比率", "卡玛比率", "年化波动"],
                "FOF组合": [f"{f_stats['ann_ret']:.2%}", f"{f_stats['sharpe']:.2f}", f"{f_stats['sortino']:.2f}", f"{f_stats['calmar']:.2f}", f"{f_stats['vol']:.2%}"],
                "基准": [f"{b_stats['ann_ret']:.2%}", f"{b_stats['sharpe']:.2f}", f"{b_stats['sortino']:.2f}", f"{b_stats['calmar']:.2f}", f"{b_stats['vol']:.2%}"]
            })
            st.table(comparison_df)
            
        with col_lb:
            st.write("**Alpha 稳定性监控**")
            st.metric("跟踪误差 (Tracking Error)", f"{f_stats['active_risk']:.2%}", help="越低代表超额越稳定，越高代表偏离基准越剧烈")
            st.metric("信息比率 (IR)", f"{f_stats['info_ratio']:.2f}")

    with t3:
        st.write("点击按钮生成深度分析快报...")
        # 此处集成 2.7.0 的全量 HTML 导出逻辑，包含新增的索提诺等指标
        report_html = f"""
        <div style="font-family: sans-serif; padding: 20px; border: 2px solid #1e3a8a;">
            <h2 style="color: #1e3a8a; text-align: center;">寻星投研 2.7.0 深度报告</h2>
            <p>对标基准: {selected_bench} | 时间区间: {s_date} to {e_date}</p>
            <hr>
            <h3>核心分析结论</h3>
            <ul>
                <li><b>组合信息比率 (IR): {f_stats['info_ratio']:.2f}</b> - 衡量超额收益的性价比。</li>
                <li><b>索提诺比率: {f_stats['sortino']:.2f}</b> - 衡量在承受相同下行风险时获得的回报。</li>
                <li><b>最长回撤周期内表现</b>: 组合最大回撤 {f_stats['mdd']:.2%}。</li>
            </ul>
        </div>
        """
        st.download_button("💾 导出 2.7.0 专业报告", report_html, "寻星投研深度版.html", "text/html")

else:
    st.info("💡 系统已准备就绪。请确保底层数据库中包含产品净值及至少一个宽基指数（如中证1000）。")
