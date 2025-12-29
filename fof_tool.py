import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import base64

# ==========================================
# 1. 深度风险分析引擎 (桥水投研风格)
# ==========================================
def analyze_advanced_stats(nav_series, benchmark_nav=None):
    """计算包含索提诺、信息比率等高级指标"""
    ret_daily = nav_series.pct_change().fillna(0)
    days = (nav_series.index[-1] - nav_series.index[0]).days
    # 累计收益
    total_ret = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1
    # 年化收益 (365.25天逻辑)
    ann_ret = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (365.25 / max(days, 1)) - 1
    
    # 最大回撤
    mdd_series = (nav_series / nav_series.cummax() - 1)
    mdd = mdd_series.min()
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
        # 确保日期对齐计算超额
        active_ret_daily = ret_daily - bench_ret_daily
        # 年化超额收益
        active_ret_ann = active_ret_daily.mean() * 252
        # 跟踪误差
        active_risk = active_ret_daily.std() * np.sqrt(252)
        info_ratio = active_ret_ann / active_risk if active_risk > 0 else 0

    return {
        "total_ret": total_ret, "ann_ret": ann_ret, "mdd": mdd, "vol": vol, 
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar, 
        "info_ratio": info_ratio, "active_risk": active_risk, "mdd_series": mdd_series
    }

# ==========================================
# 2. 系统界面布局
# ==========================================
st.set_page_config(layout="wide", page_title="寻星 2.7.0", page_icon="🏛️")

# CSS 注入美化表格
st.markdown("""<style> .metric-card { background-color: #f0f2f6; padding: 10px; border-radius: 10px; } </style>""", unsafe_allow_html=True)

st.title("🏛️ 寻星配置分析系统 2.7.0")
st.caption(f"迭代日期: {datetime.now().strftime('%Y-%m-%d')} | 桥水投研风格 | 索提诺/信息比率穿透 | 多产品对比")

uploaded_file = st.sidebar.file_uploader("1. 上传底层数据库 (Excel)", type=["xlsx"])

if uploaded_file:
    # A. 数据加载
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all').sort_index()
    all_cols = raw_df.columns.tolist()
    
    # B. 基准识别与勾选
    st.sidebar.subheader("2. 策略对标配置")
    bench_candidates = [c for c in all_cols if any(x in c for x in ["300", "500", "指数", "基准", "1000"])]
    selected_bench = st.sidebar.selectbox("选择对标基准 (Benchmark)", bench_candidates if bench_candidates else all_cols)
    
    other_funds = [c for c in all_cols if c != selected_bench]
    selected_funds = st.sidebar.multiselect("挑选拟配置产品", other_funds, default=other_funds[:min(3, len(other_funds))])
    
    if not selected_funds:
        st.warning("👈 请在左侧勾选需要分析的底层产品。")
        st.stop()

    # C. 权重分配
    st.sidebar.markdown("---")
    weights_dict = {}
    for f in selected_funds:
        weights_dict[f] = st.sidebar.number_input(f"权重: {f}", 0.0, 1.0, 1.0/len(selected_funds), step=0.05)
    
    total_w = sum(weights_dict.values())
    st.sidebar.info(f"当前总权重: {total_w:.2%}")

    s_date = st.sidebar.date_input("分析起点", value=raw_df.index.min())
    e_date = st.sidebar.date_input("分析终点", value=raw_df.index.max())
    
    # D. 核心计算
    p_nav = raw_df.loc[s_date:e_date].ffill()
    p_nav_norm = p_nav / p_nav.iloc[0]
    
    # FOF 组合净值
    w_series = pd.Series(weights_dict) / (total_w if total_w != 0 else 1)
    fof_ret = (p_nav_norm[selected_funds].pct_change().fillna(0) * w_series).sum(axis=1)
    fof_nav = (1 + fof_ret).cumprod()
    
    # 基准净值
    bench_nav = p_nav_norm[selected_bench]
    
    # 统计数据
    f_stats = analyze_advanced_stats(fof_nav, bench_nav)
    b_stats = analyze_advanced_stats(bench_nav)

    # --- 功能标签页 ---
    tab1, tab2, tab3 = st.tabs(["📊 FOF 看板 (全维度对比)", "🔍 桥水风险诊断", "📄 导出深度投研报告"])

    with tab1:
        st.markdown("### 🏛️ FOF 综合配置绩效")
        # 指标卡
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("累计收益率", f"{f_stats['total_ret']:.2%}")
        c2.metric("最大回撤", f"{f_stats['mdd']:.2%}", f"基准: {b_stats['mdd']:.1%}", delta_color="inverse")
        c3.metric("索提诺比率", f"{f_stats['sortino']:.2f}", help="针对下行风险的收益比")
        c4.metric("信息比率", f"{f_stats['info_ratio']:.2f}", help="超额收益性价比")
        c5.metric("卡玛比率", f"{f_stats['calmar']:.2f}", help="年化收益/最大回撤")

        # FOF 全图表
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # 叠加底层产品细线
        for fund in selected_funds:
            fig.add_trace(go.Scatter(x=p_nav_norm.index, y=p_nav_norm[fund], name=f"底层:{fund}", 
                                     line=dict(width=1), opacity=0.3), row=1, col=1)
        
        # 叠加 FOF 和 基准粗线
        fig.add_trace(go.Scatter(x=fof_nav.index, y=fof_nav, name="🏛️ FOF组合", line=dict(color='red', width=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name=f"基准:{selected_bench}", line=dict(color='gray', width=2, dash='dot')), row=1, col=1)
        
        # 回撤图
        fig.add_trace(go.Scatter(x=f_stats['mdd_series'].index, y=f_stats['mdd_series'], name="回撤路径", fill='tozeroy', line=dict(color='rgba(255,0,0,0.2)')), row=2, col=1)
        
        fig.update_layout(height=700, hovermode="x unified", legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 🧬 桥水式风险诊断")
        la, lb = st.columns(2)
        with la:
            st.write("**风险对标矩阵**")
            compare_df = pd.DataFrame({
                "分析维度": ["年化收益率", "夏普比率", "索提诺比率", "卡玛比率", "年化波动率"],
                "FOF组合": [f"{f_stats['ann_ret']:.2%}", f"{f_stats['sharpe']:.2f}", f"{f_stats['sortino']:.2f}", f"{f_stats['calmar']:.2f}", f"{f_stats['vol']:.2%}"],
                "对标基准": [f"{b_stats['ann_ret']:.2%}", f"{b_stats['sharpe']:.2f}", f"{b_stats['sortino']:.2f}", f"{b_stats['calmar']:.2f}", f"{b_stats['vol']:.2%}"]
            })
            st.table(compare_df)
        
        with lb:
            st.write("**超额收益 (Alpha) 稳定性**")
            st.metric("跟踪误差 (Tracking Error)", f"{f_stats['active_risk']:.2%}")
            st.metric("信息比率 (Information Ratio)", f"{f_stats['info_ratio']:.2f}")
            
            # 年度胜率
            f_y = fof_nav.resample('YE').apply(lambda x: x.iloc[-1]/x.iloc[0]-1)
            b_y = bench_nav.resample('YE').apply(lambda x: x.iloc[-1]/x.iloc[0]-1)
            win_df = pd.DataFrame({"FOF": f_y, "基准": b_y})
            win_df["超额"] = win_df["FOF"] - win_df["基准"]
            win_df.index = win_df.index.year
            st.write("**年度超额统计**")
            st.dataframe(win_df.style.format("{:.2%}"), use_container_width=True)

    with tab3:
        st.markdown("### 📋 报告导出中心")
        if st.button("生成深度投研报告预览"):
            weights_html = "".join([f"<li>{k}: {v:.1%}</li>" for k, v in weights_dict.items()])
            report_html = f"""
            <div style="font-family: sans-serif; padding: 30px; border: 2px solid #1e3a8a; border-radius: 10px;">
                <h2 style="color: #1e3a8a; text-align: center;">🏛️ 寻星投研资产配置报告 (2.7.0版)</h2>
                <hr>
                <h4>一、FOF 组合概况 (对比基准: {selected_bench})</h4>
                <table style="width:100%; border-collapse: collapse; text-align: center;">
                    <tr style="background-color: #f2f2f2;"><th>指标</th><th>组合表现</th><th>基准表现</th></tr>
                    <tr><td>累计收益</td><td>{f_stats['total_ret']:.2%}</td><td>{b_stats['total_ret']:.2%}</td></tr>
                    <tr><td>最大回撤</td><td>{f_stats['mdd']:.2%}</td><td>{b_stats['mdd']:.2%}</td></tr>
                    <tr><td>索提诺比率</td><td>{f_stats['sortino']:.2f}</td><td>{b_stats['sortino']:.2f}</td></tr>
                    <tr><td>信息比率</td><td>{f_stats['info_ratio']:.2f}</td><td>--</td></tr>
                </table>
                <h4>二、资产配置权重</h4>
                <ul>{weights_html}</ul>
                <p style="color: #666; font-size: 12px; margin-top: 50px;">* 报告由寻星投研系统自动生成。历史业绩不代表未来收益。</p>
            </div>
            """
            st.markdown(report_html, unsafe_allow_html=True)
            st.download_button("💾 下载 HTML 报告 (可直接打印PDF)", report_html, "寻星深度报告.html", "text/html")

else:
    st.info("👋 寻星系统 2.7.0 已就绪。请上传清洗后的数据库 Excel 开始投研之旅。")
