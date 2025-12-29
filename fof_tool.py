import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ==========================================
# 1. 核心计算引擎
# ==========================================
def analyze_new_high_gap(nav_series):
    """计算创新高间隔及路径诊断"""
    if nav_series.empty or len(nav_series) < 2: 
        return 0, 0, "数据不足", nav_series, nav_series
    peak_series = nav_series.cummax()
    new_high_mask = nav_series >= (peak_series * 0.9995)
    new_high_dates = nav_series[new_high_mask].index
    current_gap = (nav_series.index[-1] - new_high_dates[-1]).days
    status = f"已持续 {current_gap} 天" if current_gap > 7 else "✅ 处于新高附近"
    gaps = pd.Series(new_high_dates).diff().dt.days
    max_gap = int(gaps.max()) if not gaps.empty else current_gap
    return max_gap, current_gap, status, new_high_dates, peak_series

# ==========================================
# 2. 系统界面设置
# ==========================================
st.set_page_config(layout="wide", page_title="寻星 2.5.1", page_icon="🏛️")
st.title("🏛️ 寻星配置分析系统 2.5.1")
st.caption(f"编译日期: {datetime.now().strftime('%Y-%m-%d')} | 核心指标看板 & 深度穿透分析 & 报告导出")

# 侧边栏：数据管理
uploaded_file = st.sidebar.file_uploader("1. 上传底层数据库", type=["xlsx"])

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all').sort_index()
    all_funds = raw_df.columns.tolist()

    st.sidebar.subheader("2. 组合模拟配置")
    selected_funds = st.sidebar.multiselect("挑选拟配置产品", all_funds, default=all_funds)
    
    if not selected_funds:
        st.warning("👈 请先在左侧勾选需要分析的产品。")
        st.stop()

    # 动态权重输入
    st.sidebar.markdown("---")
    weights_dict = {}
    for f in selected_funds:
        weights_dict[f] = st.sidebar.number_input(f"权重: {f}", 0.0, 1.0, 1.0/len(selected_funds), step=0.05)
    
    total_w = sum(weights_dict.values())
    st.sidebar.progress(min(total_w, 1.0), text=f"当前总权重: {total_w:.2%}")
    if abs(total_w - 1.0) > 0.001:
        st.sidebar.warning("⚠️ 注意：当前权重合计不等于 100%")

    s_date = st.sidebar.date_input("分析起点", value=raw_df.index.min())
    e_date = st.sidebar.date_input("分析终点", value=raw_df.index.max())
    
    # 数据计算准备
    period_nav = raw_df[selected_funds].loc[s_date:e_date].ffill()
    period_returns = period_nav.pct_change().fillna(0)
    w_series = pd.Series(weights_dict) / (total_w if total_w != 0 else 1)

    fof_daily_ret = (period_returns * w_series).sum(axis=1)
    fof_cum_nav = (1 + fof_daily_ret).cumprod()

    # 核心绩效指标
    total_ret = fof_cum_nav.iloc[-1] - 1
    mdd_series = (fof_cum_nav / fof_cum_nav.cummax() - 1)
    mdd = mdd_series.min()
    ann_ret = (1 + total_ret)**(365.25/max((fof_cum_nav.index[-1] - fof_cum_nav.index[0]).days, 1)) - 1
    vol = fof_daily_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / vol if vol != 0 else 0

    # --- 界面展示 ---
    tab1, tab2, tab3 = st.tabs(["📊 FOF 组合看板", "🔍 底层产品全集成分析", "📐 资产相关性"])

    with tab1:
        st.markdown("### 🏛️ 组合绩效概览")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("累计收益率", f"{total_ret:.2%}")
        c2.metric("年化收益率", f"{ann_ret:.2%}")
        c3.metric("最大回撤", f"{mdd:.2%}")
        c4.metric("夏普比率", f"{sharpe:.2f}")
        c5.metric("年化波动率", f"{vol:.2%}")

        # 组合图表
        fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
        fig_main.add_trace(go.Scatter(x=fof_cum_nav.index, y=fof_cum_nav, name="FOF净值", line=dict(color='red', width=3)), row=1, col=1)
        fig_main.add_trace(go.Scatter(x=mdd_series.index, y=mdd_series, name="回撤路径", fill='tozeroy', line=dict(color='gray')), row=2, col=1)
        fig_main.update_layout(height=600, hovermode="x unified", title="FOF组合净值与回撤走势")
        st.plotly_chart(fig_main, use_container_width=True)

    with tab2:
        st.markdown("### 🔍 底层资产深度穿透")
        sf = st.selectbox("选择目标产品", selected_funds)
        
        f_nav = period_nav[sf]
        f_ret = f_nav.pct_change().fillna(0)
        f_total_ret = (f_nav.iloc[-1]/f_nav.iloc[0]) - 1
        f_mdd = (f_nav / f_nav.cummax() - 1).min()
        
        # 1. 产品指标卡
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("该资产累计收益", f"{f_total_ret:.2%}")
        with col_b:
            st.metric("最大历史回撤", f"{f_mdd:.2%}")
        with col_c:
            st.metric("配置权重", f"{w_series[sf]:.1%}")

        # 2. 路径图
        max_g, curr_g, status, high_dates, peaks = analyze_new_high_gap(f_nav)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=f_nav.index, y=f_nav, name="实际净值", line=dict(color='#1e3a8a')))
        fig_f.add_trace(go.Scatter(x=high_dates, y=f_nav[high_dates], mode='markers', name="新高时刻", marker=dict(color='red')))
        fig_f.update_layout(title=f"{sf} 路径分析 (最长新高间隔: {max_g}天 | 当前: {status})", height=400)
        st.plotly_chart(fig_f, use_container_width=True)

        # 3. 年度收益统计
        st.markdown("##### 📅 年度收益对照")
        y_ret = f_ret.resample('YE').apply(lambda x: (1+x).prod()-1)
        y_df = pd.DataFrame(y_ret).T
        y_df.index = ["收益率"]
        y_df.columns = [d.year for d in y_df.columns]
        st.dataframe(y_df.style.format("{:.2%}"), use_container_width=True)

    with tab3:
        st.markdown("### 📊 资产配置逻辑")
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.write("相关性矩阵")
            st.dataframe(period_returns.corr().round(2).style.background_gradient(cmap='RdYlGn'), use_container_width=True)
        with col_r:
            st.write("产品贡献度排行")
            contrib = (period_returns * w_series).sum().sort_values()
            fig_c = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation='h'))
            fig_c.update_layout(xaxis_tickformat=".1%", height=400)
            st.plotly_chart(fig_c, use_container_width=True)

    # --- 报告导出逻辑 (集成版) ---
    st.sidebar.markdown("---")
    if st.sidebar.button("📊 生成全量投研报告"):
        # 生成年度统计HTML
        f_stats_html = ""
        for f in selected_funds:
            f_stats_html += f"<li><b>{f}</b>: 累计收益 {(period_nav[f].iloc[-1]/period_nav[f].iloc[0]-1):.2%}, 权重 {w_series[f]:.1%}</li>"

        report_html = f"""
        <div style="font-family: 'Microsoft YaHei', sans-serif; padding: 30px; border: 2px solid #1e3a8a; border-radius: 10px;">
            <h1 style="color: #1e3a8a; text-align: center;">🏛️ 寻星投研资产配置报告</h1>
            <p style="text-align: right; color: #666;">报告日期: {datetime.now().strftime('%Y-%m-%d')}</p>
            
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h3 style="margin-top:0;">一、组合绩效汇总 (FOF)</h3>
                <table style="width:100%; border-collapse: collapse;">
                    <tr style="background-color: #1e3a8a; color: white;">
                        <th style="padding:10px;">累计收益</th><th style="padding:10px;">年化收益</th>
                        <th style="padding:10px;">最大回撤</th><th style="padding:10px;">夏普比率</th>
                    </tr>
                    <tr style="text-align: center; border-bottom: 1px solid #ddd;">
                        <td style="padding:10px;">{total_ret:.2%}</td><td style="padding:10px;">{ann_ret:.2%}</td>
                        <td style="padding:10px;">{mdd:.2%}</td><td style="padding:10px;">{sharpe:.2f}</td>
                    </tr>
                </table>
            </div>

            <div style="margin-bottom: 20px;">
                <h3>二、配置构成及底层分析</h3>
                <ul>{f_stats_html}</ul>
            </div>

            <div style="margin-bottom: 20px;">
                <h3>三、风险提示</h3>
                <p style="color: #d9534f;">注：历史业绩不代表未来表现。模拟组合未计入交易摩擦成本及管理费。</p>
            </div>
            <p style="text-align: center; font-size: 12px; color: #999;">- 寻星自动化数据中心提供技术支持 -</p>
        </div>
        """
        st.markdown(report_html, unsafe_allow_html=True)
        st.download_button("💾 点击下载报告 (HTML版，可直接打印成PDF)", report_html, "寻星投研报告.html", "text/html")

else:
    st.info("👋 欢迎使用寻星系统。请上传清洗后的数据库文件开始投研分析。")
