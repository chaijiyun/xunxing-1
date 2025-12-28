import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. 身份验证逻辑 (密码: 281699)
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.set_page_config(page_title="身份验证", page_icon="🔐")
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center; background-color: #f0f2f6; padding: 30px; border-radius: 10px; border: 1px solid #dcdfe6;'>
                <h2 style='color: #1e3a8a;'>🏛️ 寻星投研系统 2.4</h2>
                <p style='color: #666;'>终极自适应双轴 & 布局优化版</p>
            </div>
        """, unsafe_allow_html=True)
        pwd = st.text_input("", type="password", placeholder="请输入授权码...")
        if st.button("进入系统", use_container_width=True):
            if pwd == "281699":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("密码错误")
    st.stop()

# ==========================================
# 2. 核心算法逻辑
# ==========================================
def analyze_new_high_gap(nav_series):
    """计算创新高间隔及当前状态"""
    if nav_series.empty or len(nav_series) < 2: 
        return 0, 0, "数据不足", nav_series, nav_series
    
    peak_series = nav_series.cummax()
    new_high_mask = nav_series >= (peak_series * 0.9995)
    new_high_dates = nav_series[new_high_mask].index
    
    if len(new_high_dates) >= 2:
        gaps = pd.Series(new_high_dates).diff().dt.days
        max_historical_gap = int(gaps.max())
    else:
        max_historical_gap = (nav_series.index[-1] - nav_series.index[0]).days
    
    current_gap = (nav_series.index[-1] - new_high_dates[-1]).days
    status = f"⚠️ 持续 {current_gap} 天" if current_gap > 7 else "✅ 处于新高附近"
    return max(max_historical_gap, current_gap), current_gap, status, new_high_dates, peak_series

# ==========================================
# 3. 业务主界面
# ==========================================
st.set_page_config(layout="wide", page_title="寻星 2.4 终极自适应版")

if st.sidebar.button("🔒 退出系统"):
    st.session_state["authenticated"] = False
    st.rerun()

st.title("🏛️ 寻星配置分析系统 2.4")
st.caption("2025-12-27 更新：自适应坐标轴、右轴收益率刻度、Tab2 垂直布局")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    # 加载数据并排序
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).dropna(how='all').sort_index()
    
    st.sidebar.subheader("2. 时间与配置")
    s_date = st.sidebar.date_input("开始日期", value=raw_df.index.min())
    e_date = st.sidebar.date_input("结束日期", value=raw_df.index.max())
    
    period_nav = raw_df.loc[s_date:e_date]
    period_returns = period_nav.pct_change()
    funds = period_nav.columns.tolist()
    
    # 权重设定
    target_weights = {f: st.sidebar.slider(f, 0.0, 1.0, 1.0/len(funds)) for f in funds}
    tw_total = sum(target_weights.values()) or 1
    weights_series = pd.Series({k: v / tw_total for k, v in target_weights.items()})

    # 计算FOF组合
    fof_daily_returns = period_returns.fillna(0).multiply(weights_series).sum(axis=1)
    fof_cum_nav = (1 + fof_daily_returns).cumprod()

    if not fof_cum_nav.empty:
        # --- 模块化 Tab ---
        tab1, tab2, tab3 = st.tabs(["📈 绩效看板", "📊 收益归因", "🔍 穿透诊断"])

        with tab1:
            st.subheader("净值走势与累计收益双轴对比")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 记录所有曲线的最大最小值，用于自适应坐标轴
            y1_all_values = [fof_cum_nav.max(), fof_cum_nav.min()]
            
            # 1. 绘制底层产品 (归一化)
            for fund in funds:
                f_nav = period_nav[fund].dropna()
                if not f_nav.empty:
                    f_norm = f_nav / f_nav.iloc[0]
                    y1_all_values.extend([f_norm.max(), f_norm.min()])
                    fig.add_trace(go.Scatter(
                        x=f_norm.index, y=f_norm, name=fund, 
                        line=dict(width=1.2), opacity=0.4
                    ), secondary_y=False)
            
            # 2. 绘制 FOF 组合
            fig.add_trace(go.Scatter(
                x=fof_cum_nav.index, y=fof_cum_nav, name="🏛️ FOF组合", 
                line=dict(color='red', width=3.8)
            ), secondary_y=False)
            
            # 3. 动态计算坐标轴范围 (核心修复)
            y1_max = max(y1_all_values) * 1.08  # 预留8%空间防止冲顶
            y1_min = min(y1_all_values) * 0.95  # 下方预留5%
            
            # 4. 同步计算右轴收益率范围
            y2_max = (y1_max - 1) * 100
            y2_min = (y1_min - 1) * 100

            fig.update_layout(
                height=650,
                hovermode="x unified",
                yaxis=dict(title="归一化净值 (起点=1.0)", range=[y1_min, y1_max], side="left", showgrid=True),
                yaxis2=dict(title="累计收益率 (%)", range=[y2_min, y2_max], side="right", showgrid=False, ticksuffix="%"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 走势图已自动适配最高净值产品。左轴看净值水位，右轴看累计涨幅。")

        with tab2:
            # 布局优化：上下排列
            st.subheader("📊 资产相关性矩阵")
            st.dataframe(period_returns.corr().round(2).style.background_gradient(cmap='RdYlGn'), use_container_width=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            st.subheader("🎯 资产累计收益贡献")
            contrib = period_returns.fillna(0).multiply(weights_series).sum().sort_values()
            fig_bar = go.Figure(go.Bar(
                x=contrib.values, y=contrib.index, 
                orientation='h', marker_color='#1e3a8a'
            ))
            fig_bar.update_layout(xaxis_tickformat=".2%", height=max(400, len(funds)*40))
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            st.subheader("🔍 底层产品“路径穿透”诊断")
            selected_f = st.selectbox("切换分析产品", funds)
            f_nav_single = period_nav[selected_f].dropna()
            
            max_g, curr_g, status, high_dates, peaks = analyze_new_high_gap(f_nav_single)
            
            fig_diag = go.Figure()
            fig_diag.add_trace(go.Scatter(x=f_nav_single.index, y=f_nav_single, name="实际净值", line=dict(color='#1e3a8a', width=2.5)))
            fig_diag.add_trace(go.Scatter(x=peaks.index, y=peaks, name="最高水位线", line=dict(color='rgba(255,0,0,0.2)', dash='dash')))
            fig_diag.add_trace(go.Scatter(x=high_dates, y=f_nav_single[high_dates], mode='markers', marker=dict(color='red', size=8), name="创新高时刻"))
            
            fig_diag.update_layout(title=f"{selected_f} - 历史最长无新高间隔: {max_g} 天", height=500, hovermode="x unified")
            st.plotly_chart(fig_diag, use_container_width=True)
            
            summary_list = []
            for f in funds:
                mg, cg, st_str, _, _ = analyze_new_high_gap(period_nav[f].dropna())
                summary_list.append({"产品": f, "历史最长无新高天数": f"{mg} 天", "当前状态": st_str})
            st.table(pd.DataFrame(summary_list))

else:
    st.info("👋 系统就绪。请上传 Excel 净值表开始深度分析。")
