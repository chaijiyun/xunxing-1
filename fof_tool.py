import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 身份验证逻辑
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
                <p style='color: #666;'>终极收益率对齐与穿透版</p>
            </div>
        """, unsafe_allow_html=True)
        pwd = st.text_input("", type="password", placeholder="请输入授权码并按回车...")
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
    """计算创新高间隔及当前状态（带0.05%容差）"""
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
st.set_page_config(layout="wide", page_title="寻星 2.4 终极版")

if st.sidebar.button("🔒 退出系统并锁定"):
    st.session_state["authenticated"] = False
    st.rerun()

st.title("🏛️ 寻星配置分析系统 2.4")
st.caption("专业的私募FOF分析工具 | 收益率对齐与创新高穿透专项版")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    # 加载并排序
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

    # 计算FOF组合表现
    fof_daily_returns = period_returns.fillna(0).multiply(weights_series).sum(axis=1)
    fof_cum_nav = (1 + fof_daily_returns).cumprod()

    if not fof_cum_nav.empty:
        # --- 核心指标看板 ---
        c1, c2, c3, c4 = st.columns(4)
        total_ret = fof_cum_nav.iloc[-1] - 1
        days_in_period = (fof_cum_nav.index[-1] - fof_cum_nav.index[0]).days
        ann_ret = (1 + total_ret)**(365.25/max(days_in_period, 1)) - 1
        mdd = (fof_cum_nav / fof_cum_nav.cummax() - 1).min()
        vol = fof_daily_returns.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / vol if vol != 0 else 0
        
        c1.metric("累计收益率", f"{total_ret*100:.2f}%")
        c2.metric("最大回撤", f"{mdd*100:.2f}%")
        c3.metric("年化收益率", f"{ann_ret*100:.2f}%")
        c4.metric("夏普比率", f"{sharpe:.2f}")

        # --- 模块化 Tab ---
        tab1, tab2, tab3 = st.tabs(["📈 绩效看板", "📊 收益归因", "🔍 穿透诊断"])

        with tab1:
            st.subheader("累计收益率走势 (0% 起点对齐)")
            fig_nav = go.Figure()
            # 绘制底层
            for fund in funds:
                f_nav = period_nav[fund].dropna()
                f_cum_ret = (f_nav / f_nav.iloc[0] - 1) * 100
                fig_nav.add_trace(go.Scatter(x=f_cum_ret.index, y=f_cum_ret, name=fund, 
                                             line=dict(width=1.2), opacity=0.5))
            # 绘制组合
            fof_cum_ret = (fof_cum_nav - 1) * 100
            fig_nav.add_trace(go.Scatter(x=fof_cum_ret.index, y=fof_cum_ret, name="🏛️ FOF组合", 
                                         line=dict(color='red', width=3.5)))
            
            fig_nav.update_layout(yaxis_title="累计收益率 (%)", hovermode="x unified", height=600,
                                  shapes=[dict(type='line', y0=0, y1=0, x0=fof_cum_ret.index.min(), 
                                               x1=fof_cum_ret.index.max(), line=dict(color="gray", dash="dash"))])
            st.plotly_chart(fig_nav, use_container_width=True)

        with tab2:
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.subheader("资产相关性矩阵")
                st.dataframe(period_returns.corr().round(2).style.background_gradient(cmap='RdYlGn'))
            with col_b:
                st.subheader("收益贡献拆解")
                contrib = period_returns.fillna(0).multiply(weights_series).sum().sort_values()
                fig_bar = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation='h'))
                fig_bar.update_layout(xaxis_tickformat=".2%", height=400)
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            st.subheader("底层产品“创新高周期”穿透诊断")
            selected_f = st.selectbox("选择要分析的产品", funds)
            f_nav_single = period_nav[selected_f].dropna()
            
            # 计算诊断数据
            max_g, curr_g, status, high_dates, peaks = analyze_new_high_gap(f_nav_single)
            
            # 绘图
            fig_diag = go.Figure()
            fig_diag.add_trace(go.Scatter(x=f_nav_single.index, y=f_nav_single, name="实际净值", line=dict(color='#1e3a8a', width=2.5)))
            fig_diag.add_trace(go.Scatter(x=peaks.index, y=peaks, name="历史最高水位线", line=dict(color='rgba(255,0,0,0.3)', dash='dash')))
            fig_diag.add_trace(go.Scatter(x=high_dates, y=f_nav_single[high_dates], mode='markers', marker=dict(color='red', size=8), name="创新高时刻"))
            
            fig_diag.update_layout(title=f"{selected_f} - 路径分析 (历史最长无新高间隔: {max_g} 天)", height=500, hovermode="x unified")
            st.plotly_chart(fig_diag, use_container_width=True)
            
            st.markdown("#### 🔍 全员无新高状态一览")
            summary_list = []
            for f in funds:
                mg, cg, st_str, _, _ = analyze_new_high_gap(period_nav[f].dropna())
                summary_list.append({"产品": f, "历史最长无新高天数": f"{mg} 天", "当前状态": st_str})
            st.table(pd.DataFrame(summary_list))

else:
    st.info("👋 欢迎进入寻星投研系统！请在左侧上传 Excel 净值表开始分析。")
