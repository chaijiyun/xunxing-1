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
                <h2 style='color: #1e3a8a;'>🏛️ 寻星投研系统</h2>
                <p style='color: #666;'>内部专用版 | 请输入授权码访问</p>
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
# 2. 核心计算函数
# ==========================================
def calculate_max_drawdown(returns):
    if returns.empty: return 0
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative/peak) - 1
    return drawdown.min()

# ==========================================
# 3. 主程序逻辑
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统2.2")

if st.sidebar.button("🔒 退出系统"):
    st.session_state["authenticated"] = False
    st.rerun()

st.title("🏛️ 寻星配置分析系统 2.2")
st.caption("专业的私募FOF资产配置与收益归因工具 | 2025版")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    # 强制日期转换保险
    raw_df = pd.read_excel(uploaded_file, index_col=0)
    raw_df.index = pd.to_datetime(raw_df.index) 
    raw_df = raw_df.sort_index()
    returns_df = raw_df.pct_change()

    # 日期筛选控制
    st.sidebar.subheader("2. 回测区间设置")
    min_date = raw_df.index.min().to_pydatetime()
    max_date = raw_df.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("开始日期", value=min_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date)
    
    funds = raw_df.columns.tolist()
    st.sidebar.subheader("3. 目标配置比例")
    target_weights = {f: st.sidebar.slider(f, 0.0, 1.0, 1.0/len(funds)) for f in funds}
    
    # 频率设置
    freq_option = st.sidebar.selectbox("横轴日期频率", ["月度展示", "季度展示"])
    dtick_val = "M1" if freq_option == "月度展示" else "M3"

    # 切片计算
    mask = (returns_df.index >= pd.Timestamp(start_date)) & (returns_df.index <= pd.Timestamp(end_date))
    period_returns = returns_df.loc[mask]
    
    total_tw = sum(target_weights.values()) or 1
    weights_series = pd.Series({k: v / total_tw for k, v in target_weights.items()})

    daily_contributions = period_returns.fillna(0).multiply(weights_series)
    fof_daily_returns = daily_contributions.sum(axis=1)
    fof_cum_nav = (1 + fof_daily_returns).cumprod()

    if not fof_cum_nav.empty:
        # 指标看板
        c1, c2, c3, c4 = st.columns(4)
        total_ret = fof_cum_nav.iloc[-1] - 1
        mdd = calculate_max_drawdown(fof_daily_returns)
        vol = fof_daily_returns.std() * np.sqrt(252)
        days_diff = (fof_cum_nav.index[-1] - fof_cum_nav.index[0]).days
        ann_ret = (1 + total_ret)**(365.25/max(days_diff, 1)) - 1
        sharpe = (ann_ret - 0.02) / vol if vol != 0 else 0

        c1.metric("累计收益率", f"{total_ret*100:.2f}%")
        c2.metric("年化收益率", f"{ann_ret*100:.2f}%")
        c3.metric("最大回撤", f"{mdd*100:.2f}%")
        c4.metric("夏普比率", f"{sharpe:.2f}")

        tab1, tab2 = st.tabs(["📈 净值与回撤曲线", "📊 收益贡献归因"])
        
        with tab1:
            fig = go.Figure()
            # FOF净值
            fig.add_trace(go.Scatter(x=fof_cum_nav.index, y=fof_cum_nav, name='组合净值', line=dict(color='red', width=3)))
            # 回撤填充图
            f_peak = fof_cum_nav.cummax()
            f_dd = (fof_cum_nav - f_peak) / f_peak
            fig.add_trace(go.Scatter(x=f_dd.index, y=f_dd, name='组合回撤', fill='tozeroy', line=dict(color='rgba(255,0,0,0.1)'), yaxis='y2'))
            fig.update_layout(yaxis2=dict(overlaying='y', side='right', range=[-0.6, 0], tickformat=".0%"), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            cum_contrib = daily_contributions.sum().sort_values()
            fig_contrib = go.Figure(go.Bar(x=cum_contrib.values, y=cum_contrib.index, orientation='h', marker_color='#1f77b4'))
            fig_contrib.update_layout(title="底层产品对总收益的贡献 (点数)", xaxis_tickformat=".2%")
            st.plotly_chart(fig_contrib, use_container_width=True)

        # --- 深度画像：彻底修正修复天数逻辑 ---
        st.markdown("### 🔍 底层产品深度画像")
        analysis_data = []
        for fund in funds:
            f_ret = period_returns[fund].dropna()
            if f_ret.empty: continue
            
            # 计算回撤跨度
            f_cum = (1 + f_ret).cumprod()
            f_peak = f_cum.cummax()
            f_dd = (f_cum - f_peak) / f_peak
            
            max_days = 0
            start_dt = None
            
            # 使用索引遍历，确保对齐
            for i in range(len(f_dd)):
                val = f_dd.iloc[i]
                dt = f_dd.index[i]
                
                # 只要低于高点，就视为在回撤中 (容差 1e-6)
                if val < -0.000001:
                    if start_dt is None:
                        start_dt = dt
                else:
                    # 回升到高点或创新高
                    if start_dt is not None:
                        diff = (dt - start_dt).days
                        if diff > max_days: max_days = diff
                        start_dt = None
            
            # 检查期末尚未修复的回撤
            if start_dt is not None:
                final_diff = (f_dd.index[-1] - start_dt).days
                if final_diff > max_days: max_days = final_diff

            analysis_data.append({
                "产品名称": fund,
                "本期收益贡献": f"{daily_contributions[fund].sum()*100:.2f}%",
                "正收益周占比": f"{(f_ret > 0).sum()/len(f_ret)*100:.1f}%",
                "最长回撤修复时长": f"{max_days} 天"
            })
        
        st.table(pd.DataFrame(analysis_data))
        
        st.subheader("📊 底层资产相关性矩阵")
        st.dataframe(period_returns.corr().round(2))
else:
    st.info("👋 请上传Excel文件开始分析。")
