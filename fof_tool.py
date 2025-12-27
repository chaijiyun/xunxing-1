import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 绝对优先的身份验证逻辑
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
                <p style='color: #666;'>内部专用版 | 业绩基准与回撤修正版</p>
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
# 2. 核心金融计算函数
# ==========================================
def calculate_max_drawdown(returns):
    if returns.empty: return 0
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative/peak) - 1
    return drawdown.min()

# ==========================================
# 3. 主业务代码
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统1.2")

if st.sidebar.button("🔒 退出系统并锁定"):
    st.session_state["authenticated"] = False
    st.rerun()

st.title("🏛️ 寻星配置分析系统 1.2")
st.caption("专业的私募FOF资产配置、业绩基准对比及回撤穿透工具")
st.markdown("---")

st.sidebar.header("🛠️ 系统控制面板")
uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (含指数)", type=["xlsx"])

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
    raw_df = raw_df.sort_index()
    returns_df = raw_df.pct_change()

    # 自动识别业绩基准 (列名含300或500)
    all_cols = raw_df.columns.tolist()
    benchmarks = [c for c in all_cols if "300" in str(c) or "500" in str(c)]
    funds = [c for c in all_cols if c not in benchmarks]

    st.sidebar.subheader("2. 回测区间设置")
    min_date = raw_df.index.min().to_pydatetime()
    max_date = raw_df.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("开始日期", value=min_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date)
    
    st.sidebar.subheader("3. 目标配置比例")
    target_weights = {f: st.sidebar.slider(f"{f}", 0.0, 1.0, 1.0/len(funds)) for f in funds}
    
    freq_option = st.sidebar.selectbox("横轴日期频率", ["月度展示", "季度展示"])
    dtick_val = "M1" if freq_option == "月度展示" else "M3"

    # --- 核心计算 ---
    mask = (returns_df.index >= pd.Timestamp(start_date)) & (returns_df.index <= pd.Timestamp(end_date))
    period_returns = returns_df.loc[mask]

    # 组合计算
    total_tw = sum(target_weights.values()) or 1
    weights_series = pd.Series({k: v / total_tw for k, v in target_weights.items()})
    
    def calculate_dynamic_fof(daily_ret):
        available = daily_ret[funds].notna() 
        if not available.any(): return 0.0
        actual_w = weights_series[available] / weights_series[available].sum()
        return (daily_ret[available] * actual_w).sum()

    fof_daily_returns = period_returns.apply(calculate_dynamic_fof, axis=1)
    fof_cum_nav = (1 + fof_daily_returns).cumprod()

    if not fof_cum_nav.empty:
        # 指标看板
        c1, c2, c3, c4 = st.columns(4)
        total_ret = fof_cum_nav.iloc[-1] - 1
        mdd = calculate_max_drawdown(fof_daily_returns)
        days_span = max((fof_cum_nav.index[-1] - fof_cum_nav.index[0]).days, 1)
        ann_ret = (1 + total_ret)**(365.25/days_span)-1
        vol = fof_daily_returns.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / vol if vol != 0 else 0

        c1.metric("组合累计收益", f"{total_ret*100:.2f}%")
        c2.metric("组合年化收益", f"{ann_ret*100:.2f}%")
        c3.metric("组合最大回撤", f"{mdd*100:.2f}%")
        c4.metric("组合夏普比率", f"{sharpe:.2f}")

        # 净值曲线图
        fig = go.Figure()
        for f in funds:
            f_cum = (1 + period_returns[f].dropna()).cumprod()
            fig.add_trace(go.Scatter(x=f_cum.index, y=f_cum, name=f'底层-{f}', line=dict(width=1), opacity=0.3))
        for b in benchmarks:
            b_cum = (1 + period_returns[b].dropna()).cumprod()
            fig.add_trace(go.Scatter(x=b_cum.index, y=b_cum, name=f'基准-{b}', line=dict(dash='dash', width=2)))
        fig.add_trace(go.Scatter(x=fof_cum_nav.index, y=fof_cum_nav, name='寻星组合', line=dict(color='red', width=4)))
        
        fig.update_layout(title="净值对比曲线", xaxis=dict(dtick=dtick_val, tickformat="%Y-%m"), hovermode="x unified", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # 深度画像 (回撤逻辑修正)
        st.markdown("### 🔍 深度指标分析 (回撤修复状态穿透)")
        analysis_data = []
        for item in (funds + benchmarks):
            ret = period_returns[item].dropna()
            if ret.empty: continue
            cum = (1 + ret).cumprod()
            
            # --- 精准回撤算法 ---
            dd = (cum - cum.cummax()) / cum.cummax()
            max_rec, ongoing, is_dd, start_dt = 0, 0, False, None
            
            for dt, val in dd.items():
                if val < -0.0001: # 入水 (万分之一容差)
                    if not is_dd:
                        is_dd, start_dt = True, dt
                else: # 出水
                    if is_dd:
                        max_rec = max(max_rec, (dt - start_dt).days)
                        is_dd, start_dt = False, None
            
            if is_dd: # 至今未出水
                ongoing = (dd.index[-1] - start_dt).days
            
            final_days = max(max_rec, ongoing)
            status = "⚠️未修复" if ongoing >= max_rec and ongoing > 0 else "✅已修复"
            
            analysis_data.append({
                "名称": item,
                "性质": "底层产品" if item in funds else "业绩基准",
                "本期累计收益": f"{(cum.iloc[-1]-1)*100:.2f}%",
                "正收益概率": f"{(ret > 0).sum()/len(ret)*100:.1f}%",
                "最长回撤修复时长": f"{final_days}天",
                "回撤当前状态": status
            })
        st.table(pd.DataFrame(analysis_data))

        # 相关性
        st.subheader("📊 资产相关性矩阵")
        st.dataframe(period_returns.corr().style.background_gradient(cmap='RdYlGn').format("{:.2f}"))
else:
    st.info("👋 请上传包含净值数据及指数(列名含300/500)的Excel。")
