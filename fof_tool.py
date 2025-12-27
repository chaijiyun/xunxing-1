import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 访问安全设置（新增登录逻辑）
# ==========================================
# 你可以在这里修改你的专属访问密码
ACCESS_PASSWORD = "xunxing2025" 

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login():
    """显示登录界面"""
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center; background-color: #f0f2f6; padding: 30px; border-radius: 10px; border: 1px solid #dcdfe6;'>
                <h2 style='color: #1e3a8a;'>🏛️ 寻星投研系统</h2>
                <p style='color: #666;'>内部专用版 | 请输入授权码访问</p>
            </div>
        """, unsafe_allow_html=True)
        pwd = st.text_input("", type="password", placeholder="输入密码后按回车...")
        if st.button("确认登录", use_container_width=True):
            if pwd == ACCESS_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("密码错误，请联系管理员。")
    st.stop()

# 检查是否已登录
if not st.session_state["authenticated"]:
    login()

# ==========================================
# 2. 自定义金融计算函数 (已为你保留)
# ==========================================
def calculate_sharpe(returns):
    if returns.std() == 0: return 0
    return (returns.mean() / returns.std()) * (252 ** 0.5)

def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative/peak) - 1
    return drawdown.min()

# ==========================================
# 3. 业务逻辑代码 (你原本的寻星系统)
# ==========================================
# --- 系统配置 ---
st.set_page_config(layout="wide", page_title="寻星配置分析系统1.0")

# --- 网页标题 ---
st.title("🏛️ 寻星配置分析系统 1.0")
st.caption("专业的私募FOF资产配置与深度产品画像工具 | 内部专用版")

# 添加退出登录按钮在右上角
if st.sidebar.button("🔒 退出登录"):
    st.session_state["authenticated"] = False
    st.rerun()

st.markdown("---")

# --- 侧边栏：数据与参数 ---
st.sidebar.header("🛠️ 系统控制面板")
uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    # 加载原始数据
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
    raw_df = raw_df.sort_index()
    
    # 提取收益率
    returns_df = raw_df.pct_change()

    # 2. 日期筛选
    st.sidebar.subheader("2. 回测区间设置")
    min_date = raw_df.index.min().to_pydatetime()
    max_date = raw_df.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("开始日期", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date, min_value=min_date, max_value=max_date)
    
    # 3. 权重设置
    funds = raw_df.columns.tolist()
    st.sidebar.subheader("3. 目标配置比例")
    target_weights = {}
    for fund in funds:
        target_weights[fund] = st.sidebar.slider(f"{fund}", 0.0, 1.0, 1.0/len(funds))
    
    # 4. 刻度频率选择
    st.sidebar.subheader("4. 图表显示设置")
    freq_option = st.sidebar.selectbox("横轴日期频率", ["月度展示", "季度展示"])
    dtick_val = "M1" if freq_option == "月度展示" else "M3"

    # --- 核心计算逻辑 ---
    mask = (returns_df.index >= pd.Timestamp(start_date)) & (returns_df.index <= pd.Timestamp(end_date))
    period_returns = returns_df.loc[mask]

    # 权重归一化
    total_tw = sum(target_weights.values()) if sum(target_weights.values()) != 0 else 1
    weights_series = pd.Series({k: v / total_tw for k, v in target_weights.items()})

    def calculate_dynamic_fof(daily_ret):
        available = daily_ret.notna() 
        if not available.any(): return 0.0
        curr_w = weights_series[available]
        if curr_w.sum() == 0: return 0.0
        actual_w = curr_w / curr_w.sum()
        return (daily_ret[available] * actual_w).sum()

    fof_daily_returns = period_returns.apply(calculate_dynamic_fof, axis=1)
    fof_cum_nav = (1 + fof_daily_returns).cumprod()

    # --- 1. 指标展示 ---
    if not fof_cum_nav.empty:
        c1, c2, c3, c4 = st.columns(4)
        days_span = (fof_cum_nav.index[-1] - fof_cum_nav.index[0]).days
        years_span = max(days_span / 365.25, 0.01)
        total_ret = fof_cum_nav[-1] - 1
        ann_ret = (1 + total_ret)**(1/years_span)-1
        
        # 使用你自定义的计算函数
        mdd = calculate_max_drawdown(fof_daily_returns)
        vol = fof_daily_returns.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / vol if vol != 0 else 0

        c1.metric("累计收益率", f"{total_ret*100:.2f}%")
        c2.metric("年化收益率", f"{ann_ret*100:.2f}%")
        c3.metric("最大回撤", f"{mdd*100:.2f}%")
        c4.metric("夏普比率", f"{sharpe:.2f}")

        # --- 2. 绘图逻辑 ---
        fig = go.Figure()

        for fund in funds:
            f_ret = period_returns[fund].dropna()
            if not f_ret.empty:
                f_cum = (1 + f_ret).cumprod()
                fig.add_trace(go.Scatter(x=f_cum.index, y=f_cum, name=f'底层-{fund}', 
                                         line=dict(dash='dot', width=1.2), opacity=0.4, yaxis='y1'))

        fig.add_trace(go.Scatter(x=fof_cum_nav.index, y=fof_cum_nav, name='寻星组合净值', 
                                 line=dict(color='red', width=3.5), yaxis='y1'))
        
        rolling_max = fof_cum_nav.cummax()
        dd_series = (fof_cum_nav - rolling_max) / rolling_max
        fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series, name='组合回撤(右轴)', 
                                 fill='tozeroy', line=dict(color='rgba(255,0,0,0.1)'), yaxis='y2'))

        fig.update_layout(
            title=f"寻星组合分析图 (当前频率: {freq_option})",
            xaxis=dict(
                title="日期", 
                tickformat
