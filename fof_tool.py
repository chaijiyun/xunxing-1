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
                <h2 style='color: #1e3a8a;'>🏛️ 寻星投研系统 1.6</h2>
                <p style='color: #666;'>空值修复与基准对比终极版</p>
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
# 2. 核心暴力算法：处理空值、精度与回撤
# ==========================================
def robust_recovery_calc(series):
    """
    专门对付含空值、浮点数误差的回撤修复计算函数
    """
    # 强制数值化 + 线性插值补全(处理中间空值) + 前后填充(处理两头空值)
    s = pd.to_numeric(series, errors='coerce').interpolate(method='linear').ffill().bfill()
    
    if s.empty or len(s) < 2:
        return 0, 0
    
    max_rec = 0
    ongoing = 0
    peak_val = -np.inf
    peak_dt = None
    in_dd = False
    
    # 逐行扫描判定
    for dt, val in s.items():
        # 容差判定：回升到最高点的 99.95% 视为修复，防止微小误差导致不回正
        if val >= peak_val or (peak_val > 0 and (val / peak_val) >= 0.9995):
            if in_dd and peak_dt is not None:
                duration = (dt - peak_dt).days
                max_rec = max(max_rec, duration)
                in_dd = False
            peak_val = val
            peak_dt = dt
        else:
            in_dd = True
            
    # 计算当前仍未修复的时长
    if in_dd and peak_dt is not None:
        ongoing = (s.index[-1] - peak_dt).days
        
    return max_rec, ongoing

# ==========================================
# 3. 主程序逻辑
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统1.6")

if st.sidebar.button("🔒 退出系统"):
    st.session_state["authenticated"] = False
    st.rerun()

st.title("🏛️ 寻星配置分析系统 1.6")
st.caption("空值自动修复 | 业绩基准对比 | 2025 投研版")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (含空值/指数)", type=["xlsx"])

if uploaded_file:
    # A. 原始数据加载与初步清洗
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index()
    
    # 自动识别资产类型
    all_cols = raw_df.columns.tolist()
    benchmarks = [c for c in all_cols if any(x in str(c) for x in ["300", "500"])]
    funds = [c for c in all_cols if c not in benchmarks]

    # B. 侧边栏交互设置
    st.sidebar.subheader("2. 区间设置")
    min_date, max_date = raw_df.index.min().to_pydatetime(), raw_df.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("开始日期", value=min_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date)
    
    freq_option = st.sidebar.selectbox("横轴频率", ["月度展示", "季度展示"])
    dtick_val = "M1" if freq_option == "月度展示" else "M3"

    st.sidebar.subheader("3. 目标配置比例")
    target_weights = {f: st.sidebar.slider(f, 0.0, 1.0, 1.0/len(funds)) for f in funds}

    # C. 数据平滑处理（用于计算组合收益）
    # 组合计算需要连续的净值流
    smooth_df = raw_df.interpolate(method='linear').ffill().bfill()
    mask = (smooth_df.index >= pd.Timestamp(start_date)) & (smooth_df.index <= pd.Timestamp(end_date))
    period_df = smooth_df.loc[mask]
    returns_df = period_df.pct_change().fillna(0)

    # D. 组合计算
    w_sum = sum(target_weights.values()) or 1
    w_series = pd.Series({k: v/w_sum for k, v in target_weights.items()})
    fof_ret = (returns_df[funds] * w_series).sum(axis=1)
    fof_cum = (1 + fof_ret).cumprod()

    if not fof_cum.empty:
        # --- 1. 核心看板 ---
        c1, c2, c3, c4 = st.columns(4)
        total_ret = fof_cum.iloc[-1] - 1
        # 计算组合的最大回撤
        peak = fof_cum.expanding().max()
        mdd_val = ((fof_cum / peak) - 1).min()
        
        days_span = max((fof_cum.index[-1] - fof_cum.index[0]).days, 1)
        ann_ret = (1 + total_ret)**(365.25/days_span) - 1
        vol = fof_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / (vol if vol != 0 else 1)

        c1.metric("组合累计收益", f"{total_ret*100:.2f}%")
        c2.metric("组合年化收益", f"{ann_ret*100:.2f}%")
        c3.metric("组合最大回撤", f"{mdd_val*100:.2f}%")
        c4.metric("组合夏普比率", f"{sharpe:.2f}")

        # --- 2. 净值曲线图 ---
        fig = go.Figure()
        # 基准线
        for b in benchmarks:
            b_nav = (period_df[b] / period_df[b].iloc[0])
            fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav, name=f'基准-{b}', line=dict(dash='dash', width=2)))
        # 组合线
        fig.add_trace(go.Scatter(x=fof_cum.index, y=fof_cum, name='寻星组合', line=dict(color='red', width=4)))
        
        fig.update_layout(title="组合 vs 业绩基准 (数据已自动平滑处理)", xaxis=dict(dtick=dtick_val, tickformat="%Y-%m"), hovermode="x unified", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # --- 3. 深度分析表 (解决空值导致的 294 天问题) ---
        st.markdown("### 🔍 深度画像排查 (空值穿透算法)")
        analysis_data = []
        for item in (funds + benchmarks):
            # 关键：传入原始全量数据 raw_df[item]，让算法能看到 3 月之后的高点
            max_h, ongoing = robust_recovery_calc(raw_df[item])
            
            # 区间表现
            sub_nav = period_df[item]
            p_ret = (sub_nav.iloc[-1] / sub_nav.iloc[0] - 1) if len(sub_nav) > 1 else 0

            analysis_data.append({
                "名称": item,
                "性质": "底层产品" if item in funds else "业绩基准",
                "所选区间收益": f"{p_ret*100:.2f}%",
                "历史最长修复": f"{max_h} 天",
                "当前回撤持续": f"{ongoing} 天" if ongoing > 0 else "✅ 已创新高",
                "回撤状态": "⚠️ 正在回撤" if ongoing > 0 else "✅ 表现稳健"
            })
        st.table(pd.DataFrame(analysis_data))

        # --- 4. 相关性矩阵 ---
        st.subheader("📊 资产相关性矩阵")
        st.dataframe(returns_df[funds].corr().style.background_gradient(cmap='RdYlGn').format("{:.2f}"))

else:
    st.info("👋 请上传包含净值数据及指数的 Excel 文件。系统将自动修复空值。")
