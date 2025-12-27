import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 版本标志与身份验证 (确保看到 2.0)
# ==========================================
VERSION = "2.0-FINAL" 

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.set_page_config(page_title="寻星投研验证", page_icon="🏛️")
    st.markdown(f"<h2 style='text-align:center; margin-top:50px;'>🏛️ 寻星配置分析系统 {VERSION}</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pwd = st.text_input("内部授权码", type="password", placeholder="请输入授权码...")
        if st.button("登录系统", use_container_width=True):
            if pwd == "281699":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("授权码错误")
    st.stop()

# ==========================================
# 2. 核心算法：解决 294 天死锁的精密计算
# ==========================================
def get_precision_stats(series):
    """
    针对宁泉、宽远等连续净值产品设计的精密回撤计算
    """
    # 强制数值化并剔除空值
    s = pd.to_numeric(series, errors='coerce').dropna().sort_index()
    if s.empty: return None
    
    # 计算滚动最高点
    roll_max = s.cummax()
    # 只要当前值比最高点差值小于 0.0001 (万分之一)，就视为回正
    is_recovered = s >= (roll_max - 0.0001)
    
    max_rec_days = 0
    current_ongoing = 0
    last_peak_dt = s.index[0]
    in_pit = False
    
    for dt, recovered in is_recovered.items():
        if recovered:
            if in_pit:
                # 计算从掉坑前的高点到回正当天的自然日天数
                duration = (dt - last_peak_dt).days
                max_rec_days = max(max_rec_days, duration)
                in_pit = False
            last_peak_dt = dt # 更新最高点时间锚点
        else:
            in_pit = True
            
    if in_pit:
        current_ongoing = (s.index[-1] - last_peak_dt).days
        
    return {
        "max_rec": max_rec_days,
        "curr_ong": current_ongoing,
        "peak_v": s.max(),
        "last_v": s.iloc[-1]
    }

# ==========================================
# 3. 主界面布局
# ==========================================
st.set_page_config(layout="wide", page_title=f"寻星系统 {VERSION}")
st.title(f"🏛️ 寻星配置分析系统 {VERSION}")
st.caption("针对连续净值产品优化 | 自动精度对齐 | 2025 官方版")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("1. 上传净值 Excel", type=["xlsx"])

if uploaded_file:
    # A. 数据预处理
    df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index()
    benchmarks = [c for c in df_raw.columns if any(x in str(c) for x in ["300", "500"])]
    funds = [c for c in df_raw.columns if c not in benchmarks]

    # B. 参数设置
    st.sidebar.subheader("2. 策略参数")
    start_dt = st.sidebar.date_input("分析起点", value=df_raw.index.min())
    end_dt = st.sidebar.date_input("分析终点", value=df_raw.index.max())
    weights = {f: st.sidebar.slider(f, 0.0, 1.0, 1.0/len(funds)) for f in funds}

    # C. 深度画像分析（解决 294 天问题的核心展示）
    st.subheader("🔍 产品深度回撤画像")
    analysis_results = []
    
    # 数据切片供收益计算
    mask = (df_raw.index >= pd.Timestamp(start_dt)) & (df_raw.index <= pd.Timestamp(end_dt))
    df_period = df_raw.loc[mask].ffill()

    for item in (funds + benchmarks):
        res = get_precision_stats(df_raw[item])
        if not res: continue
        
        # 区间收益计算
        p_sub = df_period[item].dropna()
        p_ret = (p_sub.iloc[-1] / p_sub.iloc[0] - 1) if len(p_sub) > 1 else 0

        analysis_results.append({
            "名称": item,
            "历史最长修复": f"{res['max_rec']} 天",
            "当前回撤持续": f"{res['curr_ong']} 天" if res['curr_ong'] > 0 else "✅ 已创新高",
            "历史最高": f"{res['peak_v']:.4f}",
            "最新净值": f"{res['last_v']:.4f}",
            "区间累计收益": f"{p_ret*100:.2f}%"
        })
    st.table(pd.DataFrame(analysis_results))

    # D. 组合表现
    w_sum = sum(weights.values()) or 1
    w_vec = np.array([weights[f]/w_sum for f in funds])
    returns = df_period[funds].pct_change().fillna(0)
    fof_ret = returns.dot(w_vec)
    fof_cum = (1 + fof_ret).cumprod()

    # E. 净值曲线图
    fig = go.Figure()
    for b in benchmarks:
        b_nav = df_period[b].dropna()
        fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav/b_nav.iloc[0], name=f'基准-{b}', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=fof_cum.index, y=fof_cum, name='寻星组合', line=dict(color='red', width=4)))
    st.plotly_chart(fig, use_container_width=True)

    # F. 相关性
    st.subheader("📊 资产相关性矩阵")
    st.dataframe(returns.corr().style.background_gradient(cmap='RdYlGn').format("{:.2f}"))

else:
    st.info("👋 请上传 Excel 净值表。系统将自动处理宁泉、宽远等产品的连续净值逻辑。")
