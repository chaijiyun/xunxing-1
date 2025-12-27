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
                <h2 style='color: #1e3a8a;'>🏛️ 寻星投研系统 1.3</h2>
                <p style='color: #666;'>精度修正与基准对比完整版</p>
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
# 2. 核心金融计算函数
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
st.set_page_config(layout="wide", page_title="寻星配置分析系统1.3")

if st.sidebar.button("🔒 退出系统"):
    st.session_state["authenticated"] = False
    st.rerun()

st.title("🏛️ 寻星配置分析系统 1.3")
st.caption("回撤修复精度优化版 | 业绩基准对比 | 2025版")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("1. 上传净值数据 (Excel)", type=["xlsx"])

if uploaded_file:
    # 预处理：强制平滑和精度
    raw_df = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
    raw_df = raw_df.sort_index().ffill()
    # 识别基准和产品
    all_cols = raw_df.columns.tolist()
    benchmarks = [c for c in all_cols if any(x in str(c) for x in ["300", "500"])]
    funds = [c for c in all_cols if c not in benchmarks]

    # 侧边栏设置
    st.sidebar.subheader("2. 区间与频率")
    min_date, max_date = raw_df.index.min().to_pydatetime(), raw_df.index.max().to_pydatetime()
    start_date = st.sidebar.date_input("开始日期", value=min_date)
    end_date = st.sidebar.date_input("结束日期", value=max_date)
    
    freq_option = st.sidebar.selectbox("横轴日期频率", ["月度展示", "季度展示"])
    dtick_val = "M1" if freq_option == "月度展示" else "M3"

    st.sidebar.subheader("3. 目标配置比例")
    target_weights = {f: st.sidebar.slider(f, 0.0, 1.0, 1.0/len(funds)) for f in funds}

    # 核心计算切片
    mask = (raw_df.index >= pd.Timestamp(start_date)) & (raw_df.index <= pd.Timestamp(end_date))
    period_df = raw_df.loc[mask]
    returns_df = period_df.pct_change().fillna(0)

    # 权重归一化
    w_sum = sum(target_weights.values()) or 1
    w_series = pd.Series({k: v/w_sum for k, v in target_weights.items()})

    # 计算组合
    fof_ret = (returns_df[funds] * w_series).sum(axis=1)
    fof_cum = (1 + fof_ret).cumprod()

    if not fof_cum.empty:
        # --- 1. 指标看板 ---
        c1, c2, c3, c4 = st.columns(4)
        total_ret = fof_cum.iloc[-1] - 1
        mdd = calculate_max_drawdown(fof_ret)
        days_span = max((fof_cum.index[-1] - fof_cum.index[0]).days, 1)
        ann_ret = (1 + total_ret)**(365.25/days_span) - 1
        vol = fof_ret.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / (vol if vol != 0 else 1)

        c1.metric("组合累计收益", f"{total_ret*100:.2f}%")
        c2.metric("组合年化收益", f"{ann_ret*100:.2f}%")
        c3.metric("组合最大回撤", f"{mdd*100:.2f}%")
        c4.metric("组合夏普比率", f"{sharpe:.2f}")

        # --- 2. 净值对比图 ---
        fig = go.Figure()
        # 绘制指数基准
        for b in benchmarks:
            b_nav = (period_df[b] / period_df[b].iloc[0])
            fig.add_trace(go.Scatter(x=b_nav.index, y=b_nav, name=f'基准-{b}', line=dict(dash='dash', width=2)))
        # 绘制组合
        fig.add_trace(go.Scatter(x=fof_cum.index, y=fof_cum, name='寻星组合', line=dict(color='red', width=4)))
        
        fig.update_layout(title="组合 vs 业绩基准 收益曲线", xaxis=dict(dtick=dtick_val, tickformat="%Y-%m"), hovermode="x unified", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # --- 3. 深度分析表（回撤修复精度修正） ---
        st.markdown("### 🔍 深度指标排查 (回撤修复状态穿透)")
        analysis_data = []
        for item in (funds + benchmarks):
            item_nav = (period_df[item] / period_df[item].iloc[0]).round(5) # 强行精度对齐
            item_ret = item_nav.pct_change().fillna(0)
            
            # 计算回撤
            pk = item_nav.cummax()
            dd = (item_nav - pk) / pk
            
            # 统计逻辑：容差 0.0005 (0.05%)
            max_his, ongoing, start_dt = 0, 0, None
            for dt, val in dd.items():
                if val < -0.0005: # 入水
                    if start_dt is None: start_dt = dt
                else: # 出水 (只要回到 99.95% 就算修复)
                    if start_dt is not None:
                        max_his = max(max_his, (dt - start_dt).days)
                        start_dt = None
            
            if start_dt is not None: # 截止到最后数据还没回正
                ongoing = (dd.index[-1] - start_dt).days

            analysis_data.append({
                "名称": item,
                "性质": "底层产品" if item in funds else "业绩基准",
                "累计收益": f"{(item_nav.iloc[-1]-1)*100:.2f}%",
                "历史最长修复": f"{max_his} 天",
                "当前回撤持续": f"{ongoing} 天" if ongoing > 0 else "✅ 已创新高",
                "状态": "⚠️ 正在经历回撤" if ongoing > 0 else "✅ 表现稳健"
            })
        st.table(pd.DataFrame(analysis_data))

        # --- 4. 相关性矩阵 ---
        st.subheader("📊 底层资产相关性 (1.0 代表完全相关)")
        st.dataframe(returns_df[funds].corr().style.background_gradient(cmap='RdYlGn').format("{:.2f}"))

else:
    st.info("👋 请上传包含“沪深300”或“中证500”列的 Excel 文件。")
