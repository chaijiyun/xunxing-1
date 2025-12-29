import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. 核心指标计算引擎 (升级版)
# ==========================================
def get_max_drawdown_recovery_days(nav_series):
    """
    计算最大回撤修复天数 (严格定义)
    逻辑：从最大回撤发生的【谷底日期】开始，寻找净值首次回升到【造成该回撤的前高】所需的天数。
    """
    if nav_series.empty or len(nav_series) < 2: return 0, "数据不足"
    
    # 1. 计算回撤序列
    cummax = nav_series.cummax()
    drawdown = (nav_series / cummax) - 1
    
    # 2. 找到最大回撤发生的日期 (谷底)
    if drawdown.min() == 0: return 0, "无回撤"
    mdd_date = drawdown.idxmin()
    
    # 3. 找到坑口水位 (谷底之前的最高点)
    # 注意：这里要找的是造成这次深坑的那个“前高”，即 mdd_date 对应的 cummax 值
    peak_val = cummax.loc[mdd_date]
    
    # 4. 寻找爬坑结束日 (从谷底之后开始找)
    post_mdd_data = nav_series.loc[mdd_date:]
    # 排除掉 mdd_date 当天
    post_mdd_data = post_mdd_data[post_mdd_data.index > mdd_date]
    
    recovery_mask = post_mdd_data >= peak_val
    
    if recovery_mask.any():
        recover_date = recovery_mask.idxmax() # 找到第一个爬出来的日期
        days = (recover_date - mdd_date).days
        return days, f"{days}天"
    else:
        return 9999, "尚未修复"

def get_longest_new_high_interval(nav_series):
    """计算最长创新高间隔天数"""
    if nav_series.empty: return 0
    cummax = nav_series.cummax()
    # 找出所有等于当前历史最高值的日期
    high_dates = nav_series[nav_series == cummax].index.to_series()
    
    if len(high_dates) < 2:
        return (nav_series.index[-1] - nav_series.index[0]).days # 如果一直没创新高
    
    # 计算日期之间的间隔
    diffs = high_dates.diff().dt.days
    return int(diffs.max()) if not pd.isna(diffs.max()) else 0

def calculate_metrics(nav):
    """全维度绩效指标计算"""
    nav = nav.dropna()
    if len(nav) < 2: return {}
    
    # 基础收益
    total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
    days = (nav.index[-1] - nav.index[0]).days
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / max(days, 1)) - 1
    
    # 风险指标
    returns = nav.pct_change().fillna(0)
    cummax = nav.cummax()
    mdd = (nav / cummax - 1).min()
    vol = returns.std() * np.sqrt(252)
    
    # 高级比率
    rf = 0.02 # 无风险利率假设 2%
    sharpe = (ann_ret - rf) / vol if vol > 0 else 0
    calmar = ann_ret / abs(mdd) if abs(mdd) > 0 else 0
    
    # 索提诺比率 (只考虑下行波动)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = (ann_ret - rf) / downside_vol if downside_vol > 0 else 0
    
    # 特殊指标
    repair_days_val, repair_days_str = get_max_drawdown_recovery_days(nav)
    high_gap = get_longest_new_high_interval(nav)
    
    return {
        "总收益率": total_ret, 
        "年化收益": ann_ret, 
        "最大回撤": mdd, 
        "夏普比率": sharpe, 
        "卡玛比率": calmar, 
        "年化波动率": vol, 
        "索提诺比率": sortino,
        "回撤修复天数": repair_days_str, 
        "最长新高间隔": f"{high_gap}天"
    }

# ==========================================
# 2. 系统 UI 布局
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统 v2.12", page_icon="🏛️")

# --- 侧边栏：数据与组合控制 ---
st.sidebar.title("🏛️ 寻星控制台")
uploaded_file = st.sidebar.file_uploader("📂 加载清洗后的数据底座 (xlsx)", type=["xlsx"])

if uploaded_file:
    # 1. 数据加载
    try:
        df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
        # 简单清洗列名
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        # 填充缺失值 (以防万一)
        df_raw = df_raw.sort_index().ffill()
    except Exception as e:
        st.error(f"数据读取失败: {e}")
        st.stop()
    
    all_cols = df_raw.columns.tolist()
    
    # 2. 基准与配置选择
    st.sidebar.markdown("---")
    default_bench = '沪深300' if '沪深300' in all_cols else all_cols[0]
    sel_bench = st.sidebar.selectbox("基准指数", all_cols, index=all_cols.index(default_bench))
    
    fund_pool = [c for c in all_cols if c != sel_bench]
    st.sidebar.subheader("🛠️ 构建寻星组合")
    sel_funds = st.sidebar.multiselect("选择持仓产品", fund_pool, default=fund_pool[:min(2, len(fund_pool))])
    
    if not sel_funds:
        st.warning("请至少选择一个产品。")
        st.stop()

    # 3. 权重设置
    weights = {}
    st.sidebar.markdown("#### ⚖️ 权重配置")
    for f in sel_funds:
        weights[f] = st.sidebar.number_input(f"{f}", 0.0, 1.0, 1.0/len(sel_funds), step=0.05, format="%.2f")
    
    # 4. 组合计算逻辑
    total_w = sum(weights.values())
    w_series = pd.Series(weights) / (total_w if total_w > 0 else 1) # 归一化权重
    
    # 计算组合净值 (基于日收益率加权)
    # 先将所有产品归一化到起点 1.0 方便计算收益率
    norm_df = df_raw.div(df_raw.iloc[0])
    daily_ret = norm_df[sel_funds].pct_change().fillna(0)
    
    star_port_ret = (daily_ret * w_series).sum(axis=1)
    star_nav = (1 + star_port_ret).cumprod()
    star_nav.name = "寻星配置组合"
    
    bench_nav = norm_df[sel_bench] # 也是归一化的

    # ==========================================
    # 3. 主界面 Tabs (精简架构)
    # ==========================================
    # 删除原 Tab2, 5, 6，仅保留核心与升级后的全量对比
    tabs = st.tabs(["🚀 组合驾驶舱", "🔍 底层产品透视", "🧩 权重与归因", "⚔️ 资产池全量比武(Pro)"])

    # --- Tab 1: 组合驾驶舱 (原Tab0) ---
    with tabs[0]:
        st.subheader("📊 寻星配置组合 · 核心表现")
        
        # 指标卡片
        m = calculate_metrics(star_nav)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("总收益率", f"{m['总收益率']:.2%}", help="成立以来累计收益")
        col2.metric("年化收益", f"{m['年化收益']:.2%}")
        col3.metric("最大回撤", f"{m['最大回撤']:.2%}", delta_color="inverse")
        col4.metric("夏普比率", f"{m['夏普比率']:.2f}")
        col5.metric("卡玛比率", f"{m['卡玛比率']:.2f}")
        col6.metric("修复天数", m['回撤修复天数'])
        
        # 走势图
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="寻星配置组合", line=dict(color='#2563EB', width=3)))
        fig.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav, name=sel_bench, line=dict(color='#9CA3AF', dash='dot')))
        fig.update_layout(title="组合 vs 基准 (净值走势)", template="plotly_white", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # 风险指标补充
        c1, c2, c3 = st.columns(3)
        c1.metric("索提诺比率", f"{m['索提诺比率']:.2f}", help="衡量下行风险调整后收益")
        c2.metric("年化波动率", f"{m['年化波动率']:.2%}")
        c3.metric("最长新高间隔", m['最长新高间隔'])

    # --- Tab 2: 底层产品透视 (原Tab1) ---
    with tabs[1]:
        st.subheader("🔬 组合成分深度分析")
        
        # 仅展示当前组合内的产品
        df_sel = df_raw[sel_funds].dropna()
        # 重新归一化绘图
        df_sel_norm = df_sel.div(df_sel.iloc[0])
        
        fig_sub = px.line(df_sel_norm, title="成分产品净值走势 (同期归一)")
        st.plotly_chart(fig_sub, use_container_width=True)
        
        st.markdown("#### 成分相关性热力图")
        corr = df_sel.pct_change().corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- Tab 3: 权重与归因 (原Tab3/4保留) ---
    with tabs[2]:
        st.subheader("⚖️ 配置逻辑透视")
        col_w1, col_w2 = st.columns([1, 2])
        
        with col_w1:
            st.markdown("##### 当前静态权重")
            w_df = pd.DataFrame.from_dict(weights, orient='index', columns=['权重'])
            st.dataframe(w_df.style.format("{:.2%}"), use_container_width=True)
            
            fig_pie = px.pie(w_df, values='权重', names=w_df.index, hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_w2:
            st.info("💡 此模块展示组合构建的初始比例。若需要查看动态贡献度归因，请确保数据包含完整的时间序列。")

    # --- Tab 4: 资产池全量比武 (Tab7 豪华升级版) ---
    with tabs[3]:
        st.subheader("⚔️ 全天候资产池 · 深度比武场")
        st.markdown("在此模块，您可以从整个数据库中任意挑选产品（单只或多只）进行同台竞技。系统将自动进行**时空对齐**与**净值归一**。")
        
        # 1. 交互选品区
        st.markdown("##### 1️⃣ 挑选参赛选手")
        # 默认选中当前组合的成分，方便对比
        compare_pool = st.multiselect(
            "请选择对比对象 (支持全库搜索)", 
            all_cols, 
            default=sel_funds
        )
        
        if compare_pool:
            # 2. 数据预处理引擎 (时空对齐)
            # 提取数据
            df_comp = df_raw[compare_pool].copy()
            # 找到这些产品的共同交集区间 (Common Range)
            df_comp_common = df_comp.dropna()
            
            if df_comp_common.empty:
                st.error("⚠️ 所选产品之间没有共同的存续时间段（交集为空），无法进行同维走势对比。请重新选择时间重叠的产品。")
            else:
                # 3. 净值走势对比 (归一化)
                st.markdown(f"##### 2️⃣ 净值走势擂台 (基准日: {df_comp_common.index[0].date()})")
                
                # 【核心逻辑】归一化：所有产品除以共同起点的第一天净值
                # 这样所有线条都从 1.0 开始，涨幅高低一目了然
                df_normalized = df_comp_common.div(df_comp_common.iloc[0])
                
                fig_comp = go.Figure()
                for col in df_normalized.columns:
                    # 如果是组合本身或基准，线条加粗
                    width = 3 if col == "寻星配置组合" or col == sel_bench else 1.5
                    fig_comp.add_trace(go.Scatter(
                        x=df_normalized.index, 
                        y=df_normalized[col], 
                        name=col,
                        line=dict(width=width)
                    ))
                
                fig_comp.update_layout(
                    title="区间收益率对比 (消除净值绝对值差异)",
                    yaxis_title="累计净值 (起点=1.0)",
                    template="plotly_white",
                    hovermode="x unified",
                    height=600
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # 4. 全维度指标PK (Table)
                st.markdown("##### 3️⃣ 核心指标 · 详细战报")
                
                metrics_data = []
                for col in compare_pool:
                    # 指标计算：建议基于该产品的【全历史数据】还是【当前对比区间】？
                    # 通常看产品能力看全历史，看对比看当前区间。
                    # 这里我们提供【当前对比区间】的指标，以保证公平性。
                    
                    # 使用 df_comp_common (交集数据) 进行指标计算，保证比赛时间公平
                    m_comp = calculate_metrics(df_comp_common[col])
                    
                    row = {
                        "产品": col,
                        "区间收益": m_comp['总收益率'],
                        "年化收益": m_comp['年化收益'],
                        "最大回撤": m_comp['最大回撤'],
                        "夏普比率": m_comp['夏普比率'],
                        "卡玛比率": m_comp['卡玛比率'],
                        "索提诺": m_comp['索提诺比率'],
                        "波动率": m_comp['年化波动率'],
                        "回撤修复": m_comp['回撤修复天数'],
                        "新高间隔": m_comp['最长新高间隔']
                    }
                    metrics_data.append(row)
                
                # 格式化展示
                res_df = pd.DataFrame(metrics_data).set_index("产品")
                
                # 对数值列进行高亮格式化
                st.dataframe(
                    res_df.style.format({
                        "区间收益": "{:.2%}", "年化收益": "{:.2%}", "最大回撤": "{:.2%}",
                        "夏普比率": "{:.2f}", "卡玛比率": "{:.2f}", "索提诺": "{:.2f}", 
                        "波动率": "{:.2%}"
                    }).background_gradient(subset=['区间收益', '年化收益', '夏普比率', '卡玛比率'], cmap='Reds')
                      .background_gradient(subset=['最大回撤', '波动率'], cmap='Greens', high=0.5), # 回撤越小越绿
                    use_container_width=True
                )
                
                st.caption(f"注：以上指标基于共同时间段 ({df_comp_common.index[0].date()} 至 {df_comp_common.index[-1].date()}) 计算，确保对比公平。")

else:
    st.info("👈 请在左侧上传‘寻星配置底座’Excel文件以启动分析。")
