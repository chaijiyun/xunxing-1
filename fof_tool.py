import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. 核心指标计算引擎
# ==========================================
def get_max_drawdown_recovery_days(nav_series):
    if nav_series.empty or len(nav_series) < 2: return 0, "数据不足"
    cummax = nav_series.cummax()
    drawdown = (nav_series / cummax) - 1
    if drawdown.min() == 0: return 0, "无回撤"
    mdd_date = drawdown.idxmin()
    peak_val = cummax.loc[mdd_date]
    post_mdd_data = nav_series.loc[mdd_date:]
    post_mdd_data = post_mdd_data[post_mdd_data.index > mdd_date]
    recovery_mask = post_mdd_data >= peak_val
    if recovery_mask.any():
        recover_date = recovery_mask.idxmax()
        days = (recover_date - mdd_date).days
        return days, f"{days}天"
    else:
        return 9999, "尚未修复"

def get_longest_new_high_interval(nav_series):
    if nav_series.empty: return 0
    cummax = nav_series.cummax()
    high_dates = nav_series[nav_series == cummax].index.to_series()
    if len(high_dates) < 2: return (nav_series.index[-1] - nav_series.index[0]).days
    diffs = high_dates.diff().dt.days
    return int(diffs.max()) if not pd.isna(diffs.max()) else 0

def calculate_metrics(nav):
    nav = nav.dropna()
    if len(nav) < 2: return {}
    total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
    days = (nav.index[-1] - nav.index[0]).days
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / max(days, 1)) - 1
    returns = nav.pct_change().fillna(0)
    cummax = nav.cummax()
    mdd = (nav / cummax - 1).min()
    vol = returns.std() * np.sqrt(252)
    rf = 0.02
    sharpe = (ann_ret - rf) / vol if vol > 0 else 0
    calmar = ann_ret / abs(mdd) if abs(mdd) > 0 else 0
    downside_vol = returns[returns < 0].std() * np.sqrt(252)
    sortino = (ann_ret - rf) / downside_vol if downside_vol > 0 else 0
    rep_v, rep_s = get_max_drawdown_recovery_days(nav)
    high_gap = get_longest_new_high_interval(nav)
    return {
        "总收益率": total_ret, "年化收益": ann_ret, "最大回撤": mdd, 
        "夏普比率": sharpe, "卡玛比率": calmar, "年化波动率": vol, 
        "索提诺比率": sortino, "回撤修复天数": rep_s, "最长新高间隔": f"{high_gap}天"
    }

# ==========================================
# 2. UI 布局与侧边栏
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统 v2.14", page_icon="🏛️")

st.sidebar.title("🏛️ 寻星控制台")
uploaded_file = st.sidebar.file_uploader("📂 加载清洗后的数据底座 (xlsx)", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    all_cols = df_raw.columns.tolist()
    
    # 侧边栏：基准与组合构建
    st.sidebar.markdown("---")
    default_bench = '沪深300' if '沪深300' in all_cols else all_cols[0]
    sel_bench = st.sidebar.selectbox("业绩基准", all_cols, index=all_cols.index(default_bench))
    
    fund_pool = [c for c in all_cols if c != sel_bench]
    st.sidebar.subheader("🛠️ 构建寻星配置组合")
    # 这里选中的产品只影响 Tab 1, 2, 3
    sel_funds = st.sidebar.multiselect("挑选组合成分", fund_pool, default=fund_pool[:min(4, len(fund_pool))])
    
    if not sel_funds:
        st.warning("请在左侧侧边栏选择成分产品。")
        st.stop()

    weights = {}
    st.sidebar.markdown("#### ⚖️ 比例分配")
    for f in sel_funds:
        weights[f] = st.sidebar.number_input(f"{f}", 0.0, 1.0, 1.0/len(sel_funds), step=0.05)
    
    # --- 【Tab 1 专属计算：虚拟组合合成】 ---
    df_portfolio_common = df_raw[sel_funds].dropna()
    portfolio_rets = df_portfolio_common.pct_change().fillna(0)
    norm_weights = pd.Series(weights) / sum(weights.values())
    star_rets = (portfolio_rets * norm_weights).sum(axis=1)
    star_nav = (1 + star_rets).cumprod()
    star_nav.name = "寻星配置组合"
    # 基准对齐
    bench_nav_sync = df_raw.loc[star_nav.index, sel_bench]
    bench_nav_norm = bench_nav_sync / bench_nav_sync.iloc[0]

    # ==========================================
    # 3. 功能标签页 (更名后)
    # ==========================================
    tabs = st.tabs(["🚀 寻星配置组合全景图", "🔍 寻星配置底层产品分析", "🧩 权重与归因", "⚔️ 配置池产品分析"])

    # --- Tab 1: 寻星配置组合全景图 ---
    with tabs[0]:
        st.subheader("📊 寻星配置组合整体表现")
        m = calculate_metrics(star_nav)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("总收益率", f"{m['总收益率']:.2%}")
        c2.metric("年化收益", f"{m['年化收益']:.2%}")
        c3.metric("最大回撤", f"{m['最大回撤']:.2%}")
        c4.metric("夏普比率", f"{m['夏普比率']:.2f}")
        c5.metric("卡玛比率", f"{m['卡玛比率']:.2f}")
        c6.metric("修复天数", m['回撤修复天数'])
        
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="寻星配置组合", line=dict(color='#1E40AF', width=3.5)))
        fig_main.add_trace(go.Scatter(x=bench_nav_norm.index, y=bench_nav_norm, name=f"基准: {sel_bench}", line=dict(color='#9CA3AF', dash='dot')))
        fig_main.update_layout(template="plotly_white", hovermode="x unified", height=550, title="资产配置组合模拟运行净值 (基于左侧比例配置)")
        st.plotly_chart(fig_main, use_container_width=True)

    # --- Tab 2: 寻星配置底层产品分析 ---
    with tabs[1]:
        st.subheader("🔬 组合成分深度拆解")
        df_comp_norm = df_portfolio_common.div(df_portfolio_common.iloc[0])
        fig_sub = px.line(df_comp_norm, title="选中成分产品走势 (同期起点归一)")
        st.plotly_chart(fig_sub, use_container_width=True)
        
        st.markdown("##### 成分相关性热力图")
        corr = df_portfolio_common.pct_change().corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)

    # --- Tab 3: 权重与归因 ---
    with tabs[2]:
        st.subheader("🧩 组合架构逻辑")
        cw1, cw2 = st.columns(2)
        with cw1:
            fig_p = px.pie(names=list(weights.keys()), values=list(weights.values()), hole=0.4, title="当前组合权重分布")
            st.plotly_chart(fig_p, use_container_width=True)
        with cw2:
            st.write("##### 权重明细")
            st.table(pd.DataFrame.from_dict(weights, orient='index', columns=['所占比例']))

    # --- Tab 4: 配置池产品分析 (独立模块) ---
    with tabs[3]:
        st.subheader("⚔️ 配置池单品/多品对比")
        st.markdown("💡 此模块用于在全库内自由勾选产品进行素质分析，**不受左侧组合设置影响**。")
        
        # 页面内独立多选框
        compare_pool = st.multiselect("请选择要分析的产品 (支持全库搜索单只或多只)", all_cols, default=fund_pool[0] if fund_pool else None)
        
        if compare_pool:
            # 自动提取选定产品的共同交集区间进行公平PK
            df_compare = df_raw[compare_pool].dropna()
            
            if not df_compare.empty:
                # 1. 独立归一化净值图
                df_c_norm = df_compare.div(df_compare.iloc[0])
                fig_c = px.line(df_c_norm, title=f"所选产品对比走势 (起点: {df_compare.index[0].date()})")
                fig_c.update_layout(yaxis_title="归一化净值 (起点=1.0)", template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig_c, use_container_width=True)
                
                # 2. 全方位绩效战报
                st.markdown("##### 🏆 核心素质PK表")
                res_list = []
                for col in compare_pool:
                    m_p = calculate_metrics(df_compare[col])
                    m_p['产品名称'] = col
                    res_list.append(m_p)
                
                # 格式化展示
                res_df = pd.DataFrame(res_list).set_index('产品名称')
                st.dataframe(
                    res_df.style.format({
                        "总收益率": "{:.2%}", "年化收益": "{:.2%}", "最大回撤": "{:.2%}",
                        "夏普比率": "{:.2f}", "卡玛比率": "{:.2f}", "索提诺比率": "{:.2f}", 
                        "年化波动率": "{:.2%}"
                    }).background_gradient(cmap='RdYlGn', subset=['夏普比率', '卡玛比率']),
                    use_container_width=True
                )
            else:
                st.error("⚠️ 选中的产品在时间上没有重叠区间，无法同台对比。")
        else:
            st.info("请上方搜索框中勾选配置池中的产品。")

else:
    st.info("👋 请在左侧上传‘寻星配置底座’Excel文件以启动系统。")
