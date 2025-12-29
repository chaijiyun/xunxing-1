import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. 核心指标计算引擎 (寻星专用版)
# ==========================================
def get_max_drawdown_recovery_days(nav_series):
    """计算最大回撤修复天数：从谷底到回升至前高的时间"""
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
    """计算最长创新高间隔天数"""
    if nav_series.empty: return 0
    cummax = nav_series.cummax()
    high_dates = nav_series[nav_series == cummax].index.to_series()
    if len(high_dates) < 2: return (nav_series.index[-1] - nav_series.index[0]).days
    diffs = high_dates.diff().dt.days
    return int(diffs.max()) if not pd.isna(diffs.max()) else 0

def calculate_metrics(nav):
    """计算全维度绩效指标"""
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
# 2. UI 界面与侧边栏控制
# ==========================================
st.set_page_config(layout="wide", page_title="寻星配置分析系统 v2.16", page_icon="🏛️")

st.sidebar.title("🏛️ 寻星控制台")
uploaded_file = st.sidebar.file_uploader("📂 加载寻星配置底座 (xlsx)", type=["xlsx"])

if uploaded_file:
    # 加载数据
    df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    all_cols = df_raw.columns.tolist()
    
    # 业绩基准选择 (默认勾选沪深300)
    st.sidebar.markdown("---")
    default_bench = '沪深300' if '沪深300' in all_cols else all_cols[0]
    sel_bench = st.sidebar.selectbox("业绩基准", all_cols, index=all_cols.index(default_bench))
    
    # 挑选组合成分 (默认不勾选)
    fund_pool = [c for c in all_cols if c != sel_bench]
    st.sidebar.subheader("🛠️ 构建寻星配置组合")
    sel_funds = st.sidebar.multiselect("挑选组合成分", fund_pool, default=[])
    
    # 动态分配权重 (默认平均分配)
    weights = {}
    star_nav = None
    if sel_funds:
        st.sidebar.markdown("#### ⚖️ 比例分配")
        avg_w = 1.0 / len(sel_funds)
        for f in sel_funds:
            weights[f] = st.sidebar.number_input(f"{f}", 0.0, 1.0, avg_w, step=0.05)
        
        # 预计算组合净值
        df_port = df_raw[sel_funds].dropna()
        if not df_port.empty:
            port_rets = df_port.pct_change().fillna(0)
            norm_w = pd.Series(weights) / sum(weights.values())
            star_rets = (port_rets * norm_w).sum(axis=1)
            star_nav = (1 + star_rets).cumprod()
            star_nav.name = "寻星配置组合"
            # 同步基准
            bench_sync = df_raw.loc[star_nav.index, sel_bench]
            bench_norm = bench_sync / bench_sync.iloc[0]

    # ==========================================
    # 3. 功能标签页布局
    # ==========================================
    tabs = st.tabs(["🚀 寻星配置组合全景图", "🔍 寻星配置底层产品分析", "🧩 权重与归因", "⚔️ 配置池产品分析"])

    # --- Tab 1: 寻星配置组合全景图 ---
    with tabs[0]:
        if star_nav is not None:
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
            fig_main.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm, name=f"基准: {sel_bench}", line=dict(color='#9CA3AF', dash='dot')))
            fig_main.update_layout(template="plotly_white", hovermode="x unified", height=550, title="资产配置组合模拟运行净值 (基于左侧比例配置)")
            st.plotly_chart(fig_main, use_container_width=True)
            
            with st.expander("更多组合风险指标"):
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("索提诺比率", f"{m['索提诺比率']:.2f}")
                cc2.metric("年化波动率", f"{m['年化波动率']:.2%}")
                cc3.metric("最长新高间隔", m['最长新高间隔'])
        else:
            st.info("👈 请先在左侧侧边栏【挑选组合成分】并分配比例，以生成寻星配置组合全景图。")

    # --- Tab 2: 寻星配置底层产品分析 ---
    with tabs[1]:
        if sel_funds:
            st.subheader("🔬 组合成分深度拆解")
            df_sub_norm = df_raw[sel_funds].dropna()
            if not df_sub_norm.empty:
                df_sub_norm = df_sub_norm.div(df_sub_norm.iloc[0])
                fig_sub = px.line(df_sub_norm, title="选中成分产品走势 (同期起点归一)")
                st.plotly_chart(fig_sub, use_container_width=True)
                
                corr = df_raw[sel_funds].dropna().pct_change().corr()
                st.write("##### 成分相关性热力图")
                st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)
        else:
            st.info("👈 请先在左侧勾选成分产品以进行分析。")

    # --- Tab 3: 权重与归因 ---
    with tabs[2]:
        if sel_funds:
            st.subheader("🧩 寻星配置架构")
            cw1, cw2 = st.columns(2)
            with cw1:
                fig_pie = px.pie(names=list(weights.keys()), values=list(weights.values()), hole=0.4, title="当前组合权重分布")
                st.plotly_chart(fig_pie, use_container_width=True)
            with cw2:
                st.write("##### 权重明细表")
                st.table(pd.DataFrame.from_dict(weights, orient='index', columns=['所占比例']).style.format("{:.2%}"))
        else:
            st.info("👈 请先在左侧勾选成分产品。")

    # --- Tab 4: 配置池产品分析 (独立模块) ---
    with tabs[3]:
        st.subheader("⚔️ 配置池单品/多品对比")
        st.markdown("💡 此模块用于在全库内自由勾选产品进行分析，**独立于左侧组合设置**。")
        
        # 独立的多选框，默认不选或选第一个
        compare_pool = st.multiselect("请选择要分析的产品 (支持单选/多选)", all_cols, default=[])
        
        if compare_pool:
            df_comp_raw = df_raw[compare_pool].dropna()
            if not df_comp_raw.empty:
                # 归一化走势对比
                df_comp_norm = df_comp_raw.div(df_comp_raw.iloc[0])
                fig_c = px.line(df_comp_norm, title=f"配置池对比走势 (起点: {df_comp_raw.index[0].date()})")
                fig_c.update_layout(yaxis_title="归一化净值 (起点=1.0)", template="plotly_white", hovermode="x unified", height=600)
                st.plotly_chart(fig_c, use_container_width=True)
                
                # 核心绩效PK表
                st.markdown("##### 🏆 核心素质PK战报")
                res_list = []
                for col in compare_pool:
                    m_p = calculate_metrics(df_comp_raw[col])
                    m_p['产品名称'] = col
                    res_list.append(m_p)
                
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
                st.warning("⚠️ 选中的产品没有共同的存续时间段，无法同台对比。")
        else:
            st.info("🔎 请在上方搜索框勾选配置池中的产品进行展示或比较。")

else:
    st.info("👋 请在左侧上传‘寻星配置底座’Excel文件以启动系统。")
