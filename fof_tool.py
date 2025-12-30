import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 0. 登录验证
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.markdown("<h1 style='text-align: center; color: #1E40AF;'>寻星配置分析系统</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                pwd_input = st.text_input("系统访问密码", type="password")
                if st.form_submit_button("立即登录", use_container_width=True):
                    if pwd_input == "281699":
                        st.session_state["password_correct"] = True
                        st.rerun()
                    else:
                        st.error("密码错误")
        return False
    return True

if check_password():
    # ==========================================
    # 1. 核心计算函数 (含索提诺比率)
    # ==========================================
    def calculate_metrics(nav, bench_nav=None):
        nav = nav.dropna()
        if len(nav) < 2: return {}
        total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
        days = (nav.index[-1] - nav.index[0]).days
        ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / max(days, 1)) - 1
        returns = nav.pct_change().fillna(0)
        vol = returns.std() * np.sqrt(252)
        mdd = (nav / nav.cummax() - 1).min()
        
        # 风险指标计算
        rf = 0.02
        sharpe = (ann_ret - rf) / vol if vol > 0 else 0
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (ann_ret - rf) / downside_std if downside_std > 0 else 0
        calmar = ann_ret / abs(mdd) if abs(mdd) > 0 else 0
        
        # 修复天数计算
        cummax = nav.cummax()
        drawdown = (nav / cummax) - 1
        mdd_recovery = "尚未修复"
        if mdd < 0:
            mdd_date = drawdown.idxmin()
            peak_val = nav.loc[:mdd_date].max()
            recovery_data = nav.loc[mdd_date:]
            recovered = recovery_data[recovery_data >= peak_val]
            if not recovered.empty:
                mdd_recovery = f"{(recovered.index[0] - mdd_date).days}天"

        metrics = {
            "总收益率": total_ret, "年化收益": ann_ret, "最大回撤": mdd, 
            "夏普比率": sharpe, "索提诺比率": sortino, "卡玛比率": calmar, 
            "年化波动": vol, "回撤修复": mdd_recovery
        }
        
        # 捕获率计算
        if bench_nav is not None:
            b_rets = bench_nav.reindex(nav.index).pct_change().fillna(0)
            up_mask, down_mask = b_rets > 0, b_rets < 0
            metrics["上行捕获"] = returns[up_mask].mean() / b_rets[up_mask].mean() if up_mask.any() else 0
            metrics["下行捕获"] = returns[down_mask].mean() / b_rets[down_mask].mean() if down_mask.any() else 0
        return metrics

    # ==========================================
    # 2. 侧边栏配置
    # ==========================================
    st.set_page_config(layout="wide", page_title="寻星配置分析系统")
    st.sidebar.title("🏛️ 寻星配置系统")
    uploaded_file = st.sidebar.file_uploader("上传寻星配置数据库", type=["xlsx"])

    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
        all_cols = list(df_raw.columns)
        
        sel_bench = st.sidebar.selectbox("选择业绩基准", all_cols, index=0)
        sel_funds = st.sidebar.multiselect("挑选寻星配置组合成分", [c for c in all_cols if c != sel_bench])
        
        weights = {}
        if sel_funds:
            for f in sel_funds:
                weights[f] = st.sidebar.number_input(f"{f} 权重", 0.0, 1.0, 1.0/len(sel_funds))
        
        start_d = st.sidebar.date_input("起始日期", df_raw.index.min())
        end_d = st.sidebar.date_input("结束日期", df_raw.index.max())
        df = df_raw.loc[start_d:end_d].copy()

        # 计算寻星配置组合净值
        star_nav = None
        if sel_funds and not df.empty:
            w_sum = sum(weights.values())
            norm_w = {k: v/w_sum for k, v in weights.items()}
            star_rets = (df[sel_funds].pct_change().fillna(0) * pd.Series(norm_w)).sum(axis=1)
            star_nav = (1 + star_rets).cumprod()
            bench_norm = df[sel_bench] / df[sel_bench].iloc[0]

        # ==========================================
        # 3. 页面主体
        # ==========================================
        tabs = st.tabs(["🚀 组合看板", "🔍 归因与性格", "⚔️ 配置池对比"])

        with tabs[0]: # 组合看板
            if star_nav is not None:
                m = calculate_metrics(star_nav)
                cols = st.columns(4)
                cols[0].metric("总收益率", f"{m['总收益率']:.2%}")
                cols[1].metric("年化收益", f"{m['年化收益']:.2%}")
                cols[2].metric("最大回撤", f"{m['最大回撤']:.2%}")
                cols[3].metric("索提诺比率", f"{m['索提诺比率']:.2f}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="寻星配置组合", line=dict(color='red', width=3)))
                fig.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm, name=sel_bench, line=dict(color='gray', dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("请在左侧选择产品。")

        with tabs[1]: # 归因与性格
            if sel_funds:
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(px.pie(names=list(weights.keys()), values=list(weights.values()), title="资金权重分配"), use_container_width=True)
                
                # 产品性格分布 (上下行捕获)
                char_data = []
                for f in sel_funds:
                    fm = calculate_metrics(df[f], df[sel_bench])
                    char_data.append({"产品": f, "上行捕获": fm['上行捕获'], "下行捕获": fm['下行捕获'], "年化收益": fm['年化收益']})
                df_char = pd.DataFrame(char_data)
                fig_char = px.scatter(df_char, x="下行捕获", y="上行捕获", text="产品", size=np.abs(df_char["年化收益"])*100, color="年化收益", title="产品性格分布图")
                fig_char.add_vline(x=1, line_dash="dot"); fig_char.add_hline(y=1, line_dash="dot")
                st.plotly_chart(fig_char, use_container_width=True)
                
                st.markdown("#### 相关性矩阵")
                st.plotly_chart(px.imshow(df[sel_funds].pct_change().corr(), text_auto=".2f"), use_container_width=True)

        with tabs[2]: # 配置池对比
            compare_pool = st.multiselect("添加产品进行对比", all_cols, default=sel_funds)
            if compare_pool:
                # 净值走势图 (Tab 3 补回)
                fig_comp = go.Figure()
                res_list = []
                for p in compare_pool:
                    p_nav = df[p] / df[p].iloc[0]
                    fig_comp.add_trace(go.Scatter(x=p_nav.index, y=p_nav, name=p))
                    
                    m = calculate_metrics(df[p])
                    res_list.append({
                        "产品名称": p, "总收益率": f"{m['总收益率']:.2%}", "年化收益": f"{m['年化收益']:.2%}",
                        "最大回撤": f"{m['最大回撤']:.2%}", "索提诺": round(m['索提诺比率'], 2),
                        "夏普": round(m['夏普比率'], 2), "波动率": f"{m['年化波动']:.2%}", "修复": m['回撤修复']
                    })
                st.plotly_chart(fig_comp, use_container_width=True)
                st.dataframe(pd.DataFrame(res_list).set_index("产品名称"), use_container_width=True)
    else:
        st.info("请上传 Excel 数据库文件。")
