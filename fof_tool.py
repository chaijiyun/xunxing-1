import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 0. 登录验证模块 (保持不变)
# ==========================================
def check_password():
    def password_entered():
        if st.session_state["password"] == "281699":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<br><br>", unsafe_allow_html=True) 
        st.markdown("<div style='text-align: center; color: #999;'>[ 此处预留公司 LOGO 位置 ]</div>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #1E40AF;'>寻星配置分析系统</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input(label="", type="password", on_change=password_entered, key="password")
            st.markdown("<p style='text-align: center; color: #666; font-size: 0.9em;'>请输入访问密码</p>", unsafe_allow_html=True)
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("<h1 style='text-align: center; color: #1E40AF;'>寻星配置分析系统</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input(label="", type="password", on_change=password_entered, key="password")
            st.markdown("<p style='text-align: center; color: #D32F2F; font-size: 0.9em;'>密码错误，请重新输入</p>", unsafe_allow_html=True)
        return False
    else:
        return True

if check_password():
    # ==========================================
    # 1. 核心指标计算引擎 (新增建议2/3算法)
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

    def calculate_metrics(nav, bench_nav=None):
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
        rep_v, rep_s = get_max_drawdown_recovery_days(nav)
        
        metrics = {
            "总收益率": total_ret, "年化收益": ann_ret, "最大回撤": mdd, 
            "夏普比率": sharpe, "卡玛比率": calmar, "年化波动率": vol, "回撤修复天数": rep_s
        }

        # 建议2逻辑：上下行捕获计算
        if bench_nav is not None:
            bench_rets = bench_nav.pct_change().fillna(0)
            up_mask = bench_rets > 0
            down_mask = bench_rets < 0
            up_cap = (returns[up_mask].mean() / bench_rets[up_mask].mean()) if up_mask.any() else 0
            down_cap = (returns[down_mask].mean() / bench_rets[down_mask].mean()) if down_mask.any() else 0
            metrics.update({"上行捕获": up_cap, "下行捕获": down_cap})
            
        return metrics

    # ==========================================
    # 2. UI 界面与侧边栏控制
    # ==========================================
    st.set_page_config(layout="wide", page_title="寻星配置分析系统", page_icon="🏛️")
    st.sidebar.title("🏛️ 寻星配置分析系统")
    uploaded_file = st.sidebar.file_uploader("📂 请上传产品数据库", type=["xlsx"])

    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        all_cols = df_raw.columns.tolist()
        
        st.sidebar.markdown("---")
        default_bench = '沪深300' if '沪深300' in all_cols else all_cols[0]
        sel_bench = st.sidebar.selectbox("业绩基准", all_cols, index=all_cols.index(default_bench))
        
        fund_pool = [c for c in all_cols if c != sel_bench]
        st.sidebar.subheader("🛠️ 构建寻星配置组合")
        sel_funds = st.sidebar.multiselect("挑选组合成分", fund_pool, default=[])
        
        weights = {}
        if sel_funds:
            st.sidebar.markdown("#### ⚖️ 比例分配")
            avg_w = 1.0 / len(sel_funds)
            for f in sel_funds:
                weights[f] = st.sidebar.number_input(f"{f}", 0.0, 1.0, avg_w, step=0.05)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📅 时间跨度选择")
        min_date, max_date = df_raw.index.min().to_pydatetime(), df_raw.index.max().to_pydatetime()
        start_date = st.sidebar.date_input("起始日期", min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("截止日期", max_date, min_value=min_date, max_value=max_date)
        
        df_db = df_raw.loc[start_date:end_date].copy()
        star_nav = None
        bench_sync_raw = df_db[sel_bench]

        if sel_funds and not df_db.empty:
            df_port = df_db[sel_funds].dropna()
            if not df_port.empty:
                port_rets = df_port.pct_change().fillna(0)
                norm_w = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) > 0 else 1)
                star_rets = (port_rets * norm_w).sum(axis=1)
                star_nav = (1 + star_rets).cumprod()
                bench_norm = bench_sync_raw.loc[star_nav.index] / bench_sync_raw.loc[star_nav.index][0]

        # ==========================================
        # 3. 功能标签页
        # ==========================================
        tabs = st.tabs(["🚀 寻星配置组合全景图", "🔍 寻星配置底层产品分析", "🧩 权重与归因", "⚔️ 配置池产品分析"])

        with tabs[0]:
            if star_nav is not None:
                st.subheader(f"📊 寻星配置组合全景图 ({start_date} 至 {end_date})")
                m = calculate_metrics(star_nav)
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("区间收益率", f"{m['总收益率']:.2%}")
                c2.metric("年化收益", f"{m['年化收益']:.2%}")
                c3.metric("区间最大回撤", f"{m['最大回撤']:.2%}")
                c4.metric("夏普比率", f"{m['夏普比率']:.2f}")
                c5.metric("卡玛比率", f"{m['卡玛比率']:.2f}")
                c6.metric("修复天数", m['回撤修复天数'])
                
                fig_main = go.Figure()
                fig_main.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="寻星配置组合", line=dict(color='#1E40AF', width=3.5)))
                fig_main.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm, name=f"基准: {sel_bench}", line=dict(color='#9CA3AF', dash='dot')))
                fig_main.update_layout(template="plotly_white", hovermode="x unified", height=500)
                st.plotly_chart(fig_main, use_container_width=True)
            else:
                st.info("👈 请在侧边栏挑选组合成分。")

        with tabs[1]:
            if sel_funds:
                st.subheader("🔍 寻星配置底层产品分析")
                df_sub = df_db[sel_funds].dropna()
                if not df_sub.empty:
                    df_sub_norm = df_sub.div(df_sub.iloc[0])
                    st.plotly_chart(px.line(df_sub_norm, title="选中成分走势"), use_container_width=True)
                    
                    st.markdown("---")
                    c_tab2_1, c_tab2_2 = st.columns([1, 1])
                    with c_tab2_1:
                        st.plotly_chart(px.imshow(df_sub.pct_change().corr(), text_auto=True, color_continuous_scale='RdBu_r', title="相关性热力图"), use_container_width=True)
                    
                    # 建议2精准落位：性格分布图
                    with c_tab2_2:
                        char_data = []
                        for f in sel_funds:
                            f_m = calculate_metrics(df_sub[f], bench_sync_raw)
                            char_data.append({"产品": f, "上行捕获": f_m['上行捕获'], "下行捕获": f_m['下行捕获'], "年化收益": f_m['年化收益']})
                        df_char = pd.DataFrame(char_data)
                        fig_char = px.scatter(df_char, x="下行捕获", y="上行捕获", size=df_char["年化收益"].clip(lower=0.01), 
                                             text="产品", title="成分产品性格分布图（横轴越左越防守，纵轴越上越进攻）",
                                             color="年化收益", color_continuous_scale='Viridis')
                        fig_char.add_vline(x=1.0, line_dash="dash", line_color="gray")
                        fig_char.add_hline(y=1.0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig_char, use_container_width=True)
            else:
                st.info("👈 请先勾选成分产品。")

        with tabs[2]:
            if sel_funds:
                st.subheader("🧩 权重与归因分析")
                cw1, cw2 = st.columns(2)
                with cw1:
                    st.plotly_chart(px.pie(names=list(weights.keys()), values=list(weights.values()), hole=0.4, title="资金权重分布 (Capital Allocation)"), use_container_width=True)
                
                # 建议3精准落位：风险贡献分析
                with cw2:
                    df_sub_rets = df_db[sel_funds].pct_change().fillna(0)
                    vol_list = df_sub_rets.std() * np.sqrt(252)
                    risk_contrib = {f: weights[f] * vol_list[f] for f in sel_funds}
                    total_risk = sum(risk_contrib.values()) if sum(risk_contrib.values()) > 0 else 1
                    risk_pct = {k: v/total_risk for k, v in risk_contrib.items()}
                    st.plotly_chart(px.pie(names=list(risk_pct.keys()), values=list(risk_pct.values()), hole=0.4, title="风险贡献分布 (Risk Contribution)", color_discrete_sequence=px.colors.sequential.RdBu), use_container_width=True)
                
                st.write("##### 详细分配数据")
                table_data = []
                for f in sel_funds:
                    table_data.append({"产品名称": f, "资金权重": weights[f], "风险贡献占比": risk_pct.get(f, 0)})
                st.table(pd.DataFrame(table_data).set_index("产品名称").style.format("{:.2%}"))
            else:
                st.info("👈 请先勾选成分产品。")

        with tabs[3]:
            st.subheader("⚔️ 配置池产品分析")
            compare_pool = st.multiselect("搜索并勾选池内产品", all_cols, default=[])
            if compare_pool:
                df_comp_raw = df_db[compare_pool].dropna()
                if not df_comp_raw.empty:
                    st.plotly_chart(px.line(df_comp_raw.div(df_comp_raw.iloc[0]), title="业绩对比"), use_container_width=True)
                    res_list = [dict(calculate_metrics(df_comp_raw[col]), **{"产品名称": col}) for col in compare_pool]
                    st.dataframe(pd.DataFrame(res_list).set_index('产品名称'), use_container_width=True)
    else:
        st.info("👋 请上传‘产品数据库’开始分析。")
