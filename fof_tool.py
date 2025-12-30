import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 0. 登录验证模块
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.markdown("<br><br>", unsafe_allow_html=True) 
        st.markdown("<h1 style='text-align: center; color: #1E40AF;'>寻星配置分析系统</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                pwd_input = st.text_input(label="系统访问密码", type="password", placeholder="请输入密码")
                submit_button = st.form_submit_button("立即登录", use_container_width=True)
                if submit_button:
                    if pwd_input == "281699":
                        st.session_state["password_correct"] = True
                        st.rerun()
                    else:
                        st.error("密码不正确")
        return False
    return True

if check_password():
    # ==========================================
    # 1. 核心指标计算引擎
    # ==========================================
    def get_drawdown_details(nav_series):
        if nav_series.empty or len(nav_series) < 2: 
            return "数据不足", "数据不足", pd.Series()
        cummax = nav_series.cummax()
        drawdown = (nav_series / cummax) - 1
        mdd_val = drawdown.min()
        if mdd_val == 0:
            mdd_recovery = "无回撤"
        else:
            mdd_date = drawdown.idxmin()
            peak_val_at_mdd = cummax.loc[mdd_date]
            post_mdd_data = nav_series.loc[mdd_date:]
            recovery_mask = post_mdd_data >= peak_val_at_mdd
            mdd_recovery = f"{(recovery_mask.idxmax() - mdd_date).days}天" if recovery_mask.any() else "尚未修复"
        
        is_at_new_high = (nav_series == cummax)
        high_dates = nav_series[is_at_new_high].index
        if len(high_dates) < 2:
            max_no_new_high = f"{(nav_series.index[-1] - nav_series.index[0]).days}天"
        else:
            intervals = (high_dates[1:] - high_dates[:-1]).days
            last_gap = (nav_series.index[-1] - high_dates[-1]).days
            max_no_new_high = f"{max(intervals.max(), last_gap) if len(intervals)>0 else last_gap}天"
        return mdd_recovery, max_no_new_high, drawdown

    def calc_win_prob(nav, days):
        """计算滚动持有n个交易日的盈利概率"""
        if len(nav) <= days: return 0.0
        diff = nav.shift(-days) / nav - 1
        win_prob = (diff.dropna() > 0).sum() / len(diff.dropna())
        return win_prob

    def calculate_metrics(nav, bench_nav=None):
        nav = nav.dropna()
        if len(nav) < 2: return {}
        returns = nav.pct_change().fillna(0)
        
        total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
        ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / max((nav.index[-1] - nav.index[0]).days, 1)) - 1
        vol = returns.std() * np.sqrt(252)
        mdd_recovery, max_no_new_high, dd_series = get_drawdown_details(nav)
        
        win_probs = {
            "正收益概率(日)": (returns > 0).sum() / len(returns),
            "持有3月胜率": calc_win_prob(nav, 63),
            "持有6月胜率": calc_win_prob(nav, 126),
            "持有12月胜率": calc_win_prob(nav, 252),
            "持有24月胜率": calc_win_prob(nav, 504),
            "持有至今胜率": ((nav.iloc[-1] > nav.iloc[:-1]).sum() / (len(nav)-1)) if len(nav)>1 else 0
        }

        metrics = {
            "总收益率": total_ret, "年化收益": ann_ret, "最大回撤": (nav / nav.cummax() - 1).min(), 
            "夏普比率": (ann_ret - 0.02) / vol if vol > 0 else 0, "卡玛比率": ann_ret / abs((nav / nav.cummax() - 1).min()) if (nav / nav.cummax() - 1).min() != 0 else 0,
            "年化波动率": vol, "最大回撤修复时间": mdd_recovery, "最大无新高持续时间": max_no_new_high, 
            "水下时间": (nav < nav.cummax()).sum() / len(nav), "dd_series": dd_series
        }
        metrics.update(win_probs)
        
        if bench_nav is not None:
            b_sync = bench_nav.reindex(nav.index).ffill()
            b_rets = b_sync.pct_change().fillna(0)
            metrics.update({
                "上行捕获": (returns[b_rets > 0].mean() / b_rets[b_rets > 0].mean()) if (b_rets > 0).any() else 0,
                "下行捕获": (returns[b_rets < 0].mean() / b_rets[b_rets < 0].mean()) if (b_rets < 0).any() else 0
            })
        return metrics

    # ==========================================
    # 2. UI 侧边栏与数据
    # ==========================================
    st.set_page_config(layout="wide", page_title="寻星配置分析系统", page_icon="🏛️")
    st.sidebar.title("🏛️ 寻星配置分析系统")
    uploaded_file = st.sidebar.file_uploader("📂 上传产品数据库", type=["xlsx"])

    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        
        sel_bench = st.sidebar.selectbox("业绩基准", df_raw.columns, index=0)
        sel_funds = st.sidebar.multiselect("挑选寻星配置组合成分", [c for c in df_raw.columns if c != sel_bench])
        
        weights = {}
        if sel_funds:
            st.sidebar.markdown("#### ⚖️ 初始比例设定")
            for f in sel_funds: weights[f] = st.sidebar.number_input(f"{f}", 0.0, 1.0, 1.0/len(sel_funds), 0.05)
        
        df_db = df_raw.loc[st.sidebar.date_input("起始", df_raw.index.min()):st.sidebar.date_input("截止", df_raw.index.max())].copy()
        
        star_nav = None
        if sel_funds and not df_db.empty:
            df_port = df_db[sel_funds].dropna()
            if not df_port.empty:
                norm_w = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) > 0 else 1)
                star_rets = (df_port.pct_change().fillna(0) * norm_w).sum(axis=1)
                star_nav = (1 + star_rets).cumprod()
                star_nav.name = "寻星配置组合"

        # ==========================================
        # 3. 标签页布局
        # ==========================================
        tabs = st.tabs(["🚀 寻星配置组合全景图", "🔍 穿透归因分析", "⚔️ 配置池产品分析"])

        with tabs[0]:
            if star_nav is not None:
                m = calculate_metrics(star_nav)
                st.subheader("📊 核心收益指标")
                c1 = st.columns(6)
                c1[0].metric("总收益率", f"{m['总收益率']:.2%}")
                c1[1].metric("年化收益", f"{m['年化收益']:.2%}")
                c1[2].metric("最大回撤", f"{m['最大回撤']:.2%}")
                c1[3].metric("年化波动", f"{m['年化波动率']:.2%}")
                c1[4].metric("夏普比率", f"{m['夏普比率']:.2f}")
                c1[5].metric("卡玛比率", f"{m['卡玛比率']:.2f}")

                fig_main = go.Figure()
                fig_main.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="寻星配置组合", line=dict(color='red', width=4)))
                bench_v = df_db.loc[star_nav.index, sel_bench]
                bench_norm = bench_v / bench_v.iloc[0]
                fig_main.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm, name=f"基准: {sel_bench}", line=dict(color='#9CA3AF', dash='dot')))
                fig_main.update_layout(title="累计净值走势", template="plotly_white", height=450)
                st.plotly_chart(fig_main, use_container_width=True)

                st.subheader("🛡️ 风险体验与多期限胜率")
                c_risk, c_win = st.columns([1, 1.5])
                with c_risk:
                    st.markdown("**持有体验指标**")
                    st.write(f"最大回撤修复时间: **{m['最大回撤修复时间']}**")
                    st.write(f"最大无新高持续时间: **{m['最大无新高持续时间']}**")
                    st.write(f"水下时间占比: **{m['水下时间']:.1%}**")
                    st.write(f"日度正收益概率: **{m['正收益概率(日)']:.1%}**")
                with c_win:
                    st.markdown("**持有盈利概率拆解**")
                    win_df = pd.DataFrame({
                        "持有期限": ["3个月", "6个月", "12个月", "24个月", "至今"],
                        "盈利概率": [f"{m['持有3月胜率']:.1%}", f"{m['持有6月胜率']:.1%}", f"{m['持有12月胜率']:.1%}", f"{m['持有24月胜率']:.1%}", f"{m['持有至今胜率']:.1%}"]
                    })
                    st.table(win_df)
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=m['dd_series'].index, y=m['dd_series'], fill='tozeroy', line=dict(color='rgba(220, 38, 38, 0.8)'), fillcolor='rgba(220, 38, 38, 0.3)'))
                fig_dd.update_layout(title="水下分布", yaxis_tickformat=".1%", template="plotly_white", height=250)
                st.plotly_chart(fig_dd, use_container_width=True)

        with tabs[1]:
            if sel_funds:
                st.subheader("🔍 寻星配置穿透归因分析")
                df_sub = df_db[sel_funds].dropna()
                w_init = pd.Series(weights) / sum(weights.values())
                w_now = (w_init * (df_sub.iloc[-1]/df_sub.iloc[0])) / (w_init * (df_sub.iloc[-1]/df_sub.iloc[0])).sum()
                
                cw = st.columns(2)
                cw[0].plotly_chart(px.pie(names=w_init.index, values=w_init.values, hole=0.4, title="初始配置比例"), use_container_width=True)
                cw[1].plotly_chart(px.pie(names=w_now.index, values=w_now.values, hole=0.4, title="最新配置比例(漂移)"), use_container_width=True)
                
                rk = w_init * (df_sub.pct_change().std() * np.sqrt(252))
                gn = w_init * ((df_sub.iloc[-1]/df_sub.iloc[0])-1)
                ca = st.columns(2)
                ca[0].plotly_chart(px.pie(names=rk.index, values=rk.values, hole=0.4, title="风险贡献归因"), use_container_width=True)
                ca[1].plotly_chart(px.pie(names=gn.index, values=gn.abs(), hole=0.4, title="收益贡献归因"), use_container_width=True)

                st.plotly_chart(px.scatter(pd.DataFrame([calculate_metrics(df_sub[f], df_db[sel_bench]) for f in sel_funds]).assign(产品=sel_funds), x="下行捕获", y="上行捕获", size="年化收益", text="产品", color="年化收益", title="产品性格象限图", height=600), use_container_width=True)
                st.plotly_chart(px.imshow(df_sub.pct_change().corr(), text_auto=".2f", color_continuous_scale='RdBu_r', title="产品相关性矩阵", height=600), use_container_width=True)

        with tabs[2]:
            st.subheader("⚔️ 配置池产品分析")
            pool = st.multiselect("搜索池内产品", df_raw.columns)
            if pool:
                is_a = st.checkbox("对齐起始日期")
                df_c = df_db[pool].dropna() if is_a else df_db[pool]
                if not df_c.empty:
                    fig_c = go.Figure()
                    for c in pool: fig_c.add_trace(go.Scatter(x=df_c.index, y=df_c[c]/df_c[c].dropna().iloc[0], name=c))
                    st.plotly_chart(fig_c.update_layout(title="业绩对比", template="plotly_white"), use_container_width=True)
                
                res = []
                for c in pool:
                    m = calculate_metrics(df_db[c])
                    res.append({
                        "名称": c, "总收益": f"{m['总收益率']:.2%}", "年化": f"{m['年化收益']:.2%}", "回撤": f"{m['最大回撤']:.2%}", "夏普": round(m['夏普比率'], 2), "卡玛": round(m['卡玛比率'], 2),
                        "修复": m['最大回撤修复时间'], "无新高": m['最大无新高持续时间'], "水下": f"{m['水下时间']:.1%}", 
                        "3月胜率": f"{m['持有3月胜率']:.1%}", "6月胜率": f"{m['持有6月胜率']:.1%}", "12月胜率": f"{m['持有12月胜率']:.1%}", "24月胜率": f"{m['持有24月胜率']:.1%}", "至今": f"{m['持有至今胜率']:.1%}"
                    })
                df_res = pd.DataFrame(res).set_index("名称")
                st.markdown("#### 1. 收益风险表")
                st.dataframe(df_res[["总收益", "年化", "回撤", "夏普", "卡玛"]], use_container_width=True)
                st.markdown("#### 2. 体验与滚动胜率表")
                st.dataframe(df_res[["修复", "无新高", "水下", "3月胜率", "6月胜率", "12月胜率", "24月胜率", "至今"]], use_container_width=True)
    else:
        st.info("👋 请上传‘产品数据库’。")
