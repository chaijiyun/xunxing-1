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
        """核心逻辑修正：计算任意一点买入，持有N个交易日后的盈利概率"""
        if len(nav) <= days: return 0.0
        # 使用 diff 计算：(N天后的价格 / 当前价格) - 1
        # 用 shift(-days) 将未来的价格对齐到当前行
        future_nav = nav.shift(-days)
        returns = (future_nav / nav) - 1
        valid_returns = returns.dropna()
        if len(valid_returns) == 0: return 0.0
        return (valid_returns > 0).sum() / len(valid_returns)

    def calculate_metrics(nav, bench_nav=None):
        nav = nav.dropna()
        if len(nav) < 2: return {}
        returns = nav.pct_change().fillna(0)
        
        total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
        days_count = (nav.index[-1] - nav.index[0]).days
        ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / max(days_count, 1)) - 1
        vol = returns.std() * np.sqrt(252)
        mdd = (nav / nav.cummax() - 1).min()
        
        rf = 0.02
        sharpe = (ann_ret - rf) / vol if vol > 0 else 0
        # 修正索提诺比率计算逻辑
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0.0001
        sortino = (ann_ret - rf) / downside_std if downside_std > 0 else 0
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0
        
        mdd_rec, max_nh, dd_s = get_drawdown_details(nav)
        
        metrics = {
            "总收益率": total_ret, "年化收益": ann_ret, "最大回撤": mdd, 
            "夏普比率": sharpe, "索提诺比率": sortino, "卡玛比率": calmar, "年化波动率": vol,
            "最大回撤修复时间": mdd_rec, "最大无新高持续时间": max_nh,
            "正收益概率(日)": (returns > 0).sum() / len(returns),
            "持有3月胜率": calc_win_prob(nav, 63),
            "持有6月胜率": calc_win_prob(nav, 126),
            "持有12月胜率": calc_win_prob(nav, 252),
            "持有24月胜率": calc_win_prob(nav, 504),
            "持有至今胜率": ((nav.iloc[-1] > nav.iloc[:-1]).sum() / (len(nav)-1)) if len(nav)>1 else 0,
            "dd_series": dd_s
        }

        if bench_nav is not None:
            b_sync = bench_nav.reindex(nav.index).ffill()
            b_rets = b_sync.pct_change().fillna(0)
            up_mask, down_mask = b_rets > 0, b_rets < 0
            up_cap = (returns[up_mask].mean() / b_rets[up_mask].mean()) if up_mask.any() else 0
            down_cap = (returns[down_mask].mean() / b_rets[down_mask].mean()) if down_mask.any() else 0
            metrics.update({"上行捕获": up_cap, "下行捕获": down_cap})
        return metrics

    # ==========================================
    # 2. UI 界面与侧边栏
    # ==========================================
    st.set_page_config(layout="wide", page_title="寻星配置分析系统", page_icon="🏛️")
    st.sidebar.title("🏛️ 寻星配置分析系统")
    uploaded_file = st.sidebar.file_uploader("📂 请上传产品数据库", type=["xlsx"])

    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
        all_cols = [str(c).strip() for c in df_raw.columns]
        df_raw.columns = all_cols
        
        st.sidebar.markdown("---")
        default_bench = '沪深300' if '沪深300' in all_cols else all_cols[0]
        sel_bench = st.sidebar.selectbox("业绩基准", all_cols, index=all_cols.index(default_bench))
        sel_funds = st.sidebar.multiselect("挑选寻星配置组合成分", [c for c in all_cols if c != sel_bench])
        
        weights = {}
        if sel_funds:
            st.sidebar.markdown("#### ⚖️ 初始比例设定")
            avg_w = 1.0 / len(sel_funds)
            for f in sel_funds:
                weights[f] = st.sidebar.number_input(f"{f}", 0.0, 1.0, avg_w, step=0.05)
        
        df_db = df_raw.loc[st.sidebar.date_input("起始日期", df_raw.index.min()):st.sidebar.date_input("截止日期", df_raw.index.max())].copy()
        
        star_nav = None
        if sel_funds and not df_db.empty:
            df_port = df_db[sel_funds].dropna()
            if not df_port.empty:
                norm_w = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) > 0 else 1)
                star_rets = (df_port.pct_change().fillna(0) * norm_w).sum(axis=1)
                star_nav = (1 + star_rets).cumprod()
                star_nav.name = "寻星配置组合"
                bn_sync = df_db.loc[star_nav.index, sel_bench]
                bn_norm = bn_sync / bn_sync.iloc[0]

        # ==========================================
        # 3. 标签页布局
        # ==========================================
        tabs = st.tabs(["🚀 寻星配置组合全景图", "🔍 穿透归因分析", "⚔️ 配置池产品分析"])

        with tabs[0]:
            if star_nav is not None:
                st.subheader("📊 寻星配置组合全景图")
                m = calculate_metrics(star_nav)
                c_top = st.columns(7)
                c_top[0].metric("总收益率", f"{m['总收益率']:.2%}")
                c_top[1].metric("年化收益", f"{m['年化收益']:.2%}")
                c_top[2].metric("最大回撤", f"{m['最大回撤']:.2%}")
                c_top[3].metric("夏普比率", f"{m['夏普比率']:.2f}")
                c_top[4].metric("索提诺", f"{m['索提诺比率']:.2f}")
                c_top[5].metric("卡玛比率", f"{m['卡玛比率']:.2f}")
                c_top[6].metric("年化波动", f"{m['年化波动率']:.2%}")
                
                fig_main = go.Figure()
                fig_main.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="寻星配置组合", line=dict(color='red', width=4)))
                fig_main.add_trace(go.Scatter(x=bn_norm.index, y=bn_norm, name=f"基准: {sel_bench}", line=dict(color='#9CA3AF', dash='dot')))
                fig_main.update_layout(title="累计净值走势", template="plotly_white", hovermode="x unified", height=450)
                st.plotly_chart(fig_main, use_container_width=True)

                st.markdown("#### 🛡️ 风险体验与持有盈利概率")
                c_risk, c_win = st.columns([1, 1.5])
                with c_risk:
                    st.write(f"最大回撤修复时间: **{m['最大回撤修复时间']}**")
                    st.write(f"最大无新高持续时间: **{m['最大无新高持续时间']}**")
                    st.write(f"日度正收益概率: **{m['正收益概率(日)']:.1%}**")
                with c_win:
                    win_df = pd.DataFrame({
                        "持有期限": ["3个月", "6个月", "12个月", "24个月", "持有至今"],
                        "盈利概率": [f"{m['持有3月胜率']:.1%}", f"{m['持有6月胜率']:.1%}", f"{m['持有12月胜率']:.1%}", f"{m['持有24月胜率']:.1%}", f"{m['持有至今胜率']:.1%}"]
                    })
                    st.table(win_df)
            else:
                st.info("👈 请在左侧侧边栏配置组合成分。")

        with tabs[1]:
            if sel_funds:
                st.subheader("🔍 寻星配置穿透归因分析")
                df_sub_prices = df_db[sel_funds].dropna()
                initial_w_series = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) > 0 else 1)
                
                growth_factors = df_sub_prices.iloc[-1] / df_sub_prices.iloc[0]
                latest_values = initial_w_series * growth_factors
                latest_w_series = latest_values / latest_values.sum()

                col_w1, col_w2 = st.columns(2)
                col_w1.plotly_chart(px.pie(names=initial_w_series.index, values=initial_w_series.values, hole=0.4, title="初始配置比例"), use_container_width=True)
                col_w2.plotly_chart(px.pie(names=latest_w_series.index, values=latest_w_series.values, hole=0.4, title="最新配置比例(漂移)"), use_container_width=True)

                df_sub_rets = df_sub_prices.pct_change().fillna(0)
                risk_vals = initial_w_series * (df_sub_rets.std() * np.sqrt(252))
                contribution_vals = initial_w_series * ((df_sub_prices.iloc[-1] / df_sub_prices.iloc[0]) - 1)

                col_attr1, col_attr2 = st.columns(2)
                col_attr1.plotly_chart(px.pie(names=risk_vals.index, values=risk_vals.values, hole=0.4, title="风险贡献归因"), use_container_width=True)
                col_attr2.plotly_chart(px.pie(names=contribution_vals.index, values=contribution_vals.abs(), hole=0.4, title="收益贡献归因"), use_container_width=True)

                st.markdown("---")
                st.markdown("#### 底层产品走势对比")
                df_sub_norm = df_sub_prices.div(df_sub_prices.iloc[0])
                fig_sub_compare = go.Figure()
                for col in df_sub_norm.columns:
                    fig_sub_compare.add_trace(go.Scatter(x=df_sub_norm.index, y=df_sub_norm[col], name=col, opacity=0.6))
                if star_nav is not None:
                    fig_sub_compare.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="寻星配置组合", line=dict(color='red', width=4)))
                st.plotly_chart(fig_sub_compare.update_layout(template="plotly_white", height=500), use_container_width=True)
                
                st.markdown("---")
                char_data = []
                for f in sel_funds:
                    f_metrics = calculate_metrics(df_sub_prices[f], df_db[sel_bench])
                    f_metrics['产品'] = f
                    char_data.append(f_metrics)
                st.plotly_chart(px.scatter(pd.DataFrame(char_data), x="下行捕获", y="上行捕获", size="年化收益", text="产品", color="年化收益", title="产品性格象限分布", height=600), use_container_width=True)
                st.plotly_chart(px.imshow(df_sub_rets.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', title="产品相关性矩阵", height=600), use_container_width=True)

        with tabs[2]:
            st.subheader("⚔️ 配置池产品分析")
            compare_pool = st.multiselect("搜索池内产品", all_cols, default=[])
            if compare_pool:
                is_aligned = st.checkbox("对齐起始日期比较", value=False)
                df_comp = df_db[compare_pool].dropna() if is_aligned else df_db[compare_pool]
                if not df_comp.empty:
                    fig_p = go.Figure()
                    for col in compare_pool:
                        s = df_comp[col].dropna()
                        if not s.empty: fig_p.add_trace(go.Scatter(x=s.index, y=s/s.iloc[0], name=col))
                    st.plotly_chart(fig_p.update_layout(title="业绩对比", template="plotly_white", height=500), use_container_width=True)
                
                res_data = []
                for col in compare_pool:
                    k = calculate_metrics(df_db[col])
                    res_data.append({
                        "产品名称": col, "总收益": f"{k['总收益率']:.2%}", "年化": f"{k['年化收益']:.2%}", 
                        "回撤": f"{k['最大回撤']:.2%}", "夏普": round(k['夏普比率'], 2), 
                        "波动": f"{k['年化波动率']:.2%}", "3M胜率": f"{k['持有3月胜率']:.1%}", 
                        "6M胜率": f"{k['持有6月胜率']:.1%}", "12M胜率": f"{k['持有12月胜率']:.1%}",
                        "24M胜率": f"{k['持有24月胜率']:.1%}", "至今胜率": f"{k['持有至今胜率']:.1%}"
                    })
                st.dataframe(pd.DataFrame(res_data).set_index('产品名称'), use_container_width=True)
    else:
        st.info("👋 请上传‘产品数据库’。")
