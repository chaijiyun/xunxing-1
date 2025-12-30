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
        
        # 1. 最大回撤修复时间
        mdd_val = drawdown.min()
        if mdd_val == 0:
            mdd_recovery = "无回撤"
        else:
            mdd_date = drawdown.idxmin()
            peak_val_at_mdd = cummax.loc[mdd_date]
            post_mdd_data = nav_series.loc[mdd_date:]
            recovery_mask = post_mdd_data >= peak_val_at_mdd
            if recovery_mask.any():
                recover_date = recovery_mask.idxmax()
                mdd_recovery = f"{(recover_date - mdd_date).days}天"
            else:
                mdd_recovery = "尚未修复"
        
        # 2. 最大无新高持续时间
        is_at_new_high = (nav_series == cummax)
        high_dates = nav_series[is_at_new_high].index
        if len(high_dates) < 2:
            max_no_new_high = f"{(nav_series.index[-1] - nav_series.index[0]).days}天"
        else:
            intervals = (high_dates[1:] - high_dates[:-1]).days
            last_gap = (nav_series.index[-1] - high_dates[-1]).days
            max_no_new_high = f"{max(intervals.max(), last_gap) if len(intervals)>0 else last_gap}天"
            
        return mdd_recovery, max_no_new_high, drawdown

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
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (ann_ret - rf) / downside_std if downside_std > 0 else 0
        
        calmar = ann_ret / abs(mdd) if abs(mdd) > 0 else 0
        mdd_recovery, max_no_new_high, dd_series = get_drawdown_details(nav)
        tuw_ratio = (nav < cummax).sum() / len(nav)
        
        metrics = {
            "总收益率": total_ret, "年化收益": ann_ret, "最大回撤": mdd, 
            "夏普比率": sharpe, "索提诺比率": sortino, "卡玛比率": calmar, "年化波动率": vol, 
            "回撤修复时间": mdd_recovery, "无新高持续时间": max_no_new_high, "水下时间": tuw_ratio,
            "dd_series": dd_series
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
            st.sidebar.markdown("#### ⚖️ 比例分配")
            avg_w = 1.0 / len(sel_funds)
            for f in sel_funds:
                weights[f] = st.sidebar.number_input(f"{f}", 0.0, 1.0, avg_w, step=0.05)
        
        start_date = st.sidebar.date_input("起始日期", df_raw.index.min())
        end_date = st.sidebar.date_input("截止日期", df_raw.index.max())
        df_db = df_raw.loc[start_date:end_date].copy()
        
        star_nav = None
        if sel_funds and not df_db.empty:
            df_port = df_db[sel_funds].dropna()
            if not df_port.empty:
                port_rets = df_port.pct_change().fillna(0)
                norm_w = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) > 0 else 1)
                star_rets = (port_rets * norm_w).sum(axis=1)
                star_nav = (1 + star_rets).cumprod()
                star_nav.name = "寻星配置组合"
                bench_sync = df_db.loc[star_nav.index, sel_bench]
                bench_norm = bench_sync / bench_sync.iloc[0]

        # ==========================================
        # 3. 标签页布局
        # ==========================================
        tabs = st.tabs(["🚀 寻星配置组合全景图", "🔍 穿透归因分析", "⚔️ 配置池产品分析"])

        with tabs[0]:
            if star_nav is not None:
                st.subheader("📊 寻星配置组合全景图")
                m = calculate_metrics(star_nav)
                c = st.columns(10)
                c[0].metric("总收益率", f"{m['总收益率']:.2%}")
                c[1].metric("年化收益", f"{m['年化收益']:.2%}")
                c[2].metric("最大回撤", f"{m['最大回撤']:.2%}")
                c[3].metric("夏普比率", f"{m['夏普比率']:.2f}")
                c[4].metric("索提诺", f"{m['索提诺比率']:.2f}")
                c[5].metric("卡玛比率", f"{m['卡玛比率']:.2f}")
                c[6].metric("年化波动", f"{m['年化波动率']:.2%}")
                c[7].metric("回撤修复", m['回撤修复时间'])
                c[8].metric("无新高期", m['无新高持续时间'])
                c[9].metric("水下时间", f"{m['水下时间']:.1%}")
                
                # 净值走势图
                fig_main = go.Figure()
                fig_main.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="寻星配置组合", line=dict(color='red', width=4)))
                fig_main.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm, name=f"基准: {sel_bench}", line=dict(color='#9CA3AF', dash='dot')))
                fig_main.update_layout(title="累计净值走势", template="plotly_white", hovermode="x unified", height=450)
                st.plotly_chart(fig_main, use_container_width=True)

                # 🚀 新增：水下时间分布图（回撤区域图）
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=m['dd_series'].index, y=m['dd_series'],
                    fill='tozeroy', mode='lines', name='回撤深度',
                    line=dict(color='rgba(220, 38, 38, 0.8)', width=1),
                    fillcolor='rgba(220, 38, 38, 0.3)'
                ))
                fig_dd.update_layout(
                    title="水下时间分布（红色区域代表无新高区间）",
                    yaxis_tickformat=".1%",
                    template="plotly_white",
                    height=250,
                    margin=dict(t=40, b=0)
                )
                st.plotly_chart(fig_dd, use_container_width=True)
            else:
                st.info("👈 请在左侧侧边栏配置组合成分。")

        with tabs[1]:
            if sel_funds:
                st.subheader("🔍 寻星配置穿透归因分析")
                st.markdown("#### 1. 初始配置与风险贡献")
                ca1, ca2 = st.columns(2)
                with ca1:
                    st.plotly_chart(px.pie(names=list(weights.keys()), values=list(weights.values()), hole=0.4, title="资金权重分配"), use_container_width=True)
                with ca2:
                    df_sub_rets = df_db[sel_funds].pct_change().fillna(0)
                    vol_list = df_sub_rets.std() * np.sqrt(252)
                    risk_contrib = {f: weights[f] * vol_list[f] for f in sel_funds}
                    total_r = sum(risk_contrib.values()) if sum(risk_contrib.values()) > 0 else 1
                    risk_pct = {k: v/total_r for k, v in risk_contrib.items()}
                    st.plotly_chart(px.pie(names=list(risk_pct.keys()), values=list(risk_pct.values()), hole=0.4, title="风险贡献归因"), use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### 2. 底层产品走势对比")
                df_sub = df_db[sel_funds].dropna()
                df_sub_norm = df_sub.div(df_sub.iloc[0])
                fig_sub_compare = go.Figure()
                for col in df_sub_norm.columns:
                    fig_sub_compare.add_trace(go.Scatter(x=df_sub_norm.index, y=df_sub_norm[col], name=col, opacity=0.6, line=dict(width=1.5)))
                if star_nav is not None:
                    fig_sub_compare.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="寻星配置组合", line=dict(color='red', width=4)))
                fig_sub_compare.update_layout(template="plotly_white", hovermode="x unified", height=500)
                st.plotly_chart(fig_sub_compare, use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### 3. 产品性格分布图")
                char_data = []
                for f in sel_funds:
                    f_m = calculate_metrics(df_sub[f], df_db[sel_bench])
                    char_data.append({"产品": f, "上行捕获": f_m['上行捕获'], "下行捕获": f_m['下行捕获'], "年化收益": f_m['年化收益']})
                df_char = pd.DataFrame(char_data)
                fig_char = px.scatter(df_char, x="下行捕获", y="上行捕获", size=df_char["年化收益"].clip(lower=0.01), 
                                     text="产品", color="年化收益", color_continuous_scale='Viridis', height=600)
                fig_char.add_vline(x=1.0, line_dash="dash"); fig_char.add_hline(y=1.0, line_dash="dash")
                st.plotly_chart(fig_char, use_container_width=True)
                
                st.markdown("#### 4. 产品相关性矩阵")
                st.plotly_chart(px.imshow(df_sub.pct_change().corr(), text_auto=".2f", color_continuous_scale='RdBu_r', height=600), use_container_width=True)

        with tabs[2]:
            st.subheader("⚔️ 配置池产品分析")
            compare_pool = st.multiselect("搜索池内产品", all_cols, default=[])
            
            if compare_pool:
                is_aligned = st.checkbox("对齐共同起始日期比较", value=False)
                df_comp = df_db[compare_pool].dropna() if is_aligned else df_db[compare_pool]
                
                if not df_comp.empty:
                    fig_comp_lines = go.Figure()
                    for col in compare_pool:
                        series = df_comp[col].dropna()
                        if not series.empty:
                            norm_series = series / series.iloc[0]
                            fig_comp_lines.add_trace(go.Scatter(x=norm_series.index, y=norm_series, name=col))
                    fig_comp_lines.update_layout(title="配置池产品业绩走势对比", template="plotly_white", hovermode="x unified", height=500)
                    st.plotly_chart(fig_comp_lines, use_container_width=True)
                
                res_data = []
                for col in compare_pool:
                    metrics = calculate_metrics(df_db[col])
                    res_data.append({
                        "产品名称": col,
                        "总收益率": f"{metrics['总收益率']:.2%}",
                        "年化收益": f"{metrics['年化收益']:.2%}",
                        "最大回撤": f"{metrics['最大回撤']:.2%}",
                        "夏普比率": round(metrics['夏普比率'], 2),
                        "索提诺": round(metrics['索提诺比率'], 2),
                        "卡玛比率": round(metrics['卡玛比率'], 2),
                        "回撤修复时间": metrics['回撤修复时间'],
                        "最大无新高": metrics['无新高持续时间'],
                        "年化波动": f"{metrics['年化波动率']:.2%}"
                    })
                st.dataframe(pd.DataFrame(res_data).set_index('产品名称'), use_container_width=True)
    else:
        st.info("👋 请上传‘产品数据库’开始分析。")
