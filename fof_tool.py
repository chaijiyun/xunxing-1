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
    # 1. 核心计算引擎
    # ==========================================
    def get_drawdown_details(nav_series):
        if nav_series.empty or len(nav_series) < 2: return "数据不足", "数据不足"
        cummax = nav_series.cummax()
        drawdown = (nav_series / cummax) - 1
        mdd_val = drawdown.min()
        if mdd_val == 0:
            mdd_recovery = "无回撤"
        else:
            mdd_date = drawdown.idxmin()
            peak_before_mdd = nav_series.loc[:mdd_date].idxmax()
            peak_val = nav_series.loc[peak_before_mdd]
            post_mdd_data = nav_series.loc[mdd_date:]
            recovery_mask = post_mdd_data >= peak_val
            mdd_recovery = f"{(recovery_mask.idxmax() - peak_before_mdd).days}天" if recovery_mask.any() else "尚未修复"
        
        is_new_high = nav_series == cummax
        high_dates = is_new_high[is_new_high].index
        max_interval = f"{(high_dates[1:] - high_dates[:-1]).days.max()}天" if len(high_dates) > 1 else f"{(nav_series.index[-1] - nav_series.index[0]).days}天"
        return mdd_recovery, max_interval

    def calculate_metrics(nav, bench_nav=None):
        nav = nav.dropna()
        if len(nav) < 2: return {}
        total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
        days = (nav.index[-1] - nav.index[0]).days
        ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / max(days, 1)) - 1
        returns = nav.pct_change().fillna(0)
        vol = returns.std() * np.sqrt(252)
        mdd = (nav / nav.cummax() - 1).min()
        
        rf = 0.02
        sharpe = (ann_ret - rf) / vol if vol > 0 else 0
        downside_std = returns[returns < 0].std() * np.sqrt(252)
        sortino = (ann_ret - rf) / downside_std if downside_std > 0 else 0
        calmar = ann_ret / abs(mdd) if abs(mdd) > 0 else 0
        mdd_rec, max_peak_int = get_drawdown_details(nav)
        
        metrics = {
            "总收益率": total_ret, "年化收益": ann_ret, "最大回撤": mdd, 
            "夏普比率": sharpe, "索提诺比率": sortino, "卡玛比率": calmar, 
            "年化波动": vol, "回撤修复": mdd_rec, "新高间隔": max_peak_int,
            "水下占比": (nav < nav.cummax()).sum() / len(nav)
        }
        
        if bench_nav is not None:
            b_sync = bench_nav.reindex(nav.index).ffill()
            b_rets = b_sync.pct_change().fillna(0)
            up_mask, down_mask = b_rets > 0, b_rets < 0
            metrics["上行捕获"] = (returns[up_mask].mean() / b_rets[up_mask].mean()) if up_mask.any() and b_rets[up_mask].mean() !=0 else 0
            metrics["下行捕获"] = (returns[down_mask].mean() / b_rets[down_mask].mean()) if down_mask.any() and b_rets[down_mask].mean() !=0 else 0
        return metrics

    # ==========================================
    # 2. 侧边栏及配置
    # ==========================================
    st.set_page_config(layout="wide", page_title="寻星配置分析系统", page_icon="🏛️")
    st.sidebar.title("🏛️ 寻星配置分析系统")
    uploaded_file = st.sidebar.file_uploader("📂 上传寻星配置数据库", type=["xlsx"])

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
            for f in sel_funds: weights[f] = st.sidebar.number_input(f"{f}", 0.0, 1.0, 1.0/len(sel_funds), step=0.05)
        
        start_date = st.sidebar.date_input("起始日期", df_raw.index.min())
        end_date = st.sidebar.date_input("截止日期", df_raw.index.max())
        df_db = df_raw.loc[start_date:end_date].copy()
        
        star_nav = None
        if sel_funds and not df_db.empty:
            df_port = df_db[sel_funds].dropna()
            if not df_port.empty:
                norm_w = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) >0 else 1)
                star_rets = (df_port.pct_change().fillna(0) * norm_w).sum(axis=1)
                star_nav = (1 + star_rets).cumprod()
                star_nav.name = "寻星配置组合"
                bench_norm = df_db.loc[star_nav.index, sel_bench] / df_db.loc[star_nav.index, sel_bench].iloc[0]

        # ==========================================
        # 3. 页面主体
        # ==========================================
        tabs = st.tabs(["🚀 寻星配置组合全景图", "🔍 穿透归因分析", "⚔️ 配置池产品分析"])

        with tabs[0]:
            if star_nav is not None:
                st.subheader("📊 寻星配置组合全景图")
                m = calculate_metrics(star_nav)
                c = st.columns(9)
                metrics_list = [
                    ("总收益率", f"{m['总收益率']:.2%}"), ("年化收益", f"{m['年化收益']:.2%}"),
                    ("最大回撤", f"{m['最大回撤']:.2%}"), ("夏普比率", f"{m['夏普比率']:.2f}"),
                    ("索提诺", f"{m['索提诺比率']:.2f}"), ("卡玛比率", f"{m['卡玛比率']:.2f}"),
                    ("修复天数", m['回撤修复']), ("新高间隔", m['新高间隔']),
                    ("水下时间", f"{m['水下占比']:.1%}")
                ]
                for i, (label, value) in enumerate(metrics_list): c[i].metric(label, value)
                
                fig_main = go.Figure()
                fig_main.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="寻星配置组合", line=dict(color='red', width=4)))
                fig_main.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm, name=f"基准: {sel_bench}", line=dict(color='#9CA3AF', dash='dot')))
                fig_main.update_layout(template="plotly_white", hovermode="x unified", height=550)
                st.plotly_chart(fig_main, use_container_width=True)
            else:
                st.info("👈 请在左侧侧边栏配置组合成分并选择日期。")

        with tabs[1]:
            if sel_funds:
                st.subheader("🔍 寻星配置穿透归因分析")
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(px.pie(names=list(weights.keys()), values=list(weights.values()), hole=0.4, title="资金权重分配"), use_container_width=True)
                with c2:
                    vol_list = df_db[sel_funds].pct_change().std() * np.sqrt(252)
                    risk_contrib = {f: weights[f] * vol_list[f] for f in sel_funds}
                    st.plotly_chart(px.pie(names=list(risk_contrib.keys()), values=list(risk_contrib.values()), hole=0.4, title="风险贡献归因"), use_container_width=True)
                
                st.markdown("---")
                st.markdown("#### 🎯 产品性格分布图 (Capture Ratio)")
                char_list = []
                for f in sel_funds:
                    m_f = calculate_metrics(df_db[f], df_db[sel_bench])
                    char_list.append({"产品": f, "上行捕获": m_f.get('上行捕获', 0), "下行捕获": m_f.get('下行捕获', 0), "年化收益": m_f.get('年化收益', 0)})
                df_char = pd.DataFrame(char_list)
                fig_char = px.scatter(df_char, x="下行捕获", y="上行捕获", size=df_char["年化收益"].clip(lower=0.01), text="产品", color="年化收益", color_continuous_scale='Viridis', height=600)
                fig_char.add_vline(x=1.0, line_dash="dash", line_color="gray"); fig_char.add_hline(y=1.0, line_dash="dash", line_color="gray")
                fig_char.update_layout(xaxis_title="下行捕获 (越小越防御)", yaxis_title="上行捕获 (越大越进攻)")
                st.plotly_chart(fig_char, use_container_width=True)

                st.markdown("---")
                st.markdown("#### 🔗 产品相关性矩阵")
                st.plotly_chart(px.imshow(df_db[sel_funds].pct_change().corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
            else:
                st.info("👈 请在左侧侧边栏选择成分产品。")

        with tabs[2]:
            st.subheader("⚔️ 配置池产品分析")
            compare_pool = st.multiselect("搜索并勾选池内产品", all_cols)
            if compare_pool:
                res_data = []
                for col in compare_pool:
                    m = calculate_metrics(df_db[col])
                    res_data.append({
                        "产品名称": col, "总收益率": f"{m['总收益率']:.2%}", "年化收益": f"{m['年化收益']:.2%}",
                        "最大回撤": f"{m['最大回撤']:.2%}", "夏普比率": round(m['夏普比率'], 2),
                        "索提诺": round(m['索提诺比率'], 2), "卡玛比率": round(m['卡玛比率'], 2),
                        "年化波动": f"{m['年化波动']:.2%}", "回撤修复": m['回撤修复'],
                        "新高间隔": m['新高间隔'], "水下占比": f"{m['水下占比']:.1%}"
                    })
                st.dataframe(pd.DataFrame(res_data).set_index('产品名称'), use_container_width=True)
    else:
        st.info("👋 欢迎！请上传‘寻星配置数据库’开始你的专业配置之旅。")
