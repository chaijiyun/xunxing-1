import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 0. 全局产品费率库 (标准刊例价)
# ==========================================
# 这里存储的是“默认标准”，系统启动时会先加载这些
# 格式：'产品名称': {'mgmt': 年管理费率(1.5%=0.015), 'perf': 业绩报酬(20%=0.20)}
PRESET_FEES = {
    # --- 核心底仓 ---
    "合绎期权套利": {"mgmt": 0.00, "perf": 0.30},
    "平方和多策略6号(市场中性+多策略）": {"mgmt": 0.00, "perf": 0.18},
    
    # --- 股票多头 ---
    "开思沪港深优选": {"mgmt": 0.015, "perf": 0.17},
    "蓝墨长河1号": {"mgmt": 0.00, "perf": 0.20},
    "宁泉特定策略1号": {"mgmt": 0.00, "perf": 0.15},
    "睿郡节节高11号": {"mgmt": 0.00, "perf": 0.20},
    "宽远优势成长10号": {"mgmt": 0.00, "perf": 0.20},
    
    # --- 量化/中性 ---
    "孝庸中性策略": {"mgmt": 0.00, "perf": 0.20},
    "孝庸中性+cta": {"mgmt": 0.00, "perf": 0.20},
    "平方和市场中性": {"mgmt": 0.00, "perf": 0.20},
    
    # --- 指数增强 ---
    "孝庸500指增": {"mgmt": 0.00, "perf": 0.20},
    "孝庸1000指增": {"mgmt": 0.00, "perf": 0.20},
    "平方和1000指数增强": {"mgmt": 0.00, "perf": 0.20},
}

DEFAULT_FEE = {"mgmt": 0.00, "perf": 0.20} 

# ==========================================
# 1. 登录验证模块
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.markdown("<br><br>", unsafe_allow_html=True) 
        st.markdown("<h1 style='text-align: center; color: #1E40AF;'>寻星配置分析系统 v5.10</h1>", unsafe_allow_html=True)
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
    # 2. 核心指标计算引擎
    # ==========================================
    def calculate_net_nav_series(gross_nav_series, mgmt_fee_rate=0.0, perf_fee_rate=0.0):
        """费后净值计算函数 (高水位法)"""
        if gross_nav_series.empty: return gross_nav_series
        base_nav = gross_nav_series.iloc[0]
        gross_norm = gross_nav_series / base_nav
        
        net_nav = [1.0]
        high_water_mark = 1.0
        dates = gross_nav_series.index
        
        gross_returns = gross_norm.pct_change().fillna(0)
        
        days_diff = (dates[-1] - dates[0]).days
        periods = len(dates)
        avg_days = days_diff / periods if periods > 0 else 7
        freq_factor = 365.0 / avg_days if avg_days > 0 else 52.0

        for i in range(1, len(gross_returns)):
            r_gross = gross_returns.iloc[i]
            mgmt_cost = mgmt_fee_rate / freq_factor
            nav_after_mgmt = net_nav[-1] * (1 + r_gross - mgmt_cost)
            
            fee_perf = 0.0
            if nav_after_mgmt > high_water_mark:
                excess = nav_after_mgmt - high_water_mark
                fee_perf = excess * perf_fee_rate
                high_water_mark = nav_after_mgmt - fee_perf 
            
            nav_final = nav_after_mgmt - fee_perf
            if nav_final < 0: nav_final = 0
            net_nav.append(nav_final)
        
        return pd.Series(net_nav, index=dates)

    def get_drawdown_details(nav_series):
        if nav_series.empty or len(nav_series) < 2: 
            return "数据不足", "数据不足", pd.Series(dtype='float64')
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
            "dd_series": dd_s,
            "Beta": 0.0, "Current_Beta": 0.0,
            "Rolling_Beta_Series": pd.Series(dtype='float64')
        }

        if bench_nav is not None:
            b_sync = bench_nav.reindex(nav.index).ffill()
            b_rets = b_sync.pct_change().fillna(0)
            
            up_mask, down_mask = b_rets > 0, b_rets < 0
            up_cap = (returns[up_mask].mean() / b_rets[up_mask].mean()) if up_mask.any() else 0
            down_cap = (returns[down_mask].mean() / b_rets[down_mask].mean()) if down_mask.any() else 0
            
            cov_mat = np.cov(returns, b_rets)
            beta = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat.shape == (2, 2) and cov_mat[1, 1] != 0 else 0
            
            window = 126
            rolling_betas = []
            rolling_dates = []
            if len(returns) > window:
                for i in range(window, len(returns)):
                    r_win = returns.iloc[i-window:i]
                    b_win = b_rets.iloc[i-window:i]
                    var_b = b_win.var()
                    cov_rb = r_win.cov(b_win)
                    rb = cov_rb / var_b if var_b != 0 else 0
                    rolling_betas.append(rb)
                    rolling_dates.append(returns.index[i])
                curr_beta = rolling_betas[-1]
                rb_series = pd.Series(rolling_betas, index=rolling_dates)
            else:
                curr_beta = beta
                rb_series = pd.Series([beta]*len(returns), index=returns.index)
                
            metrics.update({
                "上行捕获": up_cap, "下行捕获": down_cap, 
                "Beta": beta, "Current_Beta": curr_beta,
                "Rolling_Beta_Series": rb_series
            })
            
        return metrics

    # ==========================================
    # 3. UI 界面与侧边栏
    # ==========================================
    st.set_page_config(layout="wide", page_title="寻星配置分析系统", page_icon="🏛️")
    st.sidebar.title("🏛️ 寻星配置分析系统")
    uploaded_file = st.sidebar.file_uploader("📂 请上传产品数据库", type=["xlsx"])

    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
        all_cols = [str(c).strip() for c in df_raw.columns]
        df_raw.columns = all_cols
        
        st.sidebar.markdown("---")
        
        # === v5.10 新增：后台费率库管理面板 ===
        with st.sidebar.expander("⚙️ 全局费率库管理 / 修订", expanded=False):
            st.caption("👇 此处可修改本产品的费率，仅对本次计算生效。")
            
            # 1. 准备数据：将 PRESET_FEES 转换为 DataFrame 方便编辑
            # 为了美观，我们把小数 (0.015) 转为 百分比 (1.5) 展示
            fee_list = []
            for name, fee in PRESET_FEES.items():
                fee_list.append({
                    "产品名称": name,
                    "年管理费(%)": fee['mgmt'] * 100,
                    "业绩报酬(%)": fee['perf'] * 100
                })
            
            # 如果有些产品在Excel里有，但在代码库里没有，可以把它们也加进来方便编辑
            known_names = set(PRESET_FEES.keys())
            for col in all_cols:
                if col not in known_names and col != '沪深300' and col != '日期':
                    fee_list.append({
                        "产品名称": col,
                        "年管理费(%)": DEFAULT_FEE['mgmt'] * 100,
                        "业绩报酬(%)": DEFAULT_FEE['perf'] * 100
                    })
            
            df_fee_edit = pd.DataFrame(fee_list).set_index("产品名称")
            
            # 2. 显示可编辑表格
            edited_fee_df = st.data_editor(
                df_fee_edit, 
                use_container_width=True,
                height=200,
                key="fee_editor" # 保证状态
            )
            
            # 3. 将编辑后的结果转回为计算字典 (百分比 -> 小数)
            ACTIVE_FEE_DICT = {}
            for name, row in edited_fee_df.iterrows():
                ACTIVE_FEE_DICT[name] = {
                    "mgmt": row["年管理费(%)"] / 100.0,
                    "perf": row["业绩报酬(%)"] / 100.0
                }
            
            st.success("✅ 费率表已加载 (修改即时生效)")
        
        # ==========================================

        default_bench = '沪深300' if '沪深300' in all_cols else all_cols[0]
        sel_bench = st.sidebar.selectbox("业绩基准", all_cols, index=all_cols.index(default_bench))
        sel_funds = st.sidebar.multiselect("挑选寻星配置组合成分", [c for c in all_cols if c != sel_bench])
        
        weights = {}
        
        fee_mode = "不考虑费率 (Gross)"
        if sel_funds:
            st.sidebar.markdown("#### ⚖️ 初始比例设定")
            avg_w = 1.0 / len(sel_funds)
            for f in sel_funds:
                weights[f] = st.sidebar.number_input(f"{f}", 0.0, 1.0, avg_w, step=0.05)
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("#### 💰 费率与净值展示模式")
            fee_mode = st.sidebar.radio(
                "选择计算模式", 
                ("不考虑费率 (Gross)", "考虑费率 (Net)", "费率磨损对比 (Analysis)"),
                index=0
            )
            if fee_mode != "不考虑费率 (Gross)":
                st.sidebar.caption("✅ 已调用【上方管理面板】中的费率进行计算")

        df_db = df_raw.loc[st.sidebar.date_input("起始日期", df_raw.index.min()):st.sidebar.date_input("截止日期", df_raw.index.max())].copy()
        
        star_nav = None
        star_nav_gross = None
        star_nav_net = None

        if sel_funds and not df_db.empty:
            df_port = df_db[sel_funds].dropna()
            
            if not df_port.empty:
                norm_w = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) > 0 else 1)
                
                # 1. Gross
                star_rets_gross = (df_port.pct_change().fillna(0) * norm_w).sum(axis=1)
                star_nav_gross = (1 + star_rets_gross).cumprod()
                star_nav_gross.name = "寻星配置组合 (费前)"

                # 2. Net
                if fee_mode != "不考虑费率 (Gross)":
                    net_funds_df = pd.DataFrame(index=df_port.index)
                    for f in sel_funds:
                        gross_series = df_port[f]
                        
                        # === 核心改动：使用 ACTIVE_FEE_DICT (即用户编辑过的表) ===
                        # 优先从编辑表里取，取不到就用默认
                        f_conf = ACTIVE_FEE_DICT.get(f, DEFAULT_FEE)
                        
                        net_series = calculate_net_nav_series(gross_series, f_conf['mgmt'], f_conf['perf'])
                        net_funds_df[f] = net_series
                    
                    star_rets_net = (net_funds_df.pct_change().fillna(0) * norm_w).sum(axis=1)
                    star_nav_net = (1 + star_rets_net).cumprod()
                    star_nav_net.name = "寻星配置组合 (费后)"

                # 3. Mode Selection
                if fee_mode == "不考虑费率 (Gross)":
                    star_nav = star_nav_gross
                else:
                    star_nav = star_nav_net
                
                bn_sync = df_db.loc[star_nav.index, sel_bench]
                bn_norm = bn_sync / bn_sync.iloc[0]

        # ==========================================
        # 4. Tabs
        # ==========================================
        tabs = st.tabs(["🚀 寻星配置组合全景图", "🔍 穿透归因分析", "⚔️ 配置池产品分析"])

        if star_nav is not None:
            m = calculate_metrics(star_nav, bn_sync)

        with tabs[0]:
            if star_nav is not None:
                title_suffix = ""
                if fee_mode == "不考虑费率 (Gross)": title_suffix = "(费前)"
                elif fee_mode == "考虑费率 (Net)": title_suffix = "(实盘费后)"
                
                st.subheader(f"📊 寻星配置组合全景图 {title_suffix}")
                
                c_top = st.columns(7)
                c_top[0].metric("总收益率", f"{m['总收益率']:.2%}")
                c_top[1].metric("年化收益", f"{m['年化收益']:.2%}")
                c_top[2].metric("最大回撤", f"{m['最大回撤']:.2%}")
                c_top[3].metric("夏普比率", f"{m['夏普比率']:.2f}")
                c_top[4].metric("索提诺", f"{m['索提诺比率']:.2f}")
                c_top[5].metric("卡玛比率", f"{m['卡玛比率']:.2f}")
                c_top[6].metric("年化波动", f"{m['年化波动率']:.2%}")
                
                fig_main = go.Figure()
                
                if fee_mode == "费率磨损对比 (Analysis)":
                    fig_main.add_trace(go.Scatter(x=star_nav_net.index, y=star_nav_net, name="寻星组合 (实盘费后)", line=dict(color='red', width=3)))
                    fig_main.add_trace(go.Scatter(x=star_nav_gross.index, y=star_nav_gross, name="寻星组合 (原始费前)", line=dict(color='gray', width=2, dash='dash')))
                    loss_amt = star_nav_gross.iloc[-1] - star_nav_net.iloc[-1]
                    loss_pct = 1 - (star_nav_net.iloc[-1] / star_nav_gross.iloc[-1])
                    st.info(f"💡 **费率磨损分析**：在当前周期内，费率导致净值少赚了 **{loss_amt:.3f}** (收益折损约 {loss_pct:.2%})。")
                else:
                    line_name = "寻星配置组合"
                    if fee_mode == "考虑费率 (Net)": line_name += " (实盘费后)"
                    fig_main.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name=line_name, line=dict(color='red', width=4)))

                fig_main.add_trace(go.Scatter(x=bn_norm.index, y=bn_norm, name=f"基准: {sel_bench}", line=dict(color='#9CA3AF', dash='dot')))
                fig_main.update_layout(title="累计净值走势", template="plotly_white", hovermode="x unified", height=450)
                st.plotly_chart(fig_main, use_container_width=True)

                st.markdown("#### 🛡️ 风险体验与风格监控")
                c_risk = st.columns(4)
                c_risk[0].metric("最大回撤修复时间", m['最大回撤修复时间'])
                c_risk[1].metric("最大无新高持续时间", m['最大无新高持续时间'])
                c_risk[2].metric("日度正收益概率", f"{m['正收益概率(日)']:.1%}")
                c_risk[3].metric("当前 Beta (近半年)", f"{m['Current_Beta']:.2f}", delta_color="off")
                
                beta_drift = abs(m['Current_Beta'] - m['Beta'])
                if beta_drift > 0.1:
                    st.warning(f"⚠️ **风格漂移预警**：当前 Beta ({m['Current_Beta']:.2f}) 与全周期均值 ({m['Beta']:.2f}) 偏差 {beta_drift:.2f} (超过阈值 0.1)，请前往 TAB 2 查看详细漂移路径。")

            else:
                st.info("👈 请在左侧侧边栏配置组合成分。")

        with tabs[1]:
            if sel_funds:
                st.subheader("🔍 寻星配置穿透归因分析")
                if fee_mode == "不考虑费率 (Gross)":
                    df_attr = df_port
                else:
                    df_attr = net_funds_df

                initial_w_series = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) > 0 else 1)
                
                growth_factors = df_attr.iloc[-1] / df_attr.iloc[0]
                latest_values = initial_w_series * growth_factors
                latest_w_series = latest_values / latest_values.sum()

                col_w1, col_w2 = st.columns(2)
                col_w1.plotly_chart(px.pie(names=initial_w_series.index, values=initial_w_series.values, hole=0.4, title="初始配置比例"), use_container_width=True)
                col_w2.plotly_chart(px.pie(names=latest_w_series.index, values=latest_w_series.values, hole=0.4, title="最新配置比例(漂移)"), use_container_width=True)

                if not m['Rolling_Beta_Series'].empty:
                    st.markdown("#### 📉 风格动态归因：Beta 漂移路径")
                    fig_beta = go.Figure()
                    fig_beta.add_trace(go.Scatter(x=m['Rolling_Beta_Series'].index, y=m['Rolling_Beta_Series'], name="滚动半年 Beta", line=dict(color='#2563EB', width=2)))
                    fig_beta.add_hline(y=m['Beta'], line_dash="dash", line_color="green", annotation_text="全周期均值 (初心)")
                    if beta_drift > 0.05: 
                         fig_beta.add_hrect(y0=m['Beta']-0.1, y1=m['Beta']+0.1, line_width=0, fillcolor="yellow", opacity=0.1, annotation_text="正常波动区间")
                    fig_beta.update_layout(template="plotly_white", height=350, hovermode="x unified")
                    st.plotly_chart(fig_beta, use_container_width=True)

                df_sub_rets = df_attr.pct_change().fillna(0)
                risk_vals = initial_w_series * (df_sub_rets.std() * np.sqrt(252))
                contribution_vals = initial_w_series * ((df_attr.iloc[-1] / df_attr.iloc[0]) - 1)

                col_attr1, col_attr2 = st.columns(2)
                col_attr1.plotly_chart(px.pie(names=risk_vals.index, values=risk_vals.values, hole=0.4, title="风险贡献归因"), use_container_width=True)
                col_attr2.plotly_chart(px.pie(names=contribution_vals.index, values=contribution_vals.abs(), hole=0.4, title="收益贡献归因"), use_container_width=True)

                st.markdown("---")
                st.markdown("#### 底层产品走势对比")
                df_sub_norm = df_attr.div(df_attr.iloc[0])
                fig_sub_compare = go.Figure()
                for col in df_sub_norm.columns:
                    fig_sub_compare.add_trace(go.Scatter(x=df_sub_norm.index, y=df_sub_norm[col], name=col, opacity=0.6))
                
                line_color = 'red' if fee_mode != "不考虑费率 (Gross)" else 'blue'
                if star_nav is not None:
                    fig_sub_compare.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name="寻星配置组合", line=dict(color=line_color, width=4)))
                st.plotly_chart(fig_sub_compare.update_layout(template="plotly_white", height=500), use_container_width=True)
                
                st.markdown("---")
                char_data = []
                for f in sel_funds:
                    f_metrics = calculate_metrics(df_attr[f], df_db[sel_bench])
                    f_metrics['产品'] = f
                    char_data.append(f_metrics)
                st.plotly_chart(px.scatter(pd.DataFrame(char_data), x="下行捕获", y="上行捕获", size="年化收益", text="产品", color="年化收益", title="产品性格象限分布", height=600), use_container_width=True)
                st.plotly_chart(px.imshow(df_sub_rets.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', title="产品相关性矩阵", height=600), use_container_width=True)

        with tabs[2]:
            st.subheader("⚔️ 配置池产品分析")
            compare_pool = st.multiselect("搜索池内产品 (费前对比)", all_cols, default=[])
            if compare_pool:
                is_aligned = st.checkbox("对齐起始日期比较", value=False)
                df_comp = df_db[compare_pool].dropna() if is_aligned else df_db[compare_pool]
                if not df_comp.empty:
                    fig_p = go.Figure()
                    for col in compare_pool:
                        s = df_comp[col].dropna()
                        if not s.empty: fig_p.add_trace(go.Scatter(x=s.index, y=s/s.iloc[0], name=col))
                    st.plotly_chart(fig_p.update_layout(title="业绩对比 (费前)", template="plotly_white", height=500), use_container_width=True)
                
                res_data = []
                for col in compare_pool:
                    k = calculate_metrics(df_db[col])
                    res_data.append({
                        "产品名称": col, "总收益": f"{k['总收益率']:.2%}", "年化": f"{k['年化收益']:.2%}", 
                        "回撤": f"{k['最大回撤']:.2%}", "夏普": round(k['夏普比率'], 2), 
                        "索提诺": round(k['索提诺比率'], 2), "卡玛": round(k['卡玛比率'], 2), 
                        "波动": f"{k['年化波动率']:.2%}", 
                        "最大回撤修复时间": k['最大回撤修复时间'], "最大无新高持续时间": k['最大无新高持续时间']
                    })
                st.dataframe(pd.DataFrame(res_data).set_index('产品名称'), use_container_width=True)
    else:
        st.info("👋 请上传‘产品数据库’。")
