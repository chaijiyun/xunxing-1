import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
from datetime import datetime

# ==========================================
# 0. 全局配置与存储架构 (CTO层)
# ==========================================
# 默认主数据 (含费率+流动性参数)
PRESET_MASTER_DEFAULT = [
    {"产品名称": "合绎期权套利", "年管理费(%)": 0.0, "业绩报酬(%)": 30.0, "开放频率": "月度", "锁定期(月)": 6, "赎回效率(T+n)": 5},
    {"产品名称": "平方和多策略6号(市场中性+多策略）", "年管理费(%)": 0.0, "业绩报酬(%)": 18.0, "开放频率": "月度", "锁定期(月)": 0, "赎回效率(T+n)": 5},
    {"产品名称": "开思沪港深优选", "年管理费(%)": 1.5, "业绩报酬(%)": 17.0, "开放频率": "月度", "锁定期(月)": 3, "赎回效率(T+n)": 7},
    {"产品名称": "蓝墨长河1号", "年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "月度", "锁定期(月)": 6, "赎回效率(T+n)": 5},
    {"产品名称": "宁泉特定策略1号", "年管理费(%)": 0.0, "业绩报酬(%)": 15.0, "开放频率": "月度", "锁定期(月)": 12, "赎回效率(T+n)": 10},
    {"产品名称": "睿郡节节高11号", "年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "月度", "锁定期(月)": 6, "赎回效率(T+n)": 5},
    {"产品名称": "宽远优势成长10号", "年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "月度", "锁定期(月)": 3, "赎回效率(T+n)": 5},
    {"产品名称": "孝庸中性策略", "年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "周度", "锁定期(月)": 0, "赎回效率(T+n)": 3},
    {"产品名称": "孝庸中性+cta", "年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "周度", "锁定期(月)": 0, "赎回效率(T+n)": 3},
    {"产品名称": "平方和市场中性", "年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "月度", "锁定期(月)": 0, "赎回效率(T+n)": 5},
    {"产品名称": "孝庸500指增", "年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "月度", "锁定期(月)": 6, "赎回效率(T+n)": 5},
    {"产品名称": "孝庸1000指增", "年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "月度", "锁定期(月)": 6, "赎回效率(T+n)": 5},
    {"产品名称": "平方和1000指数增强", "年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "月度", "锁定期(月)": 12, "赎回效率(T+n)": 5},
    {"产品名称": "合骥500对冲A期", "年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "月度", "锁定期(月)": 3, "赎回效率(T+n)": 5},
    {"产品名称": "玖鹏宏图1号", "年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "月度", "锁定期(月)": 6, "赎回效率(T+n)": 5},
]
DEFAULT_MASTER_ROW = {"年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "月度", "锁定期(月)": 6, "赎回效率(T+n)": 5}

# 初始化Session
if 'master_data' not in st.session_state:
    st.session_state.master_data = pd.DataFrame(PRESET_MASTER_DEFAULT)
if 'portfolios_data' not in st.session_state:
    st.session_state.portfolios_data = pd.DataFrame(columns=['组合名称', '产品名称', '权重'])

# ==========================================
# 1. 登录验证模块
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.markdown("<br><br>", unsafe_allow_html=True) 
        st.markdown("<h1 style='text-align: center; color: #1E40AF;'>寻星配置分析系统 v6.1.6 <small>(Fix Loop)</small></h1>", unsafe_allow_html=True)
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
    # 2. 核心计算引擎 (完全体)
    # ==========================================
    def calculate_net_nav_series(gross_nav_series, mgmt_fee_rate=0.0, perf_fee_rate=0.0):
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
            
            # Beta 滚动计算逻辑 (保持完整)
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

    def calculate_liquidity_risk(weights, master_df):
        w_series = pd.Series(weights)
        w_norm = w_series / w_series.sum()
        weighted_lockup = 0.0
        worst_lockup = 0
        liquidity_notes = []
        for p, w in w_norm.items():
            info = master_df[master_df['产品名称'] == p]
            if not info.empty:
                lock = info.iloc[0].get('锁定期(月)', 6)
                weighted_lockup += lock * w
                if lock > worst_lockup: worst_lockup = lock
                if lock >= 12: liquidity_notes.append(f"⚠️ {p}({lock}个月)")
            else:
                weighted_lockup += 6 * w 
        return weighted_lockup, worst_lockup, liquidity_notes

    # ==========================================
    # 3. UI 界面与侧边栏
    # ==========================================
    st.set_page_config(layout="wide", page_title="寻星配置分析系统 v6.1.6", page_icon="🏛️")
    st.sidebar.title("🏛️ 寻星 v6.1.6 · 驾驶舱")
    uploaded_file = st.sidebar.file_uploader("📂 第一步：上传净值数据库 (.xlsx)", type=["xlsx"])

    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
        all_cols = [str(c).strip() for c in df_raw.columns]
        df_raw.columns = all_cols
        
        st.sidebar.markdown("---")
        
        # === 配置中心 (v6.1.6 Fix Loop：openpyxl 全量备份 + 移除 rerun) ===
        with st.sidebar.expander("⚙️ 系统配置中心 (费率/组合/备份)", expanded=False):
            st.info("💡 系统采用 Excel 全量备份，包含费率与组合。")
            
            # --- 备份恢复 (Excel 版) ---
            col_bk1, col_bk2 = st.columns(2)
            uploaded_backup = col_bk1.file_uploader("📥 恢复全量备份", type=['xlsx'])
            if uploaded_backup:
                try:
                    # 读取 Master Sheet
                    df_master_new = pd.read_excel(uploaded_backup, sheet_name='Master_Data')
                    st.session_state.master_data = df_master_new
                    
                    # 读取 Portfolios Sheet (尝试读取，如果没有也不报错)
                    try:
                        df_port_new = pd.read_excel(uploaded_backup, sheet_name='Portfolios')
                        st.session_state.portfolios_data = df_port_new
                        st.toast("✅ 费率与组合数据已全部恢复！", icon="🎉")
                    except:
                        st.toast("⚠️ 仅恢复了费率，未找到组合数据。", icon="ℹ️")
                    
                    # 关键修改：删除了 st.rerun()，防止无限循环
                except Exception as e:
                    st.error(f"恢复失败: {e}")

            # 主数据编辑
            current_products = st.session_state.master_data['产品名称'].tolist()
            new_products = [p for p in all_cols if p not in current_products and p not in ['沪深300', '日期']]
            if new_products:
                new_rows = []
                for p in new_products:
                    row = DEFAULT_MASTER_ROW.copy()
                    row['产品名称'] = p
                    new_rows.append(row)
                st.session_state.master_data = pd.concat([st.session_state.master_data, pd.DataFrame(new_rows)], ignore_index=True)
            
            edited_master = st.data_editor(
                st.session_state.master_data,
                column_config={"开放频率": st.column_config.SelectboxColumn(options=["周度", "月度", "季度", "半年", "1年", "3年封闭"])},
                use_container_width=True, hide_index=True, key="master_editor_v614"
            )
            if not edited_master.equals(st.session_state.master_data):
                st.session_state.master_data = edited_master
            
            # --- 下载全量备份 (Excel 版 - 修复引擎为 openpyxl) ---
            # 使用 BytesIO 生成内存中的 Excel 文件
            buffer = io.BytesIO()
            # 关键修改：使用 openpyxl 引擎，避免云端安装 xlsxwriter 失败
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.master_data.to_excel(writer, sheet_name='Master_Data', index=False)
                st.session_state.portfolios_data.to_excel(writer, sheet_name='Portfolios', index=False)
            
            st.download_button(
                label="💾 下载全量数据备份 (.xlsx)",
                data=buffer,
                file_name="寻星_全量系统备份.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # 字典化
            MASTER_DICT = {}
            for _, row in st.session_state.master_data.iterrows():
                MASTER_DICT[row['产品名称']] = row.to_dict()

        st.sidebar.markdown("---")
        
        # === 组合管理 (保持逻辑) ===
        st.sidebar.markdown("### 💼 组合配置")
        saved_names = st.session_state.portfolios_data['组合名称'].unique().tolist() if not st.session_state.portfolios_data.empty else []
        mode_options = ["🛠️ 自定义/新建"] + saved_names
        selected_mode = st.sidebar.selectbox("选择模式:", mode_options)
        
        sel_funds = []
        weights = {}
        default_bench = '沪深300' if '沪深300' in all_cols else all_cols[0]
        sel_bench = st.sidebar.selectbox("业绩基准", all_cols, index=all_cols.index(default_bench))
        
        if selected_mode == "🛠️ 自定义/新建":
            available_funds = [c for c in all_cols if c != sel_bench]
            available_funds.sort()
            sel_funds = st.sidebar.multiselect("挑选成分基金", available_funds)
            if sel_funds:
                st.sidebar.markdown("#### ⚖️ 权重")
                avg_w = 1.0 / len(sel_funds)
                for f in sel_funds: weights[f] = st.sidebar.number_input(f"{f}", 0.0, 1.0, avg_w, step=0.05)
                
                with st.sidebar.expander("💾 保存组合", expanded=True):
                    new_p_name = st.text_input("组合名称", placeholder="如: 稳健1号")
                    if st.button("保存"):
                        if new_p_name and sel_funds:
                            new_records = [{'组合名称': new_p_name, '产品名称': f, '权重': w} for f, w in weights.items()]
                            old_df = st.session_state.portfolios_data
                            new_df = pd.DataFrame(new_records)
                            updated_df = pd.concat([old_df[old_df['组合名称']!=new_p_name], new_df], ignore_index=True)
                            st.session_state.portfolios_data = updated_df
                            st.toast(f"组合 {new_p_name} 已保存 (请记得下载备份)", icon="✅")
                            st.rerun()
        else:
            subset = st.session_state.portfolios_data[st.session_state.portfolios_data['组合名称'] == selected_mode]
            valid_subset = subset[subset['产品名称'].isin(all_cols)]
            sel_funds = valid_subset['产品名称'].tolist()
            weights = {row['产品名称']: row['权重'] for _, row in valid_subset.iterrows()}
            st.sidebar.table(valid_subset[['产品名称', '权重']].set_index('产品名称').style.format("{:.1%}"))
            if st.sidebar.button("🗑️ 删除此组合"):
                updated = st.session_state.portfolios_data[st.session_state.portfolios_data['组合名称'] != selected_mode]
                st.session_state.portfolios_data = updated
                st.rerun()

        # 颜色与费率模式
        color_map = {}
        if sel_funds:
            colors = px.colors.qualitative.Plotly 
            for i, f in enumerate(sel_funds): color_map[f] = colors[i % len(colors)]

        st.sidebar.markdown("---")
        fee_mode_label = "客户实得回报 (实盘费后)"
        if sel_funds:
            fee_mode_label = st.sidebar.radio("展示视角", ("客户实得回报 (实盘费后)", "组合策略表现 (底层净值)", "收益与运作成本分析"), index=0)

        # ==========================================
        # 计算逻辑
        # ==========================================
        df_db = df_raw.loc[st.sidebar.date_input("起始日期", df_raw.index.min()):st.sidebar.date_input("截止日期", df_raw.index.max())].copy()
        star_nav = None; star_nav_gross = None; star_nav_net = None

        if sel_funds and not df_db.empty:
            df_port = df_db[sel_funds].dropna()
            if not df_port.empty:
                norm_w = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) > 0 else 1)
                
                # Gross
                star_rets_gross = (df_port.pct_change().fillna(0) * norm_w).sum(axis=1)
                star_nav_gross = (1 + star_rets_gross).cumprod()
                star_nav_gross.name = "组合策略表现 (底层净值)"

                # Net
                if fee_mode_label != "组合策略表现 (底层净值)":
                    net_funds_df = pd.DataFrame(index=df_port.index)
                    for f in sel_funds:
                        gross_series = df_port[f]
                        # 核心修改：从主数据字典获取费率
                        info = MASTER_DICT.get(f, DEFAULT_MASTER_ROW)
                        mgmt = info.get('年管理费(%)', 0) / 100.0
                        perf = info.get('业绩报酬(%)', 0) / 100.0
                        net_funds_df[f] = calculate_net_nav_series(gross_series, mgmt, perf)
                    star_rets_net = (net_funds_df.pct_change().fillna(0) * norm_w).sum(axis=1)
                    star_nav_net = (1 + star_rets_net).cumprod()
                    star_nav_net.name = "客户实得回报 (费后)"

                star_nav = star_nav_gross if fee_mode_label == "组合策略表现 (底层净值)" else star_nav_net
                bn_sync = df_db.loc[star_nav.index, sel_bench]
                bn_norm = bn_sync / bn_sync.iloc[0]

        # ==========================================
        # Tabs 可视化 (v6.1一致)
        # ==========================================
        tabs = st.tabs(["🚀 组合全景图", "🔍 穿透归因分析", "⚔️ 配置池产品分析"])

        if star_nav is not None:
            m = calculate_metrics(star_nav, bn_sync)
            avg_lock, worst_lock, lock_notes = calculate_liquidity_risk(weights, st.session_state.master_data)

        with tabs[0]:
            if star_nav is not None:
                st.subheader(f"📊 {star_nav.name}")
                
                # 指标行 (保持 v5.20 7个指标)
                c_top = st.columns(7)
                c_top[0].metric("总收益率", f"{m['总收益率']:.2%}")
                c_top[1].metric("年化收益", f"{m['年化收益']:.2%}")
                c_top[2].metric("最大回撤", f"{m['最大回撤']:.2%}")
                c_top[3].metric("夏普比率", f"{m['夏普比率']:.2f}")
                c_top[4].metric("索提诺", f"{m['索提诺比率']:.2f}")
                c_top[5].metric("卡玛比率", f"{m['卡玛比率']:.2f}")
                c_top[6].metric("年化波动", f"{m['年化波动率']:.2%}")
                
                # 主图
                fig_main = go.Figure()
                if fee_mode_label == "收益与运作成本分析":
                    fig_main.add_trace(go.Scatter(x=star_nav_net.index, y=star_nav_net, name="客户实得权益 (红线)", line=dict(color='red', width=3)))
                    fig_main.add_trace(go.Scatter(x=star_nav_gross.index, y=star_nav_gross, name="策略名义表现 (灰线)", line=dict(color='gray', width=2, dash='dash')))
                    loss_amt = star_nav_gross.iloc[-1] - star_nav_net.iloc[-1]
                    loss_pct = 1 - (star_nav_net.iloc[-1] / star_nav_gross.iloc[-1])
                    st.info(f"💡 **成本分析**：在此期间，组合的策略运作与配置服务成本约为 **{loss_amt:.3f}** (费效比 {loss_pct:.2%})。")
                else:
                    fig_main.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name=star_nav.name, line=dict(color='red', width=4)))
                fig_main.add_trace(go.Scatter(x=bn_norm.index, y=bn_norm, name=f"基准: {sel_bench}", line=dict(color='#9CA3AF', dash='dot')))
                fig_main.update_layout(title="账户权益走势", template="plotly_white", hovermode="x unified", height=450)
                st.plotly_chart(fig_main, use_container_width=True)

                # 风控行
                st.markdown("#### 🛡️ 风险体验与风格监控")
                c_risk = st.columns(5) 
                c_risk[0].metric("最大回撤修复", m['最大回撤修复时间'])
                c_risk[1].metric("最长创新高间隔", m['最大无新高持续时间'])
                c_risk[2].metric("日胜率", f"{m['正收益概率(日)']:.1%}")
                c_risk[3].metric("Current Beta", f"{m['Current_Beta']:.2f}")
                c_risk[4].metric("平均锁定期", f"{avg_lock:.1f}个月", help="[CIO风控] 加权平均锁定期")
                
                # 漂移与流动性警报
                beta_drift = abs(m['Current_Beta'] - m['Beta'])
                if beta_drift > 0.1: st.warning(f"⚠️ **风格漂移预警**：Beta 偏差 {beta_drift:.2f}。")
                if lock_notes: st.warning(f"⚠️ **流动性警示**：{' '.join(lock_notes)}")

            else: st.info("👈 请在左侧选择或加载组合。")

        with tabs[1]:
            if sel_funds:
                st.subheader("🔍 寻星配置穿透归因分析")
                if fee_mode_label == "组合策略表现 (底层净值)": df_attr = df_port
                else: df_attr = net_funds_df
                initial_w_series = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) > 0 else 1)
                growth_factors = df_attr.iloc[-1] / df_attr.iloc[0]
                latest_values = initial_w_series * growth_factors
                latest_w_series = latest_values / latest_values.sum()

                col_w1, col_w2 = st.columns(2)
                col_w1.plotly_chart(px.pie(names=initial_w_series.index, values=initial_w_series.values, hole=0.4, title="初始配置比例", color=initial_w_series.index, color_discrete_map=color_map), use_container_width=True)
                col_w2.plotly_chart(px.pie(names=latest_w_series.index, values=latest_w_series.values, hole=0.4, title="最新配置比例(漂移)", color=latest_w_series.index, color_discrete_map=color_map), use_container_width=True)

                if not m['Rolling_Beta_Series'].empty:
                    st.markdown("#### 📉 风格动态归因：Beta 漂移路径")
                    fig_beta = go.Figure()
                    fig_beta.add_trace(go.Scatter(x=m['Rolling_Beta_Series'].index, y=m['Rolling_Beta_Series'], name="滚动半年 Beta", line=dict(color='#2563EB', width=2)))
                    fig_beta.add_hline(y=m['Beta'], line_dash="dash", line_color="green", annotation_text="全周期均值")
                    fig_beta.update_layout(template="plotly_white", height=350, hovermode="x unified")
                    st.plotly_chart(fig_beta, use_container_width=True)

                df_sub_rets = df_attr.pct_change().fillna(0)
                risk_vals = initial_w_series * (df_sub_rets.std() * np.sqrt(252))
                contribution_vals = initial_w_series * ((df_attr.iloc[-1] / df_attr.iloc[0]) - 1)

                col_attr1, col_attr2 = st.columns(2)
                col_attr1.plotly_chart(px.pie(names=risk_vals.index, values=risk_vals.values, hole=0.4, title="风险贡献归因", color=risk_vals.index, color_discrete_map=color_map), use_container_width=True)
                col_attr2.plotly_chart(px.pie(names=contribution_vals.index, values=contribution_vals.abs(), hole=0.4, title="收益贡献归因", color=contribution_vals.index, color_discrete_map=color_map), use_container_width=True)

                st.markdown("---")
                st.markdown("#### 底层产品走势对比")
                df_sub_norm = df_attr.div(df_attr.iloc[0])
                fig_sub_compare = go.Figure()
                for col in df_sub_norm.columns:
                    fig_sub_compare.add_trace(go.Scatter(x=df_sub_norm.index, y=df_sub_norm[col], name=col, opacity=0.6, line=dict(color=color_map.get(col))))
                if star_nav is not None:
                    fig_sub_compare.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name=star_nav.name, line=dict(color='red', width=4)))
                st.plotly_chart(fig_sub_compare.update_layout(template="plotly_white", height=500), use_container_width=True)
                
                st.markdown("---")
                char_data = []
                for f in sel_funds:
                    f_metrics = calculate_metrics(df_attr[f], df_db[sel_bench])
                    f_metrics['产品'] = f
                    char_data.append(f_metrics)
                st.plotly_chart(px.scatter(pd.DataFrame(char_data), x="下行捕获", y="上行捕获", size="年化收益", text="产品", color="产品", color_discrete_map=color_map, title="产品性格象限分布", height=600), use_container_width=True)
                st.plotly_chart(px.imshow(df_sub_rets.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', title="产品相关性矩阵", height=600), use_container_width=True)

        with tabs[2]:
            st.subheader("⚔️ 配置池产品分析")
            pool_options = [c for c in all_cols if c != sel_bench]
            pool_options.sort()
            compare_pool = st.multiselect("搜索池内产品 (费前对比)", pool_options, default=[])
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
                        k = calculate_metrics(df_comp[col]) 
                        if k: res_data.append({"产品名称": col, "总收益": f"{k['总收益率']:.2%}", "年化收益": f"{k['年化收益']:.2%}", "最大回撤": f"{k['最大回撤']:.2%}", "夏普": round(k['夏普比率'], 2), "索提诺": round(k['索提诺比率'], 2), "卡玛": round(k['卡玛比率'], 2), "波动率": f"{k['年化波动率']:.2%}", "最大回撤修复时间": k['最大回撤修复时间'], "最大无新高持续时间": k['最大无新高持续时间']})
                    if res_data: st.dataframe(pd.DataFrame(res_data).set_index('产品名称'), use_container_width=True)
                    
                    st.markdown("#### 📅 分年度收益率统计")
                    yearly_data = {}
                    for col in compare_pool:
                        s = df_comp[col].dropna()
                        groups = s.groupby(s.index.year)
                        y_vals = {}
                        for year, group in groups: y_vals[year] = (group.iloc[-1] / group.iloc[0]) - 1
                        yearly_data[col] = y_vals
                    if yearly_data: st.dataframe(pd.DataFrame(yearly_data).T.sort_index().style.format("{:.2%}"), use_container_width=True)
                else: st.warning("⚠️ 数据不足")
    else: st.info("👋 请上传‘产品数据库’。")
