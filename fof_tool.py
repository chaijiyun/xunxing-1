import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import os
from datetime import datetime, timedelta

# ==========================================
# 寻星配置分析系统 v7.0.2 - Core Logic
# Author: 寻星架构师
# Context: Web全栈 / 量化金融 / 极度求真
# Update: 修复 TAB3 年度收益计算逻辑 (解决跨年缝隙问题)
# ==========================================

# ------------------------------------------
# 0. 全局常量与预设 (Configuration)
# ------------------------------------------
CONFIG_FILE_PATH = "xunxing_config.pkl"

# [Factory Reset] 出厂预设值 (基于最新提供的费率表)
PRESET_MASTER_DEFAULT = [
    {'产品名称': '国富瑞合1号', '年管理费(%)': 0, '业绩报酬(%)': 16, '开放频率': '周度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '合骥500对冲A期', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '合绎期权套利', '年管理费(%)': 0, '业绩报酬(%)': 30, '开放频率': '月度', '锁定期(月)': 6, '赎回效率(T+n)': 4},
    {'产品名称': '玖鹏宏图1号', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '开思沪港深优选', '年管理费(%)': 0, '业绩报酬(%)': 17, '开放频率': '月度', '锁定期(月)': 1, '赎回效率(T+n)': 4},
    {'产品名称': '宽远优势成长10号', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '蓝墨长河1号', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 1, '赎回效率(T+n)': 4},
    {'产品名称': '宁泉特定策略1号', '年管理费(%)': 0, '业绩报酬(%)': 15, '开放频率': '月度', '锁定期(月)': 12, '赎回效率(T+n)': 4},
    {'产品名称': '平方和1000指数增强', '年管理费(%)': 0, '业绩报酬(%)': 16, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '平方和多策略', '年管理费(%)': 0, '业绩报酬(%)': 16, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '平方和量化选股', '年管理费(%)': 0, '业绩报酬(%)': 16, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '平方和市场中性', '年管理费(%)': 0, '业绩报酬(%)': 16, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '秦川1号', '年管理费(%)': 0, '业绩报酬(%)': 15, '开放频率': '周度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '睿郡节节高11号', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 6, '赎回效率(T+n)': 4},
    {'产品名称': '孝庸1000指增', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 12, '赎回效率(T+n)': 4},
    {'产品名称': '孝庸500指增', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 12, '赎回效率(T+n)': 4},
    {'产品名称': '孝庸中性+cta', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '周度', '锁定期(月)': 12, '赎回效率(T+n)': 4},
    {'产品名称': '孝庸中性策略', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '周度', '锁定期(月)': 12, '赎回效率(T+n)': 4},
    {'产品名称': '孝庸量选', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '周度', '锁定期(月)': 12, '赎回效率(T+n)': 4},
]
DEFAULT_MASTER_ROW = {"年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "月度", "锁定期(月)": 6, "赎回效率(T+n)": 5}

# ------------------------------------------
# 1. 持久化引擎 (Persistence Engine)
# ------------------------------------------
def load_local_config():
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            return pd.read_pickle(CONFIG_FILE_PATH)
        except Exception:
            return pd.DataFrame(PRESET_MASTER_DEFAULT)
    return pd.DataFrame(PRESET_MASTER_DEFAULT)

def save_local_config(df):
    try:
        df.to_pickle(CONFIG_FILE_PATH)
    except Exception as e:
        st.error(f"配置保存失败: {e}")

if 'master_data' not in st.session_state:
    st.session_state.master_data = load_local_config()
if 'portfolios_data' not in st.session_state:
    st.session_state.portfolios_data = pd.DataFrame(columns=['组合名称', '产品名称', '权重'])

# ------------------------------------------
# 2. 登录与安全 (Security)
# ------------------------------------------
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.markdown("<br><br>", unsafe_allow_html=True) 
        st.markdown("<h1 style='text-align: center; color: #1E40AF;'>寻星配置分析系统 v7.0.2 <small>(Stable)</small></h1>", unsafe_allow_html=True)
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
                        st.error("密码错误：访问拒绝。")
        return False
    return True

if check_password():
    # ------------------------------------------
    # 3. 核心计算引擎 (Calculation Engine)
    # ------------------------------------------
    
    def calculate_net_nav_series(gross_nav_series, mgmt_fee_rate=0.0, perf_fee_rate=0.0):
        if gross_nav_series.empty: return gross_nav_series
        dates = gross_nav_series.index
        gross_vals = gross_nav_series.values
        entry_price = gross_vals[0] 
        net_vals = np.zeros(len(gross_vals))
        net_vals[0] = entry_price 
        asset_after_mgmt = np.zeros(len(gross_vals))
        asset_after_mgmt[0] = entry_price
        prev_date = dates[0]
        for i in range(1, len(gross_vals)):
            r_interval = gross_vals[i] / gross_vals[i-1] - 1
            curr_date = dates[i]
            days_delta = (curr_date - prev_date).days
            mgmt_cost = mgmt_fee_rate * (days_delta / 365.0)
            asset_after_mgmt[i] = asset_after_mgmt[i-1] * (1 + r_interval - mgmt_cost)
            prev_date = curr_date
        profits = asset_after_mgmt - entry_price
        liabilities = np.where(profits > 0, profits * perf_fee_rate, 0.0)
        net_vals = asset_after_mgmt - liabilities
        net_vals = np.maximum(net_vals, 0)
        return pd.Series(net_vals, index=dates)

    def get_drawdown_details(nav_series):
        if nav_series.empty or len(nav_series) < 2: 
            return "数据不足", "数据不足", pd.Series(dtype='float64')
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax 
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
        dates = nav.index
        days_diff = (dates[-1] - dates[0]).days
        if days_diff <= 0: return {}
        count = len(dates) - 1
        avg_interval = days_diff / count if count > 0 else 1
        if avg_interval <= 1.5: freq_factor = 252.0
        elif avg_interval <= 8: freq_factor = 52.0 
        elif avg_interval <= 35: freq_factor = 12.0
        else: freq_factor = 252.0 / avg_interval
        
        returns = nav.pct_change().dropna()
        total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
        ann_ret = (1 + total_ret) ** (365.25 / days_diff) - 1
        vol = returns.std() * np.sqrt(freq_factor)
        mdd_rec, max_nh, dd_s = get_drawdown_details(nav)
        mdd = dd_s.min()
        rf = 0.019 
        sharpe = (ann_ret - rf) / vol if vol > 0 else 0
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(freq_factor) if not downside_returns.empty else 1e-6
        sortino = (ann_ret - rf) / downside_std
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0
        win_days = returns[returns > 0]
        loss_days = returns[returns < 0]
        win_rate = len(win_days) / len(returns) if len(returns) > 0 else 0
        avg_win = win_days.mean() if not win_days.empty else 0
        avg_loss = abs(loss_days.mean()) if not loss_days.empty else 0
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        var_95 = np.percentile(returns, 5) 

        metrics = {
            "总收益率": total_ret, "年化收益": ann_ret, "最大回撤": mdd, 
            "夏普比率": sharpe, "索提诺比率": sortino, "卡玛比率": calmar, "年化波动率": vol,
            "最大回撤修复时间": mdd_rec, "最大无新高持续时间": max_nh,
            "正收益概率(日)": win_rate, "盈亏比": pl_ratio, "VaR(95%)": var_95,
            "dd_series": dd_s, "Beta": 0.0, "Current_Beta": 0.0, "Alpha": 0.0,
            "上行捕获": 0.0, "下行捕获": 0.0, "Rolling_Beta_Series": pd.Series(dtype='float64')
        }
        
        if bench_nav is not None:
            common_idx = nav.index.intersection(bench_nav.index)
            if len(common_idx) > 10:
                p_rets = nav.loc[common_idx].pct_change().dropna()
                b_rets = bench_nav.loc[common_idx].pct_change().dropna()
                valid_idx = p_rets.index.intersection(b_rets.index)
                p_rets = p_rets.loc[valid_idx]
                b_rets = b_rets.loc[valid_idx]
                if not p_rets.empty:
                    cov_mat = np.cov(p_rets, b_rets)
                    beta = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat.shape == (2, 2) and cov_mat[1, 1] != 0 else 0
                    bench_total_ret = (bench_nav.loc[common_idx[-1]]/bench_nav.loc[common_idx[0]])**(365.25/(common_idx[-1]-common_idx[0]).days) - 1
                    alpha = ann_ret - (rf + beta * (bench_total_ret - rf))
                    window = int(freq_factor / 2)
                    if window < 10: window = 10
                    rolling_betas = []
                    rolling_dates = []
                    if len(p_rets) > window:
                        for i in range(window, len(p_rets)):
                            r_win = p_rets.iloc[i-window:i]
                            b_win = b_rets.iloc[i-window:i]
                            var_b = b_win.var()
                            cov_rb = r_win.cov(b_win)
                            rb = cov_rb / var_b if var_b != 0 else 0
                            rolling_betas.append(rb)
                            rolling_dates.append(p_rets.index[i])
                        curr_beta = rolling_betas[-1] if rolling_betas else beta
                        rb_series = pd.Series(rolling_betas, index=rolling_dates)
                    else:
                        curr_beta = beta
                        rb_series = pd.Series([beta]*len(p_rets), index=p_rets.index)
                    up_mask = b_rets > 0
                    down_mask = b_rets < 0
                    up_cap = (p_rets[up_mask].mean() / b_rets[up_mask].mean()) if up_mask.any() and abs(b_rets[up_mask].mean()) > 1e-6 else 0
                    down_cap = (p_rets[down_mask].mean() / b_rets[down_mask].mean()) if down_mask.any() and abs(b_rets[down_mask].mean()) > 1e-6 else 0
                    metrics.update({
                        "上行捕获": up_cap, "下行捕获": down_cap, 
                        "Beta": beta, "Current_Beta": curr_beta, "Alpha": alpha,
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

    # [Improved] 智能频率感知的蒙特卡洛引擎
    def run_monte_carlo(historical_returns, n_simulations=1000, n_years=3, initial_capital=1000000):
        if historical_returns.empty or len(historical_returns) < 10: return None, 0, 0, ""
        try:
            dates = historical_returns.index
            days_interval = (dates[-1] - dates[0]).days / len(dates)
        except:
            days_interval = 1
        if days_interval > 4:
            freq = 52.0
            dt_label = "周"
        else:
            freq = 252.0
            dt_label = "天"
        mu = historical_returns.mean() * freq 
        sigma = historical_returns.std() * np.sqrt(freq)
        dt = 1 / freq
        n_steps = int(n_years * freq)
        S = np.zeros((n_steps + 1, n_simulations))
        S[0] = initial_capital
        Z = np.random.normal(0, 1, (n_steps, n_simulations))
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        daily_returns = np.exp(drift + diffusion)
        path_matrix = initial_capital * np.cumprod(np.vstack([np.ones((1, n_simulations)), daily_returns]), axis=0)
        last_date = historical_returns.index[-1]
        step_days = 7 if freq == 52 else 1
        future_dates = [last_date + timedelta(days=x*step_days) for x in range(n_steps + 1)]
        return pd.DataFrame(path_matrix, index=future_dates), mu, sigma, dt_label

    # ------------------------------------------
    # 4. UI 界面与交互 (Interface)
    # ------------------------------------------
    st.set_page_config(layout="wide", page_title="寻星配置分析系统 v7.0.2", page_icon="🏛️")
    st.sidebar.title("🏛️ 寻星 v7.0.2 · 驾驶舱")
    uploaded_file = st.sidebar.file_uploader("📂 第一步：上传净值数据库 (.xlsx)", type=["xlsx"])

    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
        all_cols = [str(c).strip() for c in df_raw.columns]
        df_raw.columns = all_cols
        
        st.sidebar.markdown("---")
        
        # === 配置中心 ===
        with st.sidebar.expander("⚙️ 系统配置中心 (费率/组合/备份)", expanded=False):
            st.info("💡 系统已启用自动记忆：您在此处的修改会自动保存。")
            col_bk1, col_bk2 = st.columns(2)
            uploaded_backup = col_bk1.file_uploader("📥 恢复全量备份", type=['xlsx'])
            if uploaded_backup:
                try:
                    df_master_new = pd.read_excel(uploaded_backup, sheet_name='Master_Data')
                    st.session_state.master_data = df_master_new
                    save_local_config(df_master_new)
                    try:
                        df_port_new = pd.read_excel(uploaded_backup, sheet_name='Portfolios')
                        st.session_state.portfolios_data = df_port_new
                        st.toast("✅ 数据恢复并保存成功", icon="🎉")
                    except: st.toast("⚠️ 仅恢复了费率", icon="ℹ️")
                except Exception as e: st.error(f"恢复失败: {e}")

            current_products = st.session_state.master_data['产品名称'].tolist()
            new_products = [p for p in all_cols if p not in current_products and p not in ['沪深300', '日期']]
            if new_products:
                new_rows = []
                for p in new_products:
                    row = DEFAULT_MASTER_ROW.copy()
                    row['产品名称'] = p
                    new_rows.append(row)
                st.session_state.master_data = pd.concat([st.session_state.master_data, pd.DataFrame(new_rows)], ignore_index=True)
                save_local_config(st.session_state.master_data)
            
            edited_master = st.data_editor(
                st.session_state.master_data,
                column_config={"开放频率": st.column_config.SelectboxColumn(options=["周度", "月度", "季度", "半年", "1年", "3年封闭"])},
                use_container_width=True, hide_index=True, key="master_editor_v702"
            )
            if not edited_master.equals(st.session_state.master_data):
                st.session_state.master_data = edited_master
                save_local_config(edited_master)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.master_data.to_excel(writer, sheet_name='Master_Data', index=False)
                st.session_state.portfolios_data.to_excel(writer, sheet_name='Portfolios', index=False)
            st.download_button("💾 下载全量数据备份 (.xlsx)", data=buffer, file_name="寻星_全量系统备份.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            MASTER_DICT = {}
            for _, row in st.session_state.master_data.iterrows():
                MASTER_DICT[row['产品名称']] = row.to_dict()

        st.sidebar.markdown("---")
        
        # === 组合管理 ===
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
                            st.toast(f"组合 {new_p_name} 已保存", icon="✅")
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

        # Color & Fee Mode
        color_map = {}
        if sel_funds:
            colors = px.colors.qualitative.Plotly 
            for i, f in enumerate(sel_funds): color_map[f] = colors[i % len(colors)]

        st.sidebar.markdown("---")
        fee_mode_label = "客户实得回报 (实盘费后)"
        if sel_funds:
            fee_mode_label = st.sidebar.radio("展示视角", ("客户实得回报 (实盘费后)", "组合策略表现 (底层净值)", "收益与运作成本分析"), index=0)

        # ==========================================
        # 计算逻辑执行
        # ==========================================
        df_db = df_raw.loc[st.sidebar.date_input("起始日期", df_raw.index.min()):st.sidebar.date_input("截止日期", df_raw.index.max())].copy()
        star_nav = None; star_nav_gross = None; star_nav_net = None
        star_rets_for_mc = None 

        if sel_funds and not df_db.empty:
            df_port = df_db[sel_funds].ffill().dropna(how='all')
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
                        info = MASTER_DICT.get(f, DEFAULT_MASTER_ROW)
                        mgmt = info.get('年管理费(%)', 0) / 100.0
                        perf = info.get('业绩报酬(%)', 0) / 100.0
                        net_funds_df[f] = calculate_net_nav_series(gross_series, mgmt, perf)
                    
                    star_rets_net = (net_funds_df.pct_change().fillna(0) * norm_w).sum(axis=1)
                    star_nav_net = (1 + star_rets_net).cumprod()
                    star_nav_net.name = "寻星配置实得回报"
                    star_rets_for_mc = star_rets_net 
                else:
                    star_rets_for_mc = star_rets_gross 

                star_nav = star_nav_gross if fee_mode_label == "组合策略表现 (底层净值)" else star_nav_net
                bn_sync = df_db.loc[star_nav.index, sel_bench]
                bn_norm = bn_sync / bn_sync.iloc[0]

        # ==========================================
        # 可视化 (Visualization)
        # ==========================================
        tabs = st.tabs(["🚀 组合全景图", "🔍 穿透归因分析", "⚔️ 配置池产品分析", "🔮 蒙特卡洛模拟"])

        if star_nav is not None:
            m = calculate_metrics(star_nav, bn_sync)
            avg_lock, worst_lock, lock_notes = calculate_liquidity_risk(weights, st.session_state.master_data)

        with tabs[0]:
            if star_nav is not None:
                st.subheader(f"📊 {star_nav.name}")
                c_top = st.columns(8)
                c_top[0].metric("总收益率", f"{m['总收益率']:.2%}")
                c_top[1].metric("年化收益", f"{m['年化收益']:.2%}")
                c_top[2].metric("最大回撤", f"{m['最大回撤']:.2%}")
                c_top[3].metric("夏普比率", f"{m['夏普比率']:.2f}")
                c_top[4].metric("索提诺", f"{m['索提诺比率']:.2f}")
                c_top[5].metric("卡玛比率", f"{m['卡玛比率']:.2f}")
                c_top[6].metric("年化波动", f"{m['年化波动率']:.2%}")
                c_top[7].metric("组合Beta", f"{m['Beta']:.2f}", help="组合全周期历史Beta")
                
                fig_main = go.Figure()
                if fee_mode_label == "收益与运作成本分析":
                    fig_main.add_trace(go.Scatter(x=star_nav_net.index, y=star_nav_net, name="寻星配置实得回报", line=dict(color='red', width=3)))
                    fig_main.add_trace(go.Scatter(x=star_nav_gross.index, y=star_nav_gross, name="策略名义表现 (灰线)", line=dict(color='gray', width=2, dash='dash')))
                    loss_amt = star_nav_gross.iloc[-1] - star_nav_net.iloc[-1]
                    loss_pct = 1 - (star_nav_net.iloc[-1] / star_nav_gross.iloc[-1])
                    st.info(f"💡 **成本分析**：在此期间，组合的策略运作与配置服务成本约为 **{loss_amt:.3f}** (费效比 {loss_pct:.2%})。")
                else:
                    fig_main.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name=star_nav.name, line=dict(color='red', width=4)))
                fig_main.add_trace(go.Scatter(x=bn_norm.index, y=bn_norm, name=f"基准: {sel_bench}", line=dict(color='#1F2937', width=2, dash='solid'), opacity=0.6))
                fig_main.update_layout(title="账户权益走势", template="plotly_white", hovermode="x unified", height=450)
                st.plotly_chart(fig_main, use_container_width=True)

                st.markdown("#### 🛡️ 风险体验与风格监控")
                c_risk = st.columns(5) 
                c_risk[0].metric("最大回撤修复", m['最大回撤修复时间'])
                c_risk[1].metric("最长创新高间隔", m['最大无新高持续时间'])
                c_risk[2].metric("盈亏比", f"{m['盈亏比']:.2f}", help="平均盈利/平均亏损")
                c_risk[3].metric("Current Beta", f"{m['Current_Beta']:.2f}", help="组合近半年滚动Beta (当前状态)")
                c_risk[4].metric("VaR (95%)", f"{m['VaR(95%)']:.2%}", help="历史最差5%的日均亏损")
                beta_drift = abs(m['Current_Beta'] - m['Beta'])
                if beta_drift > 0.1: st.warning(f"⚠️ **风格漂移预警**：Beta 偏差 {beta_drift:.2f} (初心 {m['Beta']:.2f} vs 现状 {m['Current_Beta']:.2f})。")
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
                st.plotly_chart(px.imshow(df_sub_rets.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title="产品相关性矩阵 (Pearson)", height=600), use_container_width=True)

        with tabs[2]:
            c_t1, c_t2 = st.columns([3, 1])
            with c_t1: st.subheader("⚔️ 配置池产品分析")
            with c_t2: comp_fee_mode = st.selectbox("展示视角", ["费前 (Gross)", "费后 (Net)"], index=0)
            pool_options = [c for c in all_cols if c != sel_bench]
            pool_options.sort()
            compare_pool = st.multiselect("搜索池内产品", pool_options, default=[])
            if compare_pool:
                is_aligned = st.checkbox("对齐起始日期比较", value=False)
                df_comp_raw = df_db[compare_pool].dropna() if is_aligned else df_db[compare_pool]
                if comp_fee_mode == "费后 (Net)":
                    df_comp = pd.DataFrame(index=df_comp_raw.index)
                    for p in compare_pool:
                        s_raw = df_comp_raw[p].dropna()
                        if s_raw.empty: continue
                        info = MASTER_DICT.get(p, DEFAULT_MASTER_ROW)
                        m_rate = info.get('年管理费(%)', 0) / 100.0
                        p_rate = info.get('业绩报酬(%)', 0) / 100.0
                        df_comp[p] = calculate_net_nav_series(s_raw, m_rate, p_rate)
                else: df_comp = df_comp_raw
                if not df_comp.empty:
                    fig_p = go.Figure()
                    for col in compare_pool:
                        if col in df_comp.columns:
                            s = df_comp[col].dropna()
                            if not s.empty: fig_p.add_trace(go.Scatter(x=s.index, y=s/s.iloc[0], name=col))
                    st.plotly_chart(fig_p.update_layout(title=f"业绩对比 ({comp_fee_mode})", template="plotly_white", height=500), use_container_width=True)
                    res_data = []
                    for col in compare_pool:
                        if col in df_comp.columns:
                            k = calculate_metrics(df_comp[col], df_db[sel_bench]) 
                            if k: 
                                res_data.append({"产品名称": col, "总收益": f"{k['总收益率']:.2%}", "年化收益": f"{k['年化收益']:.2%}", "最大回撤": f"{k['最大回撤']:.2%}", "夏普": round(k['夏普比率'], 2), "盈亏比": f"{k['盈亏比']:.2f}", "胜率": f"{k['正收益概率(日)']:.1%}", "VaR(95%)": f"{k['VaR(95%)']:.2%}", "上行捕获": f"{k['上行捕获']:.2f}", "下行捕获": f"{k['下行捕获']:.2f}", "Alpha": f"{k['Alpha']:.2%}", "Beta": f"{k['Beta']:.2f}"})
                    if res_data: st.dataframe(pd.DataFrame(res_data).set_index('产品名称'), use_container_width=True)
                    
                    # -------------------------------------------------------
                    # [Fix] 修复年度收益计算逻辑 v7.0.2
                    # 旧逻辑会丢失跨年周的收益，新逻辑基于 "End/Prev_End"
                    # -------------------------------------------------------
                    st.markdown("#### 📅 分年度收益率统计")
                    yearly_data = {}
                    for col in compare_pool:
                        if col in df_comp.columns:
                            s = df_comp[col].dropna()
                            if s.empty: continue
                            
                            # 1. 取每年的最后一个净值
                            try:
                                yearly_nav = s.resample('YE').last() # Pandas 2.x
                            except:
                                yearly_nav = s.resample('A').last()  # Pandas 1.x Fallback
                                
                            if yearly_nav.empty: continue

                            # 2. 计算基于上年末的涨跌幅
                            yearly_rets = yearly_nav.pct_change()

                            # 3. 修正第一年 (成立年) 的收益
                            # 第一年的收益 = 第一年年末 / 成立日净值 - 1
                            try:
                                first_year = yearly_nav.index[0].year
                                first_val = s.iloc[0]
                                end_val_first_year = yearly_nav.iloc[0]
                                first_year_ret = (end_val_first_year / first_val) - 1
                                yearly_rets.iloc[0] = first_year_ret
                            except: pass

                            # 4. 格式化输出
                            y_vals = {}
                            for date, ret in yearly_rets.items():
                                if not pd.isna(ret):
                                    y_vals[date.year] = ret
                            
                            yearly_data[col] = y_vals
                    
                    if yearly_data:
                        df_yearly = pd.DataFrame(yearly_data).T
                        # 按年份排序
                        df_yearly = df_yearly[sorted(df_yearly.columns)]
                        st.dataframe(df_yearly.style.format("{:.2%}"), use_container_width=True)
                    # -------------------------------------------------------

                else: st.warning("⚠️ 数据不足")
            st.markdown("---")
            with st.expander("📚 寻星·量化指标权威速查字典 (CIO解读版)", expanded=False):
                st.markdown("### ... (保持原有文案) ...")

        with tabs[3]:
            st.subheader("🔮 蒙特卡洛模拟 (Monte Carlo Forecasting)")
            st.markdown("基于**几何布朗运动 (GBM)**，推演组合在未来可能的 1,000 种财富路径。")
            
            if star_nav is not None and star_rets_for_mc is not None:
                col_mc1, col_mc2 = st.columns(2)
                with col_mc1:
                    sim_years = st.slider("预测年限 (Years)", 1, 10, 3)
                    init_amt = st.number_input("模拟本金 (元)", value=1000000, step=100000)
                with col_mc2:
                    sim_count = st.selectbox("模拟路径数", [500, 1000, 2000, 5000], index=1)
                    show_paths = st.checkbox("显示所有路径 (可能会卡顿)", value=False)

                if st.button("🚀 开始推演"):
                    with st.spinner("正在生成平行宇宙..."):
                        mc_df, mu_est, sigma_est, dt_label = run_monte_carlo(star_rets_for_mc, sim_count, sim_years, init_amt)
                        
                        if mc_df is not None:
                            fig_mc = go.Figure()
                            p10 = mc_df.quantile(0.1, axis=1)
                            p50 = mc_df.quantile(0.5, axis=1)
                            p90 = mc_df.quantile(0.9, axis=1)
                            
                            fig_mc.add_trace(go.Scatter(x=mc_df.index, y=p90, mode='lines', line=dict(width=0), showlegend=False))
                            fig_mc.add_trace(go.Scatter(x=mc_df.index, y=p10, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,255,0.2)', name='80% 置信区间 (P10-P90)'))
                            fig_mc.add_trace(go.Scatter(x=mc_df.index, y=p50, mode='lines', line=dict(color='red', width=2), name='中性预期 (P50 Median)'))
                            
                            if show_paths:
                                sample_paths = mc_df.sample(n=min(100, sim_count), axis=1)
                                for col in sample_paths.columns:
                                    fig_mc.add_trace(go.Scatter(x=mc_df.index, y=sample_paths[col], mode='lines', line=dict(color='gray', width=0.5), opacity=0.1, showlegend=False))
                            
                            fig_mc.update_layout(title=f"未来 {sim_years} 年财富路径推演 (基于 {fee_mode_label})", yaxis_title="账户权益", template="plotly_white")
                            st.plotly_chart(fig_mc, use_container_width=True)
                            
                            final_values = mc_df.iloc[-1]
                            loss_prob = (final_values < init_amt).mean()
                            exp_ret_annual = (p50.iloc[-1] / init_amt) ** (1/sim_years) - 1
                            
                            st.success(f"✅ 模拟完成！检测到数据为 **{dt_label}频**，已自动校准年化参数 (Vol: {sigma_est:.2%})。")
                            
                            c_m1, c_m2, c_m3 = st.columns(3)
                            c_m1.metric("😭 破产概率 (亏损概率)", f"{loss_prob:.1%}", help="期末本金低于初始本金的概率")
                            c_m2.metric("😐 中性预期期末资产", f"¥{p50.iloc[-1]:,.0f}", help="50% 概率能达到的金额")
                            c_m3.metric("😊 乐观预期 (P90)", f"¥{p90.iloc[-1]:,.0f}", help="最好的 10% 情况下能达到的金额")
                            
                            st.info(f"**CIO 解读**：在未来 {sim_years} 年中，如果您持有此组合，有 **{1-loss_prob:.1%}** 的概率盈利。中性预期下，年化回报约为 **{exp_ret_annual:.2%}**。")
            else:
                st.info("👈 请先在左侧加载并生成组合。")
    else: st.info("👋 请上传‘产品数据库’以启动引擎。")
