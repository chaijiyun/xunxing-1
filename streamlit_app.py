import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import os
from datetime import datetime, timedelta

# ==========================================
# 寻星配置分析系统 v7.2.0 (Decoupled Simulation)
# Author: 寻星架构师
# Update Log:
#   v7.2.0: [New] 风险实验室新增“采样窗口”控制，解决短久期资产在长回测周期下指标被稀释的问题。
#   v7.1.4: [Fix] 频率自动侦测。
# ==========================================

# ------------------------------------------
# 0. 全局常量与预设 (Configuration)
# ------------------------------------------
CONFIG_FILE_PATH = "xunxing_config.pkl"

PRESET_MASTER_DEFAULT = [
    {'产品名称': '国富瑞合1号', '策略标签': '主观多头', '年管理费(%)': 0, '业绩报酬(%)': 16, '开放频率': '周度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '合骥500对冲A期', '策略标签': '量化对冲', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '合绎期权套利', '策略标签': '期权套利', '年管理费(%)': 0, '业绩报酬(%)': 30, '开放频率': '月度', '锁定期(月)': 6, '赎回效率(T+n)': 4},
    {'产品名称': '玖鹏宏图1号', '策略标签': '主观多头', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '开思沪港深优选', '策略标签': '主观多头', '年管理费(%)': 0, '业绩报酬(%)': 17, '开放频率': '月度', '锁定期(月)': 1, '赎回效率(T+n)': 4},
    {'产品名称': '宽远优势成长10号', '策略标签': '主观多头', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '蓝墨长河1号', '策略标签': '主观多头', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 1, '赎回效率(T+n)': 4},
    {'产品名称': '宁泉特定策略1号', '策略标签': '主观多头', '年管理费(%)': 0, '业绩报酬(%)': 15, '开放频率': '月度', '锁定期(月)': 12, '赎回效率(T+n)': 4},
    {'产品名称': '平方和1000指数增强', '策略标签': '量化指增', '年管理费(%)': 0, '业绩报酬(%)': 16, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '平方和多策略', '策略标签': '多策略', '年管理费(%)': 0, '业绩报酬(%)': 16, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '平方和量化选股', '策略标签': '量化选股', '年管理费(%)': 0, '业绩报酬(%)': 16, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '平方和市场中性', '策略标签': '量化对冲', '年管理费(%)': 0, '业绩报酬(%)': 16, '开放频率': '月度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '秦川1号', '策略标签': '主观多头', '年管理费(%)': 0, '业绩报酬(%)': 15, '开放频率': '周度', '锁定期(月)': 3, '赎回效率(T+n)': 4},
    {'产品名称': '睿郡节节高11号', '策略标签': '主观多头', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 6, '赎回效率(T+n)': 4},
    {'产品名称': '孝庸1000指增', '策略标签': '量化指增', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 12, '赎回效率(T+n)': 4},
    {'产品名称': '孝庸500指增', '策略标签': '量化指增', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '月度', '锁定期(月)': 12, '赎回效率(T+n)': 4},
    {'产品名称': '孝庸中性+cta', '策略标签': '多策略', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '周度', '锁定期(月)': 12, '赎回效率(T+n)': 4},
    {'产品名称': '孝庸中性策略', '策略标签': '量化对冲', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '周度', '锁定期(月)': 12, '赎回效率(T+n)': 4},
    {'产品名称': '孝庸量选', '策略标签': '量化选股', '年管理费(%)': 0, '业绩报酬(%)': 20, '开放频率': '周度', '锁定期(月)': 12, '赎回效率(T+n)': 4},
]
DEFAULT_MASTER_ROW = {"策略标签": "未分类", "年管理费(%)": 0.0, "业绩报酬(%)": 20.0, "开放频率": "月度", "锁定期(月)": 6, "赎回效率(T+n)": 5}

# ------------------------------------------
# 1. 持久化引擎 (Persistence Engine)
# ------------------------------------------
def load_local_config():
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            df = pd.read_pickle(CONFIG_FILE_PATH)
            if '策略标签' not in df.columns: df.insert(1, '策略标签', '未分类')
            return df
        except Exception: return pd.DataFrame(PRESET_MASTER_DEFAULT)
    return pd.DataFrame(PRESET_MASTER_DEFAULT)

def save_local_config(df):
    try: df.to_pickle(CONFIG_FILE_PATH)
    except Exception as e: st.error(f"配置保存失败: {e}")

if 'master_data' not in st.session_state: st.session_state.master_data = load_local_config()
if 'portfolios_data' not in st.session_state: st.session_state.portfolios_data = pd.DataFrame(columns=['组合名称', '产品名称', '权重'])

# ------------------------------------------
# 2. UI 组件封装 (UI Component)
# ------------------------------------------
def render_grouped_selector(label, options, master_df, key_prefix, default_selections=None):
    if default_selections is None: default_selections = []
    strategy_map = {}
    for p in options:
        tag = "未分类"
        if '策略标签' in master_df.columns:
            info = master_df[master_df['产品名称'] == p]
            if not info.empty: tag = info.iloc[0]['策略标签']
        if pd.isna(tag): tag = "未分类"
        if tag not in strategy_map: strategy_map[tag] = []
        strategy_map[tag].append(p)
    sorted_strategies = sorted(strategy_map.keys(), key=lambda x: (x == "未分类", x))
    final_selection = []
    st.markdown(f"**{label}**")
    for strat in sorted_strategies:
        funds_in_group = strategy_map[strat]
        default_in_group = [f for f in funds_in_group if f in default_selections]
        with st.expander(f"📂 {strat} ({len(funds_in_group)}支)", expanded=False):
            selected = st.multiselect(f"选择 {strat}", options=funds_in_group, default=default_in_group, key=f"{key_prefix}_{strat}", label_visibility="collapsed")
            final_selection.extend(selected)
    return final_selection

# ------------------------------------------
# 3. 登录与安全 (Security)
# ------------------------------------------
def check_password():
    if "password_correct" not in st.session_state: st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.markdown("<br><br>", unsafe_allow_html=True) 
        st.markdown("<h1 style='text-align: center; color: #1E40AF;'>寻星配置分析系统 v7.2.0 <small>(Decoupled)</small></h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                pwd_input = st.text_input(label="系统访问密码", type="password", placeholder="请输入密码")
                if st.form_submit_button("立即登录", use_container_width=True):
                    if pwd_input == "281699":
                        st.session_state["password_correct"] = True
                        st.rerun()
                    else: st.error("密码错误：访问拒绝。")
        return False
    return True

if check_password():
    # ------------------------------------------
    # 4. 核心计算引擎 (Calculation Engine)
    # ------------------------------------------
    def calculate_net_nav_series(gross_nav_series, mgmt_fee_rate=0.0, perf_fee_rate=0.0):
        if gross_nav_series.empty: return gross_nav_series
        dates = gross_nav_series.index
        gross_vals = gross_nav_series.values
        entry_price = gross_vals[0] 
        net_vals = np.zeros(len(gross_vals)); net_vals[0] = entry_price 
        asset_after_mgmt = np.zeros(len(gross_vals)); asset_after_mgmt[0] = entry_price
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
        net_vals = np.maximum(asset_after_mgmt - liabilities, 0)
        return pd.Series(net_vals, index=dates)

    def get_drawdown_details(nav_series):
        if nav_series.empty or len(nav_series) < 2: return "数据不足", "数据不足", pd.Series(dtype='float64')
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax 
        mdd_val = drawdown.min()
        if mdd_val == 0: mdd_recovery = "无回撤"
        else:
            mdd_date = drawdown.idxmin()
            peak_val_at_mdd = cummax.loc[mdd_date]
            post_mdd_data = nav_series.loc[mdd_date:]
            recovery_mask = post_mdd_data >= peak_val_at_mdd
            mdd_recovery = f"{(recovery_mask.idxmax() - mdd_date).days}天" if recovery_mask.any() else "尚未修复"
        
        is_at_new_high = (nav_series == cummax)
        high_dates = nav_series[is_at_new_high].index
        if len(high_dates) < 2: max_no_new_high = f"{(nav_series.index[-1] - nav_series.index[0]).days}天"
        else:
            intervals = (high_dates[1:] - high_dates[:-1]).days
            last_gap = (nav_series.index[-1] - high_dates[-1]).days
            max_no_new_high = f"{max(intervals.max(), last_gap) if len(intervals)>0 else last_gap}天"
        return mdd_recovery, max_no_new_high, drawdown

    def calculate_capture_stats(nav_series, bench_series, period_name):
        """
        [v7.1.3 Fix] 智能捕获率算法
        """
        if nav_series.empty or len(nav_series) < 2:
            return {"时段": period_name, "上行捕获": np.nan, "下行捕获": np.nan, "CIO点评": "数据不足"}
        
        p_rets = nav_series.pct_change().dropna()
        b_rets = bench_series.pct_change().dropna()
        valid_idx = p_rets.index.intersection(b_rets.index)
        
        if len(valid_idx) < 1:
            return {"时段": period_name, "上行捕获": np.nan, "下行捕获": np.nan, "CIO点评": "无重叠数据"}
            
        p_rets = p_rets.loc[valid_idx]
        b_rets = b_rets.loc[valid_idx]

        def safe_capture_ratio(p_segment, b_segment):
            if b_segment.empty: return 0.0
            b_mean = b_segment.mean()
            p_mean = p_segment.mean()
            if abs(b_mean) < 0.0005: return 0.0 
            return p_mean / b_mean

        up_mask = b_rets > 0
        down_mask = b_rets < 0
        
        up_cap = safe_capture_ratio(p_rets[up_mask], b_rets[up_mask])
        down_cap = safe_capture_ratio(p_rets[down_mask], b_rets[down_mask])
            
        comment = "正常"
        if abs(down_cap) > 5.0: comment = "⚠️ 数据异常(基准微动)"
        elif down_cap < 0: comment = "🛡️ 逆市收益 (Alpha)"
        elif down_cap > 1.0 and up_cap < 0.8: comment = "⚠️ 策略失效"
        elif down_cap < 0.8 and up_cap > 0.9: comment = "💎 攻守兼备"
        
        return {"时段": period_name, "上行捕获": up_cap, "下行捕获": down_cap, "CIO点评": comment}

    # [New] 蒙特卡洛模拟核心算法 (Updated for Frequency)
    def run_monte_carlo(period_returns, n_sims=1000, n_steps=252):
        if period_returns.empty: return None
        
        mu = period_returns.mean()
        sigma = period_returns.std()
        last_price = 1.0 
        
        # 几何布朗运动 (Geometric Brownian Motion)
        # 这里的 n_steps 代表未来的“周期数”，而非天数
        dt = 1 
        drift = (mu - 0.5 * sigma**2) * dt
        shock = sigma * np.sqrt(dt) * np.random.normal(0, 1, (n_steps, n_sims))
        
        period_returns_sim = np.exp(drift + shock)
        price_paths = np.zeros((n_steps + 1, n_sims))
        price_paths[0] = last_price
        
        for t in range(1, n_steps + 1):
            price_paths[t] = price_paths[t-1] * period_returns_sim[t-1]
            
        return price_paths

    def get_freq_factor(nav):
        # 辅助函数：计算年化因子
        if len(nav) < 2: return 252.0
        dates = nav.index
        count = len(dates) - 1
        days_diff = (dates[-1] - dates[0]).days
        avg_interval = days_diff / count if count > 0 else 1
        
        if avg_interval <= 1.5: return 252.0  # Daily
        elif avg_interval <= 8: return 52.0   # Weekly
        elif avg_interval <= 35: return 12.0  # Monthly
        else: return 252.0 / avg_interval

    def calculate_metrics(nav, bench_nav=None):
        nav = nav.dropna()
        if len(nav) < 2: return {}
        
        dates = nav.index
        days_diff = (dates[-1] - dates[0]).days
        if days_diff <= 0: return {}
        
        freq_factor = get_freq_factor(nav)
        
        returns = nav.pct_change().dropna()
        total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
        ann_ret = (1 + total_ret) ** (365.25 / days_diff) - 1
        vol = returns.std() * np.sqrt(freq_factor)
        mdd_rec, max_nh, dd_s = get_drawdown_details(nav)
        mdd = dd_s.min()
        
        rf = 0.015 
        excess_ret = ann_ret - rf
        sharpe = excess_ret / vol if vol > 1e-6 else 0.0
        
        downside_diff = returns - (rf / freq_factor)
        downside_diff = downside_diff[downside_diff < 0]
        if not downside_diff.empty:
            downside_std = np.sqrt((downside_diff ** 2).mean()) * np.sqrt(freq_factor)
        else: downside_std = 1e-6
        sortino = excess_ret / downside_std if downside_std > 1e-6 else 0.0
        
        calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-6 else 0.0
        
        win_days = returns[returns > 0]; loss_days = returns[returns < 0]
        avg_win = win_days.mean() if not win_days.empty else 0
        avg_loss = abs(loss_days.mean()) if not loss_days.empty else 0
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        var_95 = np.percentile(returns, 5) 

        metrics = {
            "总收益率": total_ret, "年化收益": ann_ret, "最大回撤": mdd, 
            "夏普比率": sharpe, "索提诺比率": sortino, "卡玛比率": calmar, "年化波动率": vol,
            "最大回撤修复时间": mdd_rec, "最大无新高持续时间": max_nh,
            "正收益概率(日)": len(win_days) / len(returns) if len(returns) > 0 else 0,
            "盈亏比": pl_ratio, "VaR(95%)": var_95, "dd_series": dd_s,
            "Beta": 0.0, "Current_Beta": 0.0, "Alpha": 0.0, "上行捕获": 0.0, "下行捕获": 0.0,
            "Rolling_Beta_Series": pd.Series(dtype='float64'),
            "Rolling_Up_Cap": pd.Series(dtype='float64'), "Rolling_Down_Cap": pd.Series(dtype='float64'),
            "freq_factor": freq_factor
        }
        
        if bench_nav is not None:
            common_idx = nav.index.intersection(bench_nav.index)
            if len(common_idx) > 10:
                p_rets = nav.loc[common_idx].pct_change().dropna()
                b_rets = bench_nav.loc[common_idx].pct_change().dropna()
                valid_idx = p_rets.index.intersection(b_rets.index)
                p_rets = p_rets.loc[valid_idx]; b_rets = b_rets.loc[valid_idx]
                
                if not p_rets.empty:
                    cov_mat = np.cov(p_rets, b_rets)
                    beta = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat.shape == (2, 2) and cov_mat[1, 1] != 0 else 0
                    bench_total_ret = (bench_nav.loc[common_idx[-1]]/bench_nav.loc[common_idx[0]])**(365.25/(common_idx[-1]-common_idx[0]).days) - 1
                    alpha = ann_ret - (rf + beta * (bench_total_ret - rf))

                    window = int(freq_factor / 2)
                    if window < 10: window = 10
                    rolling_betas = []; rolling_dates = []; rolling_up_cap = []; rolling_down_cap = []

                    if len(p_rets) > window:
                        for i in range(window, len(p_rets)):
                            r_win = p_rets.iloc[i-window:i]
                            b_win = b_rets.iloc[i-window:i]
                            current_date = p_rets.index[i]
                            
                            var_b = b_win.var()
                            cov_rb = r_win.cov(b_win)
                            rb = cov_rb / var_b if var_b != 0 else 0
                            
                            up_mask_win = b_win > 0; down_mask_win = b_win < 0
                            r_up_val = (r_win[up_mask_win].mean() / b_win[up_mask_win].mean()) if (up_mask_win.any() and abs(b_win[up_mask_win].mean()) > 1e-6) else 0
                            r_down_val = (r_win[down_mask_win].mean() / b_win[down_mask_win].mean()) if (down_mask_win.any() and abs(b_win[down_mask_win].mean()) > 1e-6) else 0
                                
                            rolling_betas.append(rb)
                            rolling_up_cap.append(r_up_val)
                            rolling_down_cap.append(r_down_val)
                            rolling_dates.append(current_date)
                            
                        curr_beta = rolling_betas[-1] if rolling_betas else beta
                        rb_series = pd.Series(rolling_betas, index=rolling_dates)
                        ru_series = pd.Series(rolling_up_cap, index=rolling_dates)
                        rd_series = pd.Series(rolling_down_cap, index=rolling_dates)
                    else:
                        curr_beta = beta
                        rb_series = pd.Series([beta]*len(p_rets), index=p_rets.index)
                        ru_series = pd.Series(dtype='float64'); rd_series = pd.Series(dtype='float64')
                    
                    up_mask = b_rets > 0; down_mask = b_rets < 0
                    up_cap = (p_rets[up_mask].mean() / b_rets[up_mask].mean()) if (up_mask.any() and abs(b_rets[up_mask].mean()) > 1e-6) else 0
                    down_cap = (p_rets[down_mask].mean() / b_rets[down_mask].mean()) if (down_mask.any() and abs(b_rets[down_mask].mean()) > 1e-6) else 0

                    metrics.update({
                        "上行捕获": up_cap, "下行捕获": down_cap, "Beta": beta, "Current_Beta": curr_beta, "Alpha": alpha,
                        "Rolling_Beta_Series": rb_series, "Rolling_Up_Cap": ru_series, "Rolling_Down_Cap": rd_series    
                    })
        return metrics

    def calculate_liquidity_risk(weights, master_df):
        w_series = pd.Series(weights)
        w_norm = w_series / w_series.sum()
        weighted_lockup = 0.0; worst_lockup = 0; liquidity_notes = []
        for p, w in w_norm.items():
            info = master_df[master_df['产品名称'] == p]
            if not info.empty:
                lock = info.iloc[0].get('锁定期(月)', 6)
                weighted_lockup += lock * w
                if lock > worst_lockup: worst_lockup = lock
                if lock >= 12: liquidity_notes.append(f"⚠️ {p}({lock}个月)")
            else: weighted_lockup += 6 * w 
        return weighted_lockup, worst_lockup, liquidity_notes

    # ------------------------------------------
    # 5. UI 界面与交互 (Interface)
    # ------------------------------------------
    st.set_page_config(layout="wide", page_title="寻星配置分析系统 v7.2.0", page_icon="🏛️")
    st.sidebar.title("🏛️ 寻星 v7.2.0 · 驾驶舱")
    uploaded_file = st.sidebar.file_uploader("📂 第一步：上传净值数据库 (.xlsx)", type=["xlsx"])

    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        all_cols = df_raw.columns.tolist()
        
        st.sidebar.markdown("---")
        with st.sidebar.expander("⚙️ 寻星配置参数", expanded=False):
            st.info("💡 系统已启用自动记忆：您在此处的修改会自动保存，下次无需重新输入。")
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
                        st.toast("✅ 费率与组合数据恢复成功！", icon="🎉")
                    except: st.toast("⚠️ 仅恢复了费率，未找到组合数据。", icon="ℹ️")
                except Exception as e: st.error(f"恢复失败: {e}")

            current_products = st.session_state.master_data['产品名称'].tolist()
            new_products = [p for p in all_cols if p not in current_products and p not in ['沪深300', '日期']]
            if new_products:
                new_rows = []
                for p in new_products:
                    row = DEFAULT_MASTER_ROW.copy(); row['产品名称'] = p
                    new_rows.append(row)
                st.session_state.master_data = pd.concat([st.session_state.master_data, pd.DataFrame(new_rows)], ignore_index=True)
                save_local_config(st.session_state.master_data) 
            
            edited_master = st.data_editor(st.session_state.master_data, column_config={
                "策略标签": st.column_config.SelectboxColumn(options=["主观多头", "量化指增", "量化中性", "量化对冲", "量化选股", "期权套利", "CTA", "多策略", "未分类"], required=True),
                "开放频率": st.column_config.SelectboxColumn(options=["周度", "月度", "季度", "半年", "1年", "3年封闭"])
            }, use_container_width=True, hide_index=True, key="master_editor_v700")
            
            if not edited_master.equals(st.session_state.master_data):
                st.session_state.master_data = edited_master
                save_local_config(edited_master) 
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.master_data.to_excel(writer, sheet_name='Master_Data', index=False)
                st.session_state.portfolios_data.to_excel(writer, sheet_name='Portfolios', index=False)
            st.download_button(label="💾 下载寻星配置参数 (.xlsx)", data=buffer, file_name="寻星_全量系统备份.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            MASTER_DICT = {row['产品名称']: row.to_dict() for _, row in st.session_state.master_data.iterrows()}

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💼 组合配置")
        saved_names = st.session_state.portfolios_data['组合名称'].unique().tolist() if not st.session_state.portfolios_data.empty else []
        mode_options = ["🛠️ 自定义/新建"] + saved_names
        selected_mode = st.sidebar.selectbox("选择模式:", mode_options)
        
        sel_funds = []; weights = {}
        default_bench = '沪深300' if '沪深300' in all_cols else all_cols[0]
        sel_bench = st.sidebar.selectbox("业绩基准", all_cols, index=all_cols.index(default_bench))
        
        if selected_mode == "🛠️ 自定义/新建":
            available_funds = sorted([c for c in all_cols if c != sel_bench])
            with st.sidebar:
                sel_funds = render_grouped_selector("挑选成分基金 (按策略)", available_funds, st.session_state.master_data, key_prefix="sidebar_select")
            if sel_funds:
                st.sidebar.markdown("#### ⚖️ 权重")
                avg_w = 1.0 / len(sel_funds)
                for f in sel_funds: weights[f] = st.sidebar.number_input(f"{f}", 0.0, 1.0, avg_w, step=0.05)
                with st.sidebar.expander("💾 保存组合", expanded=True):
                    new_p_name = st.text_input("组合名称", placeholder="如: 稳健1号")
                    if st.button("保存") and new_p_name and sel_funds:
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

        color_map = {}
        if sel_funds:
            colors = px.colors.qualitative.Plotly 
            for i, f in enumerate(sel_funds): color_map[f] = colors[i % len(colors)]

        st.sidebar.markdown("---")
        fee_mode_label = "组合实得回报"
        if sel_funds: fee_mode_label = st.sidebar.radio("展示视角", ("组合实得回报", "组合策略表现", "收益与运作成本分析"), index=0)

        # ==========================================
        # [Critical Fix v7.1.1] 默认视角锚定 2020-01-01
        # ==========================================
        st.sidebar.markdown("### ⏳ 回测区间 (Global Time Window)")
        
        # 1. 获取数据的绝对边界
        data_min_date = df_raw.index.min().date()
        data_max_date = df_raw.index.max().date()
        
        # 2. 智能计算默认起始日 (Target: 2020-01-01)
        target_start = pd.Timestamp("2020-01-01").date()
        
        if target_start < data_min_date:
            default_start = data_min_date  
        elif target_start > data_max_date:
            default_start = data_min_date  
        else:
            default_start = target_start   # ✅ 正常命中 2020-01-01
        
        # 3. 渲染日期选择器
        start_date = st.sidebar.date_input(
            "起始日期", 
            value=default_start,    # 智能默认值
            min_value=data_min_date, 
            max_value=data_max_date
        )
        end_date = st.sidebar.date_input(
            "截止日期", 
            value=data_max_date, 
            min_value=data_min_date, 
            max_value=data_max_date
        )
        
        # 4. 逻辑防呆
        if start_date >= end_date:
            st.error("❌ 错误：起始日期必须早于截止日期。")
            st.stop()
            
        # 5. 执行切片
        df_db = df_raw.loc[start_date:end_date].copy()
        
        # ==========================================
        
        star_nav = None; star_nav_gross = None; star_nav_net = None

        if sel_funds and not df_db.empty:
            df_port = df_db[sel_funds].ffill().dropna(how='all') 
            
            if not df_port.empty:
                norm_w = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) > 0 else 1)
                
                # Gross Calculation
                star_rets_gross = (df_port.pct_change().fillna(0) * norm_w).sum(axis=1) 
                star_nav_gross = (1 + star_rets_gross).cumprod()
                star_nav_gross.name = "组合策略表现"

                # Net Calculation
                net_funds_df = pd.DataFrame(index=df_port.index)
                for f in sel_funds:
                    s_raw = df_db[f].dropna()
                    if s_raw.empty: continue
                    s_segment = s_raw.reindex(df_port.index)
                    s_segment = s_segment.fillna(method='bfill').fillna(1.0)
                    
                    info = MASTER_DICT.get(f, DEFAULT_MASTER_ROW)
                    mgmt = info.get('年管理费(%)', 0) / 100.0
                    perf = info.get('业绩报酬(%)', 0) / 100.0
                    net_funds_df[f] = calculate_net_nav_series(s_segment, mgmt, perf)

                if fee_mode_label != "组合策略表现":
                    star_rets_net = (net_funds_df.pct_change().fillna(0) * norm_w).sum(axis=1)
                    star_nav_net = (1 + star_rets_net).cumprod()
                    star_nav_net.name = "组合实得回报"

                star_nav = star_nav_gross if fee_mode_label == "组合策略表现" else star_nav_net
                bn_sync = df_db.loc[star_nav.index, sel_bench]
                bn_norm = bn_sync / bn_sync.iloc[0]

        tabs = st.tabs(["⚔️ 配置池产品分析", "🚀 组合全景图", "🔍 穿透归因分析", "🌪️ 风险风洞实验室"])

        if star_nav is not None:
            m = calculate_metrics(star_nav, bn_sync)
            avg_lock, worst_lock, lock_notes = calculate_liquidity_risk(weights, st.session_state.master_data)

        # === Tab 1 ===
        with tabs[0]:
            c_t1, c_t2 = st.columns([3, 1])
            with c_t1: st.subheader("⚔️ 配置池产品分析")
            with c_t2: comp_fee_mode = st.selectbox("展示视角", ["费前 (Gross)", "费后 (Net)"], index=0)
            pool_options = sorted([c for c in all_cols if c != sel_bench])
            compare_pool = render_grouped_selector("搜索池内产品 (按策略)", pool_options, st.session_state.master_data, key_prefix="pool_select")
            
            if compare_pool:
                is_aligned = st.checkbox("对齐起始日期比较", value=False)
                df_comp_raw = df_db[compare_pool].dropna() if is_aligned else df_db[compare_pool]
                
                if comp_fee_mode == "费后 (Net)":
                    df_comp = pd.DataFrame(index=df_comp_raw.index)
                    for p in compare_pool:
                        s_raw = df_comp_raw[p].dropna()
                        if s_raw.empty: continue
                        info = MASTER_DICT.get(p, DEFAULT_MASTER_ROW)
                        df_comp[p] = calculate_net_nav_series(s_raw, info.get('年管理费(%)', 0)/100.0, info.get('业绩报酬(%)', 0)/100.0)
                else: df_comp = df_comp_raw

                if not df_comp.empty:
                    fig_p = go.Figure()
                    for col in compare_pool:
                        if col in df_comp.columns:
                            s = df_comp[col].dropna()
                            if not s.empty: fig_p.add_trace(go.Scatter(x=s.index, y=s/s.iloc[0], name=col))
                    
                    if sel_bench in df_db.columns:
                        s_bench = df_db[sel_bench].reindex(df_comp.index).ffill()
                        if not s_bench.empty:
                            s_bench = s_bench / s_bench.iloc[0]
                            fig_p.add_trace(go.Scatter(x=s_bench.index, y=s_bench, name=f"基准: {sel_bench}", line=dict(color='#1890FF', width=2, dash='solid'), opacity=0.8))

                    st.plotly_chart(fig_p.update_layout(title=f"业绩对比 ({comp_fee_mode})", template="plotly_white", height=500), use_container_width=True)
                    
                    res_data = []
                    for col in compare_pool:
                        if col in df_comp.columns:
                            s_full = df_comp[col].dropna()
                            if s_full.empty: continue
                            
                            b_full = df_db[sel_bench].reindex(s_full.index).dropna()
                            common_idx = s_full.index.intersection(b_full.index)
                            s_final = s_full.loc[common_idx]
                            b_final = b_full.loc[common_idx]
                            if len(s_final) < 10: continue

                            k = calculate_metrics(s_final, b_final)
                            if not k: continue

                            freq_n = int(k.get('freq_factor', 252)) 
                            window_1y = freq_n
                            window_6m = max(int(freq_n / 2), 1)

                            if len(s_final) >= window_1y:
                                cap_1y = calculate_capture_stats(s_final.iloc[-window_1y:], b_final.iloc[-window_1y:], "L1Y")
                                l1y_up = f"{cap_1y['上行捕获']:.2%}"
                                l1y_down = f"{cap_1y['下行捕获']:.2%}"
                            else: l1y_up, l1y_down = "-", "-"

                            if len(s_final) >= window_6m:
                                cap_6m = calculate_capture_stats(s_final.iloc[-window_6m:], b_final.iloc[-window_6m:], "L6M")
                                l6m_up = f"{cap_6m['上行捕获']:.2%}"
                                l6m_down = f"{cap_6m['下行捕获']:.2%}"
                            else: l6m_up, l6m_down = "-", "-"

                            res_data.append({
                                "产品名称": col, 
                                "总收益": f"{k['总收益率']:.2%}", "年化收益": f"{k['年化收益']:.2%}", "最大回撤": f"{k['最大回撤']:.2%}",
                                "卡玛": f"{k['卡玛比率']:.2f}", "夏普": f"{k['夏普比率']:.2f}", "索提诺": f"{k['索提诺比率']:.2f}",
                                "全景上行": f"{k['上行捕获']:.2%}", "全景下行": f"{k['下行捕获']:.2%}",
                                "近1年上行": l1y_up, "近1年下行": l1y_down,
                                "近半年上行": l6m_up, "近半年下行": l6m_down,
                                "胜率": f"{k['正收益概率(日)']:.1%}", "VaR(95%)": f"{k['VaR(95%)']:.2%}",
                                "Alpha": f"{k['Alpha']:.2%}", "Beta": f"{k['Beta']:.2f}"
                            })
                    if res_data: st.dataframe(pd.DataFrame(res_data).set_index('产品名称'), use_container_width=True)
                    
                    st.markdown("#### 📅 分年度收益率统计")
                    yearly_data = {}
                    for col in compare_pool:
                        if col in df_comp.columns:
                            s = df_comp[col].dropna()
                            groups = s.groupby(s.index.year)
                            y_vals = {}
                            for year, group in groups: y_vals[year] = (group.iloc[-1] / group.iloc[0]) - 1
                            yearly_data[col] = y_vals
                    if yearly_data:
                        df_yearly = pd.DataFrame(yearly_data).T
                        st.dataframe(df_yearly[sorted(df_yearly.columns)].style.format("{:.2%}"), use_container_width=True)
                else: st.warning("⚠️ 数据不足")
            st.markdown("---"); st.info("📚 寻星·量化指标说明：全站已统一为百分比格式，并支持周频/月频数据的短期指标计算。")

        # === Tab 2 ===
        with tabs[1]:
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
                c_top[7].metric("组合Beta", f"{m['Beta']:.2f}")
                
                fig_main = go.Figure()
                if fee_mode_label == "收益与运作成本分析":
                    fig_main.add_trace(go.Scatter(x=star_nav_net.index, y=star_nav_net, name="组合实得回报", line=dict(color='red', width=3)))
                    fig_main.add_trace(go.Scatter(x=star_nav_gross.index, y=star_nav_gross, name="组合策略表现", line=dict(color='gray', width=2, dash='dash')))
                else:
                    fig_main.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name=star_nav.name, line=dict(color='red', width=4)))
                
                fig_main.add_trace(go.Scatter(x=bn_norm.index, y=bn_norm, name=f"基准: {sel_bench}", line=dict(color='#1890FF', width=2, dash='solid'), opacity=0.8))
                fig_main.update_layout(title="账户权益走势", template="plotly_white", hovermode="x unified", height=450)
                st.plotly_chart(fig_main, use_container_width=True)

                st.markdown("#### 🛡️ 风险体验与风格监控")
                c_risk = st.columns(5) 
                c_risk[0].metric("最大回撤修复", m['最大回撤修复时间'])
                c_risk[1].metric("最长创新高间隔", m['最大无新高持续时间'])
                c_risk[2].metric("盈亏比", f"{m['盈亏比']:.2f}")
                c_risk[3].metric("Current Beta", f"{m['Current_Beta']:.2f}")
                c_risk[4].metric("VaR (95%)", f"{m['VaR(95%)']:.2%}")
                if abs(m['Current_Beta'] - m['Beta']) > 0.1: st.warning(f"⚠️ **风格漂移预警**：Beta 偏差 {abs(m['Current_Beta'] - m['Beta']):.2f}。")
                if lock_notes: st.warning(f"⚠️ **流动性警示**：{' '.join(lock_notes)}")
            else: st.info("👈 请在左侧选择或加载组合。")

        # === Tab 3 ===
        with tabs[2]:
            if sel_funds:
                st.subheader("🔍 寻星配置穿透归因分析")
                if fee_mode_label == "组合策略表现": df_attr = df_port
                else: df_attr = net_funds_df
                
                # [Core Logic: Contribution View uses Cash Filled Data]
                growth_factors = pd.Series(index=df_attr.columns, dtype=float)
                for col in df_attr.columns:
                    s = df_attr[col]
                    if not s.empty: growth_factors[col] = s.iloc[-1] / s.iloc[0]
                    else: growth_factors[col] = 1.0 

                initial_w_series = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) > 0 else 1)
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

                if not m['Rolling_Up_Cap'].empty and not m['Rolling_Down_Cap'].empty:
                    st.markdown("#### 🌊 动态攻守能力分析 (Dynamic Capture Analysis)")
                    
                    st.markdown("##### 1. 分时段攻守能力雷达 (Static Period Radar)")
                    st.info("💡 **架构师注**：以下指标基于各基金**实际成立/存续区间**计算 (Raw Data)，已剔除未投入期的现金拖累。")
                    
                    # [Dual-Track: Asset Analysis View uses Raw Data]
                    metrics_list = []
                    for col in sel_funds:
                        s_raw = df_db[col].dropna()
                        if s_raw.empty: continue
                        b_raw = df_db[sel_bench].reindex(s_raw.index).dropna()
                        common_idx = s_raw.index.intersection(b_raw.index)
                        s_final = s_raw.loc[common_idx]
                        b_final = b_raw.loc[common_idx]
                        if len(s_final) < 10: continue
                        
                        cap_stats = calculate_capture_stats(s_final, b_final, "全周期")
                        m_real = calculate_metrics(s_final, b_final)
                        
                        metrics_list.append({
                            "产品名称": col,
                            "存续时长": f"{(s_final.index[-1] - s_final.index[0]).days}天",
                            "年化收益": f"{m_real['年化收益']:.2%}",
                            "最大回撤": f"{m_real['最大回撤']:.2%}",
                            "卡玛比率": f"{m_real['卡玛比率']:.2f}",
                            "夏普比率": f"{m_real['夏普比率']:.2f}",
                            "索提诺": f"{m_real['索提诺比率']:.2f}",
                            "上行捕获": f"{cap_stats['上行捕获']:.2%}",
                            "下行捕获": f"{cap_stats['下行捕获']:.2%}",
                            "胜率": f"{m_real['正收益概率(日)']:.1%}",
                            "CIO点评": cap_stats['CIO点评']
                        })
                    if metrics_list:
                        st.dataframe(pd.DataFrame(metrics_list).set_index("产品名称"), use_container_width=True)

                    st.markdown("##### 2. 滚动捕获率趋势 (Rolling Trend)")
                    fig_cap = go.Figure()
                    fig_cap.add_trace(go.Scatter(x=m['Rolling_Up_Cap'].index, y=m['Rolling_Up_Cap'], name="上行捕获 (进攻)", line=dict(color='#1890FF', width=2), fill='tozeroy', fillcolor='rgba(24, 144, 255, 0.1)'))
                    fig_cap.add_trace(go.Scatter(x=m['Rolling_Down_Cap'].index, y=m['Rolling_Down_Cap'], name="下行捕获 (防守)", line=dict(color='#D0021B', width=2), fill='tozeroy', fillcolor='rgba(208, 2, 27, 0.1)'))
                    fig_cap.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="基准水平 (100%)")
                    fig_cap.update_layout(template="plotly_white", height=400, hovermode="x unified", yaxis=dict(title="捕获率 (Capture Ratio)", tickformat=".2f"))
                    st.plotly_chart(fig_cap, use_container_width=True)

                # [Dual-Track: Risk/Return Contribution uses Cash Filled]
                # [Fix v7.1.4] Use dynamic frequency factor instead of hardcoded 252
                df_sub_rets = df_attr.pct_change().fillna(0) 
                
                # Detect frequency for risk scaling
                if not df_attr.empty and len(df_attr) > 1:
                    freq_f = get_freq_factor(df_attr.iloc[:,0]) # approximate from first column
                else:
                    freq_f = 252.0
                    
                risk_vals = initial_w_series * (df_sub_rets.std() * np.sqrt(freq_f)) 
                
                contribution_vals = pd.Series(index=df_attr.columns, dtype=float)
                for col in df_attr.columns:
                    s = df_attr[col]
                    if not s.empty: contribution_vals[col] = (s.iloc[-1] / s.iloc[0]) - 1
                    else: contribution_vals[col] = 0.0
                contribution_vals = initial_w_series * contribution_vals

                col_attr1, col_attr2 = st.columns(2)
                col_attr1.plotly_chart(px.pie(names=risk_vals.index, values=risk_vals.values, hole=0.4, title="风险贡献归因", color=risk_vals.index, color_discrete_map=color_map), use_container_width=True)
                col_attr2.plotly_chart(px.pie(names=contribution_vals.index, values=contribution_vals.abs(), hole=0.4, title="收益贡献归因", color=contribution_vals.index, color_discrete_map=color_map), use_container_width=True)

                st.markdown("---")
                st.markdown("#### 底层产品走势对比 (独立归一化)")
                fig_sub_compare = go.Figure()
                # [Dual-Track: Line Chart uses Raw Data for Independent Normalization]
                for col in sel_funds:
                    s_raw = df_db[col].dropna()
                    # Filter to user selected range to keep X-axis consistent
                    s_raw = s_raw.loc[s_raw.index >= df_db.index[0]] 
                    if not s_raw.empty:
                        s_norm = s_raw / s_raw.iloc[0] 
                        fig_sub_compare.add_trace(go.Scatter(x=s_norm.index, y=s_norm, name=col, opacity=0.6, line=dict(color=color_map.get(col))))
                
                if star_nav is not None:
                    fig_sub_compare.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name=star_nav.name, line=dict(color='red', width=4)))
                st.plotly_chart(fig_sub_compare.update_layout(template="plotly_white", height=500), use_container_width=True)
                
                st.plotly_chart(px.imshow(df_sub_rets.corr(), text_auto=".2f", color_continuous_scale=[[0.0, '#1890FF'], [0.5, '#FFFFFF'], [1.0, '#D0021B']], zmin=-1, zmax=1, title="产品相关性矩阵 (Pearson)", height=600), use_container_width=True)

        # === Tab 4: 风险风洞实验室 (Enhanced v7.2.0) ===
        with tabs[3]:
            if star_nav is not None:
                st.subheader("🌪️ 风险风洞实验室 (Risk Lab)")
                
                # [New v7.2.0] Simulation Window Control
                st.markdown("##### 1. 训练数据采样窗口 (Training Window)")
                
                sim_options = ["全量数据 (不推荐)", "最近 5 年", "最近 3 年", "最近 1 年", "最近 6 个月"]
                # 默认选最近 1 年，因为这通常反映了产品当前的真实策略特征
                sim_period = st.select_slider(
                    "请选择用于训练蒙特卡洛模型的数据长度：",
                    options=sim_options,
                    value="最近 1 年"
                )
                
                # 1. 准备数据: 计算组合日收益率 (Cash Filled)
                star_rets = star_nav.pct_change().dropna()
                
                # [Core Logic] Data Slicing based on Selection
                slice_date = star_rets.index.min()
                if sim_period == "最近 5 年":
                    slice_date = star_rets.index.max() - timedelta(days=365*5)
                elif sim_period == "最近 3 年":
                    slice_date = star_rets.index.max() - timedelta(days=365*3)
                elif sim_period == "最近 1 年":
                    slice_date = star_rets.index.max() - timedelta(days=365)
                elif sim_period == "最近 6 个月":
                    slice_date = star_rets.index.max() - timedelta(days=180)
                
                # Apply Slice
                star_rets_trained = star_rets[star_rets.index >= slice_date]
                
                if star_rets_trained.empty:
                    st.error(f"❌ 数据不足：所选窗口内无有效数据。请选择更长的时间窗口。")
                else:
                    st.caption(f"📅 实际训练区间: {star_rets_trained.index.min().date()} 至 {star_rets_trained.index.max().date()} (样本数: {len(star_rets_trained)})")
                    
                    # [Fix v7.1.4] 智能侦测数据频率
                    dates_mc = star_rets_trained.index
                    sim_steps = 252 # Default
                    freq_label = "交易日"
                    
                    if len(dates_mc) > 1:
                        avg_days = (dates_mc[-1] - dates_mc[0]).days / (len(dates_mc) - 1)
                        if avg_days <= 1.5:
                            sim_steps = 252; freq_label = "交易日 (Daily)"
                        elif avg_days <= 8:
                            sim_steps = 52; freq_label = "周 (Weekly)"
                        elif avg_days <= 35:
                            sim_steps = 12; freq_label = "月 (Monthly)"
                        else:
                            sim_steps = int(365 / avg_days); freq_label = "期 (Periods)"
                    
                    # 2. 运行模拟 (Monte Carlo)
                    if st.button("🚀 启动蒙特卡洛模拟引擎"):
                        with st.spinner(f"正在基于 {freq_label} 频率进行 1,000 次平行宇宙推演..."):
                            sim_paths = run_monte_carlo(star_rets_trained, n_sims=1000, n_steps=sim_steps)
                            
                            if sim_paths is not None:
                                # 3. 可视化: 扇形图 (Fan Chart)
                                p5 = np.percentile(sim_paths, 5, axis=1)
                                p25 = np.percentile(sim_paths, 25, axis=1)
                                p50 = np.percentile(sim_paths, 50, axis=1)
                                p75 = np.percentile(sim_paths, 75, axis=1)
                                p95 = np.percentile(sim_paths, 95, axis=1)
                                
                                x_axis = list(range(len(p50)))
                                
                                fig_mc = go.Figure()
                                # 90% 置信区间
                                fig_mc.add_trace(go.Scatter(x=x_axis, y=p95, mode='lines', line=dict(width=0), showlegend=False, name='95%'))
                                fig_mc.add_trace(go.Scatter(x=x_axis, y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(200, 200, 200, 0.2)', name='90% Range'))
                                
                                # 50% 置信区间
                                fig_mc.add_trace(go.Scatter(x=x_axis, y=p75, mode='lines', line=dict(width=0), showlegend=False, name='75%'))
                                fig_mc.add_trace(go.Scatter(x=x_axis, y=p25, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(100, 100, 255, 0.3)', name='50% Range'))
                                
                                # 中位数路径
                                fig_mc.add_trace(go.Scatter(x=x_axis, y=p50, mode='lines', line=dict(color='#1890FF', width=2), name='中性预期 (Median)'))
                                fig_mc.add_trace(go.Scatter(x=[0], y=[1.0], mode='markers', marker=dict(color='black', size=5), showlegend=False))

                                fig_mc.update_layout(
                                    title=f"未来1年财富路径推演 (Steps={sim_steps})",
                                    xaxis_title=f"未来 {freq_label}",
                                    yaxis_title="净值预期 (归一化)",
                                    template="plotly_white",
                                    height=500
                                )
                                st.plotly_chart(fig_mc, use_container_width=True)
                                
                                # 4. VaR 指标计算
                                final_values = sim_paths[-1, :]
                                var_95_val = np.percentile(final_values, 5) - 1
                                var_99_val = np.percentile(final_values, 1) - 1
                                
                                c_var1, c_var2, c_var3 = st.columns(3)
                                c_var1.metric("中性预期收益 (Median)", f"{(np.median(final_values)-1):.2%}")
                                c_var2.metric("VaR (95%置信度)", f"{var_95_val:.2%}", help="有5%的概率，未来一年亏损超过此数值")
                                c_var3.metric("VaR (99%置信度)", f"{var_99_val:.2%}", help="有1%的概率，未来一年亏损超过此数值")
                                
                                if var_95_val < -0.2:
                                    st.error(f"⚠️ **风控预警**：极端情况下 (95% VaR)，组合可能面临 **{abs(var_95_val):.1%}** 的回撤风险，请检查杠杆或高波资产权重。")
                                else:
                                    st.success(f"✅ **风控评估**：在 95% 置信度下，未来一年潜在最大亏损控制在 **{abs(var_95_val):.1%}** 以内，属于稳健区间。")

            else: st.info("👈 请在左侧加载组合以启动实验室。")

    else: st.info("👋 请上传‘产品数据库’以启动引擎。")
