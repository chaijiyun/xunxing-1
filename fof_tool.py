import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
from datetime import datetime

# ==========================================
# 0. CTO架构层：全局数据结构定义
# ==========================================
# 定义主数据文件的列结构，确保数据一致性
MASTER_COLUMNS = [
    '产品名称', 
    '年管理费(%)', '业绩报酬(%)', 
    '开放频率', '锁定期(月)', '赎回效率(T+n)'
]

PORTFOLIO_COLUMNS = ['组合名称', '产品名称', '权重']

# 默认主数据 (CIO层：预设了常见的流动性参数)
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

# 全局变量初始化
if 'master_data' not in st.session_state:
    st.session_state.master_data = pd.DataFrame(PRESET_MASTER_DEFAULT)
if 'portfolios_data' not in st.session_state:
    st.session_state.portfolios_data = pd.DataFrame(columns=PORTFOLIO_COLUMNS)

# ==========================================
# 1. 登录验证模块
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.markdown("<br><br>", unsafe_allow_html=True) 
        st.markdown("<h1 style='text-align: center; color: #1E40AF;'>寻星配置分析系统 v6.0 <small>(Security & Risk)</small></h1>", unsafe_allow_html=True)
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
    # 2. 核心逻辑引擎
    # ==========================================
    
    # 2.1 净值计算引擎
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

    # 2.2 指标计算引擎
    def calculate_metrics(nav, bench_nav=None):
        nav = nav.dropna()
        if len(nav) < 2: return {}
        returns = nav.pct_change().fillna(0)
        total_ret = (nav.iloc[-1] / nav.iloc[0]) - 1
        days_count = (nav.index[-1] - nav.index[0]).days
        ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / max(days_count, 1)) - 1
        vol = returns.std() * np.sqrt(252)
        cummax = nav.cummax()
        drawdown = (nav / cummax) - 1
        mdd = drawdown.min()
        
        mdd_rec = "无回撤"
        if mdd != 0:
            mdd_date = drawdown.idxmin()
            recovery_mask = nav.loc[mdd_date:] >= cummax.loc[mdd_date]
            mdd_rec = f"{(recovery_mask.idxmax() - mdd_date).days}天" if recovery_mask.any() else "尚未修复"
        
        is_at_new_high = (nav == cummax)
        high_dates = nav[is_at_new_high].index
        if len(high_dates) < 2: max_nh = f"{days_count}天"
        else:
            intervals = (high_dates[1:] - high_dates[:-1]).days
            last_gap = (nav.index[-1] - high_dates[-1]).days
            max_nh = f"{max(intervals.max(), last_gap) if len(intervals)>0 else last_gap}天"

        rf = 0.02
        sharpe = (ann_ret - rf) / vol if vol > 0 else 0
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0.0001
        sortino = (ann_ret - rf) / downside_std if downside_std > 0 else 0
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0
        
        metrics = {
            "总收益率": total_ret, "年化收益": ann_ret, "最大回撤": mdd, 
            "夏普比率": sharpe, "索提诺比率": sortino, "卡玛比率": calmar, "年化波动率": vol,
            "最大回撤修复时间": mdd_rec, "最大无新高持续时间": max_nh,
            "正收益概率(日)": (returns > 0).sum() / len(returns),
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

    # 2.3 [CIO核心] 流动性风控计算
    def calculate_liquidity_risk(weights, master_df):
        # 权重归一化
        w_series = pd.Series(weights)
        w_norm = w_series / w_series.sum()
        
        weighted_lockup = 0.0
        worst_lockup = 0
        liquidity_notes = []
        
        for p, w in w_norm.items():
            info = master_df[master_df['产品名称'] == p]
            if not info.empty:
                lock = info.iloc[0]['锁定期(月)']
                freq = info.iloc[0]['开放频率']
                
                weighted_lockup += lock * w
                if lock > worst_lockup: worst_lockup = lock
                
                if lock >= 12:
                    liquidity_notes.append(f"⚠️ {p} (锁{lock}个月)")
            else:
                # 缺失数据默认处理
                weighted_lockup += 6 * w # 默认6个月
        
        return weighted_lockup, worst_lockup, liquidity_notes

    # ==========================================
    # 3. UI 界面
    # ==========================================
    st.set_page_config(layout="wide", page_title="寻星配置分析系统 v6.0", page_icon="🛡️")
    st.sidebar.title("🛡️ 寻星 v6.0 · 配置驾驶舱")
    
    # === 全局数据加载与处理 (Sidebar Top) ===
    uploaded_file = st.sidebar.file_uploader("📂 第一步：上传净值数据库", type=["xlsx"])
    
    if uploaded_file:
        # 数据读取
        df_raw = pd.read_excel(uploaded_file, index_col=0, parse_dates=True).sort_index().ffill()
        all_cols = [str(c).strip() for c in df_raw.columns]
        df_raw.columns = all_cols
        
        st.sidebar.markdown("---")
        
        # === 核心模块：配置中心 (取代了原本的费率上传) ===
        with st.sidebar.expander("⚙️ 系统配置中心 (数据/费率/流动性)", expanded=False):
            st.info("💡 这是一个安全的数据沙箱。所有修改都在内存中进行，请定期下载备份。")
            
            # 1. 备份恢复功能
            col_bk1, col_bk2 = st.columns(2)
            uploaded_backup = col_bk1.file_uploader("📥 恢复备份", type=['csv'])
            if uploaded_backup:
                try:
                    df_backup = pd.read_csv(uploaded_backup)
                    # 简单判断是主数据还是组合数据，或者混合(这里简化为只恢复主数据，实际可做zip包)
                    # v6.0 简化逻辑：我们只提供主数据的下载上传，组合数据另行管理
                    if '锁定期(月)' in df_backup.columns:
                        st.session_state.master_data = df_backup
                        st.toast("主数据已恢复！", icon="✅")
                except:
                    st.error("备份文件格式不识别")

            # 2. 主数据编辑 (费率 + 流动性)
            # 自动补充新发现的产品
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
                column_config={
                    "开放频率": st.column_config.SelectboxColumn(options=["周度", "月度", "季度", "半年", "1年", "3年封闭"]),
                },
                use_container_width=True,
                hide_index=True,
                key="master_editor"
            )
            st.session_state.master_data = edited_master
            
            # 下载备份按钮
            csv_master = st.session_state.master_data.to_csv(index=False).encode('utf-8-sig')
            st.download_button("💾 下载全量配置备份 (防丢失)", csv_master, "寻星_系统配置备份.csv", "text/csv")
            
            # 构建快速查询字典
            MASTER_DICT = {}
            for _, row in st.session_state.master_data.iterrows():
                MASTER_DICT[row['产品名称']] = row.to_dict()

        st.sidebar.markdown("---")

        # === 组合管理驾驶舱 (v5.20继承并升级) ===
        st.sidebar.markdown("### 💼 组合配置")
        
        # 模式选择
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
                            # 更新内存中的组合数据
                            old_df = st.session_state.portfolios_data
                            new_df = pd.DataFrame(new_records)
                            st.session_state.portfolios_data = pd.concat([old_df[old_df['组合名称']!=new_p_name], new_df], ignore_index=True)
                            st.toast(f"组合 {new_p_name} 已保存", icon="✅")
                            st.rerun()
        else:
            subset = st.session_state.portfolios_data[st.session_state.portfolios_data['组合名称'] == selected_mode]
            valid_subset = subset[subset['产品名称'].isin(all_cols)]
            sel_funds = valid_subset['产品名称'].tolist()
            weights = {row['产品名称']: row['权重'] for _, row in valid_subset.iterrows()}
            st.sidebar.table(valid_subset[['产品名称', '权重']].set_index('产品名称').style.format("{:.1%}"))

        # 颜色映射
        color_map = {}
        if sel_funds:
            colors = px.colors.qualitative.Plotly 
            for i, f in enumerate(sel_funds): color_map[f] = colors[i % len(colors)]

        # 费率模式
        st.sidebar.markdown("---")
        fee_mode_label = "客户实得回报 (实盘费后)"
        if sel_funds:
            fee_mode_label = st.sidebar.radio("展示视角", ("客户实得回报 (实盘费后)", "组合策略表现 (底层净值)", "收益与运作成本分析"), index=0)

        # ==========================================
        # 主计算逻辑
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
                        # 从主数据读取费率
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
        # Tabs 显示
        # ==========================================
        tabs = st.tabs(["🚀 组合全景图 (含流动性风控)", "🔍 穿透归因", "⚔️ 配置池"])

        if star_nav is not None:
            m = calculate_metrics(star_nav, bn_sync)
            # [CIO] 计算流动性
            avg_lock, worst_lock, lock_notes = calculate_liquidity_risk(weights, st.session_state.master_data)

        with tabs[0]:
            if star_nav is not None:
                st.subheader(f"📊 {star_nav.name}")
                
                # 第一行：业绩指标
                c1 = st.columns(7)
                c1[0].metric("总收益率", f"{m['总收益率']:.2%}")
                c1[1].metric("年化收益", f"{m['年化收益']:.2%}")
                c1[2].metric("最大回撤", f"{m['最大回撤']:.2%}")
                c1[3].metric("夏普比率", f"{m['夏普比率']:.2f}")
                c1[4].metric("索提诺", f"{m['索提诺比率']:.2f}")
                c1[5].metric("卡玛比率", f"{m['卡玛比率']:.2f}")
                c1[6].metric("年化波动", f"{m['年化波动率']:.2%}")
                
                # 第二行：[CIO新增] 风控仪表盘
                st.markdown("#### 🛡️ 风险与流动性仪表盘")
                c2 = st.columns(4)
                c2[0].metric("⏳ 平均锁定期", f"{avg_lock:.1f} 个月", help="按权重计算的加权平均资金冻结时间")
                c2[1].metric("🔒 最长单品锁定", f"{worst_lock} 个月", help="组合中流动性最差的那个产品")
                c2[2].metric("Current Beta", f"{m['Current_Beta']:.2f}")
                c2[3].metric("最大回撤修复", m['最大回撤修复时间'])
                
                if lock_notes:
                    st.warning(f"⚠️ **流动性警示**：组合中包含长期锁定资产：{'、'.join(lock_notes)}。请务必确认客户资金使用期限匹配。")
                
                # 图表
                fig_main = go.Figure()
                line_color = 'red' if '费后' in star_nav.name else 'blue'
                fig_main.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name=star_nav.name, line=dict(color=line_color, width=4)))
                fig_main.add_trace(go.Scatter(x=bn_norm.index, y=bn_norm, name=f"基准: {sel_bench}", line=dict(color='gray', dash='dot')))
                st.plotly_chart(fig_main, use_container_width=True)

        with tabs[1]:
            # (保留 v5.20 逻辑)
            if sel_funds:
                st.subheader("🔍 穿透归因")
                # ... (此处省略与 v5.20 相同的绘图代码，保持不变以节省空间，实际运行时请保留)
                # 为确保代码完整运行，此处补全核心绘图
                if fee_mode_label == "组合策略表现 (底层净值)": df_attr = df_port
                else: df_attr = net_funds_df
                initial_w_series = pd.Series(weights) / (sum(weights.values()) if sum(weights.values()) > 0 else 1)
                growth_factors = df_attr.iloc[-1] / df_attr.iloc[0]
                latest_values = initial_w_series * growth_factors
                latest_w_series = latest_values / latest_values.sum()

                c_pi1, c_pi2 = st.columns(2)
                c_pi1.plotly_chart(px.pie(names=initial_w_series.index, values=initial_w_series.values, title="初始配置", color=initial_w_series.index, color_discrete_map=color_map), use_container_width=True)
                c_pi2.plotly_chart(px.pie(names=latest_w_series.index, values=latest_w_series.values, title="最新漂移后", color=latest_w_series.index, color_discrete_map=color_map), use_container_width=True)
                
                # 走势对比图 (红线增强)
                df_sub_norm = df_attr.div(df_attr.iloc[0])
                fig_sub = go.Figure()
                for col in df_sub_norm.columns:
                    fig_sub.add_trace(go.Scatter(x=df_sub_norm.index, y=df_sub_norm[col], name=col, opacity=0.5, line=dict(color=color_map.get(col))))
                fig_sub.add_trace(go.Scatter(x=star_nav.index, y=star_nav, name=star_nav.name, line=dict(color=line_color, width=4)))
                st.plotly_chart(fig_sub, use_container_width=True)

        with tabs[2]:
            # (保留 v5.19 增强版逻辑)
            st.subheader("⚔️ 配置池")
            # ... (代码逻辑同 v5.20，省略部分重复代码)
            pool_options = [c for c in all_cols if c != sel_bench]
            pool_options.sort()
            compare_pool = st.multiselect("搜索产品", pool_options)
            if compare_pool:
                is_aligned = st.checkbox("对齐起始日", value=False)
                df_comp = df_db[compare_pool].dropna() if is_aligned else df_db[compare_pool]
                if not df_comp.empty:
                    # 分年度统计表
                    st.markdown("#### 📅 分年度收益")
                    yearly_data = {}
                    for col in compare_pool:
                        s = df_comp[col].dropna()
                        groups = s.groupby(s.index.year)
                        y_vals = {year: (g.iloc[-1]/g.iloc[0])-1 for year, g in groups}
                        yearly_data[col] = y_vals
                    st.dataframe(pd.DataFrame(yearly_data).T.sort_index().style.format("{:.2%}"), use_container_width=True)
    else:
        st.info("👋 请先在左上角上传净值数据 Excel。")
