import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SalesForge AI — Forecasting Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS — GLASSMORPHISM DARK THEME & VISIBILITY FIXES
# =========================================================

st.markdown("""
<style>

@import url('https://www.telemetrytv.com/posts/sales-team-digital-signage/');

/* Global */

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
}

.stApp {
    background: #06080F;
}

/* Ambient glow */

.stApp::before,
.stApp::after {
    content: '';
    position: fixed;
    border-radius: 50%;
    filter: blur(120px);
    opacity: 0.10;
    pointer-events: none;
    z-index: 0;
}

.stApp::before {
    width: 50vw;
    height: 50vw;
    max-width: 600px;
    max-height: 600px;
    background: #06B6D4;
    top: -10vw;
    left: -10vw;
}

.stApp::after {
    width: 40vw;
    height: 40vw;
    max-width: 500px;
    max-height: 500px;
    background: #8B5CF6;
    bottom: -10vw;
    right: -10vw;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: rgba(15, 18, 30, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(20px);
}

[data-testid="stSidebar"] * {
    color: #D5DDED !important;
}

/* Headings */

h1 {
    background: linear-gradient(135deg, #06B6D4, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 2.8rem !important;
    letter-spacing: -1px;
}

h2 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}

h3 {
    color: #B8C4D6 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.85rem !important;
}

/* Cards */

.glass-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 28px;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}

.glass-card:hover {
    background: rgba(255, 255, 255, 0.07);
    border-color: rgba(6, 182, 212, 0.25);
    box-shadow: 0 8px 32px rgba(6, 182, 212, 0.08);
}

/* Metric Cards & Container Fixes */

[data-testid="stMetric"], 
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    backdrop-filter: blur(12px) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stMetric"]:hover,
div[data-testid="metric-container"]:hover {
    border-color: rgba(6, 182, 212, 0.4) !important;
    box-shadow: 0 4px 20px rgba(6, 182, 212, 0.15) !important;
    transform: translateY(-2px) !important;
}

/* Metric Label */
[data-testid="stMetricLabel"] *, 
[data-testid="stMetricLabel"] {
    color: #BFCBDA !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* FIX FOR LOW VISIBILITY NUMBERS */
[data-testid="stMetricValue"], 
[data-testid="stMetricValue"] > div, 
[data-testid="stMetricValue"] *,
div[data-testid="metric-container"] [data-testid="stMetricValue"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    fill: #FFFFFF !important;
    opacity: 1 !important;
    visibility: visible !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
}

/* Metric Delta */
[data-testid="stMetricDelta"] *, 
[data-testid="stMetricDelta"] {
    color: #E2E8F0 !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
            
/* Custom hover KPI cards */

.kpi-hover-card {
    position: relative;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 16px;
    padding: 20px 15px;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 100%; 
    min-height: 120px;
    overflow: hidden; 
}

.kpi-hover-card:hover {
    border-color: rgba(6, 182, 212, 0.4);
    box-shadow: 0 4px 20px rgba(6, 182, 212, 0.15);
    transform: translateY(-2px);
}

.kpi-label {
    color: #BFCBDA;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
}

.kpi-value {
    color: #FFFFFF;
    font-size: clamp(18px, 2vw, 26px); 
    font-weight: 800;
    line-height: 1.2;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    word-wrap: break-word;
}

.kpi-tooltip {
    position: absolute;
    left: 50%;
    bottom: calc(100% + 12px);
    transform: translateX(-50%) translateY(8px);
    background: rgba(8, 12, 24, 0.98);
    border: 1px solid rgba(6, 182, 212, 0.35);
    border-radius: 14px;
    padding: 12px 14px;
    min-width: 190px;
    text-align: center;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    opacity: 0;
    visibility: hidden;
    transition: all 0.22s ease;
    z-index: 9999;
    pointer-events: none;
}

.kpi-hover-card:hover .kpi-tooltip {
    opacity: 1;
    visibility: visible;
    transform: translateX(-50%) translateY(0);
}

.kpi-tooltip-title {
    color: #9FB3C8;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

.kpi-tooltip-value {
    color: #FFFFFF;
    font-size: 20px;
    font-weight: 800;
}

.kpi-tooltip::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border-width: 7px;
    border-style: solid;
    border-color: rgba(8, 12, 24, 0.98) transparent transparent transparent;
}

/* Buttons */

.stButton > button,
.stDownloadButton > button {
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    padding: 12px 28px;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button {
    background: linear-gradient(135deg, #06B6D4, #8B5CF6);
    box-shadow: 0 4px 15px rgba(6, 182, 212, 0.25);
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #8B5CF6, #06B6D4);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
}

/* DataFrame */

[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    width: 100%;
}

/* Text */

p, li {
    color: #AAB6C8 !important;
    font-size: 16px;
    line-height: 1.7;
}

strong {
    color: #F1F5F9 !important;
}

/* Tags */

.feature-tag {
    display: inline-block;
    background: rgba(6, 182, 212, 0.12);
    color: #06B6D4;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 14px;
    font-weight: 600;
    margin: 4px;
    border: 1px solid rgba(6, 182, 212, 0.2);
}

/* Expander */

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    font-weight: 600;
    color: #CBD5E1;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# CHART THEME
# =========================================================

CHART_BG = "#0B0F1A"
GRID_COLOR = (1, 1, 1, 0.06)
ACCENT_1 = "#06B6D4"
ACCENT_2 = "#8B5CF6"
ACCENT_3 = "#F59E0B"
ACCENT_4 = "#10B981"
TEXT_COLOR = "#B8C4D6"

PALETTE = [ACCENT_1, ACCENT_2, ACCENT_3, ACCENT_4, "#EF4444", "#EC4899"]

def style_axis(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    if title:
        ax.set_title(title, color="#F8FAFC", fontsize=16, fontweight=700, pad=16)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=12, fontweight=600)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=12, fontweight=600)

def kpi_hover_card(title, value):
    st.markdown(
        f"""
        <div class="kpi-hover-card">
            <div class="kpi-tooltip">
                <div class="kpi-tooltip-title">{title}</div>
                <div class="kpi-tooltip-value">{value}</div>
            </div>
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():
    try:
        return pd.read_csv("Superstore.csv", encoding="latin1")
    except FileNotFoundError:
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        return pd.DataFrame({
            'Order Date': dates,
            'Sales': np.random.uniform(50, 1000, 1000),
            'Profit': np.random.uniform(-50, 200, 1000),
            'Category': np.random.choice(['Technology', 'Furniture', 'Office Supplies'], 1000),
            'Sub-Category': np.random.choice(['Phones', 'Chairs', 'Binders', 'Storage', 'Art'], 1000),
            'Region': np.random.choice(['West', 'East', 'Central', 'South'], 1000),
            'Segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], 1000),
            'Ship Mode': np.random.choice(['Standard Class', 'Second Class', 'First Class'], 1000),
            'Customer ID': np.random.choice(['C1', 'C2', 'C3', 'C4'], 1000),
            'Product Name': np.random.choice(['Product A', 'Product B', 'Product C'], 1000)
        })

df = load_data()

# =========================================================
# PREPROCESSING
# =========================================================

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
df['Quarter'] = df['Order Date'].dt.quarter
df['Month Name'] = df['Order Date'].dt.strftime('%b')
df['Year-Month'] = df['Order Date'].dt.to_period('M').astype(str)

sales_df = df.groupby('Order Date', as_index=False)['Sales'].sum()

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load("prophet_model.pkl")
    except:
        return None

model = load_model()

# =========================================================
# FORMATTING FUNCTIONS
# =========================================================

def format_kpi_currency(num):
    """Formats large numbers into K or M for cleaner visuals."""
    if num >= 1_000_000:
        return f"${num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"${num/1_000:.0f}K"
    else:
        return f"${num:,.0f}"

def format_kpi_number(num):
    """Formats large integers into K or M."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.0f}K"
    else:
        return f"{num:,}"

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 35px 15px; border-radius: 12px; margin-bottom: 20px;
                background: linear-gradient(rgba(10, 15, 30, 0.1), rgba(10, 15, 30, 0.95)),
                            url('https://www.shutterstock.com/image-photo/dark-theme-features-financial-ticker-260nw-2585764759.jpg') no-repeat center center;
                background-size: cover;
                border: 1px solid rgba(6, 182, 212, 0.3); box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
        <h2 style="font-size:1.6rem; font-weight:900; color:#FFFFFF !important; margin-bottom: 0px; text-shadow: 0px 2px 4px rgba(0,0,0,0.8);">
            ⚡ SalesForge AI
        </h2>
        <p style="font-size:13px; color:#06B6D4 !important; font-weight: 700; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px;">
            Predictive Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📊 Dashboard",
            "📉 Analytics",
            "📈 Forecasting",
            "🧠 Insights"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.metric("Total Revenue", format_kpi_currency(df['Sales'].sum()))
    st.metric("Total Orders", format_kpi_number(len(df)))
    st.metric("Avg Order Value", format_kpi_currency(df['Sales'].mean()))

# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title("SalesForge AI")

    st.markdown("""
    <p style="font-size:20px; color:#64748B; margin-top:-10px;">
    AI-powered sales forecasting & analytics platform built on Prophet
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Revenue Analyzed", format_kpi_currency(df['Sales'].sum()))
    c2.metric("Total Orders", format_kpi_number(len(df)))

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#06B6D4 !important; font-size:14px !important;">📊 Real-Time Dashboard</h3>
            <p style="color:#64748B; font-size:14px;">
            Live KPIs, category breakdowns, and regional performance at a glance.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#8B5CF6 !important; font-size:14px !important;">🤖 AI Forecasting</h3>
            <p style="color:#64748B; font-size:14px;">
            Prophet-powered predictions with confidence intervals up to 365 days ahead.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#F59E0B !important; font-size:14px !important;">🧠 Smart Insights</h3>
            <p style="color:#64748B; font-size:14px;">
            Data-driven recommendations for inventory, marketing, and growth strategy.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Revenue Trajectory")

    yearly_sales = df.groupby('Year')['Sales'].sum()

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    ax.fill_between(yearly_sales.index, yearly_sales.values, alpha=0.15, color=ACCENT_1)
    ax.plot(
        yearly_sales.index, yearly_sales.values,
        color=ACCENT_1, linewidth=3, marker='o',
        markersize=10, markerfacecolor=ACCENT_1,
        markeredgecolor='white', markeredgewidth=2
    )

    for x, y in zip(yearly_sales.index, yearly_sales.values):
        ax.annotate(
            f'${y/1000:.0f}K',
            (x, y),
            textcoords="offset points",
            xytext=(0, 18),
            ha='center',
            fontsize=12,
            fontweight='700',
            color='#F8FAFC'
        )

    style_axis(ax, ylabel="Revenue ($)")
    ax.set_xlabel("")
    st.pyplot(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Capabilities")

    tags = [
        "Time Series Forecasting", "Trend Decomposition", "Seasonal Analysis",
        "Category Intelligence", "Regional Heatmaps", "Export Reports",
        "Confidence Intervals", "Interactive Filtering", "Growth Metrics"
    ]
    tag_html = "".join(f'<span class="feature-tag">{t}</span>' for t in tags)
    st.markdown(f'<div style="line-height:2.8">{tag_html}</div>', unsafe_allow_html=True)

# =========================================================
# DASHBOARD PAGE
# =========================================================

elif page == "📊 Dashboard":

    st.title("Sales Dashboard")

    total_sales = df['Sales'].sum()
    avg_sales = df['Sales'].mean()
    max_sales = df['Sales'].max()
    total_orders = len(df)
    unique_customers = df['Customer ID'].nunique()
    profit_margin = (df['Profit'].sum() / total_sales) * 100 if total_sales != 0 else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(f"""
    <div class="kpi-hover-card">
        <div class="kpi-tooltip">
            <div class="kpi-tooltip-title">Total Revenue</div>
            <div class="kpi-tooltip-value">${total_sales:,.0f}</div>
        </div>
        <div class="kpi-label">Total Revenue</div>
        <div class="kpi-value">{format_kpi_currency(total_sales)}</div>
    </div>
    """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
    <div class="kpi-hover-card">
        <div class="kpi-tooltip">
            <div class="kpi-tooltip-title">Avg Order</div>
            <div class="kpi-tooltip-value">${avg_sales:,.0f}</div>
        </div>
        <div class="kpi-label">Avg Order</div>
        <div class="kpi-value">{format_kpi_currency(avg_sales)}</div>
    </div>
    """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
    <div class="kpi-hover-card">
        <div class="kpi-tooltip">
            <div class="kpi-tooltip-title">Max Order</div>
            <div class="kpi-tooltip-value">${max_sales:,.0f}</div>
        </div>
        <div class="kpi-label">Max Order</div>
        <div class="kpi-value">{format_kpi_currency(max_sales)}</div>
    </div>
    """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
    <div class="kpi-hover-card">
        <div class="kpi-tooltip">
            <div class="kpi-tooltip-title">Total Orders</div>
            <div class="kpi-tooltip-value">{total_orders:,}</div>
        </div>
        <div class="kpi-label">Total Orders</div>
        <div class="kpi-value">{format_kpi_number(total_orders)}</div>
    </div>
    """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
    <div class="kpi-hover-card">
        <div class="kpi-tooltip">
            <div class="kpi-tooltip-title">Customers</div>
            <div class="kpi-tooltip-value">{unique_customers:,}</div>
        </div>
        <div class="kpi-label">Customers</div>
        <div class="kpi-value">{format_kpi_number(unique_customers)}</div>
    </div>
    """, unsafe_allow_html=True)

    with c6:
        st.markdown(f"""
    <div class="kpi-hover-card">
        <div class="kpi-tooltip">
            <div class="kpi-tooltip-title">Profit Margin</div>
            <div class="kpi-tooltip-value">{profit_margin:.1f}%</div>
        </div>
        <div class="kpi-label">Profit Margin</div>
        <div class="kpi-value">{profit_margin:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)


    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Category Revenue")
        category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(CHART_BG)
        ax.set_facecolor(CHART_BG)

        bars = ax.barh(
            category_sales.index,
            category_sales.values,
            color=PALETTE[:len(category_sales)],
            height=0.55,
            edgecolor='none'
        )

        for bar, val in zip(bars, category_sales.values):
            ax.text(
                val - val * 0.02, bar.get_y() + bar.get_height() / 2,
                f' ${val:,.0f}',
                va='center', ha='right',
                fontsize=12, fontweight='700', color='white'
            )

        style_axis(ax, xlabel="Revenue ($)")
        ax.tick_params(axis='y', labelsize=13, labelcolor='#F8FAFC')
        st.pyplot(fig, use_container_width=True)

    with c2:
        st.subheader("Region Revenue")
        region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(CHART_BG)
        ax.set_facecolor(CHART_BG)

        wedges, texts, autotexts = ax.pie(
            region_sales.values,
            labels=region_sales.index,
            autopct='%1.1f%%',
            colors=PALETTE[:len(region_sales)],
            startangle=90,
            textprops={'color': '#F8FAFC', 'fontsize': 12, 'fontweight': '600'},
            pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor=CHART_BG, linewidth=3)
        )

        for t in autotexts:
            t.set_fontsize(11)
            t.set_fontweight('700')
            t.set_color('white')

        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Top 15 Sub-Categories")
    subcat_sales = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(subcat_sales)))

    bars = ax.barh(
        subcat_sales.index,
        subcat_sales.values,
        color=colors,
        height=0.65,
        edgecolor='none'
    )

    for bar, val in zip(bars, subcat_sales.values):
        ax.text(
            val + total_sales * 0.003, bar.get_y() + bar.get_height() / 2,
            f'${val:,.0f}',
            va='center', ha='left',
            fontsize=11, fontweight='600', color='#D7E1EE'
        )

    style_axis(ax, xlabel="Revenue ($)")
    ax.tick_params(axis='y', labelsize=12, labelcolor='#F8FAFC')
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    with st.expander("📄 Dataset Preview", expanded=False):
        st.dataframe(
            df[['Order Date', 'Customer ID', 'Category',
                'Sub-Category', 'Region', 'Sales', 'Profit']].head(30),
            use_container_width=True
        )

# =========================================================
# ANALYTICS PAGE
# =========================================================

elif page == "📉 Analytics":

    st.title("Deep Analytics")

    st.subheader("Revenue Over Time")

    monthly_agg = df.groupby('Year-Month')['Sales'].sum().reset_index()
    monthly_agg = monthly_agg.sort_values('Year-Month')

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    ax.fill_between(range(len(monthly_agg)), monthly_agg['Sales'].values, alpha=0.12, color=ACCENT_1)
    ax.plot(range(len(monthly_agg)), monthly_agg['Sales'].values, color=ACCENT_1, linewidth=2.5)

    style_axis(ax, ylabel="Revenue ($)")
    ax.set_xticks(range(0, len(monthly_agg), max(1, len(monthly_agg)//8)))
    ax.set_xticklabels(monthly_agg['Year-Month'].values[::max(1, len(monthly_agg)//8)], rotation=45, fontsize=10)
    st.pyplot(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Monthly Revenue")
        monthly_sum = df.groupby('Month')['Sales'].sum()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(CHART_BG)
        ax.set_facecolor(CHART_BG)

        ax.bar(
            month_names, monthly_sum.values,
            color=[ACCENT_1 if v >= monthly_sum.mean() else (0.2, 0.24, 0.33, 0.6)
                   for v in monthly_sum.values],
            width=0.65,
            edgecolor='none'
        )

        ax.axhline(
            monthly_sum.mean(), color=ACCENT_2,
            linestyle='--', linewidth=1.5, alpha=0.7,
            label=f'Average: ${monthly_sum.mean():,.0f}'
        )

        ax.legend(facecolor=CHART_BG, edgecolor=GRID_COLOR, labelcolor='#CBD5E1', fontsize=11)
        style_axis(ax, ylabel="Revenue ($)")
        ax.tick_params(axis='x', labelsize=12, labelcolor='#F8FAFC')
        st.pyplot(fig, use_container_width=True)

    with c2:
        st.subheader("Quarterly Revenue")
        quarterly = df.groupby('Quarter')['Sales'].sum()
        q_labels = [f'Q{q}' for q in quarterly.index]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(CHART_BG)
        ax.set_facecolor(CHART_BG)

        ax.bar(
            q_labels, quarterly.values,
            color=[ACCENT_2, ACCENT_1, ACCENT_3, ACCENT_4][:len(quarterly)],
            width=0.5,
            edgecolor='none'
        )

        total_q = quarterly.sum()
        for i, (q, val) in enumerate(zip(q_labels, quarterly.values)):
            ax.text(
                i, val + total_q * 0.005,
                f'${val:,.0f}',
                ha='center', fontsize=12,
                fontweight='700', color='#F8FAFC'
            )

        style_axis(ax, ylabel="Revenue ($)")
        ax.tick_params(axis='x', labelsize=14, labelcolor='#F8FAFC')
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Segment Revenue")
        seg_sales = df.groupby('Segment')['Sales'].sum().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(CHART_BG)
        ax.set_facecolor(CHART_BG)

        wedges, texts, autotexts = ax.pie(
            seg_sales.values,
            labels=seg_sales.index,
            autopct='%1.1f%%',
            colors=PALETTE[:len(seg_sales)],
            startangle=90,
            textprops={'color': '#F8FAFC', 'fontsize': 13, 'fontweight': '600'},
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor=CHART_BG, linewidth=3)
        )

        for t in autotexts:
            t.set_fontsize(11)
            t.set_fontweight('700')
            t.set_color('white')

        st.pyplot(fig, use_container_width=True)

    with c2:
        st.subheader("Ship Mode Distribution")
        ship_sales = df.groupby('Ship Mode')['Sales'].sum().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(CHART_BG)
        ax.set_facecolor(CHART_BG)

        ax.barh(
            ship_sales.index, ship_sales.values,
            color=[ACCENT_4, ACCENT_1, ACCENT_2, ACCENT_3][:len(ship_sales)],
            height=0.5,
            edgecolor='none'
        )

        total_ship = ship_sales.sum()
        for i, val in enumerate(ship_sales.values):
            ax.text(
                val + total_ship * 0.003, i,
                f'${val:,.0f}',
                va='center', fontsize=11,
                fontweight='600', color='#D7E1EE'
            )

        style_axis(ax, xlabel="Revenue ($)")
        ax.tick_params(axis='y', labelsize=12, labelcolor='#F8FAFC')
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include=['number'])

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, annot=True,
        cmap='RdYlGn', center=0,
        fmt='.2f', linewidths=0.5,
        linecolor=CHART_BG,
        cbar_kws={'shrink': 0.8},
        annot_kws={'fontsize': 11, 'fontweight': '600'},
        ax=ax
    )

    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    st.pyplot(fig, use_container_width=True)

# =========================================================
# FORECASTING PAGE
# =========================================================

elif page == "📈 Forecasting":

    st.title("AI Sales Forecasting")

    st.markdown("""
    <p style="color:#64748B; font-size:16px; margin-top:-8px;">
    Prophet time-series model predicts future revenue with confidence bands
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if model is None:
        st.warning("Prophet model not found for demonstration.")
    else:
        c1, c2 = st.columns([1, 3])

        with c1:
            days = st.slider("Forecast Horizon (days)", 30, 365, 90, step=30)

            st.markdown("<br>", unsafe_allow_html=True)

            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)

            latest = forecast['yhat'].iloc[-1]
            peak = forecast['yhat'].max()
            avg = forecast['yhat'].mean()
            first_yhat = forecast['yhat'].iloc[0]
            growth = ((latest - first_yhat) / abs(first_yhat)) * 100 if first_yhat != 0 else 0.0

            st.metric("Latest Forecast", format_kpi_currency(latest))
            st.metric("Peak Forecast", format_kpi_currency(peak))
            st.metric("Average Forecast", format_kpi_currency(avg))
            st.metric("Projected Growth", f"{growth:+.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days).to_csv(index=False)

            st.download_button(
                label="⬇ Download Forecast CSV",
                data=csv,
                file_name=f'sales_forecast_{days}d.csv',
                mime='text/csv'
            )

        with c2:
            st.subheader("Forecast with Confidence Band")

            cutoff_date = df['Order Date'].max()
            start_plot_date = cutoff_date - pd.Timedelta(days=days * 2)
            forecast_tail = forecast[forecast['ds'] >= start_plot_date]

            fig, ax = plt.subplots(figsize=(14, 6))
            fig.patch.set_facecolor(CHART_BG)
            ax.set_facecolor(CHART_BG)

            ax.fill_between(
                pd.to_datetime(forecast_tail['ds']),
                forecast_tail['yhat_lower'],
                forecast_tail['yhat_upper'],
                alpha=0.12,
                color=ACCENT_1,
                label='95% Confidence'
            )

            ax.plot(
                pd.to_datetime(forecast_tail['ds']),
                forecast_tail['yhat'],
                color=ACCENT_1, linewidth=2.5,
                label='Forecast'
            )

            historical = sales_df[sales_df['Order Date'] >= start_plot_date]
            if len(historical) > 0:
                ax.scatter(
                    historical['Order Date'],
                    historical['Sales'],
                    color=ACCENT_3, s=8, alpha=0.6,
                    label='Actual', zorder=5
                )

            forecast_start_mask = forecast['ds'] > cutoff_date
            if forecast_start_mask.any():
                forecast_start = forecast.loc[forecast_start_mask, 'ds'].min()
                ax.axvline(
                    x=forecast_start, color=ACCENT_2,
                    linestyle='--', linewidth=1.5, alpha=0.7,
                    label='Forecast Start'
                )

            ax.legend(
                facecolor=CHART_BG, edgecolor=GRID_COLOR,
                labelcolor='#CBD5E1', fontsize=11, loc='upper left'
            )

            style_axis(ax, ylabel="Revenue ($)")
            st.pyplot(fig, use_container_width=True)

        st.markdown("---")

        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Trend Component")

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor(CHART_BG)
            ax.set_facecolor(CHART_BG)

            ax.plot(pd.to_datetime(forecast['ds']), forecast['trend'], color=ACCENT_2, linewidth=2.5)
            ax.fill_between(pd.to_datetime(forecast['ds']), forecast['trend'], alpha=0.08, color=ACCENT_2)

            style_axis(ax, ylabel="Trend")
            st.pyplot(fig, use_container_width=True)

        with c4:
            st.subheader("Weekly Seasonality")

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor(CHART_BG)
            ax.set_facecolor(CHART_BG)

            if 'weekly' in forecast.columns:
                weekly = forecast.groupby(forecast['ds'].dt.dayofweek)['weekly'].mean()
                days_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

                ax.bar(
                    days_labels, weekly.values,
                    color=ACCENT_3, width=0.55,
                    edgecolor='none', alpha=0.85
                )

            style_axis(ax, ylabel="Seasonal Effect")
            st.pyplot(fig, use_container_width=True)

        st.markdown("---")

        with st.expander("📄 Forecast Data Table", expanded=False):
            forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days).copy()
            forecast_data.columns = ['Date', 'Predicted', 'Lower Bound', 'Upper Bound']
            forecast_data['Date'] = pd.to_datetime(forecast_data['Date']).dt.strftime('%Y-%m-%d')
            st.dataframe(forecast_data, use_container_width=True)

# =========================================================
# INSIGHTS PAGE
# =========================================================

elif page == "🧠 Insights":

    st.title("Business Intelligence")

    top_category = df.groupby('Category')['Sales'].sum().idxmax()
    top_region = df.groupby('Region')['Sales'].sum().idxmax()
    top_segment = df.groupby('Segment')['Sales'].sum().idxmax()
    top_subcat = df.groupby('Sub-Category')['Sales'].sum().idxmax()

    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    monthly_sales = df.groupby('Month')['Sales'].sum()
    best_month = monthly_sales.idxmax()
    worst_month = monthly_sales.idxmin()
    best_month_name = month_names[best_month - 1]
    worst_month_name = month_names[worst_month - 1]

    yoy = df.groupby('Year')['Sales'].sum()
    if len(yoy) >= 2 and yoy.iloc[-2] != 0:
        latest_year = yoy.index[-1]
        prev_year = yoy.index[-2]
        yoy_growth = ((yoy.iloc[-1] - yoy.iloc[-2]) / yoy.iloc[-2]) * 100
    else:
        latest_year = yoy.index[-1]
        prev_year = yoy.index[0]
        yoy_growth = 0.0

    top_product = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(1)
    top_product_name = top_product.index[0]

    loss_products = df.groupby('Product Name')['Profit'].sum().sort_values().head(5)

    st.markdown("""
    <p style="color:#64748B; font-size:16px; margin-top:-8px; margin-bottom:30px;">
    Auto-generated insights from your data
    </p>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color:#06B6D4 !important; font-size:13px !important; margin-bottom:15px;">
                🏆 Top Performers
            </h3>
            <p><strong>Category:</strong> {top_category}</p>
            <p><strong>Sub-Category:</strong> {top_subcat}</p>
            <p><strong>Region:</strong> {top_region}</p>
            <p><strong>Segment:</strong> {top_segment}</p>
            <p><strong>Product:</strong> {top_product_name[:50]}</p>
            <p><strong>Peak Month:</strong> {best_month_name}</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color:#F59E0B !important; font-size:13px !important; margin-bottom:15px;">
                📊 Growth Metrics
            </h3>
            <p><strong>YoY Growth ({prev_year}→{latest_year}):</strong> {yoy_growth:+.1f}%</p>
            <p><strong>Peak Revenue Month:</strong> {best_month_name}</p>
            <p><strong>Lowest Revenue Month:</strong> {worst_month_name}</p>
            <p><strong>Total Unique Customers:</strong> {df['Customer ID'].nunique():,}</p>
            <p><strong>Avg Orders per Customer:</strong> {len(df)/df['Customer ID'].nunique():.1f}</p>
            <p><strong>Profit Margin:</strong> {(df['Profit'].sum()/df['Sales'].sum()*100):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("⚠️ Top 5 Loss-Making Products")

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    loss_names = [n[:35] + '...' if len(n) > 35 else n for n in loss_products.index]

    ax.barh(
        loss_names, loss_products.values,
        color='#EF4444', height=0.55,
        edgecolor='none', alpha=0.8
    )

    for i, val in enumerate(loss_products.values):
        ax.text(
            val - abs(val) * 0.02, i,
            f' ${val:,.0f}',
            va='center', ha='right',
            fontsize=11, fontweight='700', color='white'
        )

    style_axis(ax, xlabel="Profit ($)")
    ax.tick_params(axis='y', labelsize=11, labelcolor='#F8FAFC')
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Strategic Recommendations")

    recs = [
        {
            "icon": "📦",
            "title": "Inventory Optimization",
            "text": f"Stock up on {top_subcat} before {best_month_name} — the highest revenue month. Consider reducing inventory for loss-making products."
        },
        {
            "icon": "🎯",
            "title": "Targeted Marketing",
            "text": f"Focus campaigns on the {top_segment} segment and {top_region} region where ROI is highest."
        },
        {
            "icon": "📉",
            "title": "Loss Mitigation",
            "text": "Review pricing strategy for top loss-making products. Consider bundling or discontinuing underperformers."
        },
        {
            "icon": "🌍",
            "title": "Regional Expansion",
            "text": "Analyze underperforming regions and replicate strategies from top-performing areas."
        },
        {
            "icon": "🤖",
            "title": "AI-Driven Planning",
            "text": "Use Prophet forecasts to align purchasing cycles with predicted demand spikes."
        }
    ]

    for rec in recs:
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:12px;">
            <p style="margin:0; color:#F1F5F9 !important;">
                <span style="font-size:20px;">{rec['icon']}</span>
                <strong style="color:#06B6D4;">{rec['title']}</strong><br>
                <span style="color:#94A3B8; font-size:15px;">{rec['text']}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    