import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Enterprise Sales Intelligence Dashboard",
    layout="wide"
)

# --------------------------------------------------
# INR FORMAT
# --------------------------------------------------
def format_inr(x):
    if pd.isna(x):
        return "₹0"
    x = int(round(x))
    s = str(x)
    if len(s) <= 3:
        return "₹" + s
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return "₹" + ",".join(parts + [last3])

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("product_sales_dataset_final.csv")
    df.columns = df.columns.str.strip()

    df['Order_Date'] = pd.to_datetime(
        df['Order_Date'],
        errors='coerce'
    )

    df = df.dropna(subset=['Order_Date'])

    return df

df = load_data()

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
df['Year'] = df['Order_Date'].dt.year
df['Month_Period'] = df['Order_Date'].dt.to_period('M').dt.to_timestamp()
df['Month_Label'] = df['Month_Period'].dt.strftime('%b %Y')
df['Month_Name'] = df['Month_Period'].dt.strftime('%B')

# --------------------------------------------------
# HEADER
# --------------------------------------------------
logo = Image.open("download (1).jpg")

c1, c2 = st.columns([1,6])
with c1:
    st.image(logo, width=110)

with c2:
    st.markdown(
        "<h1 style='margin-top:25px;'>Enterprise Sales Intelligence Dashboard</h1>",
        unsafe_allow_html=True
    )

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Dashboard Controls 📊")

year_filter = st.sidebar.multiselect(
    "Year",
    sorted(df['Year'].unique()),
    default=sorted(df['Year'].unique())
)

category_filter = st.sidebar.multiselect(
    "Category",
    sorted(df['Category'].unique()),
    default=sorted(df['Category'].unique())
)


region_filter = st.sidebar.multiselect(
    "Region",
    sorted(df['Region'].unique()),
    default=sorted(df['Region'].unique())
)

month_filter = st.sidebar.multiselect(
    "Month",
    df.sort_values('Month_Period')['Month_Name'].unique(),
    default=df.sort_values('Month_Period')['Month_Name'].unique()
)

# --------------------------------------------------
# FILTERED DATA
# --------------------------------------------------
filtered_df = df[
    (df['Year'].isin(year_filter)) &
    (df['Category'].isin(category_filter)) &
    (df['Month_Name'].isin(month_filter))
]

# --------------------------------------------------
# MONTHLY DATA
# --------------------------------------------------
monthly_base = (
    filtered_df
    .groupby('Month_Period')['Revenue']
    .sum()
    .reset_index()
    .sort_values('Month_Period')
)

monthly_base['Growth_Rate'] = monthly_base['Revenue'].pct_change()
monthly_base['Month_Label'] = monthly_base['Month_Period'].dt.strftime('%b %Y')

# --------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------
total_revenue = monthly_base['Revenue'].sum()
avg_monthly_revenue = monthly_base['Revenue'].mean()

previous_month_revenue = 0
next_month_revenue = 0
expected_growth = 0

if len(monthly_base) >= 1:

    current_month_period = monthly_base.iloc[-1]['Month_Period']
    current_month_number = current_month_period.month

    prev_month_number = 12 if current_month_number == 1 else current_month_number - 1

    previous_month_revenue = df[
        df['Order_Date'].dt.month == prev_month_number
    ]['Revenue'].sum()

    next_month_number = 1 if current_month_number == 12 else current_month_number + 1

    next_month_revenue = df[
        (df['Order_Date'].dt.month == next_month_number) &
        (df['Year'].isin(year_filter))
    ]['Revenue'].sum()

    if len(monthly_base) >= 2:
        growth_rate = monthly_base.iloc[-1]['Growth_Rate']
        if pd.notna(growth_rate):
            expected_growth = growth_rate * 100

# --------------------------------------------------
# KPI DISPLAY
# --------------------------------------------------
st.markdown("## 📌 Key Performance Indicators")

k1,k2,k3,k4,k5 = st.columns(5)

k1.metric("Total Revenue", format_inr(total_revenue))
k2.metric("Avg Monthly Revenue", format_inr(avg_monthly_revenue))
k3.metric("Previous Month Revenue", format_inr(previous_month_revenue))
k4.metric("Next Month Revenue", format_inr(next_month_revenue))
k5.metric("Expected Growth (%)", f"{expected_growth:.2f}%")

# --------------------------------------------------
# MONTHLY TREND
# --------------------------------------------------
st.markdown("## 📈 Monthly Revenue Trend")

fig = px.line(
    monthly_base,
    x="Month_Label",
    y="Revenue",
    markers=True
)

fig.update_traces(
    hovertemplate="Month: %{x}<br>Revenue: ₹%{y:,.0f}<extra></extra>"
)

fig.update_layout(
    template="plotly_dark",
    yaxis_tickprefix="₹ ",
    xaxis_title="Month",
    yaxis_title="Revenue (INR)"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# DAILY FORECAST
# --------------------------------------------------
st.markdown("## 🔮 Day Wise Forecast – December 2024")

daily_full = (
    df.groupby('Order_Date')['Revenue']
    .sum()
    .reset_index()
    .rename(columns={'Order_Date':'ds','Revenue':'y'})
    .sort_values('ds')
)

if len(daily_full) >= 60:

    model = Prophet(daily_seasonality=True)
    model.fit(daily_full)

    future = model.make_future_dataframe(periods=31,freq='D')
    forecast = model.predict(future)

    december_daily = forecast[
        (forecast['ds']>="2024-12-01") &
        (forecast['ds']<="2024-12-31")
    ]

    forecast_fig = go.Figure()

    forecast_fig.add_trace(go.Scatter(
        x=daily_full['ds'],
        y=daily_full['y'],
        mode='lines',
        name='Actual Revenue'
    ))

    forecast_fig.add_trace(go.Scatter(
        x=december_daily['ds'],
        y=december_daily['yhat'],
        mode='lines+markers',
        name='Predicted Revenue'
    ))

    forecast_fig.update_layout(
        template="plotly_dark",
        yaxis_tickprefix="₹ ",
        xaxis_title="Date",
        yaxis_title="Revenue (INR)"
    )

    st.plotly_chart(forecast_fig,use_container_width=True)

    total_december = december_daily['yhat'].sum()

    st.success(f"📊 Estimated TOTAL December 2024 Revenue: {format_inr(total_december)}")

# --------------------------------------------------
# ENTERPRISE INSIGHT SECTION
# --------------------------------------------------
st.markdown("## 🧠 Key Business Insights")

st.success("""
• Electronics drives 40% of total revenue but has the lowest profit margin (~14%).

• Accessories and Apparel have the highest margins (33%+), making them strong profit drivers.

• November revenue is around 130% higher than average due to festive and promotional demand.

• February is the weakest month with major revenue decline compared to peak season.

• California and Arizona generate over $13.4M combined, showing heavy regional dependency.

• Two products - Tempur-Pedic Mattress and Instant Pot — contribute nearly 25% of total profit.

• Year-over-year revenue growth is only about +1.26%, indicating possible market saturation.
""")




