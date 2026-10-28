import base64
from io import BytesIO
from datetime import datetime
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fpdf import FPDF

st.set_page_config(
    page_title="Customer Behavior Analytics & Business Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGO_URL = "https://img.icons8.com/ios-filled/100/ffffff/customer-insight.png"

VALID_USERS = {
    "admin": "Business2026!",
    "analyst": "Analytics@2026",
}

AGE_GROUPS = [18, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]


def inject_css():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        body {
            background: #0b1020;
            color: #e5e7eb;
        }
        section[data-testid="stSidebar"] {
            background: rgba(12, 22, 48, 0.95);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        .stApp {
            background: linear-gradient(180deg, #09111f 0%, #0b1020 100%);
            color: #e5e7eb;
        }
        .glass-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(18px);
        }
        .kpi-card {
            padding: 24px;
            border-radius: 20px;
            background: rgba(14, 23, 44, 0.85);
            border: 1px solid rgba(255,255,255,0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
        }
        .kpi-value {
            font-size: 2.8rem;
            font-weight: 700;
            animation: glow 2.5s ease-in-out infinite alternate;
        }
        .kpi-label {
            color: rgba(226, 232, 240, 0.84);
            margin-top: 4px;
        }
        @keyframes glow {
            from {text-shadow: 0 0 10px rgba(255, 255, 255, 0.1);}
            to {text-shadow: 0 0 16px rgba(56, 189, 248, 0.35);}
        }
        html, body, .stApp {
            min-height: 100vh;
            height: 100%;
        }
        .login-page {
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 16px 0;
            position: fixed;
            inset: 0;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(6, 10, 25, 0.88), rgba(4, 8, 18, 0.88)),
                        url('https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1600&q=80');
            background-attachment: fixed;
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-blend-mode: overlay;
        }
        .login-page::before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 22%),
                        radial-gradient(circle at bottom right, rgba(16, 185, 129, 0.12), transparent 18%);
            pointer-events: none;
        }
        .login-card {
            position: relative;
            z-index: 1;
            background: rgba(8, 13, 24, 0.94);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 28px;
            padding: 26px 24px;
            max-width: 420px;
            width: 100%;
            box-shadow: 0 28px 80px rgba(0, 0, 0, 0.45);
            backdrop-filter: blur(26px);
            overflow: hidden;
            max-height: calc(100vh - 40px);
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .login-card::before {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 36px;
            padding: 1px;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(139, 92, 246, 0.16));
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: destination-out;
            mask-composite: exclude;
            pointer-events: none;
        }
        .login-card .login-header {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
            margin-bottom: 20px;
        }
        .login-card .login-tag {
            color: #38bdf8;
            font-size: 0.78rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            font-weight: 700;
        }
        .login-card .login-title {
            color: #f8fafc;
            margin: 0;
            font-size: 1.85rem;
            line-height: 1.1;
        }
        .login-card .login-subtitle {
            color: rgba(226, 232, 240, 0.78);
            margin: 0;
            font-size: 0.95rem;
            max-width: 90%;
            text-align: center;
            line-height: 1.6;
        }
        .login-card input {
            background: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            color: #f8fafc !important;
            border-radius: 16px;
            padding: 16px 18px !important;
            width: 100% !important;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .login-card input:focus {
            outline: none !important;
            border-color: rgba(59, 130, 246, 0.85) !important;
            box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.3), 0 0 0 4px rgba(59, 130, 246, 0.08);
        }
        .login-actions {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            margin-top: 12px;
            margin-bottom: 18px;
        }
        .login-actions .forgot-link {
            color: #93c5fd;
            font-size: 0.9rem;
            text-decoration: none;
        }
        .login-actions .forgot-link:hover {
            text-decoration: underline;
        }
        .login-card .login-button {
            margin-top: 8px;
        }
        .login-card .login-footer {
            margin-top: 18px;
            color: #94a3b8;
            text-align: center;
            font-size: 0.9rem;
        }
        .login-card .login-cta {
            color: #67e8f9;
            font-weight: 600;
        }
        .register-text {
            margin-top: 8px;
            color: #cbd5e1;
            font-size: 0.92rem;
        }
        .stButton>button {
            background: linear-gradient(135deg, #2563eb, #0ea5e9);
            color: white;
            border: none;
            padding: 0.8rem 1rem;
            border-radius: 12px;
            font-size: 1rem;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #1d4ed8, #0284c7);
        }
        .logo-title {
            color: #f8fafc;
        }
        .logo-subtitle {
            color: #94a3b8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_session_defaults():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "data" not in st.session_state:
        st.session_state.data = None


def authenticate(username: str, password: str) -> bool:
    return VALID_USERS.get(username) == password


def generate_sample_data(num_records=120):
    np.random.seed(42)
    categories = ["Electronics", "Apparel", "Home", "Beauty", "Sports", "Toys"]
    genders = ["Male", "Female", "Non-binary", "Unknown"]
    dates = pd.date_range(end=datetime.today(), periods=num_records, freq="D")

    return pd.DataFrame(
        {
            "customerid": np.random.randint(1001, 1040, size=num_records),
            "customername": [f"Customer {i}" for i in range(1, num_records + 1)],
            "age": np.random.randint(18, 70, size=num_records),
            "gender": np.random.choice(genders, size=num_records),
            "productcategory": np.random.choice(categories, size=num_records),
            "purchaseamount": np.round(np.random.uniform(15, 450, size=num_records), 2),
            "quantity": np.random.randint(1, 6, size=num_records),
            "purchasedate": np.random.choice(dates, size=num_records),
        }
    )


@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists("customer_shopping_behavior.csv"):
        df = pd.read_csv("customer_shopping_behavior.csv")
    else:
        df = generate_sample_data()

    df.columns = [col.strip() for col in df.columns]
    return df


def normalize_columns(df):
    lower_cols = {col: col.lower() for col in df.columns}
    df = df.rename(columns=lower_cols)

    if "revenue" not in df.columns:
        if "purchaseamount" in df.columns and "quantity" in df.columns:
            df["revenue"] = df["purchaseamount"] * df["quantity"]
        elif "totalamount" in df.columns:
            df["revenue"] = df["totalamount"]
        elif "sales" in df.columns:
            df["revenue"] = df["sales"]
        elif "purchaseamount" in df.columns:
            df["revenue"] = df["purchaseamount"]
        elif "orderamount" in df.columns:
            df["revenue"] = df["orderamount"]

    if "purchaseamount" not in df.columns:
        if "revenue" in df.columns:
            df["purchaseamount"] = df["revenue"]
        elif "orderamount" in df.columns:
            df["purchaseamount"] = df["orderamount"]

    if "customerid" not in df.columns:
        if "clientid" in df.columns:
            df = df.rename(columns={"clientid": "customerid"})
        elif "id" in df.columns:
            df = df.rename(columns={"id": "customerid"})

    if "productcategory" not in df.columns:
        for candidate in ["category", "product", "categoryname", "itemcategory"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "productcategory"})
                break

    if "purchasedate" not in df.columns:
        for candidate in ["date", "orderdate", "transactiondate"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "purchasedate"})
                break

    if "age" not in df.columns:
        for candidate in ["customerage", "agegroup"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "age"})
                break

    if "gender" not in df.columns:
        for candidate in ["sex", "customersex"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "gender"})
                break

    if "purchaseid" not in df.columns and "transactionid" in df.columns:
        df = df.rename(columns={"transactionid": "purchaseid"})

    if "purchaseid" not in df.columns and "invoice" in df.columns:
        df = df.rename(columns={"invoice": "purchaseid"})

    if "quantity" not in df.columns and "qty" in df.columns:
        df = df.rename(columns={"qty": "quantity"})

    return df


def enrich_data(df):
    df = normalize_columns(df)

    if "purchasedate" in df.columns:
        df["purchasedate"] = pd.to_datetime(df["purchasedate"], errors="coerce")
    else:
        df["purchasedate"] = pd.NaT

    if "revenue" in df.columns:
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    else:
        df["revenue"] = 0.0

    if "purchaseamount" in df.columns:
        df["purchaseamount"] = pd.to_numeric(df["purchaseamount"], errors="coerce").fillna(0.0)
    else:
        df["purchaseamount"] = df["revenue"]

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1)
    else:
        df["quantity"] = 1

    if "customerid" not in df.columns:
        df["customerid"] = np.arange(1, len(df) + 1)

    if "gender" not in df.columns:
        df["gender"] = "Unknown"
    else:
        df["gender"] = df["gender"].fillna("Unknown").astype(str)

    if "age" not in df.columns:
        df["age"] = np.nan
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df["productcategory"] = df.get("productcategory", "Unknown").fillna("Unknown").astype(str)
    df["agegroup"] = pd.cut(df["age"], bins=AGE_GROUPS, labels=AGE_LABELS, right=False)
    df["agegroup"] = df["agegroup"].astype(str).replace("nan", "Unknown")
    df["month"] = df["purchasedate"].dt.to_period("M").dt.to_timestamp()
    df["month_label"] = df["month"].dt.strftime("%Y-%m")
    df["order_count"] = 1

    if "customername" not in df.columns and "name" in df.columns:
        df = df.rename(columns={"name": "customername"})

    return df


def build_model_features(df):
    df_model = df.copy()
    gender_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    df_model["gender_encoded"] = gender_encoder.fit_transform(df_model["gender"].astype(str))
    df_model["category_encoded"] = category_encoder.fit_transform(df_model["productcategory"].astype(str))
    df_model["month_ordinal"] = df_model["month"].dt.month.fillna(0).astype(int)
    if "customer_total_revenue" not in df_model.columns:
        customer_summary = (
            df_model.groupby("customerid")["revenue"].sum().rename("customer_total_revenue")
        )
        df_model = df_model.join(customer_summary, on="customerid")
    df_model["customer_total_revenue"] = df_model["customer_total_revenue"].fillna(0)

    X = df_model[
        ["age", "gender_encoded", "category_encoded", "quantity", "customer_total_revenue", "month_ordinal"]
    ].fillna(0)
    y = df_model["purchaseamount"].fillna(0)
    return X, y, df_model, gender_encoder, category_encoder


def create_kmeans_segments(df, n_clusters=4):
    customer_agg = df.groupby("customerid").agg(
        total_revenue=("revenue", "sum"),
        average_value=("purchaseamount", "mean"),
        purchase_count=("order_count", "sum"),
        age=("age", "median"),
    )
    customer_agg = customer_agg.fillna(0)
    cluster_count = min(n_clusters, max(1, len(customer_agg)))
    scaler = StandardScaler()
    features = scaler.fit_transform(customer_agg)
    if cluster_count > 1:
        kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
    else:
        labels = np.zeros(len(customer_agg), dtype=int)
    customer_agg["segment"] = labels
    customer_agg["segment_label"] = customer_agg["segment"].apply(
        lambda x: f"Segment {x + 1}"
    )
    return df.merge(customer_agg["segment_label"], left_on="customerid", right_index=True, how="left"), customer_agg


def generate_pdf_summary(metrics, top_customers, insights):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Customer Behavior Analytics Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(169, 184, 209)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}" , ln=True, align="C")
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for label, value in metrics.items():
        pdf.cell(0, 8, f"{label}: {value}", ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Top Customers", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for idx, row in top_customers.head(6).iterrows():
        revenue_value = row.get('revenue', row.get('total_revenue', 0.0))
        pdf.cell(
            0,
            7,
            f"{idx + 1}. {row.get('customerid', 'Unknown')} - Revenue: ${float(revenue_value):,.2f}",
            ln=True,
        )
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "AI Insights", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for insight in insights:
        pdf.multi_cell(0, 7, f"- {insight}")

    content = pdf.output(dest="S").encode("latin-1")
    buffer = BytesIO(content)
    buffer.seek(0)
    return buffer


def create_download_buttons(df, summary_metrics, top_customers, insights):
    csv_buffer = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Full CSV Report",
        data=csv_buffer,
        file_name="customer_behavior_report.csv",
        mime="text/csv",
        key="download_csv",
    )

    pdf_buffer = generate_pdf_summary(summary_metrics, top_customers, insights)
    st.download_button(
        label="Download PDF Summary",
        data=pdf_buffer,
        file_name="customer_behavior_summary.pdf",
        mime="application/pdf",
        key="download_pdf",
    )


def render_header():
    st.markdown(
        f"""
        <div style='display:flex; align-items:center; gap:16px; margin-bottom: 20px;'>
            <img src='{LOGO_URL}' width='64' style='border-radius:16px; filter: drop-shadow(0 0 12px rgba(14, 165, 233, 0.45));' />
            <div>
                <h1 class='logo-title' style='margin:0; padding:0;'>Customer Behavior Analytics & Business Intelligence System</h1>
                <p class='logo-subtitle' style='margin:4px 0 0 0; font-size:16px;'>Enterprise-grade KPI monitoring, segmentation, forecasting and AI-driven insights.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_login_page():
    st.markdown("<div class='login-page'>", unsafe_allow_html=True)
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='login-header'>
            <div class='login-tag'>Welcome Back</div>
            <h2 class='login-title'>Customer Analytics</h2>
            <p class='login-subtitle'>Login with your username and password to access the dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    username = st.text_input("Username", key="login_user", placeholder="Enter your username", label_visibility="collapsed")
    password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password", label_visibility="collapsed")

    if st.button("Login", use_container_width=True):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.session_state.username = username
        else:
            st.error("Authentication failed. Please check your credentials.")

    st.markdown(
        "<div class='login-footer'>Hint: admin / Business2026! or analyst / Analytics@2026</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def show_metrics_cards(metrics):
    cols = st.columns(4, gap="large")
    card_details = {
        "Total Customers": "Unique customers represented in this dataset. Explore segment-level purchase behavior for your customer base.",
        "Total Revenue": "Total sales across the dataset. Check revenue trends, top categories, and month-over-month performance.",
        "Average Purchase Value": "Average order value per transaction. Use this metric to compare spending by cohort and category.",
        "Most Purchased Category": "Top-performing category by transaction volume. Analyze what customers buy most and why it leads revenue.",
    }
    card_icons = {
        "Total Customers": "👥",
        "Total Revenue": "💰",
        "Average Purchase Value": "📈",
        "Most Purchased Category": "🛍️",
    }

    selected_metric = None
    for idx, (label, value) in enumerate(metrics.items()):
        with cols[idx]:
            display_value = value if value not in [None, ""] else "N/A"
            detail = card_details.get(label, "Detailed comparison and analysis are available in the section below.")
            icon = card_icons.get(label, "⚡")
            st.markdown(
                f"""
                <div style='border-radius:28px; padding:24px; min-height:240px; background: linear-gradient(180deg, rgba(8,15,31,0.95), rgba(14,23,42,0.88)); border:1px solid rgba(56,189,248,0.18); box-shadow: 0 24px 48px rgba(0,0,0,0.22);'>
                    <div style='display:flex; align-items:center; gap:12px; margin-bottom:16px;'>
                        <div style='width:44px; height:44px; display:flex; align-items:center; justify-content:center; border-radius:16px; background: rgba(56,189,248,0.18); color:#7dd3fc; font-size:1.3rem;'>{icon}</div>
                        <div style='font-size:0.85rem; letter-spacing:0.18em; color:#94a3b8; text-transform:uppercase; font-weight:700;'>{label}</div>
                    </div>
                    <div style='font-size:2.4rem; font-weight:800; color:#f8fafc; line-height:1.05; margin-bottom:14px;'>{display_value}</div>
                    <div style='font-size:0.95rem; color:#cbd5e1; line-height:1.65; margin-bottom:18px;'>{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("View details", key=f"detail_{label.replace(' ', '_')}"):
                selected_metric = label
    return selected_metric


def render_main_dashboard(df, segmented_df, customer_agg, model_data):
    st.subheader("Executive Overview")
    total_customers = int(df["customerid"].nunique())
    total_revenue = df["revenue"].sum()
    avg_purchase = df["purchaseamount"].mean()
    most_purchased = df["productcategory"].mode().iloc[0] if not df["productcategory"].mode().empty else "Unknown"

    metrics = {
        "Total Customers": f"{total_customers:,}",
        "Total Revenue": f"${total_revenue:,.2f}",
        "Average Purchase Value": f"${avg_purchase:,.2f}",
        "Most Purchased Category": most_purchased,
    }
    selected_metric = show_metrics_cards(metrics)

    if selected_metric:
        detail_map = {
            "Total Customers": "This dashboard is tracking the current customer universe and can be used to compare customer count trends over time.",
            "Total Revenue": "Revenue is based on normalized purchase values and helps identify the most valuable customer cohorts.",
            "Average Purchase Value": "Average Purchase Value highlights transactional efficiency and helps identify where upsell opportunities exist.",
            "Most Purchased Category": "Use this category insight to optimize inventory, promotions, and personalized offers for best-selling products.",
        }
        st.markdown("<div class='glass-card' style='padding:24px; margin-top:24px;'>", unsafe_allow_html=True)
        st.subheader(f"Drill-down: {selected_metric}")
        st.write(detail_map.get(selected_metric, "More details are available in the report sections below."))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card' style='padding:24px; margin-top:20px;'>", unsafe_allow_html=True)
    st.subheader("Revenue Performance")
    with st.expander("Click to expand revenue insights", expanded=True):
        chart_cols = st.columns(2)
        with chart_cols[0]:
            age_revenue = (
                df.groupby("agegroup")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
            )
            fig_age = px.bar(
                age_revenue,
                x="agegroup",
                y="revenue",
                title="Revenue by Age Group",
                labels={"agegroup": "Age Group", "revenue": "Revenue"},
                template="plotly_dark",
            )
            fig_age.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_age, use_container_width=True)

        with chart_cols[1]:
            gender_revenue = df.groupby("gender")["revenue"].sum().reset_index()
            fig_gender = px.pie(
                gender_revenue,
                names="gender",
                values="revenue",
                title="Revenue by Gender",
                hole=0.4,
                template="plotly_dark",
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        st.markdown("---")
        category_revenue = df.groupby("productcategory")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        fig_category = px.bar(
            category_revenue.head(12),
            x="productcategory",
            y="revenue",
            title="Revenue by Product Category",
            labels={"productcategory": "Product Category", "revenue": "Revenue"},
            template="plotly_dark",
        )
        fig_category.update_layout(xaxis_tickangle=-45, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_category, use_container_width=True)

        monthly = df.groupby("month")["revenue"].sum().reset_index().sort_values("month")
        fig_month = px.line(
            monthly,
            x="month",
            y="revenue",
            title="Monthly Revenue Trend",
            labels={"month": "Month", "revenue": "Revenue"},
            markers=True,
            template="plotly_dark",
        )
        fig_month.update_traces(line=dict(color="#38bdf8", width=3))
        st.plotly_chart(fig_month, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card' style='padding:24px; margin-top:24px;'>", unsafe_allow_html=True)
    st.subheader("Customer Segmentation & Frequency")
    with st.expander("Click to open segmentation and frequency insights", expanded=False):
        seg_cols = st.columns(2)
        with seg_cols[0]:
            segment_summary = (
                customer_agg.groupby("segment_label")["total_revenue"].sum().reset_index().sort_values("total_revenue", ascending=False)
            )
            fig_seg = px.bar(
                segment_summary,
                x="segment_label",
                y="total_revenue",
                title="Revenue by Customer Segment",
                labels={"segment_label": "Segment", "total_revenue": "Revenue"},
                template="plotly_dark",
            )
            st.plotly_chart(fig_seg, use_container_width=True)

        with seg_cols[1]:
            freq_distribution = df.groupby("customerid")["order_count"].sum().reset_index()
            fig_freq = px.histogram(
                freq_distribution,
                x="order_count",
                nbins=15,
                title="Purchase Frequency Analysis",
                labels={"order_count": "Purchases per Customer", "count": "Number of Customers"},
                template="plotly_dark",
            )
            st.plotly_chart(fig_freq, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card' style='padding:24px; margin-top:24px;'>", unsafe_allow_html=True)
    st.subheader("Top Spending Customers")
    with st.expander("Click to reveal the top customers list", expanded=False):
        top_customers = customer_agg.sort_values("total_revenue", ascending=False).reset_index()
        top_customers_display = top_customers.head(10)[["customerid", "total_revenue", "average_value", "purchase_count", "segment_label"]]
        top_customers_display.columns = ["Customer ID", "Revenue", "Avg Purchase", "Purchases", "Segment"]
        st.dataframe(top_customers_display.style.format({"Revenue": "${:,.2f}", "Avg Purchase": "${:,.2f}"}), height=360)
    st.markdown("</div>", unsafe_allow_html=True)

    return metrics, top_customers


def render_segmentation_page(df, segmented_df, customer_agg):
    st.subheader("K-Means Customer Segmentation")
    st.markdown(
        "Customer segments are derived from revenue, average purchase value, purchase frequency, and age. Use these segments to tailor loyalty programs and premium offers.")
    seg_overview = (
        customer_agg.groupby("segment_label").agg(
            {
                "total_revenue": ["count", "sum"],
                "average_value": "mean",
            }
        )
    )
    seg_overview.columns = ["Customers", "Revenue", "Avg Purchase Value"]
    seg_overview = seg_overview.reset_index().sort_values("Revenue", ascending=False)
    seg_overview = seg_overview.sort_values("Revenue", ascending=False)
    st.dataframe(seg_overview.style.format({"Revenue": "${:,.2f}", "Avg Purchase Value": "${:,.2f}"}), height=320)

    fig_scatter = px.scatter(
        customer_agg,
        x="purchase_count",
        y="average_value",
        color="segment_label",
        size="total_revenue",
        title="Customer Segments by Purchase Frequency and Average Value",
        labels={"purchase_count": "Purchase Frequency", "average_value": "Avg Purchase Value", "total_revenue": "Total Revenue"},
        template="plotly_dark",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Segment Profiles")
    for _, row in seg_overview.iterrows():
        st.markdown(
            f"- **{row['segment_label']}**: {int(row['Customers']):,} customers, ${row['Revenue']:,.2f} total revenue, average basket ${row['Avg Purchase Value']:,.2f}."
        )


def render_prediction_page(df):
    st.subheader("Future Purchase Amount Prediction")
    st.markdown(
        "Use the model to estimate the next purchase amount for a customer profile based on historical behavior.")
    X, y, df_model, gender_encoder, category_encoder = build_model_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    st.metric("Model R² Score", f"{score:.3f}")
    st.markdown("---")
    input_cols = st.columns(3)
    with input_cols[0]:
        age = st.number_input("Age", min_value=18, max_value=99, value=35)
        gender = st.selectbox("Gender", options=sorted(df["gender"].unique()))
    with input_cols[1]:
        product_category = st.selectbox("Product Category", options=sorted(df["productcategory"].unique()))
        quantity = st.number_input("Quantity", min_value=1, max_value=20, value=1)
    with input_cols[2]:
        customer_total_revenue = st.number_input("Customer Lifetime Revenue", min_value=0.0, value=float(df["revenue"].sum() / max(1, df["customerid"].nunique())))
        month = st.selectbox("Purchase Month", options=list(df["month_label"].sort_values().unique()))

    if st.button("Predict Future Purchase Amount"):
        if gender in gender_encoder.classes_:
            gender_code = int(np.where(gender_encoder.classes_ == gender)[0][0])
        else:
            gender_code = 0

        if product_category in category_encoder.classes_:
            category_code = int(np.where(category_encoder.classes_ == product_category)[0][0])
        else:
            category_code = 0

        input_df = pd.DataFrame(
            [
                {
                    "age": age,
                    "gender_encoded": gender_code,
                    "category_encoded": category_code,
                    "quantity": quantity,
                    "customer_total_revenue": customer_total_revenue,
                    "month_ordinal": int(pd.to_datetime(month + "-01").month),
                }
            ]
        )
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted purchase amount: ${prediction:,.2f}")
        st.info("Use this prediction to identify cross-sell opportunities, set personalized offers, and forecast demand.")


def render_insights_page(df, customer_agg):
    st.subheader("AI-Driven Business Insights")
    highest_revenue_category = df.groupby("productcategory")["revenue"].sum().idxmax()
    best_segment = customer_agg.groupby("segment_label")["total_revenue"].sum().idxmax()
    top_age_group = df.groupby("agegroup")["revenue"].sum().idxmax()

    category_trend = df.groupby("productcategory")["revenue"].sum().sort_values(ascending=False)
    low_growth_categories = category_trend.tail(3).index.tolist()
    opportunity_text = (
        f"Invest in targeted campaigns for {', '.join(low_growth_categories)} if they align with your high-value segments. "
        "Improve cross-sell offers for customers in segments that have strong engagement but lower total revenue."
    )

    insights = [
        f"Highest revenue category: {highest_revenue_category}.",
        f"Best customer segment: {best_segment}.",
        f"Highest spending age group: {top_age_group}.",
        f"Growth opportunity: {opportunity_text}",
    ]

    for insight in insights:
        st.write(f"• {insight}")

    st.markdown("---")
    st.subheader("Opportunity Radar")
    opportunity_df = (
        df.groupby(["productcategory", "agegroup"])["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
    )
    st.dataframe(opportunity_df.head(12).style.format({"revenue": "${:,.2f}"}), height=320)
    return insights


def main():
    inject_css()
    set_session_defaults()

    if not st.session_state.authenticated:
        show_login_page()
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose Report",
        [
            "Dashboard",
            "Customer Segmentation",
            "Prediction",
            "AI Insights",
            "Reports",
            "About",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.write(f"Logged in as **{st.session_state.username}**")
    if st.sidebar.button("Log out"):
        st.session_state.authenticated = False
        st.session_state.username = ""

    st.sidebar.markdown("---")
    st.sidebar.write("Upload a fresh CSV to refresh the workspace:")
    uploaded_file = st.sidebar.file_uploader("Upload customer_shopping_behavior.csv", type=["csv"])
    use_sample_data = uploaded_file is None and not os.path.exists("customer_shopping_behavior.csv")
    try:
        df = load_data(uploaded_file)
    except Exception as ex:
        st.error(f"Unable to load the dataset: {ex}")
        return

    if use_sample_data:
        st.warning(
            "No local dataset was found. Showing built-in sample data so the dashboard remains visible and interactive."
        )

    df = enrich_data(df)
    segmented_df, customer_agg = create_kmeans_segments(df)

    render_header()

    if page == "Dashboard":
        summary_metrics, top_customers = render_main_dashboard(df, segmented_df, customer_agg, None)
    elif page == "Customer Segmentation":
        render_segmentation_page(df, segmented_df, customer_agg)
    elif page == "Prediction":
        render_prediction_page(df)
    elif page == "AI Insights":
        insights = render_insights_page(df, customer_agg)
    elif page == "Reports":
        _, top_customers = render_main_dashboard(df, segmented_df, customer_agg, None)
        summary_metrics = {
            "Total Customers": f"{int(df['customerid'].nunique()):,}",
            "Total Revenue": f"${df['revenue'].sum():,.2f}",
            "Average Purchase Value": f"${df['purchaseamount'].mean():,.2f}",
            "Most Purchased Category": df['productcategory'].mode().iloc[0] if not df['productcategory'].mode().empty else 'Unknown',
        }
        insights = [
            f"Highest revenue category: {df.groupby('productcategory')['revenue'].sum().idxmax()}",
            f"Best customer segment: {customer_agg.groupby('segment_label')['total_revenue'].sum().idxmax()}",
        ]
        st.subheader("Downloadable Reports")
        create_download_buttons(df, summary_metrics, top_customers, insights)
    else:
        st.subheader("About This System")
        st.markdown(
            "This corporate analytics platform delivers interactive KPIs, segmentation, forecasting, and automated business insights for customer behavior data."
        )
        st.markdown(
            "Built with Streamlit, Plotly, scikit-learn, and PDF export capabilities for enterprise-ready reporting."
        )


if __name__ == "__main__":
    main()
