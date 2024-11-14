import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Streamlit app title
st.set_page_config(page_title='Customer Conversion Prediction Tool', layout='wide', initial_sidebar_state='expanded')

# Load and display the logo and title
st.image("App/socialinsider_logo.png", width=80)
st.markdown(
    """
    <h1 style="color: white; font-size: 36px; margin-left: 20px;">Customer Conversion Prediction Tool</h1>
    """,
    unsafe_allow_html=True
)

# Styling for a sleek and modern look
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(90deg, #eef2f7, #fff);
    }
    .sidebar .sidebar-content {
        background: #eef2f7;
    }
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: white;  /* Ensuring the title remains white */
    }
    .css-1offfwp {
        color: #2c3e50;
        font-size: 14px;
    }
    .stButton > button {
        font-size: 16px;
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
    }
    .stDataFrame {
        font-size: 13px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# Step 1: Upload CSV file as a banner
st.markdown("### 🔍 Upload Event Data to Predict Conversions")
uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Step 2: Data Transformation
        st.info("✨ Processing and Transforming Data...")
        data = pd.read_csv(uploaded_file).sort_values(by="time_created")

        buy_users = data[(data["event_name"] == "New Client") | (data["event_name"] == "Buy Success")]["user_id"].tolist()
        data["successful"] = data["user_id"].apply(lambda x: 1 if x in buy_users else 0)

        user_data = (
            data.groupby("user_id")
            .agg({
                "event_name": "count",
                "country": "first"
            })
            .reset_index()
            .rename(columns={"event_name": "event_count"})
            .sort_values(by="event_count", ascending=False)
        )

        for country in ["United States", "Saudi Arabia", "India", "Britain", "Italy"]:
            user_data[f"country_{country.replace(' ', '_').replace('(', '').replace(')', '')}"] = user_data["country"].apply(
                lambda x: 1 if x == country else 0
            )

        user_data["successful"] = user_data["user_id"].apply(lambda x: 1 if x in buy_users else 0)

        average_load_time = data.groupby("user_id")["load_time"].mean().sort_values()
        max_load_time = data.groupby("user_id")["load_time"].max().sort_values()

        user_data["average_load_time"] = user_data["user_id"].map(average_load_time)
        user_data["average_load_time"].fillna(user_data["average_load_time"].median(), inplace=True)

        user_data["max_load_time"] = user_data["user_id"].map(max_load_time)
        user_data["max_load_time"].fillna(user_data["max_load_time"].median(), inplace=True)

        event_counts = data.groupby("user_id").apply(
            lambda x: pd.Series({
                "event_bench_load_success_count": (x["event_name"] == "bench load success").sum(),
                "event_profile_search_success_count": (x["event_name"] == "profile search success").sum(),
                "event_add_profile_success_count": (x["event_name"] == "add profile success").sum(),
                "event_upgrade_plan_count": (x["event_name"] == "upgrade_plan").sum(),
                "event_pricing_model_count": (x["event_name"] == "pricing modal visited").sum(),
                "event_profile_load_fail_count": (x["event_name"] == "profile load fail").sum(),
                "event_email_receipt_count": (x["event_name"] == "email receipt").sum(),
            })
        ).reset_index()

        user_data = user_data.merge(event_counts, on="user_id", how="left")

        data["platform"] = data["platform"].replace({
            "facebook": "fb", "showFacebook": "fb", "twitter": "tw", "instagram": "ig",
            "youtube": "yt", "linkedin": "li", "tiktok": "tk", "cross-platform": "xch"
        }).apply(lambda x: f"platform_{x}_count")

        platform_data = data.groupby(["user_id", "platform"]).size().unstack(fill_value=0).reset_index()
        user_data = user_data.merge(platform_data, on="user_id", how="left")

        user_data["platform_total_count"] = user_data.filter(regex="^platform_").sum(axis=1)

        data["view"] = data["view"].apply(lambda x: f"view_{x}")
        view_data = data.groupby(["user_id", "view"]).size().unstack(fill_value=0).reset_index()
        user_data = user_data.merge(view_data, on="user_id", how="left")

        all_columns = [
            'user_id', 'event_count', 'country', 'country_United_States',
            'country_Saudi_Arabia', 'country_India', 'country_Britain',
            'country_Italy', 'successful', 'average_load_time', 'max_load_time',
            'event_bench_load_success_count', 'event_profile_search_success_count',
            'event_add_profile_success_count', 'event_upgrade_plan_count',
            'event_pricing_model_count', 'event_profile_load_fail_count',
            'event_email_receipt_count', 'platform_all_count',
            'platform_brbench_count', 'platform_fb_count',
            'platform_hashtags_count', 'platform_ig._count', 'platform_ig_count',
            'platform_li_count', 'platform_meta_count', 'platform_nan_count',
            'platform_tk_count', 'platform_tw_count', 'platform_xch_count',
            'platform_yt_count', 'platform_total_count', 'view_add',
            'view_addprofiles', 'view_ads', 'view_bench', 'view_benchmark',
            'view_brands', 'view_campaigns', 'view_connect', 'view_content-pillars',
            'view_hashtag', 'view_nan', 'view_page', 'view_postsfeed',
            'view_profile', 'view_proj', 'view_projecthome', 'view_reports',
            'view_search', 'view_settings', 'view_upgradeplan'
        ]

        for col in all_columns:
            if col not in user_data.columns:
                user_data[col] = 0

        user_data = user_data[all_columns]
        user_data = user_data[user_data["event_count"] > 1]

        # Step 3: Model Inference
        st.info("🔄 Analyzing Data and Making Predictions...")
        y_test = user_data["successful"]
        X_test = user_data.drop(columns=["user_id", "country", "successful"])

        categorical_columns = [
            "country_United_States",
            "country_Saudi_Arabia",
            "country_India",
            "country_Britain",
            "country_Italy",
        ]

        X_categorical = X_test[categorical_columns]
        X_continuous = X_test.drop(columns=categorical_columns)

        with open("App/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)

        X_continuous_scaled = pd.DataFrame(scaler.transform(X_continuous))
        X_test_scaled = pd.concat(
            [X_continuous_scaled, X_categorical.reset_index(drop=True)], axis=1
        )

        X_test_scaled.columns = X_test_scaled.columns.astype(str)

        with open("App/best_model.pkl", "rb") as file:
            model = pickle.load(file)

        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

        user_data["p_converted"] = y_pred
        user_data["p_cvr"] = y_pred_prob

        columns = list(user_data.columns)
        columns.remove("successful")
        columns.remove("p_converted")
        columns.remove("p_cvr")
        columns.insert(1, "successful")
        columns.insert(2, "p_converted")
        columns.insert(3, "p_cvr")
        user_data = user_data[columns]

        # Sort by p_cvr in descending order
        user_data = user_data.sort_values(by="p_cvr", ascending=False)

        # Display the transformed data and download link
        st.success("✅ Data Transformation and Prediction Completed")
        st.dataframe(user_data, use_container_width=True)

        csv = user_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="predicted_user_data.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Please check the input CSV file and try again.")
        st.button("🔄 Retry")

if 'csv' in locals() or 'csv' in globals():
    st.markdown("""
### ⬇️ Data sorted by **p_cvr** in descending order by default. Sort data by clicking on the column headers.
### 🔢 Key Variables Explained:
- **user_id**: Identifier for each unique user.
- **successful**: Indicates real-life conversion (1 for success, 0 otherwise).
- **p_converted**: Model-predicted conversion status (1 for predicted success, 0 otherwise).
- **p_cvr**: Conversion likelihood as predicted by the model.
""")
