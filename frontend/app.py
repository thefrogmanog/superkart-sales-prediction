import os
import requests
import pandas as pd
import streamlit as st

# In Docker, "backend" is the Flask container's service name.
# For local testing, change this to http://127.0.0.1:7860.
API_URL = os.environ.get("API_URL", "http://backend:7860")

FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]

st.set_page_config(
    page_title="SuperKart Sales Predictor",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 SuperKart Sales Predictor")
st.write(
    "Estimate product sales using the trained SuperKart machine learning model."
)

with st.sidebar:
    st.header("API Connection")
    st.caption(f"Backend address: {API_URL}")

    if st.button("Check API Connection", use_container_width=True):
        try:
            response = requests.get(
                f"{API_URL}/health", timeout=5
            )
            response.raise_for_status()
            st.success("Flask API is responding.")
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"Could not connect to the API: {exc}")

single_tab, batch_tab = st.tabs(
    ["Single Prediction", "Batch Predictions"]
)

with single_tab:
    st.subheader("Enter product and store details")
    st.caption("Complete the fields below, then request a prediction.")

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            product_weight = st.number_input(
                "Product Weight",
                min_value=0.01,
                value=12.0,
                step=0.1,
            )
            sugar_content = st.selectbox(
                "Product Sugar Content",
                ["Low Fat", "Regular"],
            )
            allocated_area = st.number_input(
                "Product Allocated Area",
                min_value=0.0,
                value=0.05,
                step=0.01,
                format="%.3f",
            )
            product_type = st.selectbox(
                "Product Type",
                [
                    "Dairy", "Soft Drinks", "Meat",
                    "Fruits and Vegetables", "Household",
                    "Baking Goods", "Snack Foods", "Frozen Foods",
                    "Breakfast", "Health and Hygiene", "Hard Drinks",
                    "Canned", "Breads", "Starchy Foods", "Others",
                    "Seafood",
                ],
            )
            product_mrp = st.number_input(
                "Product MRP",
                min_value=0.0,
                value=150.0,
                step=1.0,
            )

        with right:
            store_id = st.text_input(
                "Store ID", value="OUT049"
            )
            establishment_year = st.number_input(
                "Store Establishment Year",
                min_value=1980,
                max_value=2026,
                value=2000,
                step=1,
            )
            store_size = st.selectbox(
                "Store Size", ["Small", "Medium", "High"]
            )
            city_type = st.selectbox(
                "Store Location City Type",
                ["Tier 1", "Tier 2", "Tier 3"],
            )
            store_type = st.selectbox(
                "Store Type",
                [
                    "Grocery Store",
                    "Supermarket Type1",
                    "Supermarket Type2",
                    "Supermarket Type3",
                ],
            )

        submitted = st.form_submit_button(
            "Predict Sales", use_container_width=True
        )

    if submitted:
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar_content,
            "Product_Allocated_Area": allocated_area,
            "Product_Type": product_type,
            "Product_MRP": product_mrp,
            "Store_Id": store_id.strip(),
            "Store_Establishment_Year": int(establishment_year),
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
        }

        if not payload["Store_Id"]:
            st.error("Please enter a Store ID.")
        else:
            try:
                with st.spinner("Requesting prediction..."):
                    response = requests.post(
                        f"{API_URL}/v1/predict",
                        json=payload,
                        timeout=30,
                    )

                if response.ok:
                    result = response.json()
                    prediction = result.get(
                        "Predicted_Product_Store_Sales_Total"
                    )

                    if prediction is not None:
                        st.success("Prediction completed!")
                        st.metric(
                            "Predicted Product Store Sales",
                            f"{float(prediction):,.2f}",
                        )
                        st.caption(
                            "This is a model estimate, not a guaranteed sales result."
                        )
                    else:
                        st.error(
                            "The API responded, but no prediction "
                            "field was found."
                        )
                        st.json(result)
                else:
                    st.error(f"API error ({response.status_code})")
                    st.code(response.text)

            except requests.RequestException as exc:
                st.error(f"Could not request a prediction: {exc}")

with batch_tab:
    st.subheader("Predict sales for multiple records")
    st.write(
        "Upload a CSV containing all 10 original model input columns. "
        "The API will return predictions for each row."
    )

    st.code(", ".join(FEATURE_COLUMNS))

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="batch_csv",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data")
            st.dataframe(batch_df.head(), use_container_width=True)

            missing_columns = [
                col for col in FEATURE_COLUMNS
                if col not in batch_df.columns
            ]

            if missing_columns:
                st.error(
                    "Your CSV is missing these required columns: "
                    + ", ".join(missing_columns)
                )
            elif st.button("Predict Batch Sales"):
                try:
                    with st.spinner("Processing batch..."):
                        response = requests.post(
                            f"{API_URL}/v1/predictbatch",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "text/csv",
                                )
                            },
                            timeout=120,
                        )

                    if response.ok:
                        result = response.json()

                        if "predictions" in result:
                            predictions = result["predictions"]
                            output_df = batch_df.copy()
                            output_df["Predicted_Sales"] = predictions
                            st.success(
                                f"Predicted sales for {len(output_df)} rows."
                            )
                            st.dataframe(
                                output_df, use_container_width=True
                            )
                            st.download_button(
                                "Download predictions as CSV",
                                data=output_df.to_csv(
                                    index=False
                                ).encode("utf-8"),
                                file_name="superkart_predictions.csv",
                                mime="text/csv",
                            )
                        else:
                            st.error(
                                "The API response did not contain "
                                "a predictions field."
                            )
                            st.json(result)
                    else:
                        st.error(
                            f"API error ({response.status_code})"
                        )
                        st.code(response.text)

                except requests.RequestException as exc:
                    st.error(f"Batch request failed: {exc}")

        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
