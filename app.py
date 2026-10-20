
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# 1. Project Configuration
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "used_car_price_model.joblib"
DATA_PATH = PROJECT_DIR / "data" / "used_cars.csv"

REFERENCE_YEAR = 2026

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered",
)

# -----------------------------
# 2. Load Model and Dataset
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    # Match the cleaning performed during model training
    data["fuel_type"] = data["fuel_type"].replace(
        ["–", "not supported"], pd.NA
    )

    data["price"] = (
        data["price"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    data["milage"] = (
        data["milage"]
        .str.replace(" mi.", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    return data


try:
    model = load_model()
    df = load_data()
except Exception as error:
    st.error(f"Could not load the model or dataset: {error}")
    st.stop()


# -----------------------------
# 3. App Header
# -----------------------------
st.title("🚗 Used Car Price Predictor")

st.write(
    "Estimate the price of a used car based on its "
    "specifications using machine learning."
)

st.caption("Dataset currency: USD ($) | Mileage: miles")

st.divider()


# -----------------------------
# 4. Input Form
# -----------------------------
with st.form("car_details_form"):

    st.subheader("Car Specifications")

    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox(
            "Brand",
            sorted(df["brand"].dropna().unique()),
        )

        available_models = sorted(
            df.loc[df["brand"] == brand, "model"]
            .dropna()
            .unique()
        )

        model_name = st.selectbox(
            "Model",
            available_models,
        )

        model_year = st.selectbox(
            "Model year",
            sorted(
                df["model_year"].dropna().astype(int).unique(),
                reverse=True,
            ),
        )

        mileage = st.number_input(
            "Mileage (miles)",
            min_value=0,
            max_value=1_000_000,
            value=30_000,
            step=1_000,
        )

        fuel_options = sorted(df["fuel_type"].dropna().unique())

        fuel_type = st.selectbox(
            "Fuel type",
            fuel_options,
        )

        transmission = st.selectbox(
            "Transmission",
            sorted(df["transmission"].dropna().unique()),
        )

    with col2:
        engine = st.selectbox(
            "Engine",
            sorted(df["engine"].dropna().unique()),
        )

        ext_col = st.selectbox(
            "Exterior colour",
            sorted(df["ext_col"].dropna().unique()),
        )

        int_col = st.selectbox(
            "Interior colour",
            sorted(df["int_col"].dropna().unique()),
        )

        accident = st.selectbox(
            "Accident history",
            [
                "None reported",
                "At least 1 accident or damage reported",
                "Unknown",
            ],
        )

        clean_title = st.selectbox(
            "Clean title",
            ["Yes", "Unknown"],
        )

    submitted = st.form_submit_button(
        "Estimate Price",
        use_container_width=True,
    )


# -----------------------------
# 5. Make Prediction
# -----------------------------
if submitted:
    vehicle_age = REFERENCE_YEAR - int(model_year)

    accident_mapping = {
        "None reported": "None reported",
        "At least 1 accident or damage reported":
            "At least 1 accident or damage reported",
        "Unknown": None,
    }

    title_mapping = {
        "Yes": "Yes",
        "Unknown": None,
    }

    car_data = pd.DataFrame(
        [{
            "brand": brand,
            "model": model_name,
            "model_year": int(model_year),
            "vehicle_age": vehicle_age,
            "milage": float(mileage),
            "fuel_type": fuel_type,
            "engine": engine,
            "transmission": transmission,
            "ext_col": ext_col,
            "int_col": int_col,
            "accident": accident_mapping[accident],
            "clean_title": title_mapping[clean_title],
        }]
    )

    try:
        predicted_price = float(model.predict(car_data)[0])

        if not pd.notna(predicted_price):
            st.error("The model returned an invalid prediction.")
            st.stop()

        st.divider()
        st.subheader("Estimated Car Price")

        st.metric(
            label="Predicted price",
            value=f"${predicted_price:,.0f}",
        )

        st.caption(
            "This is a machine-learning estimate, not a guaranteed "
            "market price. Accuracy may be lower for rare models "
            "and expensive luxury vehicles."
        )

        st.subheader("Your Car Details")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.write(f"**Vehicle:** {brand} {model_name}")
            st.write(f"**Model year:** {model_year}")
            st.write(f"**Mileage:** {mileage:,.0f} miles")
            st.write(f"**Vehicle age:** {vehicle_age} years")
            st.write(f"**Fuel:** {fuel_type}")

        with summary_col2:
            st.write(f"**Transmission:** {transmission}")
            st.write(f"**Engine:** {engine}")
            st.write(f"**Exterior:** {ext_col}")
            st.write(f"**Interior:** {int_col}")
            st.write(f"**Accident history:** {accident}")
            st.write(f"**Clean title:** {clean_title}")

    except Exception as error:
        st.error(f"Prediction failed: {error}")
