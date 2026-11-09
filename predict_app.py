"""
Airbnb India Market Intelligence - Price Prediction App
=======================================================

Run (default port for this app is 8502):
    streamlit run predict_app.py --server.port 8502

Requires the model bundle created by the notebook:
    models/airbnb_price_model_bundle.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "models" / "airbnb_price_model_bundle.joblib"
REQUIRED_KEYS = {
    "pipeline", "model_name", "feature_columns",
    "input_schema", "city_neighbourhoods", "metrics",
}

LABELS = {
    "city": "City",
    "neighbourhood": "Neighbourhood",
    "room_type": "Room type",
    "host_identity_verified": "Host identity status",
    "cancellation_policy": "Cancellation policy",
    "instant_bookable": "Instant booking enabled",
    "minimum_nights": "Minimum nights per booking",
    "number_of_reviews": "Total number of reviews",
    "reviews_per_month": "Reviews per month",
    "review_rate_number": "Average rating (1-5)",
    "availability_365": "Available days per year",
    "construction_year": "Construction year",
    "calculated_host_listings_count": "Listings owned by this host",
}

SECTIONS = {
    "Location": ["city", "neighbourhood"],
    "Property and booking rules": [
        "room_type", "construction_year", "minimum_nights",
        "availability_365", "instant_bookable", "cancellation_policy",
    ],
    "Host and reviews": [
        "host_identity_verified", "calculated_host_listings_count",
        "number_of_reviews", "reviews_per_month", "review_rate_number",
    ],
}

st.set_page_config(
    page_title="Airbnb India Price Predictor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource
def load_bundle(path: str) -> dict:
    """Load the model bundle and check that it contains a usable model."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise TypeError("The bundle file must contain a dictionary.")
    missing = REQUIRED_KEYS - set(bundle)
    if missing:
        raise KeyError(f"Bundle is missing keys: {sorted(missing)}")
    if not hasattr(bundle["pipeline"], "predict"):
        raise TypeError("bundle['pipeline'] does not support .predict().")
    return bundle


def label(feature: str) -> str:
    return LABELS.get(feature, feature.replace("_", " ").title())


def render_input(feature: str, spec: dict, bundle: dict, values: dict):
    """Draw one input widget, driven entirely by the saved input schema."""
    key = f"input_{feature}"

    if spec["type"] == "categorical":
        options = list(spec["options"])
        if feature == "neighbourhood" and "city" in values:
            options = bundle["city_neighbourhoods"].get(values["city"], options)
            # If the city changed, the old neighbourhood is no longer valid.
            if st.session_state.get(key) not in options:
                st.session_state.pop(key, None)
        default = spec["default"] if spec["default"] in options else options[0]
        return st.selectbox(label(feature), options,
                            index=options.index(default), key=key)

    if spec["type"] == "binary":
        options = list(spec["options"])
        return st.selectbox(
            label(feature), options,
            index=options.index(spec["default"]),
            format_func=lambda v: "Yes" if int(v) == 1 else "No",
            key=key,
        )

    # numeric
    if spec.get("integer", False):
        return st.number_input(
            label(feature),
            min_value=int(spec["minimum"]), max_value=int(spec["maximum"]),
            value=int(spec["default"]), step=int(spec["step"]), key=key,
            help=f"Allowed range: {int(spec['minimum'])} to {int(spec['maximum'])}",
        )
    return st.number_input(
        label(feature),
        min_value=float(spec["minimum"]), max_value=float(spec["maximum"]),
        value=float(spec["default"]), step=float(spec["step"]), key=key,
        help=f"Allowed range: {spec['minimum']} to {spec['maximum']}",
    )


def validate(values: dict, bundle: dict) -> tuple[list, list]:
    """Return (errors, warnings) for the user's inputs."""
    errors, warnings = [], []
    schema = bundle["input_schema"]

    for feature in bundle["feature_columns"]:
        value = values.get(feature)
        spec = schema[feature]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            errors.append(f"{label(feature)} is missing.")
            continue
        if spec["type"] in ("categorical", "binary"):
            if value not in spec["options"]:
                errors.append(f"{label(feature)}: '{value}' is not a known option.")
        elif not (spec["minimum"] <= float(value) <= spec["maximum"]):
            errors.append(
                f"{label(feature)} must be between {spec['minimum']} and {spec['maximum']}."
            )

    city, hood = values.get("city"), values.get("neighbourhood")
    if city and hood and hood not in bundle["city_neighbourhoods"].get(city, []):
        errors.append(f"{hood} is not a neighbourhood of {city} in the training data.")

    if values.get("number_of_reviews") == 0 and values.get("reviews_per_month", 0) > 0:
        warnings.append("Reviews per month is above 0 although the listing has 0 reviews.")

    return errors, warnings


# ------------------------------------------------------------------ #
# Page
# ------------------------------------------------------------------ #
st.title("🏠 Airbnb India Price Predictor")
st.write(
    "Enter the details of a listing to get an estimated nightly price "
    "from the machine learning model trained in the project notebook."
)

if not BUNDLE_PATH.exists():
    st.error(
        f"Model bundle not found at `{BUNDLE_PATH}`.\n\n"
        "Run the project notebook from top to bottom first. It creates "
        "`models/airbnb_price_model_bundle.joblib` next to this app."
    )
    st.stop()

try:
    bundle = load_bundle(str(BUNDLE_PATH))
except Exception as exc:  # show a clear message instead of a crash
    st.error(f"The model bundle could not be loaded: {exc}")
    st.stop()

schema = bundle["input_schema"]
feature_columns = bundle["feature_columns"]
metrics = bundle["metrics"]

with st.sidebar:
    st.header("Model information")
    st.write(f"**Model:** {bundle['model_name']}")
    st.write(f"**Test MAE:** ₹{metrics['MAE']:,.0f}")
    st.write(f"**Test RMSE:** ₹{metrics['RMSE']:,.0f}")
    st.write(f"**Test R²:** {metrics['R2']:.3f}")
    st.caption(
        "MAE is the average size of the error in rupees on listings "
        "the model never saw during training."
    )

values = {}
shown = set()
for section, features in SECTIONS.items():
    present = [f for f in features if f in schema]
    if not present:
        continue
    st.subheader(section)
    cols = st.columns(2)
    for i, feature in enumerate(present):
        with cols[i % 2]:
            values[feature] = render_input(feature, schema[feature], bundle, values)
        shown.add(feature)

others = [f for f in feature_columns if f not in shown]
if others:
    st.subheader("Other details")
    cols = st.columns(2)
    for i, feature in enumerate(others):
        with cols[i % 2]:
            values[feature] = render_input(feature, schema[feature], bundle, values)

st.divider()

if st.button("Predict nightly price", type="primary", key="predict_btn"):
    errors, warnings = validate(values, bundle)
    for message in warnings:
        st.warning(message)
    if errors:
        for message in errors:
            st.error(message)
    else:
        input_df = pd.DataFrame([values], columns=feature_columns)
        try:
            raw_prediction = float(bundle["pipeline"].predict(input_df)[0])
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            prediction = max(0.0, raw_prediction)
            mae = metrics["MAE"]
            st.subheader("Prediction result")
            c1, c2 = st.columns(2)
            c1.metric("Estimated price per night", f"₹{prediction:,.0f}")
            c2.metric(
                "Typical range (± test MAE)",
                f"₹{max(0, prediction - mae):,.0f} – ₹{prediction + mae:,.0f}",
            )
            st.caption(
                "The range uses the model's average error on unseen test listings. "
                "It is a rough guide, not a statistical guarantee. Real prices also "
                "depend on amenities, photos, season and demand, which are not in the data."
            )
            with st.expander("Inputs sent to the model"):
                st.dataframe(input_df, width="stretch")

st.caption("Airbnb India Market Intelligence | Price prediction application")
