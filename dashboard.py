"""
Airbnb India Market Intelligence - Analytics Dashboard
======================================================

Run (default port for this app is 8501):
    streamlit run dashboard.py --server.port 8501

Requires files created by the notebook:
    outputs/airbnb_india_cleaned.csv
    outputs/airbnb_clustered_listings.csv   (optional - adds the Segments tab)
    outputs/model_metrics.json              (optional - adds the Model tab)
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
CLEAN_PATH = BASE_DIR / "outputs" / "airbnb_india_cleaned.csv"
CLUSTER_PATH = BASE_DIR / "outputs" / "airbnb_clustered_listings.csv"
METRICS_PATH = BASE_DIR / "outputs" / "model_metrics.json"
PRICE = "price_inr_per_night"
SAMPLE_POINTS = 5000  # scatter plots use a sample so the browser stays fast

st.set_page_config(
    page_title="Airbnb India Market Intelligence",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data
def load_data(clean_path: str, cluster_path: str) -> pd.DataFrame:
    df = pd.read_csv(clean_path)
    if Path(cluster_path).exists():
        clusters = pd.read_csv(cluster_path)
        keep = [c for c in ["listing_id", "cluster", "cluster_name"] if c in clusters.columns]
        if "listing_id" in keep and len(keep) > 1:
            # Merge on listing_id (not on row position) so labels always match.
            df = df.merge(clusters[keep], on="listing_id", how="left")
    return df


@st.cache_data
def load_metrics(path: str):
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


st.title("🏠 Airbnb India Market Intelligence")
st.caption("Interactive analytics dashboard: pricing, cities, reviews and listing segments")

if not CLEAN_PATH.exists():
    st.error(
        f"Cleaned dataset not found at `{CLEAN_PATH}`.\n\n"
        "Run the project notebook from top to bottom first. It saves the "
        "cleaned data in the `outputs` folder next to this app."
    )
    st.stop()

df = load_data(str(CLEAN_PATH), str(CLUSTER_PATH))

# ------------------------------------------------------------------ #
# Sidebar filters
# ------------------------------------------------------------------ #
st.sidebar.header("Filters")

cities = sorted(df["city"].dropna().unique())
selected_cities = st.sidebar.multiselect("City", cities, default=cities, key="f_city")

rooms = sorted(df["room_type"].dropna().unique())
selected_rooms = st.sidebar.multiselect("Room type", rooms, default=rooms, key="f_room")

min_p, max_p = int(df[PRICE].min()), int(df[PRICE].max())
price_range = st.sidebar.slider(
    "Price per night (₹)", min_value=min_p, max_value=max_p,
    value=(min_p, max_p), step=100, key="f_price",
)

filtered = df[
    df["city"].isin(selected_cities)
    & df["room_type"].isin(selected_rooms)
    & df[PRICE].between(price_range[0], price_range[1])
]
st.sidebar.caption(f"Listings after filters: {len(filtered):,} of {len(df):,}")

if filtered.empty:
    st.warning("No listings match the current filters. Please widen your selection.")
    st.stop()

# ------------------------------------------------------------------ #
# KPIs
# ------------------------------------------------------------------ #
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Listings", f"{len(filtered):,}")
k2.metric("Median price", f"₹{filtered[PRICE].median():,.0f}")
k3.metric("Average price", f"₹{filtered[PRICE].mean():,.0f}")
k4.metric("Average rating", f"{filtered['review_rate_number'].mean():.2f} / 5")
k5.metric("Avg. available days", f"{filtered['availability_365'].mean():,.0f}")

metrics = load_metrics(str(METRICS_PATH))
tab_names = ["Pricing", "Cities & neighbourhoods", "Reviews & availability"]
if "cluster_name" in filtered.columns:
    tab_names.append("Segments")
if metrics:
    tab_names.append("Model")
tab_names.append("Data")
tabs = dict(zip(tab_names, st.tabs(tab_names)))

sample = filtered.sample(min(SAMPLE_POINTS, len(filtered)), random_state=42)

with tabs["Pricing"]:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered, x=PRICE, nbins=50,
                           title="Distribution of nightly prices",
                           labels={PRICE: "Price per night (₹)"})
        fig.update_yaxes(title="Number of listings")
        st.plotly_chart(fig, width="stretch")
    with c2:
        room_price = (filtered.groupby("room_type", as_index=False)[PRICE]
                      .median().sort_values(PRICE, ascending=False))
        fig = px.bar(room_price, x="room_type", y=PRICE,
                     title="Median price by room type",
                     labels={"room_type": "Room type", PRICE: "Median price (₹ per night)"})
        st.plotly_chart(fig, width="stretch")
    city_room = filtered.groupby(["city", "room_type"], as_index=False)[PRICE].median()
    fig = px.bar(city_room, x="city", y=PRICE, color="room_type", barmode="group",
                 title="Median price by city and room type",
                 labels={"city": "City", PRICE: "Median price (₹ per night)",
                         "room_type": "Room type"})
    st.plotly_chart(fig, width="stretch")

with tabs["Cities & neighbourhoods"]:
    city_data = (filtered.groupby("city", as_index=False)
                 .agg(listings=(PRICE, "size"), median_price=(PRICE, "median"))
                 .sort_values("listings", ascending=False))
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(city_data, x="city", y="listings", title="Listings by city",
                     labels={"city": "City", "listings": "Number of listings"})
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.bar(city_data, x="city", y="median_price", title="Median price by city",
                     labels={"city": "City", "median_price": "Median price (₹ per night)"})
        st.plotly_chart(fig, width="stretch")
    hood = (filtered.groupby(["city", "neighbourhood"], as_index=False)
            .agg(listings=(PRICE, "size"), median_price=(PRICE, "median"))
            .sort_values("median_price", ascending=False).head(15))
    fig = px.bar(hood, x="median_price", y="neighbourhood", color="city", orientation="h",
                 title="15 most expensive neighbourhoods (median price)",
                 labels={"median_price": "Median price (₹ per night)",
                         "neighbourhood": "Neighbourhood", "city": "City"})
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, width="stretch")

with tabs["Reviews & availability"]:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(sample, x="number_of_reviews", y=PRICE, color="room_type",
                         opacity=0.5,
                         title=f"Price vs number of reviews (sample of {len(sample):,})",
                         labels={"number_of_reviews": "Number of reviews",
                                 PRICE: "Price per night (₹)", "room_type": "Room type"})
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.histogram(filtered, x="availability_365", nbins=37,
                           title="Available days per year",
                           labels={"availability_365": "Available days per year"})
        fig.update_yaxes(title="Number of listings")
        st.plotly_chart(fig, width="stretch")
    rating = (filtered.dropna(subset=["review_rate_number"])
              .groupby("review_rate_number", as_index=False)[PRICE].median())
    fig = px.bar(rating, x="review_rate_number", y=PRICE,
                 title="Median price by rating",
                 labels={"review_rate_number": "Rating (1-5)",
                         PRICE: "Median price (₹ per night)"})
    st.plotly_chart(fig, width="stretch")

if "Segments" in tabs:
    with tabs["Segments"]:
        seg = filtered.dropna(subset=["cluster_name"])
        if seg.empty:
            st.info("No segment labels for the selected filters.")
        else:
            st.caption(
                "Segments come from K-Means clustering in the notebook. "
                "Price was one of the clustering features, so these are "
                "price-and-activity segments, not purely behavioural ones."
            )
            profile = (seg.groupby("cluster_name", as_index=False)
                       .agg(listings=(PRICE, "size"),
                            median_price=(PRICE, "median"),
                            median_reviews=("number_of_reviews", "median"),
                            median_availability=("availability_365", "median"),
                            median_min_nights=("minimum_nights", "median"))
                       .sort_values("median_price", ascending=False))
            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(profile, names="cluster_name", values="listings", hole=0.4,
                             title="Share of listings by segment")
                st.plotly_chart(fig, width="stretch")
            with c2:
                fig = px.bar(profile, x="cluster_name", y="median_price",
                             title="Median price by segment",
                             labels={"cluster_name": "Segment",
                                     "median_price": "Median price (₹ per night)"})
                st.plotly_chart(fig, width="stretch")
            st.dataframe(profile, width="stretch", hide_index=True)

if "Model" in tabs:
    with tabs["Model"]:
        st.write(f"**Final model:** {metrics['model_name']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Test MAE", f"₹{metrics['MAE']:,.0f}")
        m2.metric("Test RMSE", f"₹{metrics['RMSE']:,.0f}")
        m3.metric("Test R²", f"{metrics['R2']:.3f}")
        if "comparison" in metrics:
            st.dataframe(pd.DataFrame(metrics["comparison"]),
                         width="stretch", hide_index=True)
        st.caption("All metrics were calculated on the held-out test set in the notebook.")

with tabs["Data"]:
    st.dataframe(filtered.head(500), width="stretch")
    st.caption("Showing up to 500 filtered listings.")
