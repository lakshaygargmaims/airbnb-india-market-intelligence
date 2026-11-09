# Airbnb India Market Intelligence

**Pricing analysis, market segmentation and price prediction for Airbnb-style listings in five Indian cities**

An end-to-end data analytics and machine learning project. It covers data cleaning, exploratory analysis, hypothesis testing, leakage-free regression modelling, K-Means segmentation and two Streamlit applications: an analytics dashboard and a price predictor.

---

## Results at a glance

All numbers come from executing the notebook (random seed 42).

| | |
|---|---|
| **Dataset** | 102,599 listings × 25 columns: Delhi, Mumbai, Bengaluru, Jaipur, Hyderabad (60 neighbourhoods) |
| **Typical price** | Median ₹2,750 / night (mean ₹3,173) |
| **Most / least expensive city** | Mumbai (median ₹3,450) / Jaipur (₹2,200) |
| **Biggest price driver** | Room type. An entire home costs about 2× a private room (₹3,850 vs ₹1,900) in every city. Cohen's d = 1.41; room type explains 35% of price variation (η²) |
| **Best model** | XGBoost: **test MAE ₹954, RMSE ₹1,341, R² 0.486** |
| **Baseline (median price)** | MAE ₹1,385, RMSE ₹1,916, R² −0.049. The model cuts error by about 30% |
| **Most important features** | Room type, neighbourhood, city (about 98.5% of model importance) |
| **Segments (K-Means, K = 4)** | Active performers (32.5%), dormant listings (39.2%), open-but-under-booked (21.3%), professional long-stay hosts (7.0%) |
| **App tests** | 13 / 13 automated Streamlit tests passed |

### Model comparison (test set, 20,520 listings)

| Model | CV RMSE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| **XGBoost (selected)** | ₹1,367 | ₹954 | ₹1,341 | 0.486 |
| Random Forest | ₹1,368 | ₹954 | ₹1,341 | 0.486 |
| Linear Regression | ₹1,379 | ₹971 | ₹1,355 | 0.475 |
| Decision Tree | ₹1,461 | ₹1,012 | ₹1,434 | 0.412 |
| Baseline (median) | ₹1,931 | ₹1,385 | ₹1,916 | −0.049 |

The model was selected by **cross-validation RMSE** on the training data. The test set was used only once, to confirm the choice. Hyperparameter tuning changed CV RMSE by less than ₹2, so performance is limited by the information in the data, not by model settings.

---

## Key insights

1. **Room type and location set the price.** Ratings, reviews, availability, instant booking, cancellation policy and host verification show no meaningful price difference in this data.
2. **Neighbourhood matters within a city.** The most expensive neighbourhood is 1.36× (Delhi) to 1.56× (Jaipur) the cheapest. The top seven neighbourhoods are all in Mumbai, led by Bandra West and Juhu.
3. **The model under-prices premium listings.** It under-predicts listings above ₹8,000 by about ₹4,600 on average, because amenities and quality are not in the data.
4. **Segments are activity-based.** Price is a clustering feature, but segments separate mainly by reviews, availability and host type. 39% of listings appear dormant.

All findings are associations, not proof of cause and effect.

---

## Project structure

```
airbnb-india-market-intelligence/
├── Airbnb_India_Market_Intelligence_FINAL.ipynb   # full analysis (35 sections)
├── airbnb_india_listings.csv                      # raw dataset (add yourself)
├── dashboard.py                                   # analytics dashboard   -> port 8501
├── predict_app.py                                 # price prediction app  -> port 8502
├── requirements.txt
├── README.md
├── outputs/        # created by the notebook
│   ├── airbnb_india_cleaned.csv
│   ├── airbnb_clustered_listings.csv
│   ├── cluster_profiles.csv
│   ├── model_metrics.json
│   ├── eda_tables/
│   └── statistical_outputs/
└── models/         # created by the notebook
    ├── airbnb_price_model_bundle.joblib
    └── airbnb_kmeans_bundle.joblib
```

## Notebook contents

| Stage | Sections |
|---|---|
| Overview, problem, objectives, dataset | 1 – 4 |
| Setup, loading, data understanding | 5 – 7 |
| Cleaning, missing values and outliers, feature engineering | 8 – 10 |
| Exploratory data analysis | 11 |
| Statistical tests (Shapiro-Wilk, Spearman, Welch t-test, ANOVA, Kruskal-Wallis, Chi-square) | 12 |
| ML problem, split, baseline, model comparison, tuning, final evaluation | 13 – 18 |
| Residual analysis (by price band, city, room type) and feature importance | 19 – 20 |
| Model bundle: save, reload and verify | 21 |
| K-Means segmentation and PCA | 22 – 23 |
| Business recommendations | 24 |
| Dashboard, prediction app, automated app tests, Colab launch | 25 – 28 |
| Deployment, presentation guide, limitations, summary, conclusion, viva questions | 29 – 35 |

Every major section contains an introduction, objective, methodology, code explanation, findings, business interpretation, a short presentation script and viva questions.

---

## How to run

### Google Colab (recommended)
1. Upload `Airbnb_India_Market_Intelligence_FINAL.ipynb` to Colab.
2. Choose **Runtime → Run all**. When asked, upload `airbnb_india_listings.csv`.
3. Section 28 starts both apps and prints a link for each:
   - Dashboard on port **8501**
   - Prediction app on port **8502**

The full notebook takes about 5–10 minutes. The hyperparameter-tuning cell is the slowest.

### Local computer
```bash
git clone https://github.com/lakshaygargmaims/airbnb-india-market-intelligence.git
cd airbnb-india-market-intelligence
pip install -r requirements.txt
jupyter notebook Airbnb_India_Market_Intelligence_FINAL.ipynb   # run all cells once

# then, in two terminals:
streamlit run dashboard.py --server.port 8501
streamlit run predict_app.py --server.port 8502
```

The apps need the `outputs/` and `models/` folders that the notebook creates. If those files are missing, each app shows a clear message instead of crashing.

---

## Applications

**Analytics dashboard (`dashboard.py`)**
- Filters for city, room type and price range
- KPIs, plus tabs for pricing, cities and neighbourhoods, reviews and availability, segments, model performance and raw data
- Cluster labels are joined on `listing_id`

**Price prediction app (`predict_app.py`)**
- Loads one bundle: the fitted scikit-learn `Pipeline`, the feature order, the input schema and the metrics
- The input form is generated from the saved schema, so inputs always match the model
- The neighbourhood list follows the selected city; all inputs are validated before predicting
- Shows the estimated nightly price and a rough range (± test MAE)

---

## Technical notes

- **No data leakage:** the train/test split comes first. Imputation, scaling and one-hot encoding happen inside a `Pipeline` fitted only on training data. Price-derived columns (outlier flag, cluster labels) are excluded from model inputs.
- **Reproducible:** `random_state = 42` everywhere. Packages are installed only if missing.
- **Model bundle:** numeric inputs always use the keys `minimum`, `maximum`, `default`, `step` and `integer`. The bundle is reloaded and its predictions are checked against the in-memory model.
- **Versions:** load the bundle with the same scikit-learn and XGBoost versions that created it. They are stored in `bundle["library_versions"]`.

## Limitations

- The data source is undocumented. Several categories are almost perfectly balanced, every city has the same room-type mix, and the two host columns contradict each other. The data may be synthetic, so results describe this dataset, not the verified Indian Airbnb market.
- There is no information on amenities, size, photos, season, occupancy or revenue, which caps R² at about 0.49.
- K-Means silhouette scores are low (about 0.17–0.19), so segments overlap.

## Future work

- Add amenities, size and text features
- Model occupancy and revenue as well as price
- Give price ranges with quantile regression
- Deploy both apps on Streamlit Community Cloud
- Validate on a documented real dataset such as Inside Airbnb

---

## Tech stack

Python · pandas · NumPy · SciPy · Matplotlib · Seaborn · Plotly · scikit-learn · XGBoost · Streamlit · joblib · Google Colab

## Author

**Lakshay Garg**, BCA student, Data Analytics and Machine Learning
GitHub: [lakshaygargmaims](https://github.com/lakshaygargmaims) · LinkedIn: [lakshaygargmaims](https://linkedin.com/in/lakshaygargmaims)

> Predicted prices are estimates for learning and demonstration purposes, not guaranteed booking prices.
