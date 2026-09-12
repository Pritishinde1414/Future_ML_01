# ============================================================
# SALES & DEMAND FORECASTING
# Superstore Sales Dataset
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Keep the CSV file in the same folder as this Python file
CSV_FILE = "Sample - Superstore.csv"

df = pd.read_csv(CSV_FILE, encoding="latin1")
print("Dataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. UNDERSTAND DATASET
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\nMissing values in each column:")
print(df.isnull().sum())


# ============================================================
# 5. CHECK DUPLICATE RECORDS
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate rows:", duplicate_count)

df = df.drop_duplicates()

print("Shape after removing duplicates:")
print(df.shape)


# ============================================================
# 6. CONVERT ORDER DATE TO DATETIME
# ============================================================

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)

print("\nOrder Date data type:")
print(df["Order Date"].dtype)

# Remove rows where date could not be converted
df = df.dropna(subset=["Order Date"])


# ============================================================
# 7. HANDLE MISSING SALES VALUES
# ============================================================

print("\nMissing Sales values before cleaning:")
print(df["Sales"].isnull().sum())

df = df.dropna(subset=["Sales"])

print("Missing Sales values after cleaning:")
print(df["Sales"].isnull().sum())


# ============================================================
# 8. STATISTICAL ANALYSIS
# ============================================================

print("\nStatistical Summary:")
print(df.describe())

print("\nSales Statistics:")
print("Total Sales:", round(df["Sales"].sum(), 2))
print("Average Sale:", round(df["Sales"].mean(), 2))
print("Minimum Sale:", round(df["Sales"].min(), 2))
print("Maximum Sale:", round(df["Sales"].max(), 2))


# ============================================================
# 9. SALES BY CATEGORY
# ============================================================

category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Category:")
print(category_sales)

plt.figure(figsize=(8, 5))

category_sales.plot(kind="bar")

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 10. SALES BY REGION
# ============================================================

region_sales = (
    df.groupby("Region")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Region:")
print(region_sales)

plt.figure(figsize=(8, 5))

region_sales.plot(kind="bar")

plt.title("Total Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 11. MONTHLY SALES
# ============================================================

# Convert individual transactions into monthly total sales

monthly_sales = (
    df.set_index("Order Date")
    .resample("MS")["Sales"]
    .sum()
    .reset_index()
)

print("\nMonthly Sales:")
print(monthly_sales.head(10))


# ============================================================
# 12. HISTORICAL SALES TREND
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    marker="o"
)

plt.title("Historical Monthly Sales")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.tight_layout()
plt.show()


# ============================================================
# 13. TIME-BASED FEATURE ENGINEERING
# ============================================================

monthly_sales["Year"] = (
    monthly_sales["Order Date"].dt.year
)

monthly_sales["Month"] = (
    monthly_sales["Order Date"].dt.month
)

monthly_sales["Quarter"] = (
    monthly_sales["Order Date"].dt.quarter
)

monthly_sales["Month_Name"] = (
    monthly_sales["Order Date"].dt.month_name()
)

monthly_sales["Time_Index"] = (
    np.arange(len(monthly_sales))
)

print("\nTime-based features:")
print(monthly_sales.head())


# ============================================================
# 14. CREATE LAG FEATURES
# ============================================================

# Previous month's sales
monthly_sales["Lag_1"] = (
    monthly_sales["Sales"].shift(1)
)

# Sales from 3 months ago
monthly_sales["Lag_3"] = (
    monthly_sales["Sales"].shift(3)
)

# Sales from 6 months ago
monthly_sales["Lag_6"] = (
    monthly_sales["Sales"].shift(6)
)

# Sales from 12 months ago
monthly_sales["Lag_12"] = (
    monthly_sales["Sales"].shift(12)
)


# ============================================================
# 15. CREATE ROLLING AVERAGE FEATURES
# ============================================================

monthly_sales["Rolling_Mean_3"] = (
    monthly_sales["Sales"]
    .shift(1)
    .rolling(window=3)
    .mean()
)

monthly_sales["Rolling_Mean_6"] = (
    monthly_sales["Sales"]
    .shift(1)
    .rolling(window=6)
    .mean()
)

print("\nDataset after feature engineering:")
print(monthly_sales.head(15))


# ============================================================
# 16. REMOVE MISSING VALUES CREATED BY LAG FEATURES
# ============================================================

model_data = monthly_sales.dropna().copy()

print("\nOriginal number of rows:", len(monthly_sales))
print("Rows used for modeling:", len(model_data))


# ============================================================
# 17. DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "Year",
    "Month",
    "Quarter",
    "Time_Index",
    "Lag_1",
    "Lag_3",
    "Lag_6",
    "Lag_12",
    "Rolling_Mean_3",
    "Rolling_Mean_6"
]

target = "Sales"


# ============================================================
# 18. TIME-BASED TRAIN/TEST SPLIT
# ============================================================

# We do NOT randomly shuffle time-series data.

split_index = int(len(model_data) * 0.80)

train_data = model_data.iloc[:split_index].copy()

test_data = model_data.iloc[split_index:].copy()

print("\nTraining records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining period:")
print(
    train_data["Order Date"].min(),
    "to",
    train_data["Order Date"].max()
)

print("\nTesting period:")
print(
    test_data["Order Date"].min(),
    "to",
    test_data["Order Date"].max()
)


# ============================================================
# 19. PREPARE TRAINING AND TESTING DATA
# ============================================================

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ============================================================
# 20. TRAIN RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# ============================================================
# 21. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

print("\nPredictions generated successfully!")


# ============================================================
# 22. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n===================================")
print("       MODEL EVALUATION")
print("===================================")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 23. ACTUAL VS PREDICTED TABLE
# ============================================================

results = pd.DataFrame({
    "Date": test_data["Order Date"].values,
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

results["Error"] = (
    results["Actual Sales"]
    - results["Predicted Sales"]
)

results["Absolute Error"] = (
    abs(results["Error"])
)

print("\nActual vs Predicted Sales:")
print(results)


# ============================================================
# 24. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    results["Date"],
    results["Actual Sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    results["Date"],
    results["Predicted Sales"],
    marker="o",
    linestyle="--",
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 25. ERROR ANALYSIS
# ============================================================

plt.figure(figsize=(14, 6))

plt.bar(
    results["Date"].dt.strftime("%Y-%m"),
    results["Absolute Error"]
)

plt.title("Forecast Error Analysis")
plt.xlabel("Month")
plt.ylabel("Absolute Error")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 26. MONTHLY SEASONALITY ANALYSIS
# ============================================================

monthly_pattern = (
    df.assign(
        Month=df["Order Date"].dt.month
    )
    .groupby("Month")["Sales"]
    .mean()
)

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_pattern.index,
    monthly_pattern.values,
    marker="o"
)

plt.title("Average Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Sales")

plt.xticks(range(1, 13))

plt.tight_layout()
plt.show()


# ============================================================
# 27. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# ============================================================
# 28. FUTURE SALES FORECAST
# ============================================================

# Number of future months to predict
forecast_months = 6

history = monthly_sales[
    ["Order Date", "Sales"]
].copy()

future_predictions = []


for i in range(forecast_months):

    # Find next month
    next_date = (
        history["Order Date"].max()
        + pd.DateOffset(months=1)
    )

    year = next_date.year

    month = next_date.month

    quarter = next_date.quarter

    time_index = (
        len(monthly_sales) + i
    )

    sales_values = (
        history["Sales"].values
    )

    # Lag features
    lag_1 = sales_values[-1]

    lag_3 = sales_values[-3]

    lag_6 = sales_values[-6]

    lag_12 = sales_values[-12]

    # Rolling averages
    rolling_mean_3 = np.mean(
        sales_values[-3:]
    )

    rolling_mean_6 = np.mean(
        sales_values[-6:]
    )

    # Create feature row
    future_features = pd.DataFrame({
        "Year": [year],
        "Month": [month],
        "Quarter": [quarter],
        "Time_Index": [time_index],
        "Lag_1": [lag_1],
        "Lag_3": [lag_3],
        "Lag_6": [lag_6],
        "Lag_12": [lag_12],
        "Rolling_Mean_3": [rolling_mean_3],
        "Rolling_Mean_6": [rolling_mean_6]
    })

    # Predict next month's sales
    prediction = model.predict(
        future_features
    )[0]

    # Store prediction
    future_predictions.append({
        "Order Date": next_date,
        "Forecasted Sales": prediction
    })

    # Add prediction to history
    # This allows recursive forecasting
    history = pd.concat(
        [
            history,
            pd.DataFrame({
                "Order Date": [next_date],
                "Sales": [prediction]
            })
        ],
        ignore_index=True
    )


# ============================================================
# 29. DISPLAY FUTURE FORECAST
# ============================================================

future_forecast = pd.DataFrame(
    future_predictions
)

future_forecast[
    "Forecasted Sales"
] = future_forecast[
    "Forecasted Sales"
].round(2)

print("\n===================================")
print("       FUTURE SALES FORECAST")
print("===================================")

print(future_forecast)


# ============================================================
# 30. FINAL FORECAST VISUALIZATION
# ============================================================

plt.figure(figsize=(15, 7))

plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    marker="o",
    label="Historical Sales"
)

plt.plot(
    future_forecast["Order Date"],
    future_forecast["Forecasted Sales"],
    marker="o",
    linestyle="--",
    label="Future Forecast"
)

# Mark forecast starting point
plt.axvline(
    monthly_sales["Order Date"].max(),
    linestyle=":",
    label="Forecast Start"
)

plt.title(
    "Sales Forecast for Next 6 Months"
)

plt.xlabel("Date")

plt.ylabel("Sales")

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 31. FORECAST GROWTH ANALYSIS
# ============================================================

first_forecast = (
    future_forecast[
        "Forecasted Sales"
    ].iloc[0]
)

last_forecast = (
    future_forecast[
        "Forecasted Sales"
    ].iloc[-1]
)

growth_percentage = (
    (last_forecast - first_forecast)
    / first_forecast
) * 100

print(
    f"\nForecasted change from first "
    f"to last month: {growth_percentage:.2f}%"
)


# ============================================================
# 32. BUSINESS INSIGHTS
# ============================================================

best_category = (
    category_sales.idxmax()
)

best_category_sales = (
    category_sales.max()
)

best_region = (
    region_sales.idxmax()
)

best_region_sales = (
    region_sales.max()
)

print("\n==========================================")
print("             BUSINESS INSIGHTS")
print("==========================================")

print(
    f"\nBest Performing Category: "
    f"{best_category}"
)

print(
    f"Total Category Sales: "
    f"₹{best_category_sales:,.2f}"
)

print(
    f"\nBest Performing Region: "
    f"{best_region}"
)

print(
    f"Total Region Sales: "
    f"₹{best_region_sales:,.2f}"
)

print(
    f"\nForecasted change over next "
    f"6 months: {growth_percentage:.2f}%"
)


if growth_percentage > 0:

    print("\nBusiness Recommendation:")
    print(
        "Sales are expected to increase."
    )
    print(
        "The business should prepare "
        "sufficient inventory."
    )
    print(
        "Additional staff may be required "
        "during high-demand periods."
    )
    print(
        "Purchasing and stock planning "
        "should consider the forecast."
    )

else:

    print("\nBusiness Recommendation:")
    print(
        "Sales are expected to decrease."
    )
    print(
        "The business should avoid "
        "excessive inventory."
    )
    print(
        "Marketing and promotional "
        "strategies can be considered."
    )


# ============================================================
# 33. FINAL FORECAST REPORT
# ============================================================

final_report = future_forecast.copy()

final_report["Month"] = (
    final_report["Order Date"]
    .dt.strftime("%B %Y")
)

final_report = final_report[
    [
        "Month",
        "Forecasted Sales"
    ]
]

print("\n===================================")
print("          FINAL FORECAST")
print("===================================")

print(final_report)


# ============================================================
# 34. SAVE FUTURE FORECAST AS CSV
# ============================================================

final_report.to_csv(
    "Future_Sales_Forecast.csv",
    index=False
)

print(
    "\nFuture forecast saved successfully "
    "as 'Future_Sales_Forecast.csv'"
)


# ============================================================
# 35. PROJECT COMPLETED
# ============================================================

print("\n==========================================")
print("       SALES FORECASTING COMPLETED")
print("==========================================")

print(
    "Data Cleaning               : Completed"
)

print(
    "Missing Value Handling      : Completed"
)

print(
    "Time-Based Features         : Completed"
)

print(
    "Trend Analysis              : Completed"
)

print(
    "Seasonality Analysis        : Completed"
)

print(
    "Forecasting Model           : Completed"
)

print(
    "Model Evaluation            : Completed"
)

print(
    "Error Analysis              : Completed"
)

print(
    "Future Sales Prediction     : Completed"
)

print(
    "Business Visualization      : Completed"
)

print(
    "Business Insights           : Completed"
)