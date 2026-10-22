import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("hotel_business_1000_rows.csv")

print("Dataset loaded successfully!")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

#-------------------------------check Missing Value---------------------
print("\nMissing Values:")
print(df.isnull().sum())

#----------------------------check Duplicate value----------------------
print("\nDuplicate Rows:")
print(df.duplicated().sum())

#----------------------Convert Date ------------------------------------
df["Date"] = pd.to_datetime(df["Date"])
print(df["Date"].min())
print(df["Date"].max())

#--------------------------Calculate Occupany --------------------------
df["Occupancy_Calculated"] = (
    df["Occupied_Rooms"] /
    df["Total_Rooms"]
) * 100

df["Occupancy_Calculated"] = df["Occupancy_Calculated"].round(2)

print(df[[
    "Hotel",
    "Total_Rooms",
    "Occupied_Rooms",
    "Occupancy_Calculated"
]].head(10))

#----------------------------Basic Hotel Analysis-----------------------

# Overall Occupancy
overall_occupancy = df["Occupancy"].mean()
print(
    "Overall Average Occupancy:",
    round(overall_occupancy, 2),
    "%"
)

# Total Revenue
total_revenue = df["Revenue"].sum()
print(
    "Total Revenue: Rs.",
    round(total_revenue, 2)
)

# Total Expenses
total_expense = df["Total_Expense"].sum()
print(
    "Total Expense: Rs.",
    round(total_expense, 2)
)

# Total Profit
total_profit = df["Profit"].sum()
print(
    "Total Profit: Rs.",
    round(total_profit, 2)
)

# Total Bookings
total_bookings = df["Bookings"].sum()
print(
    "Total Bookings:",
    int(total_bookings)
)

#-----------------HOTEL WISE ANALYSIS---------------------------
hotel_summary = df.groupby("Hotel").agg({
    "Total_Rooms": "mean",
    "Occupied_Rooms": "sum",
    "Bookings": "sum",
    "Revenue": "sum",
    "Total_Expense": "sum",
    "Profit": "sum",
    "Occupancy": "mean"
}).reset_index()

print("\nHotel Summary:")
print(hotel_summary)

#Save it
hotel_summary.to_csv(
    "hotel_summary.csv",
    index=False
)

#--------------------------Momthly Analysis-----------------------
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.month_name()

monthly_summary = df.groupby(
    ["Month", "Month_Name"]
).agg({
    "Revenue": "sum",
    "Total_Expense": "sum",
    "Profit": "sum",
    "Bookings": "sum",
    "Occupancy": "mean"
}).reset_index()

print("\nMonthly Summary:")
print(monthly_summary)

#save
monthly_summary.to_csv(
    "monthly_summary.csv",
    index=False
)

# ---------------------- Create Date Features ----------------------

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["DayOfYear"] = df["Date"].dt.dayofyear

# Weekend: Saturday/Sunday = 1, Weekday = 0
df["Weekend"] = (df["DayOfWeek"] >= 5).astype(int)


# ---------------------- Create Previous Occupancy -------------------

# Sort by Hotel and Date
df = df.sort_values(["Hotel", "Date"])

# Previous day's occupancy for each hotel
df["Previous_Occupancy"] = (
    df.groupby("Hotel")["Occupancy"]
    .shift(1)
)


# ---------------------- Create 7-Day Average Occupancy ----------------

# Average occupancy of previous 7 days
df["Occupancy_7D_Average"] = (
    df.groupby("Hotel")["Occupancy"]
    .transform(
        lambda x: x.shift(1)
        .rolling(7)
        .mean()
    )
)


# ---------------------- Check Output ----------------------

print(df[
    [
        "Hotel",
        "Date",
        "Occupancy",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "DayOfYear",
        "Weekend",
        "Previous_Occupancy",
        "Occupancy_7D_Average"
    ]
].head(20))

# ---------------------- Create Previous Bookings ---------------

df["Previous_Bookings"] = (
    df.groupby("Hotel")["Bookings"]
    .shift(1)
)

df["Bookings_7D_Average"] = (
    df.groupby("Hotel")["Bookings"]
    .transform(
        lambda x: x.shift(1)
        .rolling(7)
        .mean()
    )
)


# ---------------------- Create Previous Revenue ----------------------

df["Previous_Revenue"] = (
    df.groupby("Hotel")["Revenue"]
    .shift(1)
)

df["Revenue_7D_Average"] = (
    df.groupby("Hotel")["Revenue"]
    .transform(
        lambda x: x.shift(1)
        .rolling(7)
        .mean()
    )
)


# ---------------------- Create Previous Expense ----------------------

df["Previous_Expense"] = (
    df.groupby("Hotel")["Total_Expense"]
    .shift(1)
)

df["Expense_7D_Average"] = (
    df.groupby("Hotel")["Total_Expense"]
    .transform(
        lambda x: x.shift(1)
        .rolling(7)
        .mean()
    )
)


# ---------------------- Check Output ----------------------

print(df[
    [
        "Hotel",
        "Date",
        "Bookings",
        "Previous_Bookings",
        "Bookings_7D_Average",
        "Revenue",
        "Previous_Revenue",
        "Revenue_7D_Average",
        "Total_Expense",
        "Previous_Expense",
        "Expense_7D_Average"
    ]
].head(20))

#------------------Remove first missing rows-----------------

df = df.dropna()
print("After feature engineering:")
print(df.shape)

#---------------Encode Hotel and City--------------------
df = pd.get_dummies(
    df,
    columns=["Hotel", "City"],
    dtype=int
)
print(df.head())

#----------------Occupancy ML model-----------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

features = [
    "Total_Rooms",
    "ADR",
    "Advance_Bookings",
    "Weekend",
    "Holiday",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "DayOfYear",
    "Previous_Occupancy",
    "Occupancy_7D_Average",
    "Previous_Bookings",
    "Bookings_7D_Average",
    "Previous_Revenue",
    "Revenue_7D_Average",
    "Previous_Expense",
    "Expense_7D_Average"
]

dummy_columns = [
    column
    for column in df.columns
    if column.startswith("Hotel_")
    or column.startswith("City_")
]

features = features + dummy_columns

X = df[features]
y = df["Occupancy"]

split = int(len(df) * 0.80)
X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

occupancy_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
occupancy_model.fit(
    X_train,
    y_train
)

occupancy_prediction = occupancy_model.predict(
    X_test
)
print(
    "Predicted Occupancy:"
)
print(
    occupancy_prediction[:10]
)


mae = mean_absolute_error(
    y_test,
    occupancy_prediction
)
print(
    "MAE:",
    round(mae, 2)
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        occupancy_prediction
    )
)
print(
    "RMSE:",
    round(rmse, 2)
)


r2 = r2_score(
    y_test,
    occupancy_prediction
)
print(
    "R2 Score:",
    round(r2, 3)
)


import joblib

joblib.dump(
    occupancy_model,
    "occupancy_model.joblib"
)

occupancy_results = pd.DataFrame({
    "Actual_Occupancy": y_test.values,
    "Predicted_Occupancy": occupancy_prediction
})

occupancy_results["Difference"] = (
    occupancy_results["Actual_Occupancy"]
    -
    occupancy_results["Predicted_Occupancy"]
)

print(
    occupancy_results.head(20)
)

occupancy_results.to_csv(
    "occupancy_predictions.csv",
    index=False
)

y = df["Bookings"]
bookings_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

bookings_model.fit(
    X_train,
    y_train
)

bookings_prediction = bookings_model.predict(
    X_test
)

print(
    "Bookings MAE:",
    mean_absolute_error(
        y_test,
        bookings_prediction
    )
)

print(
    "Bookings RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test,
            bookings_prediction
        )
    )
)

print(
    "Bookings R2:",
    r2_score(
        y_test,
        bookings_prediction
    )
)

joblib.dump(
    bookings_model,
    "bookings_model.joblib"
)

y = df["Revenue"]
revenue_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

revenue_model.fit(
    X_train,
    y_train
)

revenue_prediction = revenue_model.predict(
    X_test
)

print(
    "Revenue MAE:",
    mean_absolute_error(
        y_test,
        revenue_prediction
    )
)

print(
    "Revenue RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test,
            revenue_prediction
        )
    )
)

print(
    "Revenue R2:",
    r2_score(
        y_test,
        revenue_prediction
    )
)

joblib.dump(
    revenue_model,
    "revenue_model.joblib"
)

y = df["Total_Expense"]

expense_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

expense_model.fit(
    X_train,
    y_train
)

expense_prediction = expense_model.predict(
    X_test
)

print(
    "Expense MAE:",
    mean_absolute_error(
        y_test,
        expense_prediction
    )
)

print(
    "Expense RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test,
            expense_prediction
        )
    )
)

print(
    "Expense R2:",
    r2_score(
        y_test,
        expense_prediction
    )
)

joblib.dump(
    expense_model,
    "expense_model.joblib"
)

profit_prediction = (
    revenue_prediction
    -
    expense_prediction
)

print(
    "Predicted Profit:"
)

print(
    profit_prediction[:10]
)

all_predictions = pd.DataFrame({
    "Actual_Occupancy": df["Occupancy"].iloc[split:].values,
    "Predicted_Occupancy": occupancy_prediction,

    "Actual_Bookings": df["Bookings"].iloc[split:].values,
    "Predicted_Bookings": bookings_prediction,

    "Actual_Revenue": df["Revenue"].iloc[split:].values,
    "Predicted_Revenue": revenue_prediction,

    "Actual_Expense": df["Total_Expense"].iloc[split:].values,
    "Predicted_Expense": expense_prediction,

    "Predicted_Profit": profit_prediction
})

all_predictions.to_csv(
    "ALL_ML_PREDICTIONS.csv",
    index=False
)

print("\n==============================")
print("      MODEL PERFORMANCE")
print("==============================")

print(
    "Occupancy R2:",
    round(
        r2_score(
            df["Occupancy"].iloc[split:],
            occupancy_prediction
        ),
        3
    )
)

print(
    "Bookings R2:",
    round(
        r2_score(
            df["Bookings"].iloc[split:],
            bookings_prediction
        ),
        3
    )
)

print(
    "Revenue R2:",
    round(
        r2_score(
            df["Revenue"].iloc[split:],
            revenue_prediction
        ),
        3
    )
)

print(
    "Expense R2:",
    round(
        r2_score(
            df["Total_Expense"].iloc[split:],
            expense_prediction
        ),
        3
    )
)




