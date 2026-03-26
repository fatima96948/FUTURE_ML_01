# ================================
# SALES FORECASTING PROJECT
# ================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------

try:
    df= pd.read_csv("SampleSuperstore.csv", encoding= 'latin1')
    print("Dataset is successfully loaded!")
except:
    print("Error: Make sure 'SampleSuperstore.csv' is in the same folder.")
    exit()

print(df.head())
print(df.info())    


# -------------------------------
# STEP 2: Data Cleaning
# -------------------------------

print("Cleaning Data")

# Convert Order Date to datetime
df['Order Date']= pd.to_datetime(df['Order Date'])

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Drop missing values
df.dropna(inplace=True)


# -------------------------------
# STEP 3: Convert to Time Series
# -------------------------------

print("\nCreating time series data")

sales_data= df.groupby('Order Date')['Sales'].sum().reset_index()
print(sales_data.head())


# STEP 4: Feature Engineering
# -------------------------------
sales_data['year'] = sales_data['Order Date'].dt.year
sales_data['month'] = sales_data['Order Date'].dt.month
sales_data['day'] = sales_data['Order Date'].dt.day

# Create time index
sales_data['time'] = range(len(sales_data))


# -------------------------------
# STEP 5: Visualize Sales Trend
# -------------------------------

print("\nDisplaying sales trend graph")

plt.figure(figsize=(10,5))
plt.plot(sales_data['Order Date'], sales_data['Sales'])
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()


# -------------------------------
# STEP 6: Train Model
# -------------------------------

print("\nTraining model")

x=sales_data[['time']]
y=sales_data['Sales']

model= LinearRegression()
model.fit(x,y)


# -------------------------------
# STEP 7: Evaluate Model
# -------------------------------

print("\nEvaluating model")
predictions= model.predict(x)

mae= mean_absolute_error(y, predictions)
mse= mean_squared_error(y, predictions)

print("MAE:", mae)
print("MSE:", mse)


# -------------------------------
# STEP 8: Future Forecast
# -------------------------------
print("\nForecasting future sales")

future_days = 30
future_time = np.arange(len(sales_data), len(sales_data) + future_days).reshape(-1,1)

future_predictions = model.predict(future_time)


# -------------------------------
# STEP 9: Plot Forecast
# -------------------------------
print("\nDisplaying forecast graph")

plt.figure(figsize=(10,5))

# Actual data
plt.plot(sales_data['time'], sales_data['Sales'], label ='Actual Sales')
# Forecast data
plt.plot(range(len(sales_data), len(sales_data)+future_days),
         future_predictions, label='Forecast', linestyle='--')

plt.title("Sales Forecast (Next 30 Days)")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.legend()
plt.show()


# -------------------------------
# STEP 10: Final Message
# -------------------------------
print("\nProject completed successfully!")
print("Congratulations! You have built a Sales Forecasting Model.")
