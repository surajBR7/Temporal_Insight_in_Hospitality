import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv("economic-indicators.csv")

# Remove unnecessary columns
columns_to_drop = ['total_jobs', 'unemp_rate', 'labor_force_part_rate', 'pipeline_unit', 
                   'pipeline_total_dev_cost', 'pipeline_sqft', 'pipeline_const_jobs', 
                   'foreclosure_pet', 'foreclosure_deeds', 'med_housing_price', 
                   'housing_sales_vol', 'new_housing_const_permits', 'new-affordable_housing_permits']
df_cleaned = df.drop(columns=columns_to_drop, axis=1)
print(df_cleaned.isnull().sum())
# Calculate correlation between Logan Passengers and Hotel Occupancy Rate
correlation = df_cleaned['logan_passengers'].corr(df_cleaned['hotel_occup_rate'])

# Visualizations: Total Logan Passengers and Hotel Occupancy Rate by Year
yearly_passengers = df.groupby('Year')['logan_passengers'].sum()
yearly_hotel_occ_rate = df.groupby('Year')['hotel_occup_rate'].sum()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.barh(yearly_passengers.index, yearly_passengers.values, color='skyblue')
ax1.set_xlabel('Total Logan Passengers')
ax1.set_ylabel('Year')
ax1.set_title('Total Logan Passengers by Year')
ax1.grid(axis='x')

ax2.barh(yearly_hotel_occ_rate.index, yearly_hotel_occ_rate.values, color='orange')
ax2.set_xlabel('Total Hotel Occupancy Rate')
ax2.set_ylabel('Year')
ax2.set_title('Total Hotel Occupancy Rate by Year')
ax2.grid(axis='x')

plt.tight_layout()
plt.show()

# Correlation Heatmap
selected_columns = ['logan_passengers', 'logan_intl_flights', 'hotel_occup_rate', 'hotel_avg_daily_rate']
correlation_matrix = df[selected_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Heatmap')
plt.show()

# Monthly Analysis
yearly_monthly_occupancy = df.groupby(['Year', 'Month'])['hotel_occup_rate'].mean().reset_index()
monthly_avg_occupancy = yearly_monthly_occupancy.groupby('Month')['hotel_occup_rate'].mean().reset_index()

unique_years = yearly_monthly_occupancy['Year'].unique()
for year in unique_years:
    data_year = yearly_monthly_occupancy[yearly_monthly_occupancy['Year'] == year]
    plt.figure(figsize=(10, 6))
    plt.bar(data_year['Month'], data_year['hotel_occup_rate'], color='skyblue')
    plt.xlabel('Month')
    plt.ylabel('Average Hotel Occupancy Rate')
    plt.title(f'Average Hotel Occupancy Rate per Month in {year}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

########trying to find the occupancy rate##

max_occupancy_month = monthly_avg_occupancy.loc[monthly_avg_occupancy['hotel_occup_rate'].idxmax()]['Month']
max_occupancy_rate = monthly_avg_occupancy['hotel_occup_rate'].max()

print(f"The month with the highest average hotel occupancy rate across all years is {max_occupancy_month} with an average rate of {max_occupancy_rate:.2f}")


######Time series analysis######



df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))

df.set_index('Date', inplace=True)

# Plotting the time series data
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(df['logan_passengers'], color='blue')
plt.title('Logan Passengers Over Time')
plt.ylabel('Passenger Count')

plt.subplot(2, 1, 2)
plt.plot(df['hotel_occup_rate'], color='red')
plt.title('Hotel Occupancy Rate Over Time')
plt.ylabel('Occupancy Rate')

plt.xlabel('Date')
plt.tight_layout()
plt.show()



# Convert 'Year' and 'Month' to datetime format and set as index
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
df.set_index('Date', inplace=True)


df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
df.set_index('Date', inplace=True)

# Extract the 'hotel_occup_rate' column
data = df['hotel_occup_rate']

# Plot ACF and PACF to determine p and q parameters
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(data, ax=plt.gca(), lags=40)
plt.title('Autocorrelation Function (ACF)')
plt.subplot(2, 1, 2)
plot_pacf(data, ax=plt.gca(), lags=40)
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

y_true = [0.629,0.725,0.812,0.877,0.878,0.898,0.9,0.888,0.895,0.907,0.803,0.717]
# Fit ARIMA model
order = (2, 1, 2) 
model = ARIMA(data, order=order)
result = model.fit()

# Forecast next 12 months
forecast_steps = 12
forecast = result.get_forecast(steps=forecast_steps)
forecast_values = forecast.predicted_mean

# Plotting original data and forecast
plt.figure(figsize=(10, 6))
plt.plot(data, label='Original Data')
plt.plot(forecast_values.index, forecast_values, color='red', label='Forecast')
plt.title('Hotel Occupancy Rate - Original and Forecast')
plt.xlabel('Date')
plt.ylabel('Occupancy Rate')
plt.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(y_true,forecast_values ))
print(f"RMSE: {rmse}")