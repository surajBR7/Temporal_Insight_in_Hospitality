# Temporal Insights in Hospitality: ARIMA Forecasting


## Project Overview

### This project applies ARIMA Time Series forecasting to hotel occupancy rates in Boston, using economic indicators from 2013 to 2019. By leveraging statistical analysis, we aim to provide insights and predictions to support decision-making in the hospitality industry.

## Dataset
   ### The dataset was sourced from the "Boston Analysis" website, which contains 245 datasets. We used the "Economic Indicators" dataset with 85 records covering various factors, including:

  - Logan passengers (domestic and international)
  - Hotel occupancy rate
  - Hotel average daily rate
  - Total jobs, unemployment rate
  - Housing-related indicators

## Key Factors Analyzed:

  - Logan passengers
  - International flights
  - Hotel occupancy rate
  - Hotel average daily rate

    
## Key Questions Addressed

1. Profit Maximization: How can hotels optimize profits based on the number of passengers arriving in Boston?
2. Peak Occupancy: What is the month with the highest average hotel occupancy rate across all years?
3. Forecasting: Can we build a model to predict future hotel occupancy rates?

## Findings
Peak Month: July consistently has the highest hotel occupancy rate, with an average of 90%.
Price Adjustment: A 5% increase in room rates during July could boost profits without negatively affecting occupancy.
ARIMA Forecasting: An ARIMA model was successfully built to predict hotel occupancy rates based on historical data.

## Methodology
Data Preprocessing: Cleaned the dataset by handling missing values and selecting relevant columns.
Data Visualization: Created correlation heatmaps and time series plots for initial insights.
ARIMA Model: Applied Auto-Regressive Integrated Moving Average (ARIMA) for time series analysis, focusing on hotel occupancy predictions. ACF and PACF were used for model evaluation.

## Results
Correlation Analysis: Logan international passengers showed a moderate correlation with hotel occupancy rates.
Forecast: The ARIMA model predicted occupancy trends for future years, with acceptable error margins (evaluated using RMSE).

## Discussion
Tourism is a major contributor to Boston's economy. Our analysis shows that occupancy rates follow a seasonal pattern, peaking in July. By adjusting prices, hotels can increase revenue during high-demand periods. The ARIMA model is useful for forecasting future trends and assisting decision-making.

## Installation and Code
### Clone the repository from GitHub: 
```bash
git clone https://github.com/surajBR7/MTH_Project_3

```
### requirements from
```bash
pip install -r requirements.txt
```

## Usage

- Load the dataset and preprocess it.
- Run the ARIMA model script to forecast future hotel occupancy rates.
- Visualize the results using the provided plotting functions.

## Author:
### Suraj Basavaraj Rajolad


References
[Data source](https://github.com)
 

