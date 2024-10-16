# Energy-production-Project-using-EDA-Model-building-and-Deployment
Project Overview
This project aims to analyze an energy production dataset and build a predictive model that forecasts energy output based on key environmental variables such as temperature, exhaust vacuum, ambient pressure, and relative humidity. We perform extensive Exploratory Data Analysis (EDA), build multiple machine learning models, and deploy the best model using Flask as an API for real-time predictions.

Project Pipeline
Data Cleaning:

Handled missing values, corrected data types, and standardized variables.
Exploratory Data Analysis (EDA):

Generated summary statistics.
Visualized relationships using scatter plots, pair plots, and a correlation heatmap.
Checked for multicollinearity using the Variance Inflation Factor (VIF).
Feature Engineering:

Analyzed features and transformed variables to improve model performance.
Model Building:

Trained multiple regression models including:
Linear Regression
Weighted Least Squares Regression
Multiple Linear Regression
Evaluated model performance using metrics such as Mean Squared Error (MSE) and R² Score.
Deployment:

Deployed the best model using Flask.
Created an API (EnergyPredictionAPI) for real-time energy output prediction.
