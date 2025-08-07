## Student Performance Analysis - Model Evaluation Report

## Overview
This project analyzes student performance data using various machine learning regression models to predict academic scores. The analysis evaluates models based on math, reading, and writing scores in relation to demographic and educational factors.

## Dataset Description
The dataset contains the following features about students:

**Features:**
**gender:** Student's gender (male/female)

**race_ethnicity:** Ethnic group classification (group A, B, C, etc.)

**parental_level_of_education:** Highest education level of parents

**lunch:** Type of lunch program (standard/free or reduced)

**test_preparation_course**: Whether student completed test prep (none/completed)

## Target Variables:
**math_score**: Numerical score in mathematics

**reading_score**: Numerical score in reading

**writing_score:** Numerical score in writing

## Model Performance Results
We evaluated several regression models with the following metrics:

**R2 Score**: Coefficient of determination (closer to 1 is better)

**MAE**: Mean Absolute Error (lower is better)

**MSE**: Mean Squared Error (lower is better)

## Model Comparison

| Model                      | R2 Score  | MAE     | MSE      |
|----------------------------|-----------|---------|----------|
| LinearRegression           | 0.8804    | 4.2148  | 29.0952  |
| GradientBoostingRegressor  | 0.8727    | 4.2862  | 30.9735  |
| RandomForestRegressor      | 0.8552    | 4.5774  | 35.2277  |
| CatBoostRegressor          | 0.8524    | 4.5722  | 35.9275  |
| XGBRegressor               | 0.8231    | 5.0907  | 43.0490  |
| DecisionTreeRegressor      | 0.7265    | 6.4450  | 66.5650  |
| KNeighborsRegressor        | 0.4756    | 8.6910  | 127.6038 |


## Key Findings
**Best Performing Model:** Linear Regression achieved the highest R2 score (0.8804) and lowest errors

**Tree-based Models:** Gradient Boosting and Random Forest performed nearly as well as Linear Regression

**Weakest Performer**: K-Nearest Neighbors had significantly worse performance than other models


