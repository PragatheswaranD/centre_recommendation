"""
Author: Praga
Email: 
Date Created: March 11, 2025
Last Modified: March 11, 2025
Description:
    This script is a submission for the Sales Analysis project. The script reads a CSV file containing sales data,
    performs some data transformation and analysis, and generates visualizations to help understand the data better.
    The script generates the following visualizations:
    - Pair plot to analyze the relationships between different features
    - Heatmap to visualize the correlation between different features
    - Outlier and skewness analysis for the features
 
    The author has a background in data science and has worked on various projects
    involving Python, data processing, and machine learning.

License:
    This code is licensed under the MIT License. See LICENSE file for more details.

Usage:
    To run this script, ensure you have the required libraries installed:
    python - 3.11
    - pandas
    - matplotlib
    - numpy
    - seaborn
    - sklearn

    You can execute the script using the following command:
    python submission.py
"""



# FINAL RECOMMENDATION:
# Based on the analysis, I recommend the following centres for expansion:


# ======================================================================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from pycaret.regression import *

sns.set_style('whitegrid')
np.random.seed(42)

# reading the data
current_centres = pd.read_excel("data/data.xlsx", sheet_name="Current Centres")
potential_centres = pd.read_excel("data/data.xlsx", sheet_name="Potential Centres")

print(current_centres.head())
print(potential_centres.head())


# Data Encoding
# -------------
# Idenfitied the numerical and categorical columns and encoded the categorical columns
encoding_map = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6
}
current_centres["AREA_AFFLUENCE_GRADE"] = current_centres["AREA_AFFLUENCE_GRADE"].map(encoding_map)
# This is the assumption that the grades are ordinal and can be encoded in this way that follows the 
# council tax bands in the UK A-F

# Data Checks
# ----------------

# 1. check for missing values
print(current_centres.isnull().sum())
# since the missing perc is 0.01% we can drop the rows
current_centres.dropna(inplace=True)

# 2. check for duplicates
print(current_centres.duplicated().sum())
# there were 7 duplicates which were dropped
current_centres.drop_duplicates(inplace=True)

# 3. check for skewness
print(current_centres.skew())
# the following columns have high skewness:
# - AREA_POPULATION_DENSITY_PPSKM
# - AVG_DAILY_STAFF
# - ANNUAL_REVENUE
# these columns are specially noted for further analysis


# 4. check for outliers
for column in current_centres.drop('CENTRE_NO', axis=1).columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=current_centres[column])
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.show()

# Following are the columns with large outliers:
# - AREA_POPULATION_DENSITY_PPSKM
# - AVG_DAILY_STAFF
# - AVG_SALARY
# - AREA_EV_PERC
# - ANNUAL_RENT
# - AREA_AFFLUENCE_GRADE
# these columns are specially noted for further analysis
# presence of outliers is a concern and needs to be addressed
# we will use the IQR method to remove the outliers



# 5. Correlation analysis part 1
# for the columns shortlisted in 3 and 4 are analyzed against the ANNUAL_REVENUE
columns_of_interest = [
    'AREA_POPULATION_DENSITY_PPSKM', 'AVG_DAILY_STAFF', 
    'AVG_SALARY', 'AREA_EV_PERC', 
    'ANNUAL_RENT', 'AREA_AFFLUENCE_GRADE']
for column in columns_of_interest:
    sns.scatterplot(data=current_centres, y=column, x='ANNUAL_REVENUE')
    plt.title(f'Scatter plot of {column} against ANNUAL_REVENUE')
    plt.show()

# it is observed that the above columns have some relationship with the ANNUAL_REVENUE
# where annual revenue > 2.0x1e6
# suscpiciously the outliers are also in the same range as per the box plot
# on calculation, 3.3% of the data lies where ANNUAL_REVENUE > 2.0x1e6
# this is a concern and needs to be addressed


# 6. Correlation analysis part 2: pair plot
# to understand more in detail of the relationships between the features
sns.pairplot(current_centres.drop('CENTRE_NO', axis=1), hue="ANNUAL_REVENUE")
plt.title('Pair Plot of Current Centres')
plt.show()

# 7. Correlation analysis part 3: heatmap
# to understand more in detail of the relationships between the features
sns.heatmap(current_centres.drop('CENTRE_NO', axis=1).corr(), fmt=".2f", 
            annot=True)
plt.title('Heatmap of Current Centres')
plt.show()



# 8. Data Filtering
# from the above analysis, the following filters are applied
# - ANNUAL_REVENUE <= 2.0x1e6

sub_df = current_centres[current_centres["ANNUAL_REVENUE"] <= 2.0e6]
# on sub df 3 - 8 are repeated
# the following are the findings
# - AREA_POPULATION_DENSITY_PPSKM is still highly skewed
# in total 5% of the data was removed from the original dataset
# from outlier, skewness and correlation analysis
# the data is now clean and ready for further analysis

# 8. Select the columns
columns_subset = [
    "AVG_SALARY", "AREA_EV_PERC",
    "HOURS_OPEN_PER_WEEK", "ANNUAL_RENT",
    "TOTAL_STAFF", "AREA_AFFLUENCE_GRADE",
    "MOT_BAYS", "SERVICE_BAYS"
]
# the feautres are selected based on the correlation analysis and the pair plot
# multi-collinear features are removed
# the columns are selected based on the correlation with the target column,
# the outliers and skewness

target_column = "ANNUAL_REVENUE"

# 9 - 15. are the same as 1 - 7 but on the potential centres
# not repeating the code for the sake of brevity
# following are the findings
# - There were no missing values found
# - There were no duplicates found
# - The following columns have high skewness: tolerable range considered -1 to -0.5 and 0.5 to 1
#   - AREA_POPULATION_DENSITY_PPSKM


# 16. Baseline Models
# Linear and Random Forest Regression

columns_subset = [
    "AVG_DAILY_STAFF", "AREA_EV_PERC",
    "HOURS_OPEN_PER_WEEK", "ANNUAL_RENT",
    "TOTAL_STAFF", "AREA_AFFLUENCE_GRADE",
    "MOT_BAYS", "SERVICE_BAYS"
]

target_column = "ANNUAL_REVENUE"

x_train, x_test, y_train, y_test = train_test_split(
    sub_df[columns_subset], sub_df[target_column], test_size=0.2, random_state=42,
    shuffle=True
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
model1 = RandomForestRegressor()
model1.fit(x_train, y_train)
model2 = LinearRegression()
model2.fit(x_train, y_train)

predictions = model1.predict(x_test)

print("Mean Absolute Error: ", round(
    mean_absolute_error(y_test, predictions), 3))
print("R2 Score: ", round(
    r2_score(y_test, predictions), 3))
print("Root Mean Squared Error: ", round(
    root_mean_squared_error(y_test, predictions), 3))
'''
Mean Absolute Error:  154150.595
R2 Score:  0.674
Root Mean Squared Error:  198062.855
'''

predictions = model2.predict(x_test)

print("Mean Absolute Error: ", round(
    mean_absolute_error(y_test, predictions), 3))
print("R2 Score: ", round(
    r2_score(y_test, predictions), 3))
print("Root Mean Squared Error: ", round(
    root_mean_squared_error(y_test, predictions), 3))
'''
Mean Absolute Error:  151811.773
R2 Score:  0.683
Root Mean Squared Error:  195475.632
'''

# 17. PyCaret AutoML implementation
from pycaret.regression import *

reg = setup(
    data=sub_df[columns_subset + ["ANNUAL_REVENUE"]],
    target='ANNUAL_REVENUE',
    session_id=123,
    normalize=False,
    transformation=False,
    preprocess=False,
    ignore_features=['CENTRE_NO']
)

best_model = compare_models()
# best model - GradientBoost Regressor
'''
Mean Absolute Error:  152331.692
R2 Score:  0.679
Root Mean Squared Error:  194180
'''

# **** Linear Regressor is the best model so far ****


# 18. Feature Selection
# Feature selection and importance based on the Random Forest Regressor baseline model
# since RF is good less sensitive to outliers and skewness, it is used for feature selection

x_data = sub_df.drop(["ANNUAL_REVENUE", "CENTRE_NO"], axis=1)
y_data = sub_df["ANNUAL_REVENUE"]

feature_selection_model = RandomForestRegressor()
feature_selection_model.fit(x_data, y_data)
feature_selection_model.feature_importances_

importances = feature_selection_model.feature_importances_
feature_names = x_data.columns

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Random Forest Regressor')
plt.show()

## from the feature importance plot, the following features are important
#  in the order highest to lowest
# - "TOTAL_STAFF"
# - "ANNUL_RENT"
# - "SERVICE_BAYS"
# - "HOURS_OPEN_PER_WEEK"
# - "AREA_POPULATION_DENSITY_PPSKM"
# - "AVG_DAILY_STAFF"
# - "MOT_BAYS
# - "TYRE_BAYS"
# - "AVG_SALARY"
# - "AREA_EV_PERC"
# - "AREA_AFFLUENCE_GRADE"
# only top 2 are > 0.1 and the rest are < 0.1
# this in alignment with the correlation analysis
# where TOTAL STAFF had the highest correlation with the target column
# When ANNUAL_REVENUE > 2e6 is considered, 
# top 3 are > 0.1 and the rest are < 0.1, but the order remains almost the same

# 19. Linear Model + Regularization
# since the data for training is small, contains outliers and skewness
# LASSO is used for regularization
# Also, since L1 reg reduces the coefficients to 0, it can be used for feature selection
# thus the selected features could be compared with the feature importance from RF
model3 = Lasso(alpha=0.3)
model3.fit(x_train, y_train)

predictions = model3.predict(x_test)

print("Mean Absolute Error: ", round(
    mean_absolute_error(y_test, predictions), 3))
print("R2 Score: ", round(
    r2_score(y_test, predictions), 3))
print("Root Mean Squared Error: ", round(
    root_mean_squared_error(y_test, predictions), 3))

coeff_ = model3.coef_
selected_features = x_train.columns[coeff_ != 0]
selected_coeff = coeff_[coeff_ != 0]

result_df = pd.DataFrame({'Feature': selected_features, 
                          'Coefficient': selected_coeff})
print(result_df.sort_values(by='Coefficient', ascending=False))
# Features selected are, in order of the coefficients
# - SERVICE_BAYS
# - MOT_BAYS
# - TOTAL_STAFF
# - AREA_AFFLUENCE_GRADE
# - HOURS_OPEN_PER_WEEK
# - AREA_POPULATION_DENSITY_PPSKM
# - ANNUAL_RENT
# - AVG_DAILY_STAFF
# - AREA_EV_PERC

