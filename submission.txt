"""
Author: Pragatheswaran Dhandapani
Email: praga0028@gmail.com
Date Created: March 11, 2025
Last Modified: March 11, 2025
Description:
    This script is a submission for the Sales Analysis project. The script reads a CSV file containing sales data,
    performs some data transformation and analysis, and generates visualizations to help understand the data better.
    
    The author has a background in data science and has worked on various projects
    involving Python, data processing, and machine learning.

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


Github:
You can find rough notebook and other files in the below github
https://github.com/PragatheswaranD/centre_recommendation/blob/main/data_analysis3.ipynb

"""
# ======================================================================================================================

'''

### FINAL RECOMMENDATION:
# Based on the analysis, I recommend the following centres top 10 for expansion 
# from potential centers:
CENTRE_NO
74, 79, 91, 73, 71, 61, 3, 43, 85, 63

### ASSUMPTIONs and LIMITATION
- The dataset size is smaller for an assertive recommendation

- The outliers and skewed data are managed by bounding IQR for outliers
The range for skewness considered is
     -0.5 to 0.5 - skewed
     -1 to 1 - moderate skewed
     < -1 and > 1 - highly skewed

- There were outliers in ANNUAL_REVENUE target variable, that represented a long 
disconnected tail. the rows where ANNUAL_REVENUE >2.5e6 are not considered for analysis.
Since there are no reference to validate if the values are genuine and are not due to any errors.

- The data is imputed with KNN imputer

- The data is scaled - standardized with standard scaler

======================================================================================================================

'''

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

np.random.seed(42)

# Load the data
current_centres_data = pd.read_excel('data/data.xlsx', sheet_name='Current Centres')
potential_centres = pd.read_excel("data/data.xlsx", sheet_name="Potential Centres")

# Explore the data
current_centres_data.info()

# dropping the duplicates there were 7 duplicate records
current_centres_data = current_centres_data.drop_duplicates()

# checking if there are any null value in the data
current_centres_data.isnull().sum()
# there are 12 NaN values in the AREA_EV_PERC and AREA_POPULATION_DENSITY_PPSKM

# imputing the data for missing values
imputer = KNNImputer(n_neighbors=2)

columns_to_impute = ["AREA_POPULATION_DENSITY_PPSKM", "AREA_EV_PERC"]
df_imputed = pd.DataFrame(imputer.fit_transform(current_centres_data[columns_to_impute]), 
                          columns=columns_to_impute)
current_centres_data[columns_to_impute] = df_imputed

# Data encoding for the categorical column
encoding_map = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6
}
current_centres_data["AREA_AFFLUENCE_GRADE"] = current_centres_data["AREA_AFFLUENCE_GRADE"].map(encoding_map)

# Check for data skew
current_centres_data.skew()
# ANNUAL_REVENUE and AREA_POPULATION_DENSITY_PPSKM

# check for ANNUAL_REVENUE
sns.histplot(current_centres_data["ANNUAL_REVENUE"])
plt.show()
# It is observed that value > 2.5e6 is in the disconnected tail, 

# checking the values greater than value > 2.5e6
print(current_centres_data[current_centres_data["ANNUAL_REVENUE"] > 2.5e6].shape)
current_centres_data = current_centres_data[current_centres_data["ANNUAL_REVENUE"] < 2.5e6]

# checking the skewness again
print(current_centres_data.skew())

# subset of list of columns
columns_to_treat_outliers = current_centres_data.columns[1:-1]
columns_to_treat_outliers


# Outlier treatment by using the IQR method
def remove_outliers(column):
    """
    A function to bound the outliers 

    :param column: pd.series columns
    :type column: pd.Series

    :return: pd.Series
    """
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    column[column < lower_bound] = lower_bound
    column[column > upper_bound] = upper_bound
    return column

new_df = current_centres_data.copy()
new_df[columns_to_treat_outliers] = new_df[columns_to_treat_outliers].apply(remove_outliers, axis=0)
print(new_df.skew())


# Standardize the data for numerical stability and faster convergence
scaler = StandardScaler()
x_data = scaler.fit_transform(new_df[columns_to_treat_outliers])
x_data = pd.DataFrame(x_data, columns=columns_to_treat_outliers)
y_data = new_df["ANNUAL_REVENUE"]
print(x_data.shape, y_data.shape)


# splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


# Model selection
# the models are selected from set of regressors
# KFOLD splits are implemented to avoid lucky splits of the dataset
models_list = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "Elastic Net": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "SVM": SVR(),
    "KNN": KNeighborsRegressor()
} # all are set to default hyperparameters

def model_selection(models, x, y):
    results = {}
    for model_name, model in models.items():
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_results = cross_val_score(model, x, y, cv=kfold, scoring='neg_mean_squared_error')
        results[model_name] = np.median(np.sqrt(np.abs(cv_results))) # medianRMSE
    return results

results = model_selection(models_list, x_data, y_data)
print(sorted(results.items(), key=lambda x: x[1]))
"""
[('Gradient Boosting', 194624.08827421523),
 ('Ridge', 204838.91855575616),
 ('Lasso', 205013.40182966558),
 ('Linear Regression', 205014.7091973662),
 ('Random Forest', 209077.20803576816),
 ('Elastic Net', 218992.2128394802),
 ('KNN', 219450.86848698917),
 ('Decision Tree', 303107.38739159843),
 ('SVM', 393127.0218861089)]

"""

# GradientBoostingRegressor performs the best.
# Thus, it is tuned for better hyperparameters.
# Kfold splitting is also performed to ensure and avoid lucky splits.
# GridSearchCV is implemented for searching the hyperparameters 

def model_tuning(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    gbr = GradientBoostingRegressor()

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Set up K-Fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Set up the grid search with K-Fold cross-validation
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, 
                            scoring='neg_mean_absolute_error', 
                            cv=kf, n_jobs=-1, verbose=1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best Cross-Validation Score (MAE):", -best_score)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    results = {
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "model": gbr
    }
    return results

results = model_tuning(x_test, y_test)
print(results)
"""
{'RMSE': 216743.61547136924,
 'MAE': 166102.10034086058,
 'R2': 0.6985734504076805,
 'model': GradientBoostingRegressor()}
"""

# Feature selection is performed to understand if
# the subset of feature set could be used for model prediction
# thus reducing the dimensions and simplifying the model
def model_feature_selection(x_data, y_data): 
    model = GradientBoostingRegressor(
        n_estimators=100, 
        random_state=42,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=2,
        min_samples_split=5
    ) #hyper parameters from previous tuning
    model.fit(x_data, y_data)
    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': range(x_data.shape[1]),
        'Importance': importances
    })

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)

    # only top 5 features are considered for exploration purposes
    top_n = 5
    top_features = feature_importance_df.head(top_n)['Feature'].values

    # Create a new dataset with only the selected features
    subset_columns = x_data.columns[top_features]
    return subset_columns.values

subset_x_data = model_feature_selection(x_data, y_data)
subset_x_data
"""
array(['TOTAL_STAFF', 'SERVICE_BAYS', 'ANNUAL_RENT',
       'HOURS_OPEN_PER_WEEK', 'MOT_BAYS'], dtype=object)
"""
# the model is created with the subset of the features but the evaluation metrics
# reports that the model performs poor incomparison to all the features involved.

### FINAL model training

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, shuffle=True, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=100, 
    random_state=42,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=2,
    min_samples_split=5
) #hyper parameters from previous tuning
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

results = {
    "RMSE": root_mean_squared_error(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred),
}
print(results)

#### Potential Centres Prediction
potential_centres["AREA_AFFLUENCE_GRADE"] = potential_centres["AREA_AFFLUENCE_GRADE"].map(encoding_map)
columns_to_impute = ["AREA_POPULATION_DENSITY_PPSKM", "AREA_EV_PERC"]
df_imputed = pd.DataFrame(
    imputer.fit_transform(potential_centres[columns_to_impute]), 
                          columns=columns_to_impute)
potential_centres[columns_to_impute] = df_imputed
potential_centres_scaled = scaler.transform(potential_centres[columns_to_treat_outliers])
potential_centres_scaled = pd.DataFrame(
    potential_centres_scaled, columns=columns_to_treat_outliers)

predictions = model.predict(potential_centres_scaled)
potential_centres["predictions"] = predictions
print(potential_centres.sort_values("predictions", ascending=False).head(10))

### Overlay plot of the predictions vs actual
sns.histplot(x="predictions", 
             data=potential_centres, color="blue", alpha=0.5, label="Potential", kde=True)
sns.histplot(x="ANNUAL_REVENUE", 
             data=current_centres_data, color="pink", alpha=0.5, label="Current", kde=True)
plt.xlabel("Revenue")
plt.title("Overlay plot of the potential and current")
plt.legend(title="Datasets")
plt.show()
# The plot shows that the predictions are similar to the actual ANNUAL_REVENUE distribution
# There is possiblity the model could be underfitting, but that could be a result 
# of smaller dataset size.

