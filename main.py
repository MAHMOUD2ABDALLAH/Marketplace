import pandas as pd
import numpy as np
import csv
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import float64, int64
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
import xgboost as xg
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# label encoding for categorical vars
market = pd.read_csv("G:\\output_data.csv")
encoding = LabelEncoder()
market['output_date'] = encoding.fit_transform(market['output_date'])
market['mkt_id'] = encoding.fit_transform(market['mkt_id'])

# data cleaning for the missed values
imputes = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(market)
x = market.iloc[:, :8]
y = market['output_own_price']
print(x.head())
print(y.head())

# feature selection from model
# select2 = SelectFromModel(xg.XGBRegressor())
# select2.fit_transform(x, y)
# print(select2.get_support())
#
# # splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33)

# ==============================MODELS=====================================
# 1st Linear regression model
# model = LinearRegression()
# model.fit(x_train, y_train)
# y_predict = model.predict(x_train)
# print('predict values:', y_predict)
# mse = mean_squared_error(y_train, y_predict, multioutput='uniform_average')
# print('mean square error:', np.sqrt(mse))
# ==========================================================================
# 2nd sklearn module for lasso regression
# model = Lasso(alpha=1.0)
# model.fit(x_train, y_train)
# y_predict = model.predict(x_train)
# print('predict values:', y_predict)
# mse = mean_squared_error(y_train, y_predict, multioutput='uniform_average')
# print('mean square error:', np.sqrt(mse))
# ===========================================================================
# 3rd sklearn module for  ridge regression
# model = Ridge(alpha=0.1)
# model.fit(x_train, y_train)
# y_predict = model.predict(x_train)
# print('predict values:', y_predict)
# mse = mean_squared_error(y_train, y_predict, multioutput='uniform_average')
# print('mean square error:', np.sqrt(mse))
# ============================================================================
# 4th sklearn module for elastic net regression
# model = ElasticNet(alpha=0.05)
# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)
# print('predict values:', y_predict)
# print('actual values:', y_test)
# mse = mean_squared_error(y_test, y_predict, multioutput='uniform_average')
# print('mean square error:', np.sqrt(mse))
# ==================================ENSEMBLE===========================================
# # 5th sklearn module for random forest model
model = RandomForestRegressor()
model.fit(x_train, y_train)
y_predict = model.predict(x_train)
print('predict values :', y_predict)
mse = mean_squared_error(y_train, y_predict, multioutput='uniform_average')
print('mean square error :', np.sqrt(mse))


# =====================================================================================
# 6th sklearn module for gredient boosting regressor model
#
# model = GradientBoostingRegressor(learning_rate=0.05)
# model.fit(x_train, y_train)
# y_predict = model.predict(x_train)
# print('predict values :', y_predict)
# mse = mean_squared_error(y_train, y_predict, multioutput='uniform_average')
# print('mean square error :', np.sqrt(mse))
# =====================================================================================
# 7th sklearn module for XG regressor model
# model = xg.XGBRegressor(objective='reg:linear')
# model.fit(x_train, y_train)
# y_predict = model.predict(x_train)
# print('predict values :', y_predict)
# mse = mean_squared_error(y_train, y_predict, multioutput='uniform_average')
# print('mean square error :', np.sqrt(mse))
# =====================================================================================
# 8th sklearn module for neural network regressor model
# model = MLPRegressor(random_state=33, max_iter=500)
# model.fit(x_train, y_train)
# y_predict = model.predict(x_train)
# print('predict values :', y_predict)
# mse = mean_squared_error(y_train, y_predict, multioutput='uniform_average')
# print('mean square error :', np.sqrt(mse))


# =====================================PLOTTING================================================
def plotGraph(y_train, y_pred_train, rand):
    if max(y_train) >= max(y_pred_train):
        my_range = int(max(y_train))
    else:
        my_range = int(max(y_pred_train))
    plt.scatter(range(len(y_train)), y_train, color='blue')
    plt.scatter(range(len(y_pred_train)), y_pred_train, color='red')
    plt.title(rand)
    plt.show()
    return


plotGraph(y_train, y_predict, 'Random Forest Regressor')
