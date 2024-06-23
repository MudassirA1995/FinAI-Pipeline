#Test File 

# Import specific functions from finance_analytics.py
from finance_analytics import split_dataset, data_preprocess, best_model_selector , fetch_stock_data

import pandas as pd 
import numpy as np 
import yfinance as yf 
import sklearn

#Libraries for the higher models 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


import yfinance as yf
import pandas as pd

#Fetching the stock data using the function fetch_stock_data

fetch_stock_data("AAPL","2020-01-01","2023-01-01")

#Storing the values of the fetched_stock_data in a variable 
df = fetch_stock_data("AAPL","2020-01-01","2023-01-01")

#Viewing the info about the created dataframe 

df.info()

#Using the split_data function to split the stock data , split_data(dataset,target column, test size)

# Call the function to split the dataset
split_data  = split_dataset(df,"Close",0.2)

X_train, X_test, y_train, y_test = split_data

#Using the best model selector function 

testing_models = ("LinearRegression","RidgeRegression","DecisionTree")
evaluation_method = "r2"

best_model_selector(X_train, X_test, y_train, y_test, testing_models, evaluation_method)


final_model = best_model_selector(X_train, X_test, y_train, y_test, testing_models, evaluation_method)

final_model.predict(X_test)



