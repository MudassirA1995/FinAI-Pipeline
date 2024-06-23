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

# my_stock_package/data_fetcher.py

import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches stock data for the given ticker symbol between the specified start and end dates.
    
    Args:
        ticker (str): The ticker symbol of the stock.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: A DataFrame containing the stock data.
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError("No data found for the given ticker and date range.")
        return stock_data
    except Exception as e:
        print(f"An error occurred while fetching the data: {e}")
        return pd.DataFrame()
    
    

# my_stock_package/data_splitter.py

import pandas as pd
from sklearn.model_selection import train_test_split




import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(df: pd.DataFrame, target_column: str, test_size: float = 0.2):
    """
    Splits the dataset into training and testing sets based on the target column.

    Args:
        df (pd.DataFrame): The input dataframe.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        tuple: A tuple containing the split datasets (X_train, X_test, y_train, y_test).
    """
    try:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataframe.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Ensure the function returns exactly four values
        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"An error occurred while splitting the dataset: {e}")
        return None





    
    
#Creating a function to preprocess the data before applying the models 
import pandas as pd
from sklearn.impute import SimpleImputer


def data_preprocess(X_train, X_test, y_train, y_test):
    """
    Preprocesses the data to make it compatible for model training.

    - Removes null values by replacing them with the mean of the respective columns.
    - Encodes datetime columns.
    
    Args:
        X_train (pd.DataFrame): The training input features.
        X_test (pd.DataFrame): The testing input features.
        y_train (pd.Series): The training target.
        y_test (pd.Series): The testing target.
    
    Returns:
        tuple: The preprocessed X_train, X_test, y_train, and y_test.
    """
    # Function to encode datetime columns
    def encode_datetime(df):
        for col in df.select_dtypes(include=['datetime64[ns]']).columns:
            df[col + '_year'] = df[col].dt.year
            df[col + '_month'] = df[col].dt.month
            df[col + '_day'] = df[col].dt.day
            df.drop(columns=[col], inplace=True)
        return df

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')

    # Preprocess X_train and X_test
    X_train = encode_datetime(X_train)
    X_test = encode_datetime(X_test)

    X_train[:] = imputer.fit_transform(X_train)
    X_test[:] = imputer.transform(X_test)

    # Impute missing values in y_train and y_test if any
    if y_train.isnull().sum() > 0:
        y_train[:] = y_train.fillna(y_train.mean())
    if y_test.isnull().sum() > 0:
        y_test[:] = y_test.fillna(y_test.mean())

    return X_train, X_test, y_train, y_test


    
#Creating a function that takes the x_train , x_test , y_train ,y_test as an input along with the testing 
#Also the model training parameters are selected by the user 

def best_model_selector(X_train, X_test, y_train, y_test, testing_models, evaluation_method):
    """
    Selects the best model based on the evaluation method and plots the performance of all models.
    """
    if len(testing_models) < 2:
        raise ValueError("At least two models must be selected.")

    available_models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(),
        "RidgeRegression": Ridge(),
        "RandomForest": RandomForestRegressor()
    }

    selected_models = {name: available_models[name] for name in testing_models}

    if evaluation_method not in ["r2", "mean_square", "root_mean_square"]:
        raise ValueError("Invalid evaluation method. Choose from 'r2', 'mean_square', 'root_mean_square'.")

    evaluation_functions = {
        "r2": r2_score,
        "mean_square": mean_squared_error,
        "root_mean_square": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
    }

    eval_func = evaluation_functions[evaluation_method]

    model_performance = {}

    for name, model in selected_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        performance = eval_func(y_test, y_pred)
        model_performance[name] = performance

    if evaluation_method == "r2":
        best_model_name = max(model_performance, key=model_performance.get)
    else:
        best_model_name = min(model_performance, key=model_performance.get)

    best_final_model = selected_models[best_model_name]

    print(f"The best model is: {best_model_name} with a {evaluation_method} score of {model_performance[best_model_name]:.4f}")

    # Plotting the performance of all models
    plt.figure(figsize=(10, 5))
    plt.bar(model_performance.keys(), model_performance.values(), color='skyblue')
    plt.xlabel('Model')
    plt.ylabel(f'{evaluation_method.capitalize()} Score')
    plt.title('Model Performance')
    plt.show()

    return best_final_model

