#!/usr/bin/env python
# coding: utf-8

# In[24]:


#PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
import requests
import json
import os
import warnings
warnings.filterwarnings("ignore")


# In[25]:


# Define the API endpoint
api_url = "http://172.16.5.6:8080/v1/web/test12"

# Fetch data from the API
response = requests.get(api_url)

# Check if the request was successful
if response.status_code == 200:
    # Convert the JSON response to a Python dictionary
    data = response.json()
    
    # Create a pandas DataFrame from the data
    # Assuming the API response is a list of dictionaries
    df = pd.DataFrame(data['data'])
    
else:
    print(f"Failed to fetch data: {response.status_code}")
    
data = df


# In[26]:


#PARAMETER
#EWMA
alpha_ewma = 0.4

#SES & DES
alpha_ses = 0.65  # ubah nilai alpha (semakin besar semakin berat ke data terbaru)
beta_des = 0.45   # ubah nilai beta (semakin besar semakin cepat beradaptasi, kalo rendah bisa terjadi lag)


# In[27]:


# Get mean and standard deviation of 12 periods before the last one
data['mean_12'] = data['d'].apply(lambda x: np.mean(x[-13:-1]))  # Use 12 periods before the last one
data['std_12'] = data['d'].apply(lambda x: np.std(x[-13:-1]))    # Use 12 periods before the last one

# Get upper bound from mean and std
data['ub'] = data['mean_12'] + 1.5 * data['std_12']

# Limit the original data to upper bound (using the 12 periods before the last one)
data['clipped_d'] = data.apply(lambda row: np.clip(row['d'][-13:-1], 0, row['ub']).tolist(), axis=1)


# In[28]:


# Calculate Simple Moving Average
def calculate_ma(list):
    oldData = []
    newData = []
    for i in list:
        # store calculated data to old list
        oldData.append(i)
        newData.append(np.mean(oldData))
    return newData

data['ma'] = data['clipped_d'].apply(calculate_ma)
data['ma_result'] = data['ma'].apply(lambda x: x[-1:])
data['ma_result'] = data['clipped_d'].apply(lambda x: np.mean(x))


# In[38]:


import itertools
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

data['wma_clipped_d'] = data.apply(lambda row: np.clip(row['d'][-15:], 0, row['ub']).tolist(), axis=1)

def wma_forecast_with_weights(data, weights):
    wma_values = [None] * 3
    for i in range(3, len(data)):
        forecast = np.sum(np.array(data[i-3:i]) * weights) / sum(weights)
        wma_values.append(forecast)
    return wma_values

def generate_weights(step=0.05):
    weights = []
    for w1 in np.arange(0.01, 1, step):
        for w2 in np.arange(w1 + 0.01, 1 - w1, step):
            w3 = 1 - w1 - w2
            if w3 > w2 > w1 > 0 and abs(w1 + w2 + w3 - 1) < 1e-6:
                weights.append((w1, w2, w3))
    return weights

best_weights_list = []
best_maes = []

for row in data['wma_clipped_d']:
    best_mae = float('inf')
    best_weights = None
    for weights in generate_weights(step=0.05):
        wma_values = wma_forecast_with_weights(row, weights)
        mae = mean_absolute_error(row[-12:], wma_values[-12:])
        if mae < best_mae:
            best_mae = mae
            best_weights = weights
    best_weights_list.append(best_weights)
    best_maes.append(best_mae)

data['best_weights'] = best_weights_list
data['best_mae'] = best_maes

data['wma'], data['wma_result'] = zip(*data.apply(lambda row: (
    wma_forecast_with_weights(row['wma_clipped_d'], row['best_weights'])[3:][-12:],
    wma_forecast_with_weights(row['wma_clipped_d'], row['best_weights'])[-1:]
), axis=1))
print(data)


# In[30]:


# Calculate Exponential Weighted Moving Average (EWMA)
def ewma(list, alpha = alpha_ewma):
    df = pd.DataFrame(list)
    df['ewma'] = df.ewm(alpha=alpha_ewma, adjust=False).mean()
    return df['ewma'].tolist()

def ewma_forecast(list, alpha):
    ewma_values = ewma(list, alpha)
    if len(ewma_values) > 0:
        # Prediction for the next period
        next_forecast = (1 - alpha) * ewma_values[-1]
    else:
        next_forecast = None
    return ewma_values, next_forecast

data['ewma'], data['ewma_result'] = zip(*data['clipped_d'].apply(lambda x: ewma_forecast(x, alpha_ewma)))


# In[31]:


#LINEAR REGRESSION
#  Calculate Linear Regression
def lr(x):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    model =  LinearRegression()
    model.fit(df[['x']], df['y'])
    df.loc[len(df), 'x'] = len(df) + 1
    return model.predict(df[['x']])

data['lr'] = data['clipped_d'].apply(lambda x: lr(x))
data['lr_result'] = data['lr'].apply(lambda x: x[-1:])


# In[32]:


#POLYNOMIAL 2ND AND 3RD
# Calculate Polynomial Regression
def pr(x, pr_degree):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)

    X = df[['x']]  # Independent variable (reshape to 2D array)
    y = df['y']    # Dependent variable

    poly = PolynomialFeatures(degree=pr_degree)  # Create polynomial features
    X_poly = poly.fit_transform(X)  # Transform input features
    poly_model = LinearRegression()  # Initialize linear regression model
    poly_model.fit(X_poly, y)  # Fit polynomial model

    df.loc[len(df), 'x'] = len(df) + 1
    X_all_poly = poly.transform(df[['x']])
    return poly_model.predict(X_all_poly)  

data['pr2'] = data['clipped_d'].apply(lambda x: pr(x, 2))
data['pr2_result'] = data['pr2'].apply(lambda x: x[-1:])
data['pr3'] = data['clipped_d'].apply(lambda x: pr(x, 3))
data['pr3_result'] = data['pr3'].apply(lambda x: x[-1:])


# In[33]:


#SES
def ses(x, alpha = alpha_ses):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    df.loc[len(df), 'x'] = len(df) + 1

    new_data = SimpleExpSmoothing(df['y']).fit(smoothing_level=alpha, optimized=False).fittedvalues
    return new_data.tolist()

data['ses'] = data['clipped_d'].apply(lambda x: ses(x, alpha_ses))
data['ses_result'] = data['ses'].apply(lambda x: x[-1:])


# In[34]:


#DES
def des(x, alpha = alpha_ses, beta = beta_des):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    df.loc[len(df), 'x'] = len(df) + 1

    new_data = ExponentialSmoothing(df['y'], trend='add', seasonal=None).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False).fittedvalues
    return new_data.tolist()

data['des'] = data['clipped_d'].apply(lambda x: des(x,alpha_ses, beta_des))
data['des_result'] = data['des'].apply(lambda x: x[-1:])


# In[35]:


def metric(x):
    period_length = len(x['clipped_d'])
    df = pd.DataFrame()
    df['period'] = range(1, period_length + 1)
    df['qty'] = x['clipped_d'][:period_length]  # Ground truth values
    df['ma'] = x['ma'][:period_length]
    df['wma'] = x['wma'][:period_length]
    df['ewma'] = x['ewma'][:period_length]
    df['lr'] = x['lr'][:period_length]
    df['pr2'] = x['pr2'][:period_length]
    df['pr3'] = x['pr3'][:period_length]
    df['ses'] = x['ses'][:period_length]
    df['des'] = x['des'][:period_length]

    # Calculate metrics for each model
    result = []
    for model in df.columns[2:]:  # Loop through model columns (ma, ewma, etc.)
        rmse = np.sqrt(mean_squared_error(df['qty'], df[model]))  # Calculate RMSE
        r2 = r2_score(df['qty'], df[model])  # Calculate R²
        mae = mean_absolute_error(df['qty'], df[model])  # Calculate MAE
        result.append({'model': model, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    
    # Convert result to a DataFrame
    metrics_df = pd.DataFrame(result)
    
    # Select the best model (e.g., based on RMSE)
    best_model_row = metrics_df.loc[metrics_df['MAE'].idxmin()]  # Row with the lowest RMSE
    best_model = best_model_row['model']
    
    # Add the best model and metrics to the result
    return {'best_model': best_model, 'metrics': metrics_df.to_dict(orient='records')}

# Apply the metric function
data['metric'] = data.apply(lambda x: metric(x), axis=1)

# Extract the best model and metrics for each row
data['best_model'] = data['metric'].apply(lambda x: x['best_model'])
data['metrics'] = data['metric'].apply(lambda x: x['metrics'])


# In[36]:


data['mean_12_FD'] = data['d'].apply(lambda x: np.mean(x[-12:]))
data['std_12_FD'] = data['d'].apply(lambda x: np.std(x[-12:]))

data['ub_FD'] = data['mean_12_FD'] + 1.5 * data['std_12_FD']

data['clipped_d_FD'] = data.apply(lambda row: np.clip(row['d'][-12:], 0, row['ub_FD']).tolist(), axis=1)
def apply_best_model_forecast(row):
    best_model = row['best_model']
    
    data = row['D'][-15:] if best_model == 'wma' else row['d'][-12:]
    
    ub = row['ub_FD']
    clipped_data = np.clip(data, 0, ub).tolist()
    
    if best_model == 'ma':
        ma_values = calculate_ma(clipped_data)
        forecast = ma_values[-1]
    elif best_model == 'ewma':
        alpha = 0.4
        weights = np.array([(1 - alpha) ** i for i in range(len(clipped_data))][::-1])
        forecast = np.sum(weights * clipped_data) / np.sum(weights)
    elif best_model == 'wma':
        weights = [0.2, 0.3, 0.5]
        if len(clipped_data) >= len(weights):
            forecast = np.sum(np.array(clipped_data[-3:]) * weights)
        else:
            forecast = np.nan
    elif best_model == 'lr':
        X = np.arange(len(clipped_data)).reshape(-1, 1)
        y = np.array(clipped_data)
        coef = np.polyfit(X.flatten(), y, 1)
        forecast = coef[0] * len(clipped_data) + coef[1]
    elif best_model == 'pr2':
        X = np.arange(len(clipped_data)).reshape(-1, 1)
        y = np.array(clipped_data)
        coef = np.polyfit(X.flatten(), y, 2)
        forecast = coef[0] * (len(clipped_data) ** 2) + coef[1] * len(clipped_data) + coef[2]
    elif best_model == 'pr3':
        X = np.arange(len(clipped_data)).reshape(-1, 1)
        y = np.array(clipped_data)
        coef = np.polyfit(X.flatten(), y, 3)
        forecast = (
            coef[0] * (len(clipped_data) ** 3)
            + coef[1] * (len(clipped_data) ** 2)
            + coef[2] * len(clipped_data)
            + coef[3]
        )
    elif best_model == 'ses':
        model = SimpleExpSmoothing(clipped_data).fit(smoothing_level=0.65, optimized=False)
        forecast = model.forecast(1)[0]
    elif best_model == 'des':
        model = Holt(clipped_data).fit(smoothing_level=0.65, smoothing_slope=0.45, optimized=False)
        forecast = model.forecast(1)[0]
    else:
        forecast = np.nan
    
    return forecast

data['FD_forecast'] = data.apply(apply_best_model_forecast, axis=1)

data['FD_final'] = data['FD_forecast'].apply(np.ceil)
data['FD_final'] = data['FD_final'].apply(lambda x: max(0, np.ceil(x)))

display(data[['partno','best_model', 'FD_forecast', 'FD_final']])
display(data)

