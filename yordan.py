# %%
import numpy as np
import pandas as pd

import requests
from datetime import datetime


# %%
# Function to print a message with log
def log_message(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time} - {message}")

log_message("BEGIN PYTHON FORECAST PROGRAM FOR SPAREPARTS")

# %%
log_message('BEGIN Retrieving API')

# Get Data From API
params = {
    "start-date": "2023-01-01",
    "end-date": "2024-12-01",
    "exclude-older": "2024-01-01",
    "branch": "",
    "agency": "",
    "partno": ""
}

url = "http://172.16.5.6:8080/v1/web/test9"
response = requests.get(url, params=params)
data = response.json()
df = pd.DataFrame(data['data'])

log_message(str(data['data-count']) + " Data retrived from API")

# display(df)


# %%
log_message("BEGIN Constructing All Branch Data and combine it to df")

df_all = df.groupby(['agency', 'partno'], as_index=False)['d'].apply(lambda x: np.sum(np.array(x.tolist()), axis=0))
df_all.insert(0, 'branch', 'ALL')

df = pd.concat([df, df_all], ignore_index=True)

log_message("all branch data constructed and merged with df")


# %%
def calc_sma(x):
    ser = pd.Series(x)
    sma = ser.rolling(window=4).mean()
    return sma[-1:]

log_message("Begin Forecast Calculation")

df['fd']   = df['d'].apply(calc_sma)
df['std']  = df['d'].apply(lambda x: np.std(x))
df['mean'] = df['d'].apply(lambda x: np.mean(x))
df['ub']   = df['d'].apply(lambda x: np.max(x))

log_message("Forcast Calculation Completed")


# %%
log_message("BEGIN Constructing Final Data and send it back to API")

url = "http://172.16.5.6:8080/v1/web/test9-post"

result = df.drop('d', axis=1)
result_json = result.to_dict(orient='records')

log_message("Start Sending " + str(len(result)) + " Row To API")

response = requests.post(url, json=result_json)

log_message(f"Send API Complete")
log_message(f"Status Code: {response.status_code}")
if(response.status_code == 200):
    log_message(f"Response Body: {response.text}")
else:
    log_message(f"Send Failed")

# display(result)

# %%
# display(result)


