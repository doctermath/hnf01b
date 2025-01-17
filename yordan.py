# %%

import numpy as np
import pandas as pd

import time
import sys
import requests

from datetime import datetime

# Set Display Width Longer
pd.options.display.max_colwidth = 100 # 100 for long width


# %%
# Function to print a message with log
def log_message(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time} - {message}")

log_message("BEGIN PYTHON FORECAST PROGRAM FOR SPAREPARTS")

# %%
log_message('BEGIN Retrieving API')

max_retries=8
delay=2

params = {
    "start-date": "2024-01-01",
    "end-date": "2024-12-01",
    "exclude-older": "2024-01-01",
    "branch": "",
    "agency": "",
    "partno": ""
}

url = "http://172.16.5.6:8080/v1/web/test9"
    
for attempt in range(1, max_retries + 1):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if 'data' in data and 'data-count' in data:
            log_message(str(data['data-count']) + " Data retrived from API")
            df = pd.DataFrame(data['data'])
            break
        else:
            log_message("Error: Unexpected API response format")
            break
    except requests.RequestException as e:
        log_message(f"Attempt {attempt}: API request failed - {e}")
        if attempt < max_retries:
            time.sleep(delay * (2 ** (attempt - 1)))  # Exponential backoff
        else:
            log_message("Max retries reached. Exiting.")
            sys.exit(1)

    
# display(df)


# %%
# Contruct All Branch Data and Concat It To DF
log_message("BEGIN Constructing All Branch Data and Combine It to DF")

df_all = df.groupby(['agency', 'partno'], as_index=False)['d'].apply(lambda x: np.sum(np.array(x.tolist()), axis=0))
df_all.insert(0, 'branch', 'ALL')
df = pd.concat([df, df_all], ignore_index=True)

log_message(f"All Branch Data Constructed And Merged With DF With Total Data {len(df)}")

# %%
# Calculate Forecast
log_message("BEGIN Forcast Calculation")

def calc_sma(x):
    ser = pd.Series(x)
    sma = ser.rolling(window=4).mean()
    return sma[-1:]

df['fd']   = df['d'].apply(calc_sma)
df['std']  = df['d'].apply(lambda x: np.std(x))
df['mean'] = df['d'].apply(lambda x: np.mean(x))
df['ub']   = df['d'].apply(lambda x: np.max(x))

log_message("Forcast Calculation Completed")


# %%
# Send Data Back To API
log_message("BEGIN Constructing Final Data and send it back to API")

url = "http://172.16.5.6:8080/v1/web/test9-post"

result = df.drop('d', axis=1)
result_json = result.to_dict(orient='records')

log_message("Start Sending " + str(len(result)) + " Row To API")

for attempt in range(1, max_retries + 1):
    try:
        response = requests.post(url, json=result_json)
        response.raise_for_status() 
        log_message("Send API Complete")
        log_message(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            log_message(f"Response Body: {response.text}")
        else:
            log_message("Send Failed")

        break
    except requests.RequestException as e:
        log_message(f"Attempt {attempt}: API request failed - {e}")
        if attempt < max_retries:
            time.sleep(delay * (2 ** (attempt - 1)))  # Exponential backoff
        else:
            log_message("Max retries reached. Exiting.")
            sys.exit(1)  # Stop execution after max retries



