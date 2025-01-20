# %%
import numpy as np
import pandas as pd

import time
import sys
import requests
import logging

from datetime import datetime

# %%
# Set Display Width Longer
pd.options.display.max_colwidth = 100  # 100 for long width

# Set Logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.FileHandler("logs/forecast.log"), logging.StreamHandler()],
)
logging.info("="*20)
logging.info("BEGIN PYTHON FORECAST PROGRAM FOR SPAREPARTS")

# %%
logging.info('BEGIN Retrieving API')

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
            logging.info(str(data['data-count']) + " Data retrived from API")
            df = pd.DataFrame(data['data'])
            break
        else:
            logging.info("Error: Unexpected API response format")
            break
    except requests.RequestException as e:
        logging.info(f"Attempt {attempt}: API request failed - {e}")
        if attempt < max_retries:
            time.sleep(delay * (2 ** (attempt - 1)))  # Exponential backoff
        else:
            logging.info("Max retries reached. Exiting.")
            sys.exit(1)

    
# display(df)


# %%
# Contruct All Branch Data and Concat It To DF
logging.info("BEGIN Constructing All Branch Data and Combine It to DF")

df_all = df.groupby(['agency', 'partno'], as_index=False)['d'].apply(lambda x: np.sum(np.array(x.tolist()), axis=0))
df_all.insert(0, 'branch', 'ALL')
df = pd.concat([df, df_all], ignore_index=True)

logging.info(f"All Branch Data Constructed And Merged With DF With Total Data {len(df)}")

# %%
# Calculate Forecast
logging.info("BEGIN Forcast Calculation")

def calc_sma(x):
    ser = pd.Series(x)
    sma = ser.rolling(window=4).mean()
    return sma[-1:]

df['fd']   = df['d'].apply(calc_sma)
df['std']  = df['d'].apply(lambda x: np.std(x))
df['mean'] = df['d'].apply(lambda x: np.mean(x))
df['ub']   = df['d'].apply(lambda x: np.max(x))

logging.info("Forcast Calculation Completed")


# %%
# Send Data Back To API
logging.info("BEGIN Constructing Final Data and send it back to API")

url = "http://172.16.5.6:8080/v1/web/test9-post"

result = df.drop('d', axis=1)
result_json = result.to_dict(orient='records')

logging.info("Start Sending " + str(len(result)) + " Row To API")

for attempt in range(1, max_retries + 1):
    try:
        response = requests.post(url, json=result_json)
        response.raise_for_status() 
        logging.info("Send API Complete")
        logging.info(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            logging.info(f"Response Body: {response.text}")
        else:
            logging.info("Send Failed")

        break
    except requests.RequestException as e:
        logging.info(f"Attempt {attempt}: API request failed - {e}")
        if attempt < max_retries:
            time.sleep(delay * (2 ** (attempt - 1)))  # Exponential backoff
        else:
            logging.info("Max retries reached. Exiting.")
            sys.exit(1)  # Stop execution after max retries



