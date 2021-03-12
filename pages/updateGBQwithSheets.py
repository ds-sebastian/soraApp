#%%

import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from pandas.io.pytables import DataCol
from google.cloud import bigquery
import gspread
import numpy as np
import os
import datetime

#%%
#Google Sheets Client
scope = ['https://www.googleapis.com/auth/spreadsheets', "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('G:/DataScience/Sora ML Project/soraApp/pages/sora_GSclient.json', scope)
GSclient = gspread.authorize(creds)

#Google Big Query Client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='G:/DataScience/Sora ML Project/soraApp/pages/sora_BQClient.json'
GBQclient = bigquery.Client(project = 'sorapredict')
table_id = 'soraData.soraTable'

#%%
#Clear Current GCQ
clear_GCQ = """
    TRUNCATE TABLE soraData.soraTable
"""
GBQclient.query(clear_GCQ)


#%%
## Prepare Sheets Data
#Pull Data From Sora Log
sheet = GSclient.open("SoraLog").sheet1
df = pd.DataFrame(sheet.get_all_records())

#### Data cleaning
# Intial Data frame
df['Timestamp'] = pd.to_datetime(df['Timestamp']) 
time = df['Time override (optional)'] = df['Time override (optional)'].replace(r'^\s*$', df['Timestamp'].dt.strftime("%I:%M:%S %p"), regex=True)
date = df['Date override (optional)'] = df['Date override (optional)'].replace(r'^\s*$', df['Timestamp'].dt.strftime('%m/%d/%Y'), regex=True)

#%%
# Cleaned Data Frame
data = pd.DataFrame()
data['Timestamp'] = pd.to_datetime(date + ' ' + time)
data['Activity'] = df['Activity']
data['Description'] = df['Description (optional)']
data['Location'] = df['Location (optional)']
data['SoraRank'] = df['Is Sora being a good boy?']
data['client_ip'] = '127.0.0.1'

print(data.head())
# %%
def uploadToBigQuery(df):
    # Since string columns use the "object" dtype, pass in a (partial) schema
    # to ensure the correct BigQuery data type.
    job_config = bigquery.LoadJobConfig(schema=[
        bigquery.SchemaField("Timestamp", "TIMESTAMP"),
        bigquery.SchemaField("Activity", "STRING"),
        bigquery.SchemaField("Description", "STRING"),
        bigquery.SchemaField("Location", "STRING"),
        bigquery.SchemaField("SoraRank", "INTEGER"),
        bigquery.SchemaField("client_ip", "STRING"),
    ])
    job = GBQclient.load_table_from_dataframe(
        df, table_id, job_config=job_config
    )
    # Wait for the load job to complete.
    print(job.result())
    
uploadToBigQuery(data)
