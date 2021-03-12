#%%
import pandas as pd
from google.cloud import bigquery
import numpy as np
import os
import datetime as dt
from numpy.lib.shape_base import column_stack
from sklearn.model_selection import train_test_split
import os

#%%
#Google Big Query Client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='G:/DataScience/Sora ML Project/soraApp/pages/sora_BQClient.json'
GBQclient = bigquery.Client(project = 'sorapredict')

#%%
#Clear Current GCQ
#clear_GCQ = """
#    TRUNCATE TABLE soraData.soraFeatures
#"""
#GBQclient.query(clear_GCQ)


#%%

#%%
def getAllRecords():
    sql = """
    SELECT Timestamp, Activity, SoraRank
    FROM soraData.soraTable
    """
    # Run a Standard SQL query using the environment's default project
    df = GBQclient.query(sql).to_dataframe()
    return df


def FeaturePrep():
    #Pull Data
    data = getAllRecords()

    #Current Record
    now = dt.datetime.now() #Current Time
    d = {'Timestamp': [now], 'Activity': ['Current'], 'SoraRank': [3]}
    current = pd.DataFrame(data=d)
    data = pd.concat([data,current], ignore_index=True,axis=0)
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], utc=True)

    data['DOW'] = data['Timestamp'].dt.strftime("%A")
    data = data.sort_values('Timestamp')
    data = data.reset_index(drop=True)

    data['Activity'] = data['Activity'].replace('Accident', 'Pee', regex=True)
    data['Activity'] = data['Activity'].replace('Vomit', 'Poop', regex=True)
    data['Activity'] = data['Activity'].replace('Training', 'Play', regex=True)
    data['Activity'] = data['Activity'].replace('Walk', 'Play', regex=True)
    data['Activity'] = data['Activity'].replace('Crate', 'Sleep', regex=True)
    data['Activity'] = data['Activity'].replace('Haircut', 'Grooming', regex=True)
    data['Activity'] = data['Activity'].replace('Bath', 'Grooming', regex=True)
    data['Activity'] = data['Activity'].replace('Treat', 'Eat', regex=True)
    data['Activity'] = data['Activity'].replace('Hair Brush', 'Grooming', regex=True)
    data['Activity'] = data['Activity'].replace('Play', 'Exercise', regex=True)
    data['Activity'] = data['Activity'].replace('Nail Trim', 'Grooming', regex=True)
    activities = data['Activity'].unique().tolist()
    data['elapsedTime'] = data['Timestamp'].diff().dt.seconds/3600
    #
    df = data[['Timestamp','Activity']]
    out = (
        df
        .groupby(['Timestamp', 'Activity']) # set the columns as index
        .size() # aggregate by row count
        .unstack(fill_value=0) # move 'Variable' index level to columns
        .sort_index()
    )
    #Rolling Counts of Previous Activities
    counts = out.rolling('3H', closed= "left").sum()
    counts = counts.replace(np.nan, 0, regex=True)

    #%%
    # Calculate Time Differences
    timediffs = out.copy()
    timediffs['Time_Stamp'] = timediffs.index.to_series(keep_tz=True)
    for i in activities:
        timediffs['Prev_Timestamp'] = timediffs[timediffs[i] == 1].Time_Stamp.shift(1) #include shift??
        timediffs['Prev_Timestamp'].fillna(method='ffill', inplace=True)
        timediffs['TimeDiff_'+i] = (timediffs.Time_Stamp - timediffs.Prev_Timestamp) / np.timedelta64(1, 's') / 3600
        timediffs.drop(['Prev_Timestamp'], axis=1, inplace=True)
        timediffs.drop([i], axis=1, inplace=True)
    timediffs.drop(['Time_Stamp'], axis=1, inplace=True)
    timediffs.drop(['TimeDiff_Current'], axis=1, inplace=True)
    # Merge Features
    data = pd.merge(data, counts, left_on='Timestamp', right_on=counts.index)
    data = pd.merge(data, timediffs, left_on='Timestamp', right_on=timediffs.index)
    data = data.replace(np.nan, 0, regex=True)
    data['hour'] = data['Timestamp'].dt.hour + data['Timestamp'].dt.minute/60
    data = data.set_index('Timestamp')
    data['SoraRank'] = data['SoraRank'].rolling('3H', closed= "left").mean() 
    data['SoraRank'] = data['SoraRank'].replace(np.nan, 3, regex=True)
    data = data.reset_index(drop=True)

    data = pd.get_dummies(data, columns=['DOW'])
    current = data[data['Activity'] == 'Current'] #.values
    data = data[(data['Activity'] != 'Current') & (data['Activity'] != 'Grooming')]

    return(data, current)

#_, current = FeaturePrep()

# %%
def uploadToBigQuery(current):
    # Since string columns use the "object" dtype, pass in a (partial) schema
    # to ensure the correct BigQuery data type.
    job_config = bigquery.LoadJobConfig(schema=[
        bigquery.SchemaField("Activity", "STRING"),
    ])
    job = GBQclient.load_table_from_dataframe(
        current, 'soraData.soraPredict', job_config=job_config
    )
    # Wait for the load job to complete.
    print(job.result())
    
uploadToBigQuery(current)

#%%
def get_predictions():
    sql = """
    SELECT predicted_Activity_probs FROM ML.PREDICT(MODEL soraData.automlModel,(
      SELECT SoraRank
      , elapsedTime
      , Drink
      , Eat
      , Exercise
      , Grooming
      , Pee
      , Poop
      , Sleep
      , TimeDiff_Pee
      , TimeDiff_Exercise
      , TimeDiff_Poop
      , TimeDiff_Sleep
      , TimeDiff_Drink
      , TimeDiff_Eat
      , TimeDiff_Grooming
      , hour
      , DOW_Friday
      , DOW_Monday
      , DOW_Saturday
      , DOW_Sunday
      , DOW_Thursday
      , DOW_Tuesday
      , DOW_Wednesday
    FROM soraData.soraPredict
    WHERE Activity = 'Current'
    LIMIT 1
    ))
    """
    # Run a Standard SQL query using the environment's default project
    pred = GBQclient.query(sql).to_dataframe().explode('predicted_Activity_probs')
    pred[['Activity','Prob']] = pd.DataFrame(pred.predicted_Activity_probs.tolist(), index= pred.index)
    pred.drop(['predicted_Activity_probs'], axis=1, inplace=True)
    
    return pred


#%%

test, _ = FeaturePrep()
test.to_csv('dataExport.csv')
# %%
