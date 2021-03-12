#%%

# Imports from 3rd party libraries
from google.cloud import bigquery
import dash
from dash.dependencies import Output,Input
import dash_table as dct
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
#from collections import deque
import numpy as np
import datetime as dt
from numpy.lib.shape_base import column_stack
import pandas as pd
#from oauth2client.service_account import ServiceAccountCredentials
#from pandas.io.pytables import DataCol
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler 
#from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
#rom matplotlib.animation import FuncAnimation
#import matplotlib.pyplot as plt
#from matplotlib import style
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import os
#
from app import app, server

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='G:\DataScience\Sora ML Project\soraApp\pages\sora_BQClient.json'
client = bigquery.Client(project = 'sorapredict')
table_id = 'soraData.soraTable'

sql = """
    SELECT Timestamp, Activity, SoraRank
    FROM soraData.soraTable
"""
#%%
def getAllRecords():
    # Run a Standard SQL query using the environment's default project
    df = client.query(sql).to_dataframe()
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
    current = data[data['Activity'] == 'Current'].iloc[:, 1:19] #.values
    data = data[(data['Activity'] != 'Current') & (data['Activity'] != 'Grooming')]

    return(data, current)

data, _ = FeaturePrep()

colors = pd.DataFrame({'Activity':['Poop', 'Pee', 'Exercise', 'Drink', 'Sleep', 'Eat', 'Current']
          ,'RGB':['rgb(218,247,166)','rgb(255,195,0)','rgb(255,87,51)','rgb(199,0,57)','rgb(144,12,63)', 'rgb(233,150,122)','rgb(20,30,10)']})


#%%
#OLD Machine learning
X = data.drop(data.tail(5).index).iloc[:, 1:19].values
y = data.drop(data.tail(5).index).iloc[:, 0].values

#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#
sm = RandomOverSampler() 
X_train, y_train = sm.fit_sample(X_train, y_train.ravel()) 

#
clf=RandomForestClassifier(n_estimators=1000,
                   #max_depth=2,
                   #gamma=2,
                   #eta=0.8,
                   #reg_alpha=0.5,
                   #reg_lambda=0.5
                   )

eval_set = [(X_train, y_train), (X_test, y_test)]

clf.fit(X_train, y_train
        #, eval_metric="mlogloss", eval_set=eval_set, early_stopping_rounds=10
        )
y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)
y_probs = clf.predict_proba(X_test)

kfold = StratifiedKFold(n_splits=10)
cv_results = cross_val_score(clf, X_train, y_train, cv=kfold)

def GetCurrentProb():
    _, current = FeaturePrep()
    current_prob = pd.DataFrame(clf.predict_proba(current), columns=clf.classes_)
    test = pd.merge(current_prob.T, colors, left_on=current_prob.T.index, right_on='Activity')
    test= test.sort_values(by=0,ascending=False)
    return(test)


#%%

##GBQ AUtoML
#def uploadToBigQuery(current):
#    sql = """
#    TRUNCATE TABLE soraData.soraPredict
#    """
#    client.query(sql)
#    # Since string columns use the "object" dtype, pass in a (partial) schema
#    # to ensure the correct BigQuery data type.
#    job_config = bigquery.LoadJobConfig(schema=[
#        bigquery.SchemaField("Activity", "STRING"),
#    ])
#    job = client.load_table_from_dataframe(
#        current, 'soraData.soraPredict', job_config=job_config
#    )
#    # Wait for the load job to complete.
#    print(job.result())
#
#
#def get_predictions():
#    sql = """
#    SELECT predicted_Activity_probs FROM ML.PREDICT(MODEL soraData.automlModel,(
#      SELECT SoraRank
#      , elapsedTime
#      , Drink
#      , Eat
#      , Exercise
#      , Grooming
#      , Pee
#      , Poop
#      , Sleep
#      , TimeDiff_Pee
#      , TimeDiff_Exercise
#      , TimeDiff_Poop
#      , TimeDiff_Sleep
#      , TimeDiff_Drink
#      , TimeDiff_Eat
#      , TimeDiff_Grooming
#      , hour
#      , DOW_Friday
#      , DOW_Monday
#      , DOW_Saturday
#      , DOW_Sunday
#      , DOW_Thursday
#      , DOW_Tuesday
#      , DOW_Wednesday
#    FROM soraData.soraPredict
#    WHERE Activity = 'Current'
#    LIMIT 1
#    ))
#    """
#    # Run a Standard SQL query using the environment's default project
#    pred = client.query(sql).to_dataframe().explode('predicted_Activity_probs')
#    pred[['Activity','Prob']] = pd.DataFrame(pred.predicted_Activity_probs.tolist(), index= pred.index)
#    pred.drop(['predicted_Activity_probs'], axis=1, inplace=True)
#    
#    return pred



#%%
interval = dcc.Interval(
            id='graph-update',
            interval=1*3000, # in milliseconds
            n_intervals=0
        )

bargraph = dcc.Graph(id='predictiongraph')


# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        interval,
        dcc.Markdown(
            """
        
            ## PAP Predictions

            Percentages are probabilities of each activity *(3 seconds to update)*
            
            Current Model: *Random Forest*

            """
        ),
        bargraph
    ],
    #md=4,
)

#column2 = dbc.Col(
#    [
#
#    ]
#)

@app.callback(Output('predictiongraph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_graph(input_data):
    #uploadToBigQuery(current)
    #s=get_predictions()
    s= GetCurrentProb()
    #print(s)
    data = plotly.graph_objs.Bar(
        x=list(s['Activity']),
        y=list(s[0]),
        text = list(s['Activity']),
        hoverinfo = "none",
        textposition = "outside",
        texttemplate = "%{x}<br>%{y:.0%}",
        cliponaxis = False,
        marker_color = list(s['RGB'])
    )
    return {'data': [data], 'layout' : go.Layout(
             font = {"size":15},
             #height = 700,
             xaxis = {"showline":False,"tickangle":-90, "visible":False},
             yaxis = {"showline":False, "visible":False},
             )}
             #title = 'Puppy Activity Predictor - PAP')}

layout = dbc.Row([column1])