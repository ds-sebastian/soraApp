#%%
# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_table as dct
import dash_html_components as html
from dash.dependencies import Input, Output
from google.cloud import bigquery
import os
import pandas as pd
import plotly
import numpy as np
import plotly.graph_objs as go
# Imports from this application
from app import app

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='G:\DataScience\Sora ML Project\soraApp\pages\sora_BQClient.json'
client = bigquery.Client(project = 'sorapredict')
table_id = 'soraData.soraTable'

#%%

sql_topn = """
    SELECT Timestamp, Activity, Description, Location, SoraRank
    FROM soraData.soraTable
    ORDER BY Timestamp DESC
    LIMIT 20
"""

sql_soraRank = """
    SELECT Timestamp, SoraRank
    FROM soraData.soraTable
    ORDER BY Timestamp ASC
"""


# get 100 records from BigQuery
def getRecords(sql):
    # Run a Standard SQL query using the environment's default project
    df = client.query(sql).to_dataframe()
    return df



#%%

table = dct.DataTable(id='finaltable',
                 row_selectable=False,
                 editable=False,
                 columns=[
                 {'name': 'Timestamp', 'id': 'Timestamp'},
                 {'name': 'Activity', 'id': 'Activity'},
                 {'name': 'Description', 'id': 'Description'},
                 {'name': 'Location', 'id': 'Location'},
                 {'name': 'SoraRank', 'id': 'SoraRank'},
                 ],
                 data = [],
                 style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
                },
                 style_cell={'textAlign': 'center'},
                 style_table={
                     'height': 300,
                     'overflowY': 'auto',
                     #'width': 400
                 },
                 style_as_list_view=True
                 )

interval = dcc.Interval(
            id='interval',
            interval=1*3000, # in milliseconds
            n_intervals=0
        )


graph = dcc.Graph(id='soraLine',
                  animate = True)


# 1 column layout
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [   interval,
     
        dcc.Markdown(
            """
        
            ## Historical Data Table:


            """
        ),
        table,
        graph
    ],
)

@app.callback(
    [Output('finaltable', 'data')],
    [Input(component_id='interval', component_property='n_intervals')]
)
def update_table(input_value):
    tablevals = getRecords(sql_topn)
    tablevals['Timestamp'] = tablevals['Timestamp'].dt.strftime("%m/%d/%Y %I:%M %p")
    return [tablevals.to_dict('records')]


@app.callback(Output('soraLine', 'figure'),
              [Input('interval', 'n_intervals')])
def update_graph_scatter(input_data):
    df = getRecords(sql_soraRank).set_index('Timestamp').sort_index()
    df['SoraRank'] = df['SoraRank'].rolling('24H', closed= "left").mean().replace(np.nan, 3, regex=True)
    Y = df['SoraRank']
    X = df.index
    data = plotly.graph_objs.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode= 'lines+markers'
            )
    return {'data': [data],'layout' : go.Layout(font = {"size":15},
                                                xaxis={"showline":False,"tickangle":-90, "visible":True, 'tickformat':"%b %d\n%Y"},
                                                yaxis=dict(range=[1,5]),
                                                title = "Sora's GoodBoy Level")}




layout = dbc.Row([column1])