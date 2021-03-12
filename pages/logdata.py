#%%
# Imports from 3rd party libraries
from logging import PlaceHolder
import dash
import dash_bootstrap_components as dbc
import dash_table as dct
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import datetime as dt
import pandas as pd
from google.cloud import bigquery
import os
from flask import request
#%%
from app import app, server

# Construct a BigQuery client object.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='G:\DataScience\Sora ML Project\soraApp\pages\sora_BQClient.json'
client = bigquery.Client(project = 'sorapredict')
table_id = 'soraData.soraTable'

#%%

#df = pd.DataFrame({'Timestamp': [dt.datetime.now()],
#        'Activity': ['Pee'],
#        'Description': ['N/A'],
#        'Location': ['N/A'],
#        'SoraRank': [3],
#        } )
#print(df)
#print(df.dtypes)
#
#convert_dict = {'Timestamp': 'datetime64[ns, US/Eastern]',
#                'Activity': str,
#                'Description': str,
#                'Location': str,
#                'SoraRank': 'int64',
#                } 
#
#test = df.astype(convert_dict) 

#%%
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
    job = client.load_table_from_dataframe(
        df, table_id, job_config=job_config
    )
    # Wait for the load job to complete.
    print(job.result())


#%%
# Imports from this application
from app import app

now = dt.datetime.now()

notify = dbc.Alert(
            "Submitted",
            id="alert-auto",
            is_open=False,
            duration=1000,
        )

interval = dcc.Interval(
            id='interval',
            interval=1*1000, # in milliseconds
            n_intervals=0
        )

# 1 column layout
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout

#Activity
activity = dbc.FormGroup(
    [
        dbc.Label("Activity", html_for="activity"),
        dcc.Dropdown(
            id="activity",
            options=[
                {'label': 'Sleep', 'value': 'Sleep'},
                {'label': 'Drink', 'value': 'Drink'},
                {'label': 'Eat', 'value': 'Eat'},
                {'label': 'Pee', 'value': 'Pee'},
                {'label': 'Poop', 'value': 'Poop'},
                {'label': 'Walk', 'value': 'Walk'},
                {'label': 'Play', 'value': 'Play'},
                {'label': 'Training', 'value': 'Training'},
                {'label': 'Treat', 'value': 'Treat'},
                {'label': 'Accident', 'value': 'Accident'},
                {'label': 'Crate', 'value': 'Crate'},
                {'label': 'Haircut', 'value': 'Haircut'},
                {'label': 'Bath', 'value': 'Bath'},
                {'label': 'Nail Trim', 'value': 'Nail Trim'},
                {'label': 'Hair Brush', 'value': 'Hair Brush'},
                {'label': 'Teeth Clean', 'value': 'Teeth Clean'},
                {'label': 'Vomit', 'value': 'Vomit'},
            ],
        ),
    ],
    row = False
)

#sora a good boy?
goodboy = dbc.FormGroup(
    [
        dbc.Label("Is Sora Being A Good Boy?", html_for="goodboy"),
        dcc.Slider(id="goodboy", min=1, max=5, step=1, value=3,marks={
        2: "<<<<< Daddy's Little Devil",
        4: "Momma's Little Angel >>>>>"
    }),
    ]
)

#date time override

timeoverride = dbc.Row(
    [
        dbc.Label("Date:   ", html_for="dateoverride"),
        dbc.Col(
            dbc.FormGroup(     
                    dcc.DatePickerSingle(
                        id='date',
                        initial_visible_month=dt.date.today(),
                        date=dt.date.today(),
                        placeholder = 'Date'
                    ),
                        
            ),
            #width=4,
        ),
        dbc.Label("Time:   ", html_for="timeoverride"),
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Input(
                        type="hour",
                        id="hour",
                        placeholder = now.hour,
                    ),
                ]
            ),
            #width=4,
        ),
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Input(
                        type="number",
                        id="minute",
                        placeholder= now.minute,
                    ),
                ]
            ),
            #width=4,
        ),

    ],
    form=True,
)
    


#location
location = dbc.FormGroup(
                [   dbc.Label("Location (optional):   ", html_for="location"),
                    dbc.Input(
                        type="text",
                        id="location",
                        value= 'N/A',
                    )
                ]
)


#description
description = dbc.FormGroup(
                [   dbc.Label("Description (optional):   ", html_for="description"),
                    dbc.Input(
                        type="text",
                        id="description",
                        value= 'N/A',
                    )
                ]
)


submit = dbc.Button("Submit", id = 'submit-val', color="primary", n_clicks = 0, href='/predictions')

form = dbc.Form([activity, goodboy, timeoverride, location, description])

column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            # Sora Activity Log
            
            Enter Sora's activity and click '*Submit*'. **(GOBAT FAMILY USE ONLY)**
            
            🐶 Blank activities won't be submitted.
            🐶 *Hour* and *Minute* must *both* be entered to override the time.


            """
        ),
        form,

    ],
)

#Keep PLaceholders updated
@app.callback(
    [Output('date', 'placeholder')],
    [Input(component_id='interval', component_property='n_intervals')]
)
def update_minute(input_value):
    now = dt.date.today()
    return [now]

@app.callback(
    [Output('hour', 'placeholder')],
    [Input(component_id='interval', component_property='n_intervals')]
)
def update_hour(input_value):
    now = dt.datetime.now().hour
    return [now]

@app.callback(
    [Output('minute', 'placeholder')],
    [Input(component_id='interval', component_property='n_intervals')]
)
def update_minute(input_value):
    now = dt.datetime.now().minute
    return [now]

#Final Submission
final = dct.DataTable(id='finalupload',
                 row_selectable=False,
                 editable=False,
                 columns=[
                 {'name': 'Timestamp', 'id': 'Timestamp'},
                 {'name': 'Activity', 'id': 'Activity'},
                 {'name': 'Description', 'id': 'Description'},
                 {'name': 'Location', 'id': 'Location'},
                 {'name': 'SoraRank', 'id': 'SoraRank'},
                 {'name': 'client_ip', 'id': 'client_ip'},
                 ],
                 )

@app.callback(
    Output(component_id='finalupload', component_property='data'),
    [Input(component_id='activity', component_property='value'),
     Input(component_id='goodboy', component_property='value'),
     Input(component_id='hour', component_property='value'),
     Input(component_id='minute', component_property='value'),
     Input(component_id='date', component_property='value'),
     Input(component_id='location', component_property='value'),
     Input(component_id='description', component_property='value'),
     ]
)
def update_final(activity, goodboy, hour, minute, date, location, description):
    if hour is not None and minute is not None:
        time = str(hour) + ':' + str(minute)
    else:
        time = dt.datetime.now().strftime('%H:%M')
    if date is not None:
        datefield = date
    else:
        datefield = str(dt.datetime.now().strftime('%m/%d/%Y'))

    datetime = dt.datetime.strptime(datefield + ' ' + time, '%m/%d/%Y %H:%M' )

    ip = request.remote_addr
    
    frame = pd.DataFrame(columns = ['Timestamp',
                        'Activity',
                        'Description',
                        'Location',
                        'SoraRank',
                        'client_ip'])
    frame.loc[0] = [datetime, activity, description, location, goodboy,ip]
    return frame.to_dict('records')


@app.callback(
    dash.dependencies.Output('alert-auto', 'is_open'),
    [dash.dependencies.Input('submit-val', 'n_clicks'),
     dash.dependencies.Input('finalupload', 'data')],
    [dash.dependencies.State("alert-auto", "is_open")])
def update_output(n, data, is_open):
    if n:
        test = pd.DataFrame(data)
        if test['Activity'].iloc[0] is not None:
            convert_dict = {'Timestamp': 'datetime64[ns]',
                            'Activity': str,
                            'Description': str,
                            'Location': str,
                            'SoraRank': 'int64',
                            'client_ip': str,
                            } 

            df = test.astype(convert_dict)
            print('Uploading....')
            uploadToBigQuery(df)
        return not is_open
    return is_open

layout = html.Div([notify, interval, dbc.Row([column1]), final, submit])



