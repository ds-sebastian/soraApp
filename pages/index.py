# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# Imports from this application
from app import app

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Sora

            Born: June 23, 2020

            Breed: Maltese/Yorkie Mix

            """
        ),
        dcc.Link(dbc.Button('Predictions', color='primary'), href='/predictions'),
        
        dcc.Link(dbc.Button('History', color='secondary'), href='/history'),
        
        dcc.Link(dbc.Button('Log New Data', color='warning'), href='/logdata')
    ],
    md=4,
)

image = html.Div(
        children=html.Img(
            src="https://lh3.googleusercontent.com/ND1xSoNOAx-srKDQ5xxNlSFZ9MGwYahTgwGn_qRB8gFloMvtFo6CmNKaiWN0GyXiIYWv-Hg78NhY7uD8VFt4j0AQlcDmtiPLrTRMR8bHTnOJiUKBSoogHJ5xa3-B4P4AMZozWYdYggID5bq8J9d3tayhzGdVb-jGdTueNu6XIOmOJa4AQhgkvxe7QljXtb-HZi0pGTXc6kkAjZxl3I1RfEX0UUWn_AvP6eLq7l7V0kfdG0c1csXzghNjwfYuCAJYG4MSwq0f1nzKEyIbx1rGseRzVTMCVgdkxZeqXTirx5EiEBOhKSGnBICOFmN6QjhCQ4jF79e7BF-yhZX7opucz11pljG2eg9XcX8Er-TDVf5OzJ09YouklUTbNFiWRypIbH0Gyku9-RcBJa6EXJ6XLAaa9nomAMpASg7EVcCTjaGdUR2BuALaUlPqUHK9bhbEeWzhLQYOpNnNX-za6GRkKF6m3jYi0I5D7TvaiPCBEO-y-aHxCusKjKZba9MxnjNXsbClZQQ_Zct6pEoFvmSEyqKISLfnMAN7RAH2iY2-YRFzEgCJ3EaxsXEe9eLo2OGRSKKGONIld5WqkEhYWUbQgoRsfGHUoEv0cSBc3PrdV-9ps6CwSoMIkheU1u7IPuk9Ro10IiXa1UFLRQUlIccZ0Nywas6Sgm70DJpl_IeFyypfljsXD_msr8vj1SzxBw=w696-h928-no?authuser=0",
            style={
                'maxWidth': '70%',
                'maxHeight': '70%',
                'marginLeft': 'auto',
                'marginRight': 'auto'
            }
        ),
)


column2 = dbc.Col(
    [
        image,
    ]
)

layout = dbc.Row([column1, column2])