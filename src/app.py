#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
import plotly
import plotly.express as px
import numpy as np
from collections import Counter
import geopandas as gpd #To read our data 
import folium #for interactive maps
from folium import Circle, Marker #to select the maptype we want to use
from folium.plugins import HeatMap, MarkerCluster #for plugins


# In[2]:


# pickle_in = open("water_fullDataFrame_4M.pkl", "rb")
# water = pickle.load(pickle_in)
# water = water.sample(n = 500000, random_state = 0)
# water.to_pickle("water_fullDataFrame_500K.pkl")


# In[3]:


import plotly.io as pio
pio.renderers.default = 'notebook'


# In[4]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_leaflet as dl

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc


# In[5]:


# pickle_in = open("water_fullDataFrame_500K.pkl", "rb")
# water = pickle.load(pickle_in)


# In[6]:


water = pd.read_csv("https://github.com/UpasanaDutta98/my-render-app/blob/main/src/water_fullDataFrame_500K.pkl")


# In[7]:


import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import pickle

# Preprocessing for Yearly Analysis
water_timeseries = water[['county_name', 'parameter', 'sample_date', 'result']]
water_timeseries['sample_date'] = water_timeseries['sample_date'].astype(str)
water_timeseries['year'] = water_timeseries['sample_date'].apply(lambda x: x.split("/")[2].split(" ")[0])
water_timeseries['result'] = pd.to_numeric(water_timeseries['result'], errors='coerce')
water_timeseries.dropna(subset=['result'], inplace=True)
water_timeseries.reset_index(inplace=True, drop=True)

# Using Plotly's predefined 'Plasma' color scale
color_continuous_scale=px.colors.sequential.Plasma

# Defining a custom color scale
custom_color_scale = [
    [0.0, "blue"],    # Start with blue
    [0.2, "cyan"],    # Transition to cyan
    [0.4, "lime"],    # Then lime
    [0.6, "yellow"],  # Yellow is used for middle values
    [0.8, "orange"],  # Transitioning to orange
    [1.0, "red"]      # Ending with red for the highest values
]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

parameters = water['parameter'].unique()
county_options = [{'label': county, 'value': county} for county in water_timeseries['county_name'].unique()]

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Comprehensive Water Quality Analysis Dashboard", className="text-center mb-4"))),

    # Tabs including a new one for Yearly Analysis
    dbc.Tabs([
        dbc.Tab(label='County Analysis', tab_id='tab-bar'),
        dbc.Tab(label='Station Analysis', tab_id='tab-pie'),
        dbc.Tab(label='Geo Analysis', tab_id='tab-geo'),
        dbc.Tab(label='Yearly Analysis', tab_id='tab-year'),
    ], id='tabs', active_tab='tab-bar'),

    html.Div(id='tab-content', className='p-4')
])

@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab')]
)
def render_tab_content(active_tab):
    if active_tab == 'tab-bar':
        # County Analysis Content
        return dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='parameter-dropdown',
                                     options=[{'label': i, 'value': i} for i in parameters],
                                     value=parameters[0],
                                     className="mb-4"),
                        width={"size": 6, "offset": 3}),
            ]),
            dbc.Row(dbc.Col(dcc.Graph(id='parameter-graph'), width=12)),
        ])
    elif active_tab == 'tab-pie':
        # Station Analysis Content
        return dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='parameter-selector',
                                     options=[{'label': param, 'value': param} for param in parameters],
                                     value=parameters[0],
                                     className="mb-4"),
                        width={"size": 6, "offset": 3}),
            ]),
            dbc.Row(dbc.Col(dcc.Graph(id='water-station-hardness-pie-chart'), width=12))
        ])
    elif active_tab == 'tab-geo':
        # Geo Analysis Content
        return dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='geo-parameter-selector',
                                     options=[{'label': i, 'value': i} for i in parameters],
                                     value=parameters[0],
                                     className="mb-4"),
                        width={"size": 6, "offset": 3}),
            ]),
            dbc.Row(dbc.Col(dcc.Graph(id='geo-scatter-plot'), width=12))
        ])
    elif active_tab == 'tab-year':
        # Yearly Analysis Content
        return dbc.Container(fluid=True, children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select County:", className="lead"),
                    dcc.Dropdown(
                        id='county-dropdown',
                        options=county_options,
                        value=county_options[0]['value'],
                    ),
                ], md=6),

                dbc.Col([
                    html.Label("Type Parameter:", className="lead"),
                    dcc.Input(id='parameter-input', type='text', value='pH', debounce=True),
                ], md=6)
            ], justify="center", className="mb-4"),

            dbc.Row([
                dbc.Col(dcc.Graph(id='yearly-trend-plot'), width=12),
            ], className="mt-4"),
        ])

# Callbacks for each tab's content
# Add all callback definitions here...

@app.callback(
    Output('parameter-graph', 'figure'),
    [Input('parameter-dropdown', 'value')]
)
def update_bar_graph(selected_parameter):
    filtered_df = water[water['parameter'] == selected_parameter]
    filtered_df['result'] = pd.to_numeric(filtered_df['result'], errors='coerce')
    filtered_df.dropna(subset=['result'], inplace=True)

    county_df = filtered_df.groupby('county_name', as_index=False)['result'].mean().sort_values('result', ascending=False)

    fig = px.bar(county_df, x='county_name', y='result',
                 labels={'county_name': 'County', 'result': f'Average {selected_parameter}'})
    fig.update_layout(
        font=dict(size=14),
        yaxis_title=f'Average {selected_parameter}',
        title=f'Average {selected_parameter} by County',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black'
    )

    return fig

@app.callback(
    Output('water-station-hardness-pie-chart', 'figure'),
    [Input('parameter-selector', 'value')]
)
def update_pie_chart(selected_param):
    filtered_data = water[water['parameter'] == selected_param]['station_type'].value_counts()
    fig = px.pie(
        names=filtered_data.index,
        values=filtered_data.values,
        title=f'Types of water stations in samples measured for {selected_param}',
        hole=.3
    )
    fig.update_traces(
        hoverinfo='label+percent', textinfo='percent', textfont_size=20,
        marker=dict(colors=px.colors.sequential.RdBu)
    )
    fig.update_layout(font=dict(size=20), plot_bgcolor='white', paper_bgcolor='white', font_color='black')

    return fig


@app.callback(
    Output('geo-scatter-plot', 'figure'),
    [Input('geo-parameter-selector', 'value')]
)
def update_geo_scatter(selected_parameter):
    df_filtered = water[water['parameter'] == selected_parameter]
    
    numeric_mask = pd.to_numeric(df_filtered['result'], errors='coerce').notnull()
    df_filtered = df_filtered[numeric_mask].reset_index(drop=True)
    df_filtered['result'] = pd.to_numeric(df_filtered['result'])

    df_filtered = df_filtered.sort_values(['station_id', 'sample_date']).drop_duplicates('station_id', keep='last')

    fig = px.scatter_geo(df_filtered,
                         lat='latitude',
                         lon='longitude',
                         hover_name='county_name',
                         projection="albers usa",
                         color='result',
                         color_continuous_scale=custom_color_scale)
    
    fig.update_traces(marker=dict(size=2))

    fig.update_layout(
        geo=dict(
            landcolor='white',
            lakecolor='lightblue',
            showlakes=True,
            showocean=True, oceancolor='azure',
            countrycolor='lightgrey',
            showcountries=True,
            showsubunits=True, subunitcolor="grey"
        )
    )

    return fig


@app.callback(
    Output('yearly-trend-plot', 'figure'),
    [Input('county-dropdown', 'value'),
     Input('parameter-input', 'value')]
)
def update_graph(selected_county, typed_parameter):
    # Filtering DataFrame based on selected county and typed-in parameter
    df_filtered = water_timeseries[(water_timeseries['county_name'] == selected_county) & 
                                   (water_timeseries['parameter'].str.lower() == typed_parameter.lower())]

    # Aggregating to compute the average result per year
    yearly_average = df_filtered.groupby('year').agg({'result': 'mean'}).reset_index()

    # Creating a line plot
    fig = px.line(yearly_average, x='year', y='result',
                  title=f'Yearly Average of {typed_parameter} in {selected_county}',
                  markers=False,  # Add markers for each data point
                  template="presentation"  # Use presentation template for a fancy look
                 )
    
    # Enhance aesthetics
    fig.update_layout(plot_bgcolor='floralwhite', paper_bgcolor='mintcream', 
                      xaxis_title='Year', yaxis_title='Average Result',
                      font=dict(family="Arial, sans-serif", size=15, color="RebeccaPurple"))
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lavender')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lavender')

    return fig

# Add other callbacks for updating graphs/plots for different analyses here...

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




