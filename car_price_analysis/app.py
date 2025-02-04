import os
import pandas as pd
import numpy as np
from datetime import datetime

pd.set_option('display.max_columns', None)

import joblib

# importing dash to create dashboard
import dash

# Core components to plot graphs etc;
from dash import dcc

# html components
from dash import html

# Input, Output and State functions for the dashboard application
from dash.dependencies import Input, Output, State

import base64

import re

df = pd.read_csv("assests/carscom_scrape_clean.csv").drop(['Unnamed: 0'], axis=1)

cars_df = df.copy()
cars_df = cars_df[['make', 'model', 'year', 'mileage', 'ratings', 'price']]
cars_df['price'] = [re.findall(r'\d+', i)[0] for i in cars_df['price']]
cars_df['price'] = cars_df['price'].astype('float')

# remove duplicates
cars_df.drop_duplicates(inplace=True)

cars_df['age'] = datetime.now().year - cars_df['year']

cars_df = pd.get_dummies(data = cars_df, drop_first= True)

X = cars_df.drop(['year', 'price'], axis = 1)

# load the model from disk
filename = 'assests/xgb_model.pkl' 
loaded_model = joblib.load(open(filename, 'rb'))

MAK = list(df['make'].unique())

YR = []
for l in range(2000, 2019):
    YR.append(l)   

ML = []
for m in range(0, 10000000):
    M = round(m*0.01, 2)
    ML.append(M)

RT = []
for n in range(1, 51):
    R = round(n*0.1, 2)
    RT.append(R)

MIL = []
for m in range(1, 1000):
    M = round(m*100, 2)
    MIL.append(M)



def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(454)

    # set the numerical input as they are
    enc_input[0] = data['mileage']
    enc_input[1] = data['ratings']
    enc_input[2] = datetime.now().year - data['year']

    ##################### Make #########################
    # get the array of make categories
    Make = df.make.unique()
    # redefine the the user inout to match the column name
    redefinded_user_input = 'make_' + data['make']
    # search for the index in columns name list 
    Make_column_index = X.columns.tolist().index(redefinded_user_input)
    
    # fullfill the found index with 1
    enc_input[Make_column_index] = 1
    ##################### Model ####################
    # get the array of Model
    Model = df.model.unique()
    # redefine the the user inout to match the column name
    redefinded_user_input = 'model_'+ data['model']
    # search for the index in columns name list 
    Model_column_index = X.columns.tolist().index(redefinded_user_input)
    # fullfill the found index with 1
    enc_input[Model_column_index] = 1
    return enc_input

def Price(P):
    a = input_to_one_hot(P)
    price_pred = loaded_model.predict([a])
    return(price_pred[0])


# Creating an dashboard application
app = dash.Dash(__name__)
server = app.server

b_image_filename = 'assests/car_images/Honda_Accord.png'
b_encoded_image = base64.b64encode(open(b_image_filename, 'rb').read())

app.layout = html.Div([
    
    # To give Title
    html.Div([
        
        html.Div([
            html.H1('Car Prices'),
        ])
    ], style = {'textAlign': 'center'}),
    
    html.Hr(),
    
    html.Div(style = {'background-image': 'url(https://www.shutterfly.com/ideas/wp-content/uploads/2016/08/50-happy-birthday-quotes-thumb.jpg)'}),
    
    html.Div([
        
        # To display car name
        html.Div(id = 'Car_name',
        style = {'color': 'Black'}),
        
        
        # To display car image
        html.Img(id = 'car_image')
    ], style = {'fontsize': 500, 'textAlign': 'center'}),
    
    html.Hr(),
    
    #html component for Make
    html.Div([
        html.H3('Select Make'),
        
        # Dropdown menu to select Make
        dcc.Dropdown(
            id = 'Make',
            options = [
                {'label': i, 'value': i} for i in sorted(MAK)
            ],
            value = 'Honda'
        )
    ], style={'display':'inline-block', 'verticalAlign':'top', 'width':'45%', 'textAlign':'center'}),
    
    
    #html component for Model
    html.Div([
        html.H3('Select Model'),
        
        # Dropdown menu to select Model
        dcc.Dropdown(
            id = 'Model',
        )
    ], style={'display':'inline-block', 'verticalAlign':'top', 'width':'45%', 'textAlign':'center'}),
    
    
    #html component for Year
    html.Div([
        html.H3('Select Year'),
        
        # Dropdown menu to select Year
        dcc.Dropdown(
            id = 'Year',
            options = [
                {'label': i, 'value': i} for i in YR
            ],
            value = 2010
        )
    ], style={'display':'inline-block', 'verticalAlign':'top', 'width':'45%', 'textAlign':'center'}),
    
    #html component for Milage
    html.Div([
        html.H3('Select Milage'),
        
        # Dropdown menu to select Milage
        dcc.Dropdown(
            id = 'Mileage',
            options = [
                {'label': i, 'value': i} for i in MIL
            ],
            value = 85000
        )
    ], style={'display':'inline-block', 'verticalAlign':'top', 'width':'45%', 'textAlign':'center'}),
    
    #html component for Ratings
    html.Div([
        html.H3('Select Ratings'),
        
        # Dropdown menu to select Ratings
        dcc.Dropdown(
            id = 'Ratings',
            options = [
                {'label': i, 'value': i} for i in RT
            ],
            value = 4.3
        )
    ], style={'display':'inline-block', 'verticalAlign':'top', 'width':'45%', 'textAlign':'center'}),
    
    
    
    # html division to give action button
    html.Div([
        
        html.H3('Click Submit to View Results'),
        
        html.Button(
            id='submit-button',
            n_clicks=0,
            children = 'Submit',
            style={'fontSize':18}
        ),
    ], style = {'display':'inline-block', 'width':'45%', 'textAlign':'center'}),
    
    
    # To seperate different html components
    html.Hr(),
    
    # Car Price
    html.Div([
        html.Div(id = 'Car_Price',
        style = {'color': 'darkmagenta'}),
    ], style = {'fontsize': 30, 'textAlign':'center'})
       
])



# To update the options in dropdown menu of Select model
@app.callback(
    Output('Model', 'options'),
    [Input('Make', 'value')]
)
def update_dropdown(value):

    MODEL = list(df[df['make']== value]['model'].unique())
    
    model_options = [{'label': i, 'value': i} for i in sorted(MODEL)]
    
    return model_options

# To update the selected values in the dropdown menu of Select model
@app.callback(
    Output('Model', 'value'),
    [Input('Model', 'options')])

def set_dropdown_value(available_options):
    
    Value = available_options[0]['value']
    
    return Value


# To display predicted price of selected car
@app.callback(
    
    Output('Car_Price', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('Model', 'value'),
     State('Mileage', 'value'),
     State('Year', 'value'),
     State('Ratings', 'value'),
     State('Make', 'value')]
)

# updates the predicted price of selected car

def car_pred(n_clicks, model, mileage, year, ratings, make):
    
    if model == None:
        model = 'Accord'

            
    user_input = {'mileage': mileage , 'ratings': ratings, 'year': year, 'make': make, 'model': model}
    
    price = Price(user_input)
    
    return ("Predicted price is {}".format(price))


# To display image of selected car
@app.callback(
    
    Output('Car_name', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('Model', 'value'),
     State('Make', 'value')]
)

def car_name(n_clicks, model, make):
    
    if model == None:
        model = 'Accord'
        
    Car_name = make + " " +model
    
    return ("{}".format(Car_name))

# To display image of selected car
@app.callback(
    
    Output('car_image', 'src'),
    [Input('submit-button', 'n_clicks')],
    [State('Model', 'value'),
     State('Make', 'value')]
)

def car_image(n_clicks, model, make):
    
    if model == None:
        model = 'Accord'
        
    image_filename = "assests/car_images/" + make + "_" + model + ".png"    

    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    
    src ='data:image/png;base64,{}'.format(encoded_image.decode())
    
    return src

if __name__ == '__main__':
    app.run_server()