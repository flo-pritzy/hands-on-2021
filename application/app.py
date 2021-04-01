import base64
import io
import yaml

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from PIL import Image

from constants import CLASSES

#load parameters from the yaml file
with open('app.yaml') as yaml_data:
    parameters = yaml.safe_load(yaml_data)

IMAGE_WIDTH = parameters[0]['IMAGE_WIDTH']
IMAGE_HEIGHT = parameters[1]['IMAGE_HEIGHT']
MODEL_PATH = parameters[2]['MODEL_PATH']


#Load dnn model
classifier = tf.keras.models.load_model(MODEL_PATH)


def classify_image(image, model, image_box=None):
    """Classify image by model
    
    Parameters
    ----------
    content: image content
    model: tf/keras classifier
    
    Returns
    -------
    class id returned by model classifier
    """
    images_list = []
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box) # box argument clips image to (x1, y1, x2, y2)
    image = np.array(image)
    images_list.append(image)
    
    return np.argmax(model.predict(np.array(images_list)), axis = 1)

def classify_image_for_proba(image, model, image_box=None):
    """Classify image by model
    
    Parameters
    ----------
    content: image content
    model: tf/keras classifier
    
    Returns
    -------
    probabilities per class
    """
    images_list = []
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box) # box argument clips image to (x1, y1, x2, y2)
    image = np.array(image)
    images_list.append(image)
    
    return [model.predict(np.array(images_list))]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])


pre_style = {
    'whiteSpace': 'pre-wrap',
    'wordBreak': 'break-all',
    'whiteSpace': 'normal'
}

# Define application layout
app.layout = html.Div([
    html.H1('Traffic Signs Recognition', style={'textAlign': 'center'}),
    dbc.Row(dbc.Col(html.Div(dbc.Alert("This application tends to recognize traffic signs on images", color="secondary", style={'textAlign': 'center'})))),
    html.H5('Using a pretrained Deep Neural Network, we have created  an interactive application to determine which traffic sign you upload in the space down below. After uploading you image you will find the prediction and the related probabilities'),
    dcc.Upload(
        id='bouton-chargement',
        children=html.Div([
            'Click-Drop or',
            dbc.Button('select an image', outline=True, color='primary'),
        ]),
        style={
            'width': '50%',
            'height': '80px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '50px',
            'align-items': 'center' 
        },
    ),
    html.Div(id='mon-image'),
    html.Div(id='ma-zone-resultat'),
    html.Div([
        html.H5('Do you think it is the right traffic sign?'),
        dbc.Button('Yes', id='btn-nclicks-1', outline=True, color = 'success', n_clicks = 0),
        dbc.Button('No', id='btn-nclicks-2', outline=True, color = 'danger', n_clicks = 0),
        html.Div(id='container-button-timestamp'),
    ]),
])

@app.callback(Output('mon-image', 'children'),
              [Input('bouton-chargement', 'contents')])

def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        if 'image' in content_type:
            image = Image.open(io.BytesIO(base64.b64decode(content_string)))
            predicted_class = classify_image(image, classifier)[0]
            predicted_proba = classify_image_for_proba(image, classifier)[0]
            classes_nb = list(CLASSES)
            values = CLASSES.values()
            classes_values = list(values)
            
            df = pd.DataFrame({"Class_nb": classes_nb,
                               "Class": classes_values,
                               "Probabilities": np.reshape(predicted_proba, (43,))})
            fig = px.bar(df, x="Class", y="Probabilities")
            return html.Div([
                html.Hr(),
                html.Img(src=contents),
                html.Div([
                    dbc.Row([
                        dbc.Col(html.Div(dbc.Alert("PREDICTED CLASS", color="primary"))),
                    ])
                ]),
                html.H3('Predicted Class : {}'.format(CLASSES[predicted_class])),
                html.Hr(),
                html.Div([
                    dbc.Row([
                        dbc.Col(html.Div(dbc.Alert("ASSOCIATED PROBABILITIES", color="primary")))
                    ])
                ]),
                dcc.Graph(figure=fig),
                #html.Div('Raw Content'),
                #html.Pre(contents, style=pre_style)
            ])
        else:
            try:
                # Décodage de l'image transmise en base 64 (cas des fichiers ppm)
                # fichier base 64 --> image PIL
                image = Image.open(io.BytesIO(base64.b64decode(content_string)))
                # image PIL --> conversion PNG --> buffer mémoire 
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                # buffer mémoire --> image base 64
                buffer.seek(0)
                img_bytes = buffer.read()
                content_string = base64.b64encode(img_bytes).decode('ascii')
                # Appel du modèle de classification
                predicted_class = classify_image(image, classifier)[0]
                predicted_proba = classify_image_for_proba(image, classifier)[0]
                classes_nb = list(CLASSES)
                values = CLASSES.values()
                classes_values = list(values)
                df = pd.DataFrame({"Class_nb": classes_nb,
                                   "Class": classes_values,
                                   "Probabilities": np.reshape(predicted_proba, (43,))})
                fig = px.bar(df, x="Class", y="Probabilities")
                # Affichage de l'image
                return html.Div([
                    html.Hr(),
                    html.Img(src='data:image/png;base64,' + content_string, style = {'margin':'50px'}),
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Div(dbc.Alert("PREDICTED CLASS", color="primary"))),
                        ])
                    ]),
                    html.H3('Predicted Class : {}'.format(CLASSES[predicted_class])),
                    html.Hr(),
                    html.Div([
                        dbc.Row([
                            dbc.Col(html.Div(dbc.Alert("ASSOCIATED PROBABILITIES", color="primary")))
                        ])]),
                    dcc.Graph(figure=fig),
                ])
            except:
                return html.Div([
                    html.Hr(),
                    html.Div(dbc.Alert('Uniquement des images svp : {}'.format(content_type), color="danger")),
                    html.Hr(),                
                    html.Div('Raw Content'),
                    html.Pre(contents, style=pre_style),
                ])

@app.callback(Output('container-button-timestamp', 'children'),
              Input('btn-nclicks-1', 'n_clicks'),
              Input('btn-nclicks-2', 'n_clicks'))

def displayClick(btn1, btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        msg = "Great ! Let's try again !"
    elif 'btn-nclicks-2' in changed_id:
        msg = 'Oh no... Give me another chance ok? Try again !'
    else:
        msg = 'Give an answer please'
    return html.Div(msg)

# Start the application
if __name__ == '__main__':
    app.run_server(debug=True)