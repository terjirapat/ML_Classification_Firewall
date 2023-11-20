import base64
import os
from flask import Flask, send_from_directory
from urllib.parse import quote as urlquote

import dash
from dash.dependencies import Input, Output
from dash import dcc, html, callback

import pandas as pd

UPLOAD_DIRECTORY = "./Github/ThNumber_img_classification/uploaded"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

dash.register_page("home",  path='/Home',

layout = html.Div([

        html.Div(children=[
            html.H4(children='Import your DATA (.csv)'),

            html.Div(children='your DATA must be .csv file (if not, please make it)'),
            html.Hr(),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Ul(id='output-data-upload'),
]),
])
)

##################################################################################

def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    name='df_00.csv'
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "./download/{}".format(urlquote(filename))
    return html.A(filename, href=location)

@callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'filename'),
              Input('upload-data', 'contents')
              )
def update_output(uploaded_filenames, uploaded_file_contents):
    """Save uploaded files and regenerate the file list."""

    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            save_file(name, data)

    files = uploaded_files()
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        return [html.Li(file_download_link(filename)) for filename in files]
