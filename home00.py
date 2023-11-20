from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    use_pages=True,
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

app.layout = dbc.Container([
	dbc.Row([
		html.H2(children='Machine Learning Classifier Application',
	        style = {'textAlign': 'center',}),
		html.Br(),
		html.Br(),
		
		dbc.Col([ 
			dcc.Link('Home ( Import DATA )', href='/Home'),
        ], width = 4 ,
	      style={
		      'width': '31%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '20px',
            'textAlign': 'center',
            'margin': '10px'
        }
	  ),
    dbc.Col([ 
	    dcc.Link('TRAIN', href='/Training' ),
        ], width = 4 ,
	      style={
		      'width': '31%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '20px',
            'textAlign': 'center',
            'margin': '10px'
        }
	),
    dbc.Col([
	    dcc.Link('TEST & PREDICT', href='/Testing'),
        ], width = 4 ,
	      style={
		      'width': '31%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '20px',
            'textAlign': 'center',
            'margin': '10px'
        }
	)
    ],
    style = {'textAlign': 'center',}
    ),

    html.Hr(),
    dbc.Row([
	html.Div(dash.page_container),
    ]),
    html.Hr(),
	html.Div(children=[
            html.Div(children='Start your journey with Home page'),
     ], style = {'height':'200px','textAlign': 'center',})

# End of container

])

if __name__ == '__main__':
	app.run_server(debug=True)