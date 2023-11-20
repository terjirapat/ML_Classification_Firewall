import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from collections import OrderedDict
import dash_daq as daq

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import dash_bootstrap_components as dbc

import datetime
import cv2
import base64

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, recall_score, precision_score,accuracy_score, auc
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

try :
    df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")
    df = df.iloc[: , 1:] # Drop first column of dataframe
    df = pd.DataFrame(
        OrderedDict([(name, col_data) for (name, col_data) in df.items()])
    )
    op = [{'label':x, 'value':x} for x in df.columns]
except Exception as e :
    df = None
    op = None

dash.register_page("Test",  path='/Testing',

layout = dbc.Container([
    dbc.Row([
        html.H4(children='Select your testing'),
        html.Div(children='select Y column & choose trainning number'),
        html.Hr(),
        ], style={'height':'100px','text-align':'center',}),

    dbc.Row([
        dbc.Col([
        dbc.Row([

    dcc.Store(id='store-target2', storage_type='local'),
    dcc.Store(id='store-split2', storage_type='local'),

html.P("Select Model ( For Testing )", className="control_label"),
    dcc.Dropdown(
        id="select_test",
        options=['LogistcRegression', 'RandomForestClassifier', 'ExtraTreesClassifier','SGDClassifier','LGBMClassifier','KNeighborsClassifier'],
        multi=False,
        value=None,
        clearable=True,    
),
html.Hr(),

],style={
            'height':'150px',
            'text-align':'center',
               }
),

dbc.Row([
html.P("Select Target (Y column)", className="control_label"),
    dcc.Dropdown(
        id="select_target2",
        options = op ,
        multi=False,
        value=None,
        clearable=True,       
)
], style={'height':'150px','text-align':'center',}),

dbc.Row([
html.Div(children=f'You have select \"{None}\" to be the target column',id='output-target'),
html.Hr(),
html.P(children="Please select the amount of training sample (%)"),
daq.Slider(id='slider2',
    min=0,
    max=100,
    value=100,
    handleLabel={"showCurrentValue": True,"label": "VALUE"},
    step=10 

),
html.Div(children='Please select taget model and training split first',id='output-slider'),
], style={'height':'400px','text-align':'center',})

    ], width=3 ),

dbc.Col([
html.Div([
    daq.LEDDisplay(
        id='precision',
        label="Precision",
        value=0,
        style={
                        'height':'100px',
                        'margin-left':'10px',
                        'text-align':'center',
                        'width':'32%',
                        'display':'inline-block'
               }
    ),daq.LEDDisplay(
        id='recall',
        label="Recall",
        value=0,
        style={
                        'height':'100px',
                        'margin-left':'10px',
                        'text-align':'center',
                        'width':'32%',
                        'display':'inline-block'
               }
    ),daq.LEDDisplay(
        id='accuracy',
        label="Accuracy",
        value=0,
        style={
                        'height':'100px',
                        'margin-left':'10px',
                        'text-align':'center',
                        'width':'32%',
                        'display':'inline-block'
               }
    ),

]),

dcc.Store(id='store-score', storage_type='local'),
html.Hr(),
html.Div(children="The ROC Graph will plot after selecting Y taget and training split",id="roc-grph"),
], width=9)

]), html.Hr(),

###########################################################
####################    PREDICTION  #######################
###########################################################

dbc.Row([
    dbc.Col([
        html.H3(children='IMAGE Prediction'),
        html.Div(children='Please upload image (.png) to predict'),
    dcc.Upload(
        id='upload-img',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '80%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '40px'
        },
        
        multiple=True
    ),
    html.Div(id='output-image-upload'),

    ], width = 6 , style={
        'textAlign': 'center',
        } ),

    dbc.Col([
        html.H3(children='Prediction result (By your selected model)'),
        html.Div(children='Please upload image (.png) to predict',id='text0',
                 style={
                'height': '100px',
                }),
        daq.LEDDisplay(
            id = "pred",
            label="Prediction result ( only the first uploaded file )",
            labelPosition='bottom',
            value="0",
            size=64,
            color="#FF5E5E",
            style={
            'height': '200px',
        }
            ),

    ], width = 6 )
]),

dcc.Store(id='store-score', storage_type='local'),

html.Hr(),
html.H3(children='Multiple prediction results (By your selected model)'),
html.Div([
    html.Div(id = 'namelist'),
    html.Div(children='Please upload image (.png) to use a multiple prediction results and performing aggregation',id='pred-list')
    ],
                 style={
                'height': '100px',
                }),
dbc.Row([
    dbc.Col([
        html.P("Select aggregation method", className="control_label"),
        dcc.Dropdown(
        id="ch",
        options=['Mean ( Average )', 'Sum', 'Multiply','Arrange results'],
        multi=False,
        value=None,
        clearable=True,    
),
    ], width = 6 ),
    dbc.Col([
        html.Div(id='av' ,
                 style={
                'textAlign': 'center',
                } ),
    ], width = 6 )
]),


# End of dbc_container
])
)

###################################################################

@callback(Output('output-target', 'children'),
          Input('select_target2', 'value'))
def clean_data(cc):
    if cc is not None:
        return f'You have select : \"{str(cc)}\" to be the target column'
    else :
        raise PreventUpdate
    
@callback(Output('store-target2', 'value'),
          Input('select_target2', 'value'))
def clean_data(cc):
    if cc is not None:
        return str(cc)
    else :
        raise PreventUpdate

@callback(
    Output('select-test-output', 'children'),
    Input('select_test', 'value')
)
def update_output(value):
    if value is not None :
        return f'You have selected : {value} Model'
    else :
        raise PreventUpdate

@callback(
    Output('output-slider', 'children'),
    Input('slider2', 'value')
)
def update_output(value):
    if value is not None :
        return f'You have split training set : {str(value)} %'
    else :
        raise PreventUpdate
    
@callback(
    Output('store-split2', 'value'),
    Input('slider2', 'value')
)
def update_output(value):
    if value is not None :
        return str(value)
    else :
        raise PreventUpdate
    
##################################################
###############      score       #################
##################################################
    
@callback(
    Output('precision', 'value'),
    Input('store-score', 'data')
)
def update_output(value):
    if value is not None :
        value = value['precision']
        return value
    else :
        raise PreventUpdate
    
@callback(
    Output('recall', 'value'),
    Input('store-score', 'data')
)
def update_output(value):
    if value is not None :
        value = value['recall']
        return value
    else :
        raise PreventUpdate
    
@callback(
    Output('accuracy', 'value'),
    Input('store-score', 'data')
)
def update_output(value):
    if value is not None :
        value = value['accuracy']
        return value
    else :
        raise PreventUpdate

##################################################

@callback(Output('store-score', 'data'),
              Input('store-target2', 'value'),
              Input('store-split2', 'value'),
              Input('select_test', 'value')
              )
def update_scores(targ,value,model):
    if ( targ is not None ) and ( value != '100' ) and (model is not None) :
        try :
            df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")
            df = df.iloc[: , 1:] # Drop first column of dataframe
            df = df.sample(frac=1).reset_index(drop=True)
            
            x = df.drop(columns=targ)
            y = df[targ]
            tts = 1-(int(value)/100)
            X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = tts, random_state = 42, stratify = y )

        except Exception as e :
            df = None
            X_train, X_test, y_train, y_test = None

        scora = {}

        if model == 'LogistcRegression':
            steps = [
                ('scalar', MinMaxScaler()),
                ('LogisticRegression',LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=1000,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=123, solver='lbfgs', tol=0.0001, verbose=0,
                    warm_start=False))
                            ]
            pipeline = Pipeline(steps)
            pr = pipeline.fit(X_train, y_train)
            y_pred = pr.predict(X_test)
            sc = round(precision_score(y_test, y_pred, average='macro')*100,2)
            sc1 = round(recall_score(y_test, y_pred, average='macro')*100,2)
            sc2 = round(accuracy_score(y_test, y_pred)*100,2)
            
        elif model == 'RandomForestClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('Randomforest',RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                            criterion='gini', max_depth=None, max_features='sqrt',
                            max_leaf_nodes=None, max_samples=None,
                            min_impurity_decrease=0.0, min_samples_leaf=1,
                            min_samples_split=2, min_weight_fraction_leaf=0.0,
                            n_estimators=100, n_jobs=-1, oob_score=False,
                            random_state=123, verbose=0, warm_start=False))
                            ]
            pipeline = Pipeline(steps)
            pr = pipeline.fit(X_train, y_train)
            y_pred = pr.predict(X_test)
            sc = round(precision_score(y_test, y_pred, average='macro')*100,2)
            sc1 = round(recall_score(y_test, y_pred, average='macro')*100,2)
            sc2 = round(accuracy_score(y_test, y_pred)*100,2)

        elif model == 'ExtraTreesClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('ExtraTreesClassifier',ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                      criterion='gini', max_depth=None, max_features='sqrt',
                      max_leaf_nodes=None, max_samples=None,
                      min_impurity_decrease=0.0, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=-1, oob_score=False,
                      random_state=123, verbose=0, warm_start=False))
                            ]
            pipeline = Pipeline(steps)
            pr = pipeline.fit(X_train, y_train)
            y_pred = pr.predict(X_test)
            sc = round(precision_score(y_test, y_pred, average='macro')*100,2)
            sc1 = round(recall_score(y_test, y_pred, average='macro')*100,2)
            sc2 = round(accuracy_score(y_test, y_pred)*100,2)

        elif model == 'SGDClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('SGDClassifier',SGDClassifier(alpha=0.0001, average=False, class_weight=None,
               early_stopping=False, epsilon=0.1, eta0=0.001, fit_intercept=True,
               l1_ratio=0.15, learning_rate='optimal', loss='hinge',
               max_iter=1000, n_iter_no_change=5, n_jobs=-1, penalty='l2',
               power_t=0.5, random_state=123, shuffle=True, tol=0.001,
               validation_fraction=0.1, verbose=0, warm_start=False))
                            ]
            pipeline = Pipeline(steps)
            pr = pipeline.fit(X_train, y_train)
            y_pred = pr.predict(X_test)
            sc = round(precision_score(y_test, y_pred, average='macro')*100,2)
            sc1 = round(recall_score(y_test, y_pred, average='macro')*100,2)
            sc2 = round(accuracy_score(y_test, y_pred)*100,2)

        elif model == 'LGBMClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('LGBMClassifier',lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                importance_type='split', learning_rate=0.1, max_depth=-1,
                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
                random_state=123, reg_alpha=0.0, reg_lambda=0.0, silent='warn',
                subsample=1.0, subsample_for_bin=200000, subsample_freq=0))
                            ]
            pipeline = Pipeline(steps)
            pr = pipeline.fit(X_train, y_train)
            y_pred = pr.predict(X_test)
            sc = round(precision_score(y_test, y_pred, average='macro')*100,2)
            sc1 = round(recall_score(y_test, y_pred, average='macro')*100,2)
            sc2 = round(accuracy_score(y_test, y_pred)*100,2)

        elif model == 'KNeighborsClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('KNeighborsClassifier',KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                      metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                      weights='uniform'))
                            ]
            pipeline = Pipeline(steps)
            pr = pipeline.fit(X_train, y_train)
            y_pred = pr.predict(X_test)
            sc = round(precision_score(y_test, y_pred, average='macro')*100,2)
            sc1 = round(recall_score(y_test, y_pred, average='macro')*100,2)
            sc2 = round(accuracy_score(y_test, y_pred)*100,2)

        else :
            raise PreventUpdate
        
        scora['precision'] = sc
        scora['recall'] = sc1
        scora['accuracy'] = sc2
        
        return scora
    # else :
    #     raise PreventUpdate

##################################################
###################   Graph   ####################
##################################################

@callback(Output('roc-grph', 'children'),
              Input('select_target2', 'value'),
              Input('slider2', 'value'),
              Input('select_test', 'value'),
              State('roc-grph', 'figure')
              )
def update_roc(targ,value,model,fg):
    if ( targ is not None ) and (model is not None) :
        try :
            df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")
            df = df.iloc[: , 1:] # Drop first column of dataframe
            df = df.sample(frac=1).reset_index(drop=True)
            
            x = df.drop(columns=targ)
            y = df[targ]

            y = pd.get_dummies(y).to_numpy()
            n_classes = y.shape[1]

            tts = 1-(int(value)/100)
            X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = tts, random_state = 42, stratify = y )
            
        except Exception as e :
            raise PreventUpdate

        if model == 'LogistcRegression':
            steps = [
                ('scalar', MinMaxScaler()),
                ('LogisticRegression',OneVsRestClassifier(LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=1000,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=123, solver='lbfgs', tol=0.0001, verbose=0,
                    warm_start=False)))
                            ]
            pipeline = Pipeline(steps)
            y_score = pipeline.fit(X_train,y_train).decision_function(X_test)
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute macro-average ROC curve and ROC area
            fpr["macro"], tpr["macro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            
        elif model == 'RandomForestClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('Randomforest',OneVsRestClassifier(RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                            criterion='gini', max_depth=None, max_features='sqrt',
                            max_leaf_nodes=None, max_samples=None,
                            min_impurity_decrease=0.0, min_samples_leaf=1,
                            min_samples_split=2, min_weight_fraction_leaf=0.0,
                            n_estimators=100, n_jobs=-1, oob_score=False,
                            random_state=123, verbose=0, warm_start=False)))
                            ]
            pipeline = Pipeline(steps)
            y_score = pipeline.fit(X_train,y_train).predict(X_test)
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute macro-average ROC curve and ROC area
            fpr["macro"], tpr["macro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            
        elif model == 'ExtraTreesClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('ExtraTreesClassifier',OneVsRestClassifier(ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                      criterion='gini', max_depth=None, max_features='sqrt',
                      max_leaf_nodes=None, max_samples=None,
                      min_impurity_decrease=0.0, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=-1, oob_score=False,
                      random_state=123, verbose=0, warm_start=False)))
                            ]
            pipeline = Pipeline(steps)
            y_score = pipeline.fit(X_train,y_train).predict(X_test)
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute macro-average ROC curve and ROC area
            fpr["macro"], tpr["macro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        elif model == 'SGDClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('SGDClassifier',OneVsRestClassifier(SGDClassifier(alpha=0.0001, average=False, class_weight=None,
               early_stopping=False, epsilon=0.1, eta0=0.001, fit_intercept=True,
               l1_ratio=0.15, learning_rate='optimal', loss='hinge',
               max_iter=1000, n_iter_no_change=5, n_jobs=-1, penalty='l2',
               power_t=0.5, random_state=123, shuffle=True, tol=0.001,
               validation_fraction=0.1, verbose=0, warm_start=False)))
                            ]
            pipeline = Pipeline(steps)
            y_score = pipeline.fit(X_train,y_train).decision_function(X_test)
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute macro-average ROC curve and ROC area
            fpr["macro"], tpr["macro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            
        elif model == 'LGBMClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('LGBMClassifier',OneVsRestClassifier(lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                importance_type='split', learning_rate=0.1, max_depth=-1,
                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
                random_state=123, reg_alpha=0.0, reg_lambda=0.0, silent='warn',
                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)))
                            ]
            pipeline = Pipeline(steps)
            y_score = pipeline.fit(X_train,y_train).predict(X_test)
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute macro-average ROC curve and ROC area
            fpr["macro"], tpr["macro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        elif model == 'KNeighborsClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('KNeighborsClassifier',OneVsRestClassifier(KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                      metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                      weights='uniform')))
                            ]
            pipeline = Pipeline(steps)
            y_score = pipeline.fit(X_train,y_train).predict(X_test)
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute macro-average ROC curve and ROC area
            fpr["macro"], tpr["macro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        else :
            raise PreventUpdate
        
        # Plot of a ROC curve for a specific class

        fig = go.Figure(fg)
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
            )

        fig.add_trace(go.Scatter(x=fpr["macro"], y=tpr["macro"], name="macro-average ROC curve (area={0:0.2f})".format(roc_auc["macro"]), mode='lines'))

        for i in range(n_classes):
            name = f"ROC curve of class {i} (AUC={roc_auc[i]:.2f})"
            fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], name=name, mode='lines'))
        
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            width=700, height=500
        )
            
        return html.Div([html.H3(children='Multiclass ROC Curve'),
                         html.P(children=model),
                        dcc.Graph(figure = fig ,style={'margin-left':'140px','text-align':'center'})],
                        style={'margin-left':'20px','text-align':'center'}
                        )
    else :
        return html.Div([html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                        html.P(children="The ROC Graph will plot after selecting Y taget and training split")],
                        style={'margin-left':'20px','text-align':'center'}
                        )

##################################################
##################   PREDICT   ###################
##################################################

@callback(Output('output-image-upload', 'children'),
              Input('upload-img', 'contents'),
              State('upload-img', 'filename'),
              State('upload-img', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children
    
##################################################################################

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

def moveup(img):
    while True:
        if np.all(img[0] >= 240):
            img = img[1:]
        else:
            break
    return img

def movedown(img):
    while True:
        if np.all(img[-1] >= 240):
            img = img[:-1]
        else:
            break
    return img

def moveleft(img):
    while True:
        if np.all(img[:, 0] >= 240):
            img = img[:, 1:]
        else:
            break
    return img

def moveright(img):
    while True:
        if np.all(img[:, -1] >= 240):
            img = img[:, :-1]
        else:
            break
    return img

def rescale(img):
    img = moveup(img)
    img = movedown(img)
    img = moveleft(img)
    img = moveright(img)
    return img

##################################################################################

@callback(Output('text0', 'children'),
          Output('pred', 'value'),
          Output('namelist', 'children'),
          Output('pred-list', 'children'),
          Output('av', 'children'),
            Input('select_target2', 'value'),
            Input('slider2', 'value'),
            Input('select_test', 'value'),
            Input('ch', 'value'),
            Input('upload-img', 'contents'),
            State('upload-img', 'filename'),
            )
def update_pred(ytarget,split,model,ch,list_of_contents, list_of_names):
    if ( ytarget is not None ) and (model is not None) and (list_of_contents is not None) :
        try :
            df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")
            df = df.iloc[: , 1:] # Drop first column of dataframe
            df = df.sample(frac=1).reset_index(drop=True)
            
            x = df.drop(columns=ytarget)
            y = df[ytarget]

            tts = 1-(int(split)/100)
            X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = tts, random_state = 42, stratify = y )
            
        except Exception as e :
            raise PreventUpdate

        if model == 'LogistcRegression':
            steps = [
                ('scalar', MinMaxScaler()),
                ('LogisticRegression',OneVsRestClassifier(LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=1000,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=123, solver='lbfgs', tol=0.0001, verbose=0,
                    warm_start=False)))
                            ]
            pipeline = Pipeline(steps)
            y_score = pipeline.fit(X_train,y_train)
            
        elif model == 'RandomForestClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('Randomforest',OneVsRestClassifier(RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                            criterion='gini', max_depth=None, max_features='sqrt',
                            max_leaf_nodes=None, max_samples=None,
                            min_impurity_decrease=0.0, min_samples_leaf=1,
                            min_samples_split=2, min_weight_fraction_leaf=0.0,
                            n_estimators=100, n_jobs=-1, oob_score=False,
                            random_state=123, verbose=0, warm_start=False)))
                            ]
            pipeline = Pipeline(steps)
            y_score = pipeline.fit(X_train,y_train)
            
        elif model == 'ExtraTreesClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('ExtraTreesClassifier',OneVsRestClassifier(ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                      criterion='gini', max_depth=None, max_features='sqrt',
                      max_leaf_nodes=None, max_samples=None,
                      min_impurity_decrease=0.0, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=-1, oob_score=False,
                      random_state=123, verbose=0, warm_start=False)))
                            ]
            pipeline = Pipeline(steps)
            y_score = pipeline.fit(X_train,y_train)

        elif model == 'SGDClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('SGDClassifier',OneVsRestClassifier(SGDClassifier(alpha=0.0001, average=False, class_weight=None,
               early_stopping=False, epsilon=0.1, eta0=0.001, fit_intercept=True,
               l1_ratio=0.15, learning_rate='optimal', loss='hinge',
               max_iter=1000, n_iter_no_change=5, n_jobs=-1, penalty='l2',
               power_t=0.5, random_state=123, shuffle=True, tol=0.001,
               validation_fraction=0.1, verbose=0, warm_start=False)))
                            ]
            pipeline = Pipeline(steps)
            y_score = pipeline.fit(X_train,y_train)

        elif model == 'LGBMClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('LGBMClassifier',OneVsRestClassifier(lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                importance_type='split', learning_rate=0.1, max_depth=-1,
                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
                random_state=123, reg_alpha=0.0, reg_lambda=0.0, silent='warn',
                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)))
                            ]
            pipeline = Pipeline(steps)
            y_score = pipeline.fit(X_train,y_train)

        elif model == 'KNeighborsClassifier' :
            steps = [
                ('scalar', MinMaxScaler()),
                ('KNeighborsClassifier',OneVsRestClassifier(KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                      metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                      weights='uniform')))
                            ]
            pipeline = Pipeline(steps)
            y_score = pipeline.fit(X_train,y_train)

        limg = []
        for i in list_of_contents :
            encoded_data = str(i).split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = rescale(img)
            img = cv2.resize(img, (28, 28))
            img = img.flatten()

            img = np.array(img).reshape(1,-1)
            img = pd.DataFrame(img)
            result = y_score.predict(img)
            limg.append(result[0])

        choice = ['Mean ( Average )', 'Sum', 'Multiply','Arrange results']
        if ch is None :
            av = None
        else :
            if ch == choice[3] :
                num = ''
                for i in limg:
                    num += str(i)
            elif ch == choice[0] :
                num = sum(limg)/len(limg)
            elif ch == choice[1] :
                num = sum(limg)
            elif ch == choice[2] :
                num = 1
                for i in limg:
                    num *= i

            av = num

        children = f'Model : {model} >> The result of \"{list_of_names[0]}\" is : {limg[0]}'
        return children , limg[0] , f'List of image names : {list_of_names}', f'Multiple prediction results : {limg}' , f'The answer is : {av}'
    else :
        raise PreventUpdate