import dash
from dash.dependencies import Input, Output
from dash import dcc, html, callback
from dash import dash_table
from collections import OrderedDict
from dash.exceptions import PreventUpdate
import plotly.express as px
import dash_bootstrap_components as dbc

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score,accuracy_score, auc, f1_score

import dash_daq as daq
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

dash.register_page("Train",  path='/Training',

layout = dbc.Container([ 
    dbc.Row([
        html.H4(children='Select your training'),
        html.Div(children='select Y column & choose trainning number'),
        html.Hr(),

        dcc.Store(id='store-target', storage_type='local'),
        dcc.Store(id='store-split', storage_type='local')
        ], style={'height':'100px','text-align':'center',}),
        
dbc.Row([

    dbc.Col([

html.H3(children='Comparing the Accuracy Scores of Different Models (Bar Plot)'),
html.Div(children='Select a model'),

    dcc.Graph(
        id='id2',
        figure = {},
    ),
], width=8 ),

    dbc.Col([
        
        dbc.Row([
html.P("Select Target (Y column)", className="control_label"),
dcc.Dropdown(
        id="select_target",
        options = op ,
        multi=False,
        value=None,
        clearable=True,
),
html.Div(id='dd-output-container'),
html.Hr()

], style = {'height':'200px',} ),


dbc.Row([
html.Div(children='select your training number'),

html.Div([daq.LEDDisplay(
        id='my-LED-display-1',
        label="Trainning splits",
        value=100,
    ),
    dcc.Slider(
        id='my-LED-display-slider-1',
        min=0,
        max=100,
        step=10,
        value=100
    )],style = {'height':'200px',}),
 
html.Hr()
]),

dbc.Row([

html.Div(children='select tools for Cross Validation'),
html.Div(dcc.Dropdown(
    id="selectcv",
    options = ['LogistcRegression', 'RandomForestClassifier', 'ExtraTreesClassifier','SGDClassifier','LGBMClassifier','KNeighborsClassifier'],
    multi = True ,
    value = None ,
    clearable = True
), style={'padding': 10, 'flex': 1}),

html.Div(children='Please select the models',id='cvscore'),

dcc.Store(id='output-cv2', storage_type='local'),

dcc.Store(id='acc', storage_type='local'),
dcc.Store(id='precision0', storage_type='local'),
dcc.Store(id='recall0', storage_type='local'),
dcc.Store(id='f1', storage_type='local'),
])

], width=4 ),

]),

# End of dbc_container

])
)


##################################################################################

@callback(
    Output('cvscore', 'children'),
    Input('acc', 'value'),
    Input('output-cv2', 'children')
)
def update_output(data,children):
    if (data is not None) and (children is not None) :
        scorr = {}
        for i in data:
            for j in children:
                if i==j:
                    scorr[i]=data[i]
        return f'Accuracy score : {scorr}'
    else :
        raise PreventUpdate

@callback(Output('id2', 'figure'),
          Input('acc', 'value'),
          Input('precision0', 'value'),
          Input('recall0', 'value'),
          Input('f1', 'value'),
          Input('output-cv2', 'children')
)
def upd_fig(cc,pre,rec,f1,select):
    if (cc is not None) and (select is not None):

        ac_scor = {}
        pre_scor = {}
        re_scor = {}
        f1_scor = {}

        for i in cc:
            for j in select:
                if i==j:
                    ac_scor[i]=cc[i]
                    pre_scor[i]=pre[i]
                    re_scor[i]=rec[i]
                    f1_scor[i]=f1[i]

        cc = {'model':ac_scor.keys(),'accuracy_score':ac_scor.values(), 'precision_score':pre_scor.values(), 'recall_score':re_scor.values(), 'f1_score(macro)':f1_scor.values()}
        cc = pd.DataFrame(cc)

        figure = px.bar(cc,x=['accuracy_score','precision_score','recall_score','f1_score(macro)'],y='model', barmode="group"
                        , text_auto='.2s',title="Comparing model score ( MACRO-Average )",
                        labels={
                     "model": "Classification Model",
                     "value": "Score",
                     "variable": "Type of score"
                 }
             , height=650)
        figure.update_layout(transition_duration=1)
        return figure
    else :
        raise PreventUpdate

@callback(Output('store-target', 'value'),
          Input('select_target', 'value'))
def clean_data(cc):
    if cc is not None:
        return cc
    else :
        raise PreventUpdate

@callback(
    Output('my-LED-display-1', 'value'),
    Input('my-LED-display-slider-1', 'value')
)
def update_output(value):
    return str(value)

@callback(
    Output('store-split', 'value'),
    Input('my-LED-display-slider-1', 'value')
)
def update_led(value):
    if value is not None :
        return str(value)
    else :
        raise PreventUpdate

@callback(
    Output('output-cv2', 'children'),
    Input('selectcv', 'value')
)
def update_output(value):
    if value is not None :
        return value
    else :
        raise PreventUpdate


@callback(
    Output('dd-output-container', 'children'),
    Input('select_target', 'value')
)
def update_output(value):
    return f'You have select : {value}'

############################################################

@callback(Output('acc', 'value'),Output('precision0', 'value'),Output('recall0', 'value'),Output('f1', 'value'),
              Input('store-target', 'value'),
              Input('store-split', 'value')
              )
def update_score(cc,value):
    if ( cc is not None ) and ( value != '100' ) :
        try :
            df = pd.read_csv("./Github/ThNumber_img_classification/uploaded/df_00.csv")
            df = df.iloc[: , 1:] # Drop first column of dataframe
            df = df.sample(frac=1).reset_index(drop=True)
            
            x = df.drop(columns=cc)
            y = df[cc]
            tts = 1-(int(value)/100)
            X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = tts, random_state = 42, stratify = y )

        except Exception as e :
            df = None
            X_train, X_test, y_train, y_test = None

        ac_score = {}
        re_score = {}
        pre_score = {}
        f1_sc = {}

        for i in ['LogistcRegression', 'RandomForestClassifier', 'ExtraTreesClassifier','SGDClassifier','LGBMClassifier','KNeighborsClassifier']:
            if i == 'LogistcRegression' :
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
                sc = round(accuracy_score(y_test, y_pred)*100,2)
                sc0 = round(precision_score(y_test, y_pred, average='macro')*100,2)
                sc1 = round(recall_score(y_test, y_pred, average='macro')*100,2)
                sc2 = round(f1_score(y_test, y_pred, average='macro')*100,2)

                ac_score[i]=sc
                re_score[i]=sc1
                pre_score[i]=sc0
                f1_sc[i]=sc2

            elif i == 'RandomForestClassifier' :
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
                sc = round(accuracy_score(y_test, y_pred)*100,2)
                sc0 = round(precision_score(y_test, y_pred, average='macro')*100,2)
                sc1 = round(recall_score(y_test, y_pred, average='macro')*100,2)
                sc2 = round(f1_score(y_test, y_pred, average='macro')*100,2)

                ac_score[i]=sc
                re_score[i]=sc1
                pre_score[i]=sc0
                f1_sc[i]=sc2

            elif i == 'ExtraTreesClassifier' :
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
                sc = round(accuracy_score(y_test, y_pred)*100,2)
                sc0 = round(precision_score(y_test, y_pred, average='macro')*100,2)
                sc1 = round(recall_score(y_test, y_pred, average='macro')*100,2)
                sc2 = round(f1_score(y_test, y_pred, average='macro')*100,2)

                ac_score[i]=sc
                re_score[i]=sc1
                pre_score[i]=sc0
                f1_sc[i]=sc2

            elif i == 'SGDClassifier' :
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
                sc = round(accuracy_score(y_test, y_pred)*100,2)
                sc0 = round(precision_score(y_test, y_pred, average='macro')*100,2)
                sc1 = round(recall_score(y_test, y_pred, average='macro')*100,2)
                sc2 = round(f1_score(y_test, y_pred, average='macro')*100,2)

                ac_score[i]=sc
                re_score[i]=sc1
                pre_score[i]=sc0
                f1_sc[i]=sc2

            elif i == 'LGBMClassifier' :
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
                sc = round(accuracy_score(y_test, y_pred)*100,2)
                sc0 = round(precision_score(y_test, y_pred, average='macro')*100,2)
                sc1 = round(recall_score(y_test, y_pred, average='macro')*100,2)
                sc2 = round(f1_score(y_test, y_pred, average='macro')*100,2)

                ac_score[i]=sc
                re_score[i]=sc1
                pre_score[i]=sc0
                f1_sc[i]=sc2

            elif i == 'KNeighborsClassifier' :
                steps = [
                ('scalar', MinMaxScaler()),
                ('KNeighborsClassifier',KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                      metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                      weights='uniform'))
                            ]
                pipeline = Pipeline(steps)
                pr = pipeline.fit(X_train, y_train)
                y_pred = pr.predict(X_test)
                sc = round(accuracy_score(y_test, y_pred)*100,2)
                sc0 = round(precision_score(y_test, y_pred, average='macro')*100,2)
                sc1 = round(recall_score(y_test, y_pred, average='macro')*100,2)
                sc2 = round(f1_score(y_test, y_pred, average='macro')*100,2)

                ac_score[i]=sc
                re_score[i]=sc1
                pre_score[i]=sc0
                f1_sc[i]=sc2
                
        return ac_score, pre_score, re_score, f1_sc
    else :
        raise PreventUpdate
