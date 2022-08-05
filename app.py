from xmlrpc.client import DateTime
import dash
from dash import dcc
from dash import html
from datetime import date
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input,Output,State
from models.model import train_model

app=dash.Dash(__name__)
server=app.server

item1=html.Div(
            [
                html.P("Welcome to the Stock Dash App",className="start"),
                html.Div([
                    html.P("Input Stock Code"),
                    dcc.Input(type="text",value='',id="short-stock-name"),
                    html.Button("Submit",name='Submit',n_clicks=0,id="Submit")
                ],className="stock"),
                html.Div([
                    dcc.DatePickerRange(min_date_allowed=date(1975,1,1),max_date_allowed=date.today(),initial_visible_month=date.today(),end_date=date.today(),id="datepicker")
                ]),
                html.Div([
                    html.Button("Stock Price",name='Stock Price',n_clicks=0,className="btn",id="price"),
                    html.Button("Indicators",name='Indicators',n_clicks=0,className="btn",id="EMA"),
                    dcc.Input(type="text",value='',placeholder='number of days',className="days",id="n_days"),
                    html.Button("Forecast",name='Forecast',n_clicks=0,className="btn",id="n_days_btn")
                ])
            ],
            className="nav")
item2=html.Div(
            [
                html.Div([
                    #LOGO
                    #COMPANY NAME

                ],className="header",id="header"),
                html.Div( #DESCRIPTION
                id="description",className="description_ticker"
                ),
                html.Div([
                    dcc.Graph(id="graph1")#STOCK PRICE PLOT

                ],id="graphs-content"),
                html.Div([
                    dcc.Graph(id="graph2")#INDICATOR PLOT

                ],id="main-content"),
                html.Div([
                    dcc.Graph(id="graph3")#FORECAST PLOT

                ],id="forecast-content")

            ],className="Content")

app.layout=html.Div([item1,item2],className="container")
@app.callback(
    [Output(component_id="header",component_property="children"),Output(component_id="description",component_property="children")],
    [Input(component_id="Submit",component_property="n_clicks")],
    [State(component_id="short-stock-name",component_property="value")]
)
def update_data(n,val):
    ticker=yf.Ticker(val)
    inf=ticker.info
    df=pd.DataFrame().from_dict(inf,orient="index")
    return df[0]["shortName"],df[0][["longBusinessSummary"]]

@app.callback(
    Output("graph1","figure"),
    [Input(component_id="price",component_property="n_clicks")],
    [State(component_id="datepicker",component_property="start_date"),State(component_id="datepicker",component_property="end_date"),State(component_id="short-stock-name",component_property="value")]

)
def plot_graph(n,start_date,end_date,val):
    df=yf.download(val,start=start_date,end=end_date)
    df.reset_index(inplace=True)
    df1=df[["Date","Open","Close"]].copy()
    print(df1)
    fig=get_stock_price_fig(df1)
    return fig

def get_stock_price_fig(df):
    fig=px.line(df,x="Date",y=["Open","Close"],title="Closing and Opening Price vs Date")
    return fig

@app.callback(
    Output("graph2","figure"),
    [Input(component_id="EMA",component_property="n_clicks")],
    [State(component_id="datepicker",component_property="start_date"),State(component_id="datepicker",component_property="end_date"),State(component_id="short-stock-name",component_property="value")]

)
def plot_graph(n,start_date,end_date,val):
    df=yf.download(val,start=start_date,end=end_date)
    df.reset_index(inplace=True)
    df1=df[["Date","Open","Close"]].copy()
    print(df1)
    fig=get_more(df1)
    return fig

def get_more(df):
    df['EWA_20']=df['Close'].ewm(span=20, adjust=False).mean()
    fig=px.scatter(df,x="Date",y="EWA_20")
    return fig

@app.callback(
    Output('graph3','figure'),
    [Input(component_id="n_days_btn",component_property="n_clicks")],
    [State(component_id="n_days",component_property="value"),State(component_id="short-stock-name",component_property="value")]
)

def plot_predictor_graph(n,n_days,val):
    today=date.today()
    print(type(n_days))
    dates=pd.date_range(today,periods=int(n_days)).tolist()
    predicted_price=train_model(val,today,int(n_days))
    data={'Dates':dates,'Price':predicted_price}
    df=pd.DataFrame(data)
    #print(df)
    fig=plot_points(df)
    return fig

def plot_points(df):
    fig=px.line(df,x="Dates",y="Price")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
