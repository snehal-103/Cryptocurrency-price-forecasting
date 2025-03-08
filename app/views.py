from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext
from django.shortcuts import  render, redirect
from django.db import IntegrityError

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter
from django.contrib.auth.forms import UserCreationForm
import pandas as pd
import numpy as np
import json

import yfinance as yf
import datetime as dt
import qrcode

from .models import Project
from .forms import NewUserForm
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm

from django.shortcuts import  render, redirect
from .forms import NewUserForm
from django.contrib.auth import login, authenticate #add this
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm #add this
from .forms import NewUserForm
from django.contrib.auth import login
from django.contrib import messages
from .forms import NewUserForm
from django.contrib.auth import login, logout, authenticate #add this
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm #add this
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm

from django.contrib.auth.models import User


def home(request):
        return render(request, 'home.html')

def index1(request):
        return render(request, 'index.html')
# The Home page when Server loads up
def index(request):
    # ================================================= Left Card Plot =========================================================
    # Here we use yf.download function
    data = yf.download(
        
        # passes the ticker
        tickers=['BTC-CAD', 'ETH-USD','USDT-USD','SOL-USD','ADA-USD'],
        
        group_by = 'ticker',
        
        threads=True, # Set thread value to true
        
        # used for access data[ticker]
        period='1mo', 
        interval='1d'
    
    )

    data.reset_index(level=0, inplace=True)



    fig_left = go.Figure()
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['BTC-CAD']['Adj Close'], name="BTC-CAD")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['ETH-USD']['Adj Close'], name="ETH-USD")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['USDT-USD']['Adj Close'], name="USDT-USD")
             )
    fig_left.add_trace(
                 go.Scatter(x=data['Date'], y=data['SOL-USD']['Adj Close'], name="SOL-USD")
             )
    fig_left.add_trace(
                 go.Scatter(x=data['Date'], y=data['ADA-USD']['Adj Close'], name="ADA-USD")
             )
   
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')

    # ================================================ To show recent stocks ==============================================
    
    df1 = yf.download(tickers = 'BTC-CAD', period='1d', interval='1d')
    df2 = yf.download(tickers = 'ETH-USD', period='1d', interval='1d')
    df3 = yf.download(tickers = 'USDT-USD', period='1d', interval='1d')
    df4 = yf.download(tickers = 'SOL-USD', period='1d', interval='1d')
    df5 = yf.download(tickers = 'ADA-USD', period='1d', interval='1d')

    df1.insert(0, "Ticker", "BTC-CAD")
    df2.insert(0, "Ticker", "ETH-USD")
    df3.insert(0, "Ticker", "USDT-USD")
    df4.insert(0, "Ticker", "SOL-USD")
    df5.insert(0, "Ticker", "ADA-USD")


    df = pd.concat([df1, df2,df3,df4,df5], axis=0)
    df.reset_index(level=0, inplace=True)
    df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    convert_dict = {'Date': object}
    df = df.astype(convert_dict)
    df.drop('Date', axis=1, inplace=True)

    json_records = df.reset_index().to_json(orient ='records')
    recent_stocks = []
    recent_stocks = json.loads(json_records)

    # ========================================== Page Render section =====================================================

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks
    })

def search(request):
    return render(request, 'search.html', {})

def ticker(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'ticker.html', {
        'ticker_list': ticker_list
    })


# The Predict Function to implement Machine Learning as well as Plotting
def predict(request, ticker_value, number_of_days):
    try:
        # ticker_value = request.POST.get('ticker')
        ticker_value = ticker_value.upper()
        df = yf.download(tickers = ticker_value, period='1d', interval='1m')
        print("Downloaded ticker = {} successfully".format(ticker_value))
    except:
        return render(request, 'API_Down.html', {})

    try:
        # number_of_days = request.POST.get('days')
        number_of_days = int(number_of_days)
    except:
        return render(request, 'Invalid_Days_Format.html', {})

    Valid_Ticker = ["BTC-CAD","BTC-USD","BTC=F","BITO","MBT=F","ETH-USD","USDT-USD","SOL-USD","ADA-USD","STETH-USD","AVAX-USD","DOGE-USD"
                    ,"DOT-USD","WTRX-USD","MATIC-USD","TON11419-USD","SHIB-USD","LTC-USD","DAI-USD","WEOS-USD","ATOM-USD","NEAR-USD"
                    ,"BNB-USD","XRP-USD"
    ]

    if ticker_value not in Valid_Ticker:
        return render(request, 'Invalid_Ticker.html', {})
    
    if number_of_days < 0:
        return render(request, 'Negative_Days.html', {})
    
    if number_of_days > 365:
        return render(request, 'Overflow_days.html', {})
    

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))
    fig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Bitcoin Price (USD per Shares)')
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
        )
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')



    # ========================================== Machine Learning ==========================================


    try:
        df_ml = yf.download(tickers = ticker_value, period='3mo', interval='1h')
    except:
        ticker_value = 'BTC-CAD'
        df_ml = yf.download(tickers = ticker_value, period='3mo', interval='1m')

    # Fetching ticker values from Yahoo Finance API 
    df_ml = df_ml[['Adj Close']]
    print("Data: ",df_ml)
    #df_ml = int(df_ml)
    forecast_out = int(number_of_days)
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)
    # Splitting data for Test and Train
    X = np.array(df_ml.drop(['Prediction'],axis=1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    #X = float(X)
    y = np.array(df_ml['Prediction'])
    y = y[:-forecast_out]
    #y = float(y)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3)
    # Applying Linear Regression
    #from sklearn.naive_bayes import GaussianNB
    #from sklearn.svm import SVC
    #clf = SVC()
    #clf = LinearRegression()
    #from sklearn.naive_bayes import GaussianNB  
    #clf = GaussianNB(priors=None, var_smoothing=1e-09) 
    from sklearn.svm import SVR
    clf = SVR(kernel="linear", C=100, gamma="auto")
    clf.fit(X_train,y_train)
    # Prediction Score
    confidence = clf.score(X_test, y_test)
    print(confidence)
    # Predicting for 'n' days stock data
    #forecast_prediction1 = clf.predict(X_test)


    forecast_prediction = clf.predict(X_forecast)
    forecast = forecast_prediction.tolist()
    print(forecast)

    #from sklearn.metrics import accuracy_score #,confusion_matrix, classification_report, accuracy_score,#roc_curve
    #accuracy = accuracy_score(y_test,forecast_prediction1)
    #print("Accuracy : ",accuracy*100)
    
    # print("Classification Report :\n")
    # repo = (classification_report(y_test, forecast_prediction))
    # print(repo)
    # print("Confusion Matrix :")
    # cm = confusion_matrix(y_test,forecast_prediction)
    # print(cm)


    # ========================================== Plotting predicted data ======================================


    pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])
    
    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    # ========================================== Display Ticker Info ==========================================

    ticker = pd.read_csv('app/Data/Tickers.csv')
    to_search = ticker_value
    ticker.columns = ['Symbol', 'Name', 'Percent_Change', 'Market_Cap',
                    'Country', 'Volume', 'Sector']
    for i in range(0,ticker.shape[0]):
        if ticker.Symbol[i] == to_search:
            Symbol = ticker.Symbol[i]
            Name = ticker.Name[i]
            
           
            Percent_Change = ticker.Percent_Change[i]
            Market_Cap = ticker.Market_Cap[i]
            Country = ticker.Country[i]
            
            Volume = ticker.Volume[i]
            Sector = ticker.Sector[i]
            
            break

    # ========================================== Page Render section ==========================================
    

    return render(request, "result.html", context={ 'plot_div': plot_div, 
                                                    'confidence' : confidence,
                                                    'forecast': forecast,
                                                    'ticker_value':ticker_value,
                                                    'number_of_days':number_of_days,
                                                    'plot_div_pred':plot_div_pred,
                                                    'Symbol':Symbol,
                                                    'Name':Name,
                                                    
                                                    
                                                    'Percent_Change':Percent_Change,
                                                    'Market_Cap':Market_Cap,
                                                    'Country':Country,
                                                    
                                                    'Volume':Volume,
                                                    'Sector':Sector,
                                                    
                                                    })


from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required




def LoginUser(request):
    if request.user.username=="":
        return render(request,"login.html")
    else:
        return HttpResponseRedirect("/homepage")

@login_required(login_url="/loginuser/")
def HomePage(request):
    return render(request, "home.html")



def news(request):
    return render(request, "news.html")

def clicklogin(request):
    if request.method!="POST":
        return HttpResponse("<h1> Methoid not allowed<h1>")
    else:
        username = request.POST.get('username','')
        password = request.POST.get('password','')
        
        user=authenticate(username=username,password=password)
        if user!=None:
            login(request,user)
            return HttpResponseRedirect('/homepage')
        else:
            messages.error(request, "Invalid Login")
            return HttpResponseRedirect('/loginuser')

def LogoutUser(request):
    logout(request)
    request.user=None
    return HttpResponseRedirect("/loginuser")       




def RegisterUser(request):
    if request.user==None or request.user =="" or request.user.username=="":
        return render(request,"register.html")
    else:
        return HttpResponseRedirect("/homepage")        


def ClickRegister(request):
    if request.method != "POST":
        return HttpResponse("<h2>Method Not Allowed</h2>")
    else:
        username = request.POST.get('username', '')
        email = request.POST.get('email', '')
        password = request.POST.get('password', '')

        if not (User.objects.filter(username=username).exists() or User.objects.filter(email=email).exists()):
            User.objects.create_user(username, email, password)
            messages.success(request, "User Created Successfully")
            return HttpResponseRedirect('/register_user')
        else:
            messages.error(request, "Email or Username Already Exist")
            return HttpResponseRedirect('/register_user')