import math
import yfinance as yf
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,GridSearchCV,cross_validate
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_percentage_error
from datetime import date


def train_model(val,end_date,n_days_predict):
    df=yf.download(val,start="2020-01-01",end=end_date)

    df1=df[["Adj Close"]]
    df1["Prediction"]=df[["Adj Close"]].shift(-n_days_predict)

    X=np.array(df1.drop(['Prediction'],1)) #create independent data
    X=X[:-n_days_predict]

    Y=np.array(df1["Prediction"]) #create dependent data
    Y=Y[:-n_days_predict]

    X_forecast=np.array(df1.drop(['Prediction'],1))[-n_days_predict:]

    #x_train,y_train,x_test,y_test=train_test_split(X,Y,test_size=0.1)  #splitting the data
    seed = 42
    X, Y = shuffle(X, Y, random_state=seed)
    gsc=GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C' :[0.1,1,100,1000],
            'epsilon' :[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10],
            'gamma' :[0.0001,0.001,0.005,0.1,1,3,5]
        },
        cv=5,scoring='neg_mean_squared_error',verbose=0,n_jobs=-1
    )
    grid_result=gsc.fit(X,Y)
    best_params=grid_result.best_params_
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
                   coef0=0.1, shrinking=True,
                   tol=0.001, cache_size=200, verbose=False, max_iter=-1)
    rbf_svr=best_svr
    rbf_svr.fit(X,Y)
    scoring = {
               'abs_error': 'neg_mean_absolute_error',
               'squared_error': 'neg_mean_squared_error'}
    scores=cross_validate(best_svr,X,Y, cv=10,scoring=scoring,return_train_score=True)
    svm_prediction=rbf_svr.predict(X_forecast)
    #print("MAE :", abs(scores['test_abs_error'].mean()), "| RMSE :", math.sqrt(abs(scores['test_squared_error'].mean())),"| MAPE :", my_custom_MAPE(best_svr,X,Y))
    return svm_prediction.tolist()

def my_custom_MAPE(clf, X_val, y_true,epsilon=0.000001):
      y_pred = clf.predict(X_val)
      ii = 0
      for i in y_true:
          if (i < epsilon) & (i > -epsilon):
             y_true[ii] = epsilon
          else:
             y_true[ii] = y_true[ii]
          ii = ii+1
      MAPE = (1/len(y_true))*np.sum(np.abs(y_true - y_pred)/y_true)
      return MAPE


#today=date.today()
#d1=today.strftime("%Y-%m-%d")
#print(d1)
#print(train_model("GE",d1,10))

