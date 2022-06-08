#=============交叉驗證============================
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
score=[]
kf=KFold(n_splits=4,shuffle=True,random_state=71)
for tr_idx,va_idx kf.split(train_x):        #分割訓練/驗證
    tr_x,va_x=train_x.iloc[tr_idx],train_x.iloc[va_idx]
    tr_y,va_y=train_y.iloc[tr_idx],train_y.iloc[va_idx]
    model=Model(params)
    model.fit(tr_X,tr_y)
    va_pred=model.predict(va_x)
    score=log_loss(va_y,va_pred)
    scores.append(score)
print(f'logloss:{np.mean(scores):.4f}')
#===========xgboost================================
import xgboost as xgb
from sklearn.metrics import log_loss
#特徵&標籤 轉換為Xgboost 的資料結構
dtrain=xgb.DMatrix(tr_x,label=ty_y)
dvalid=xgb.DMatix(va_x,label=va_y)
dtest=xgb.DMatix(test_x)
#設定超參數 
params={'objective':'binary:logistic','silent':1,'random_state':71}
        #目標函數: #回歸任務:'reg:squarederror'    最小化MSE
                  #二元分類:'binary:logistic'     最小化logloss
                  #多元分類:'multi:softprob'      最小化multi-class logloss
num_round=50
#在watchlist中組合訓練與驗證資料
watchlist=[(dtrain,'train'),(dvaild,'eval')]
#訓練模型，並監控分數變化
model=xgb.train(params,dtrain,num_round,evals=watchlist)
#計算valid的logloss分數
va_pred=model.predict(dvalid)
score=log_loss(va_y,va_pred)
print(f'logloss:{score:.4f}')
#對test_y預測   輸出值為機率
pred=model.predict(dtest)