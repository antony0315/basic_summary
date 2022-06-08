import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df=pd.read_csv("C:/Users/anton/OneDrive/桌面/learn_python/summary/LSTM/international-airline-passengers.csv",usecols=[1])
dataset=df.values
dataset=dataset.astype('float32')
plt.figure()
plt.plot(dataset,color='blue')
#正規化
scaler=MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(dataset)
#分稱訓練、測試
train_size=int(len(dataset)*0.67)
test_size=len(dataset)-train_size
train,test=dataset[0:train_size,:],dataset[train_size:len(dataset),:]

#製造label Y為x值下一個
def create_dataset(dataset,look_back=1):
    dataX, dataY=[],[]
    for i in range(len(dataset)-look_back-1):
        a=dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0])
    return np.array(dataX),np.array(dataY)
look_back=1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX=np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX=np.reshape(testX,(testX.shape[0],1,testX.shape[1]))

#建模、編譯、訓練
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(4,input_shape=(1,look_back)))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainX,trainY,epochs=100,batch_size=1,verbose=2)

#預測
trainxpred=model.predict(trainX)
testxpred=model.predict(testX)
#轉回原始資料
trainxpred=scaler.inverse_transform(trainxpred)
trainY=scaler.inverse_transform([trainY])
testxpred=scaler.inverse_transform(testxpred)
testY=scaler.inverse_transform([testY])

#均方誤差
trainscore=math.sqrt(mean_squared_error(trainY[0],trainxpred[:,0]))
print('train score: %.2f RMSE'%(trainscore))
testscore=math.sqrt(mean_squared_error(testY[0],testxpred[:,0]))
print('test score:%.2f RMSE'%(testscore))

#圖示化
trainxpredplot=np.empty_like(dataset)
trainxpredplot[:,:]=np.nan
trainxpredplot[look_back:len(trainxpred)+look_back,:]=trainxpred

testxpredplot=np.empty_like(dataset)
testxpredplot[:,:]=np.nan
testxpredplot[len(trainxpred)+(look_back*2)+1:len(dataset)-1,:]=testxpred

plt.figure()
plt.plot(scaler.inverse_transform(dataset),color='blue')
plt.plot(trainxpredplot,color='orange')
plt.plot(testxpredplot,color='green')













