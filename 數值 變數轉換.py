#%%     standard scaler
import matplotlib.pylab as plt
%matplotlib inline
import pandas as pd
train=pd.read_csv('C:/Users/user/Desktop/sample-data/train.csv',na_values=['','NA',-1,9999])

from sklearn.preprocessing import StandardScaler
train_x=train.drop('target',1)
num_cols=['age','height','weight','amount','medical_info_a1','medical_info_a2',
          'medical_info_a3','medical_info_b1']

scaler=StandardScaler()
scaler.fit(train_x[num_cols])
train_x[num_cols]=scaler.transform(train_x[num_cols])
#%%       Min-Max scaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
train=pd.read_csv('C:/Users/user/Desktop/sample-data/train.csv',na_values=['','NA',-1,9999])
train_x=train.drop('target',1)
num_cols=['age','height','weight','amount','medical_info_a1','medical_info_a2',
          'medical_info_a3','medical_info_b1']

scaler=MinMaxScaler()
scaler.fit(train_x[num_cols])
train_x[num_cols]=scaler.transform(train_x[num_cols])

#%%非線性轉換 log   (長尾特徵時)
import numpy as np
#取對數
np.log(x)
#加1後取對數(數值為0時)
np.log1p(x)
#取絕對值 再取log 再加上原本正負號(數值有負時)
np.sign(x)*np.log(np.abs(x))
#%%BOX-COX轉換   (數值必須>0)  轉換為常態分配
pos_cols=[c for c in num_cols if (train_x[c]>0.0).all() and (test_x[c]>)0.all()]
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer(method='box-cox')
pt.fit(train_x[pos_cols])
train_x[pos_cols]=pt.transform(train_x[pos_cols])
#YEO-Johnson轉換   (可處理負值)
from sklearn.preprocessing import PowerTransformer

pt=PowerTransformer(method='yeo-johnson')
pt.fit(train_x[num_cols])
train_x[pos_cols]=pt.transform(train_x[pos_cols])

#%%Clipping   限制上下界  1% 99%quantile
p01=train_x[num_cols].quantile(0.01)
p99=train_x[num_cols].quantile(0.99)

train_x[num_cols]=train_x[num_cols].clip(p01,p99,axis=1)

for i in num_cols:
    plt.figure()
    plt.hist(train_x[i],bins=100)
#%%Binning分組    分成幾個區間
x=np.random.random(100)
binned=pd.cut(x,3,labels=False)
print(binned)

bin_edge=[-float('inf'),3.0,5.0,float('inf')]
binned=pd.cut(x,bin_edge,labels=False)
print(binned)


#%%數值轉換為排序
x=[10,20,30,0,40,40]
#排序方式1
rank=pd.Series(x).rank()
print(rank.values)
#排序方式2
order=np.argsort(x)
rank2=np.argsort(order)

#%%rankgauss  排序後強制轉換高斯分布
from sklearn.preprocessing import QuantileTransformer
transformer=QuantileTransformer(n_quantiles=100,random_state=0,
                                output_distribution='normal')
transformer.fit(train_x[num_cols])
train_x[num_cols]=transformer.transform(train_x[num_cols])
test_x[num_cols]=transformer.transform(test_x[num_cols])


#%%One-hot-encoding
all_x=pd.concat([train_x,test_x])
all_x=pd.get_dummies(all_x,columns=cat_cols)
train_x=all_x.iloc[:train_x.shape[0],:].reset_index(drop=True)
test_x=all_x.iloc[train_x.shape[0]:,:].reset_index(drop=True)

























