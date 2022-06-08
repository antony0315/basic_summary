import numpy as np
import pandas as pd
train_x=pd.read_csv('C:/Users/user/Desktop/sample-data/train.csv')
test_x=pd.read_csv('C:/Users/user/Desktop/sample-data/test.csv')

#%%One-hot-encoding
cat_cols=['sex','product']
all_x=pd.concat([train_x,test_x])
all_x=pd.get_dummies(all_x,columns=cat_cols)
train_x=all_x.iloc[:train_x.shape[0],:].reset_index(drop=True)
test_x=all_x.iloc[train_x.shape[0]:,:].reset_index(drop=True)

#===============方法2
from sklearn.preprocessing import OneHotEncoder
cat_cols=['sex','product']
#建立one-hot物件
ohe=OneHotEncoder(sparse=False,categories='auto')
ohe.fit(train_x[cat_cols])
#建立one hot 欄位
columns=[]
for i,c in enumerate(cat_cols):
    columns+=[f'{c}_{v}' for v in ohe.categories_[i]]
#轉換後轉成dataframe並合併
dummy_vals_train=pd.DataFrame(ohe.transform(train_x[cat_cols]),columns=columns)
dummy_vals_test=pd.DataFrame(ohe.transform(test_x[cat_cols]),columns=columns)
train_x=pd.concat([train_x.drop(cat_cols,axis=1),dummy_vals_train],axis=1)
test_x=pd.concat([test_x.drop(cat_cols,axis=1),dummy_vals_train],axis=1)
#%%Label encoding GBDT使用
from sklearn.preprocessing import LabelEncoder
cat_cols=['sex','product']
for c in cat_cols:
    le=LabelEncoder()
    le.fit(train_x[c])
    train_x[c]=le.transform(train_x[c])
    test_x[c]=le.transform(test_x[c])

#%%Feature hashing
from sklearn.feature_extraction import FeatureHasher
cat_cols=['sex','product']
for c in cat_cols:
    fh=FeatureHasher(n_features=5,input_type='string')
    hash_train=fh.transform(train_x[[c]].astype(str).values)
    hash_test=fh.transform(test_x[[c]].astype(str).values)
    hash_train=pd.DataFrame(hash_train.todense(),columns=[f'{c}_{i}' for i in range(5)])
    hash_test=pd.DataFrame(hash_test.todense(),columns=[f'{c}_{i}' for i in range(5)])
    train_x=pd.concat([train_x,hash_train],axis=1)
    test_x=pd.concat([test_x,hash_test],axis=1)

train_x.drop(cat_cols,axis=1,inplace=True)
test_x.drop(cat_cols,axis=1,inplace=True)

    

#%%frequency encoding  =label encoding依照頻率排序  *注意相同頻率問題
cat_cols=['sex','product']
for c in cat_cols:
    freq=train_x[c].value_counts()
    train_x[c]=train_x[c].map(freq)
    test_x[c]=test_x[c].map(freq)
















    
