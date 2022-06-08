
#%%迴歸評價指標
#RMSE (均方根誤差)
from sklearn.metrics import mean_squared_error
import numpy as np
y_true=[1.0,1.5,2.0,1.2,1.8]
y_pred=[0.8,1.5,1.8,1.3,2.0]
mse=mean_squared_error(y_true,y_pred)   #mse 均方誤差
rmse=np.sqrt(mse)
print('rmse:',rmse)

#RMSLE(均方根對數誤差)
from sklearn.metrics import mean_squared_log_error
msle=mean_squared_error(y_true, y_pred)
rmsle=np.sqrt(msle)
print('rmsle:',rmsle)

#MAE (平均絕對值誤差)
from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_true, y_pred)
print('mae:',mae)

#R-squar(決定係數)
from sklearn.metrics import r2_score
r2=r2_score(y_true,y_pred)
print('r2:',r2)

#%%二元分類    預設值:正例、負例
#=========混淆矩陣
y_true=[1,0,1,1,0,1,1,0]
y_pred=[0,0,1,1,0,0,1,1]
#方法1
tp=np.sum((np.array(y_true)==1)&(np.array(y_pred)==1))
tn=np.sum((np.array(y_true)==0)&(np.array(y_pred)==0))
fp=np.sum((np.array(y_true)==0)&(np.array(y_pred)==1))
fn=np.sum((np.array(y_true)==1)&(np.array(y_pred)==0))

confusion_matrix=np.array([[tp,fp],
                            [fn,tn]])
print('混淆矩陣:\n',confusion_matrix)
#方法2
from sklearn.metrics import confusion_matrix
confusion_matix2=confusion_matrix(y_true,y_pred)
print('混淆矩陣:\n',confusion_matix2)     #[[tn,fp],
                                          #[fn,tn]]

#accuracy & error rate
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_true,y_pred)
error_rate=1-accuracy
print('accuracy:',accuracy)
print('error_rate:',error_rate)

#precision & recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
precision=precision_score(y_true,y_pred)
recall=recall_score(y_true,y_pred)
print('precision:',precision)
print('recall:',recall)

#F1-score Fbeta-score    (F2:beta=2,F3:beta=3)
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
f1=f1_score(y_true,y_pred)
f2=fbeta_score(y_true, y_pred, 2)
print('f1:%s'%f1)
print('fbeta:%f'%f2)

#MCC   不均衡資料
from sklearn.metrics import matthews_corrcoef
mcc=matthews_corrcoef(y_true, y_pred)
print('mcc:%f'%mcc)


#%%二元分類:預測值為正例
#logloss
from sklearn.metrics import log_loss
y_true=[1,0,1,0,1,1]
y_pred=[0.8,0.3,0.9,0,1,0.7]
logloss=log_loss(y_true,y_pred)
print('logloss:',logloss)

#AUC
from sklearn.metrics import roc_auc_score
auc=roc_auc_score(y_true,y_pred)
print('auc:',auc)

#%%多元分類
#multi_class_logloss
import numpy as np
from sklearn.metrics import log_loss
y_true=np.array([0,2,1,2,2])
y_pred=np.array([[0.68,0.32,0.00],
                 [0.00,0.00,1.00],
                 [0.60,0.40,0.00],
                 [0.00,0.00,1.00],
                 [0.28,0.12,0.60]])
multi_class=log_loss(y_true,y_pred)
print('multi_class:',multi_class)

#mean_f1 macro_f1 micro_f1
from sklearn.metrics import f1_score
y_true=np.array([[1,1,0],
                 [1,0,0],
                 [1,1,1],
                 [0,1,1],
                 [0,0,1]])
y_pred=np.array([[1,0,1],
                 [0,1,0],
                 [1,0,1],
                 [0,0,1],
                 [0,0,1]])
#mean_f1
mean_f1=np.mean([f1_score(y_true[i,:],y_pred[i,:])for i in range(len(y_true))])
#macro_f1
n_class=3
macro_f1=np.mean([f1_score(y_true[:,c],y_pred[:,c])for c in range(n_class)])
#micro_f1
micro_f1=f1_score(y_true.reshape(-1),y_pred.reshape(-1))
print(mean_f1,macro_f1,micro_f1)

#方法2
mean_f1=f1_score(y_true,y_pred,average='samples')
macro_f1=f1_score(y_true,y_pred,average='macro')
micro_f1=f1_score(y_true,y_pred,average='micro')
print(mean_f1,macro_f1,micro_f1)

#%%quadratic weighted kappa  有序分類
from sklearn.metrics import confusion_matrix,cohen_kappa_score
def quadratic_weighted_kappa(c_matrix):
    numer=0.0
    denom=0.0
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            n=c_matrix.shape[0]
            wij=((i-j)**2.0)
            oij=c_matrix[i,j]
            eij=c_matrix[i,:].sum()*c_matrix[:,j].sum()/c_matrix.sum()
            numer+=wij*oij
            denom+=wij*eij
    return 1.0-(numer/denom)
y_true=[1,2,3,4,3]
y_pred=[2,2,4,4,5]
c_matrix=confusion_matrix(y_true,y_pred,labels=[1,2,3,4,5])
kappa=quadratic_weighted_kappa(c_matrix)
print('kappa:',kappa)

#方法2
kappa2=cohen_kappa_score(y_true, y_pred,weights='quadratic')
print(kappa2)

#%%推薦任務指標:MAP@K 推薦前K個值，正確率
#k=3,資料5筆,分4類
import numpy as np
k=3
y_true=[[1,2],[1,2],[4],[1,2,3,4],[3,4]]
y_pred=[[1,2,4],[4,1,2],[1,4,3],[1,2,3],[1,2,4]]

def apk(y_i_true,y_i_pred):
    assert(len(y_i_pred)<=k)                  #assert偵測錯誤          
    assert(len(np.unique(y_i_pred))==len(y_i_pred))
    sum_precision=0.0
    num_hits=0.0
    for i,p in enumerate(y_i_pred):
        if p in y_i_true:
            num_hits+=1
            precision=num_hits/(i+1)
            sum_precision+=precision
    return sum_precision/min(len(y_i_true),k)

def mapk(y_true,y_pred):
    return np.mean([apk(y_i_true,y_i_pred)for y_i_true,y_i_pred in zip(y_true,y_pred)])
print('mapk:',mapk(y_true,y_pred))








































