#%%           MLP_model
import tensorflow as tf
#======one-hot encoding
y_train=tf.keras.utils.to_categorical(y_train,num_classes=2)
#======建模
model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=10,activation=tf.nn.relu,input_dim=1),
    #10個神經元      1個feature
    tf.keras.layers.Dense(units=10,activation=tf.nn.relu),
    tf.keras.layers.Dense(units=2,activation=tf.nn.softmax)
    #2元回歸    使用softmax計算
    ])

# 寫法2==================================================================
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# model2=tf.keras.models.Sequential()
# model2.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu,input_dim=1))
# model2.add(tf.keras.layers.Dense(units=10,activation=tf.nn.relu))
# model2.add(tf.keras.layers.Dense(units=2,activation=tf.nn.softmax))  #2種答案
# 
# 
# 
# =============================================================================
# 回歸問題=============================================================================
# learning_rate=0.0001
# opt1=tf.keras.optimizers.Nadam(lr=learning_rate)
# model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(320,activation='relu',input_shape=[x_train.shape[1]]))
# model.add(tf.keras.layers.Dense(units=640,activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(1))
# model.compile(loss='mse',optimizer=opt1,metrics=['mae'])
# history=model.fit(x_train,y_train,epochs=10000,batch_size=len(y_train))
# =============================================================================
#=======編譯
model.compile(optimizer='adam',          #adam最佳化
              loss='sparse_categorical_crossentropy',     #損失處理方法
              #若使用one-hot    改為tf.keras.losses.categorical_crossentropy
              metrics=['accuracy'])                       #設定編譯處理以正確率為主
#=======訓練
history=model.fit(x_train,y_train,epochs=20,batch_size=128)        #訓練次數    訓練比數
#========評估
score=model.evaluate(x_test,y_test,batch_size=128)
print('score:',score)           #score:[loss損失率,正確率]

#========預測
predict=model.predict(x_test)
print('predict:',predict)

predict2=model.predict_classes(x_test)
print('predict_classes:',predict2)

#%%參數
#======= 激勵函數(Sigmoid   TanH     ReLU)


#======最佳化    optimizer=  最短路徑演算法
#sgd:      tf.keras.optimizers.SGD(lr=0.01,clipnorm=1.)
#RMSprop:  tf.keras.optimizer.RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0)
#adagrad:  tf.keras.optimizers.Adagrad(lr=0.01,epsilon=None,decay=0.0)
#Adadelta: tf.keras.optimizers.Adadelta(lr=1.0,rho=0.95,epsilon=None,decay=0.0)
#Adam:     tf.keras.optimizers.Adan(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0)
#Adamax:   tf.keras.optimizers.Adamax(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0)
#Nadam:    tf.keras.optimizers.Nadam(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=None,schedule_decay=0.004)
#%%
#訓練過程圖像化
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('acc&loss')
plt.xlabel('epoch')
plt.legend(['acc','loss'],loc='upper left')

#%%tensorboard
from time import time
from tensorflow.keras,callbacks import TensorBoard

history=model.fit(x_train,y_train,epochs=20,batch_size=128,callbacks=[tensorboard],verbose=1)
#回傳至tensorboard      處理時顯示精簡資訊
#透過Dos-mode 或 terminal 移動到.py位置   執行後產生logs文件夾
#執行    tensorboard --logdir=logs/
#%%保存權重
with open("model.json","w") as json_file:
    json_file.write(model.to_json())          #存取模型
model.save_weights("model.h5")                #存取權重

#%%讀取模型
from tensorflow.keras.models import model_from_json
json_file=open('model.json','r')
loaded_model_json=json_file.read()
json.file.close()
model=model_from_json(loaded_model_json)
model.load_weights("model.h5")

model.compile(........)





#%%CNN
model=tf.keras.Squential()
model.add(tf.keras.layers.Conv2D(
    filters=3,           #三個輸出
    kernel_size=(3,3),  #卷積層濾鏡大小
    padding="same",     #相同大小  or valid
    activation='relu',   
    input_shape(28,28,1))#原圖片大小

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))#寬高縮小一半

model.add(tf.keras.layers.Conv2D(filters=9,kernel_size=(2,2),
                                 padding="same",activation='relu'))#相同大小

model.add(tf.keras.layers.Dropout(rate=0.33))#丟掉1/3的圖
model.add(tf.keras.layers.Flatten())     #2D影像轉1D
model.add(tf.keras.layers.Dense(10,activation='relu'))#使用MLP類神經
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))#10種答案


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])
history=model.fit(x_train,y_train2,batchsize=1000,epochs=200,verbose=1)
score=model.evaluate(x_test,y_test2,batch_size=128)
print(score)  #準確率
predict=model.predict_classes(x_test[:10])#預測結果
print(predict2[:10])


