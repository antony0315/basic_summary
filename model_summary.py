#==========分割資料=====================================
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(feature,label,test_size=0.2)
#==========linear_model======================================
from sklearn import linear_model
import matplotlib.pyplot as plt
x_train=
y_train=
x_test=
reg_model=linear_model.LinearRegression()
reg_model.fit(x_train,y_train)
y_test_predict=reg_model.predict(x_test)

plt.scatter(x_train,y_train)
plt.scatter(x_test,y_test_predict,color='red')
plt.plot(x_test,y_test_predict.color='blue')
plt.show()

#評估
print('coefficients:',reg_model.coef_)#相關係數
print('MSE:',np.mean((reg_model.predict(x_test))**2))
#方差係數1是完美預測
print('variance score:'%reg_model.score(x_test,y_test))

#========KNN model =======================================
from sklearn.neighbors import KNeighborsClassifier
x=[[],[],[]]
y=[]
neigh=KNeighborsClassifier(n_neightbors=3)
neigh.fit(x,y)
print(neigh.predict([[]]))   #預測
print(neigh.predict_proba([[]]))  #與樣本距離
#準確率
print('準確率:',neigh.score(x_test,y_test))

#======K-means model====================================
from sklearn.cluster import Kmeans
from sklearn import metrics
x=np.array([[]])
y=[]
kmeans_model=KMeans(n_clusters=2,random_state=0)#指定兩類資料
kmeans_model=kmeans_model.fit(x)
#準確率
score=metrics.accuracy_score(y,kmeans.predict(x))
print('準確率:',.format(score))
#======decision tree model===============================
#pip3 install graphviz
#pip3 install pydot
#pip3 install python-graphviz
from sklearn import tree
x=np.array[[],[],[]]
y=np.array['','','']#分類
clf=tree.DecisionTreeClassifier()
clf=clf.fit(x,y)
prediction=clf.predict([[x_test]])
#模型視覺化
tree.export_graphviz(clf,out_file='tree.dot')#將圖片轉成dot檔
dot_data=StringIO()
tree.export_graphviz(clf,out_file=dot_data)#將dot圖片化
graph=pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_png("tree.png")#儲存tree.png

#======Random Forest Model=================================
from sklearn.esemble import RandomForestClassifier
x=
y=

Rforest=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=2)
#註:使用100組  深度10層  輸出2結果
RForesr.fit(x,y)
#評估
print(model.feature_importance_)
print(model.predict([]))
estimator=model.estimator_[5]
#視覺化
from sklearn.tree import export_graphviz
export_graphviz(estimator,out_file='tree.dot',
                feature_names=[],              #特徵名稱
                class_names=[],                #答案名稱
                rounded=True,proportion=False, #顯示比例
                precision=2,filled=True)       #精確設定
#======Bayes'theorem======================================
from sklearn.naive_bayes import GaussianNB
x=np.array([[],[],[]])
y=np.array([,,])
model=GaussianNB()
model.fit(x,y)
print(model.class_prior_)#每個分類概率
print(model.get_params())#估算工具參數
x_test=np.array([])
predicted=model.predict(x_test)
print(predicted)
print(model.predict_proba(x_test))  #對各分類機率











