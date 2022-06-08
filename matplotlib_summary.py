#===========malplotlib=======================================
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']#中文字
#設定樣式
plt.style.use('classic')
plt.style.use('seaborn-whitegrid')
%matplotlib inline     
#不須使用plt.show()
#===============範例1
import numpy as np
x=np.linspace(0,10,100)
fig=plt.figure()
plt.plot(x,np.sin(x),'-')
plt.plot(x,np.cos(x),'--')
#儲存圖片
fig.savefig('my_figure.png')
#讀取圖片
from IPython.display import Image
Image('my_figure.png')


#%%===============1+1的介面==================================
#=======maltab型式
plt.figure()    #建立 plot figure
plt.subplot(2,1,1)      #(rows,columns,panel number)
plt.plot(x,np.sin(x))
plt.subplot(2,1,2)
plt.plot(x,np.cos(x))
#======物件導向
fig,ax=plt.subplots(2)
ax[0].plot(x,np.sin(x))
ax[1].plot(x,np.cos(x))
#設定
ax.set_xlabel()
ax.set_ylabel()
ax.set_xlim()
ax.set_ylim()
ax.set_title()
#or
ax=plt.axes()
ax.plot(x,np.sin(x))
ax.set(xlim(0,10),y_lim(-2,2),xlabel='x',ylabel='sin(X)',title='A simple plot')
#======
fig=plt.figure()
ax=plt.axes()
x=np.linspace(0,10,1000)
ax.plot(x,np.sin(x))
plt.plot(x,np.cos(x))    #加線
#============================調整顏色、樣式===========
#======color=====
plt.plot(x,np.sin(x-0),color='blue')
plt.plot(x,np.sin(x-1),color='k')   #rgbcmyk
plt.plot(x,np.sin(x-2),color='0.75')
plt.plot(x,np.sin(x-3),color='#FFDD44')     #(RRGGBB  00到FF)
plt.plot(x,np.sin(x-4),color=(1.0,0.2,0.3))  #(RGB 0~1)
plt.plot(x,np.sin(x-5),color='chartreuse')
#======linestyle==
plt.figure()
plt.plot(x,x+4,linestyle='-')
plt.plot(x,x+5,linestyle='--')
plt.plot(x,x+6,linestyle='-.')
plt.plot(x,x+7,linestyle=':')
#====color+linestyle===
plt.figure()
plt.plot(x,x+4,'-g')
plt.plot(x,x+5,'--c')
plt.plot(x,x+6,'-.k')
plt.plot(x,x+7,':r')
#======================Axes範圍================
plt.figure()
plt.plot(x,np.sin(x))
plt.xlim(-1,11)
plt.ylim(-1.5,1.5);

plt.axes([-1,11,-1.5,1.5]);
plt.axes('tight')    #equal   緊密 或 均等

#x、y反向顯示 (順序調換)
plt.xlim(10,0)    
plt.ylim(1.2,-1.2)

#====================加上標籤==================
plt.figure()
plt.plot(x,np.sin(x))
plt.title("A sine curve")
plt.xlabel("x")
plt.ylabel("sin(x)")

#====================運用圖例==============
plt.plot(x,np.sin(x),'-g',label='sin(x)')
plt.plot(x,np.cos(x),':b',label='cos(x)')
plt.axis('equal')
plt.legend();


#%%==================matplotlib散佈圖=========================================================
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
x=np.linspace(0,10,30)
y=np.sin(x)
plt.plot(x,y,'o',color='black')

#散佈圖所有圖標
rng=np.random.RandomState(0)
for marker in ['o','.',',','X','+','v','^','<','>','s','d']:
    plt.plot(rng.rand(5),rng.rand(5),marker,label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0,1.8)
plt.ylim(0,1)
#=============點+線
#1
plt.figure()
plt.plot(x,y,'-ok')
#2
plt.figure()
plt.plot(x,y,'-s',color='gray',markersize=15,linewidth=4,markerfacecolor='white',
         markeredgecolor='gray',markeredgewidth=2)
plt.ylim(-1.2,1.2)
#%%===========plt.scatter畫散佈圖=================================================
plt.scatter(x,y,marker='o')

plt.figure()
rng=np.random.RandomState(1)
x=rng.randn(100)
y=rng.randn(100)
colors=rng.rand(100)
sizes=1000*rng.rand(100)  #每個點大小
plt.scatter(x,y,c=colors,s=sizes,alpha=0.4,cmap='viridis')   #alpha(透明度)
plt.colorbar()
#==========iris範例
from sklearn.datasets import load_iris
%matplotlib inline
import matplotlib.pyplot as plt
iris=load_iris()
features=iris.data.T
iris.target_names=iris.target_names.T
plt.scatter(features[0],features[1],alpha=0.4,s=100*features[3],c=iris.target,cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

#%%============誤差圖=======================================================
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
#========離散型誤差============
x=np.linspace(0,10,50)
dy=0.8
y=np.sin(x)+0.8*np.random.randn(50)
plt.errorbar(x,y,yerr=0.8,fmt='.k')

plt.errorbar(x,y,yerr=dy,fmt='o',color='black',ecolor='lightgray',elinewidth=3,
             capsize=0)

#======連續型誤差==============
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
model=lambda x:x*np.sin(x)
xdata=np.array([1,3,5,6,8])
ydata=model(xdata)
#計算Gaussian
gp =GaussianProcessRegressor()
gp.fit(xdata[:,np.newaxis],ydata)
xfit=np.linspace(0,10,1000)
yfit,std=gp.predict(xfit[:,np.newaxis],return_std=True)
dyfit=1.96*std
plt.figure()
plt.plot(xdata,ydata,'or')
plt.plot(xfit,yfit,'-',color='gray')
plt.fill_between(xfit, yfit-dyfit,yfit+dyfit,color='blue',alpha=0.4)
plt.xlim(0,10)

#%%============================直方圖===================
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
data=np.random.randn(1000)
plt.figure()
plt.hist(data);

plt.figure()
plt.hist(data,bins=30,alpha=1,histtype='stepfilled',color='steelblue',edgecolor='none');

#=======堆疊
x1=np.random.normal(0,0.8,1000)
x2=np.random.normal(-2,1,1000)
x3=np.random.normal(3,2,1000)

kwargs=dict(histtype='stepfilled',bins=40,alpha=0.8)
plt.figure()
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs);

#%%======================2D直方圖和裝箱法=============================
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
mean=[0,0]
cov=[[1,1],[1,20]]
x,y=np.random.multivariate_normal(mean,cov,10000).T
#直方
plt.figure()
plt.hist2d(x,y,bins=30,cmap='Blues')
cb=plt.colorbar()
cb.set_label('counts in bin')            #密度顏色

#六角
plt.figure()
plt.hexbin(x,y,gridsize=30,cmap='Blues')
cb=plt.colorbar(label='count in bin')       #密度顏色

#核心密度估計
from scipy.stats import gaussian_kde
data=np.vstack([x,y])
kde=gaussian_kde(data)
xgrid=np.linspace(-3.5, 3.5,40)
ygrid=np.linspace(-6,6,40)
Xgrid,Ygrid=np.meshgrid(xgrid,ygrid)
Z=kde.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
plt.figure()
plt.imshow(Z.reshape(Xgrid.shape),
           origin='lower',aspect='auto',extent=[-3.5,3.5,-6,6],
           cmap='Blues')
cb=plt.colorbar()
cb.set_label("density")
 
#%%======================自訂圖表圖例=======================================
import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np

x=np.linspace(0,10,1000)
fig,ax=plt.subplots()
ax.plot(x,np.sin(x),'-b',label='Sin')
ax.plot(x,np.cos(x),'--r',label='Cos')
ax.axis('equal')
ax.legend() 
#or       plt.legend(lines[:2],['sin','cos'])  
#可填入
#位置    loc='upper left'     'lower center'
#不要框      frameon=False       圓角框    fancybox=True
#分幾欄   ncol=2
#框透明度  framealpha=1
#陰影   shadow=True
#字與框距   borderpad=1
#title=''

#=============多重圖例
fig,ax=plt.subplots()
lines=[]
styles=['-','--','-.',':']
x=np.linspace(0,10,1000)
for i in range(4):
    lines+=ax.plot(x,np.sin(x-i*np.pi/2),styles[i], color='blue')
ax.axis('equal')
ax.legend(lines[:2],['lineA','lineB'],loc='upper right',frameon=False)
from matplotlib.legend import Legend
leg=Legend(ax,lines[2:],['line C','line D'],
           loc='lower right',frameon=False)
ax.add_artist(leg);

#%%多重子圖表
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
%matplotlib inline
import numpy as np

ax1=plt.axes()                           #母圖表
ax2=plt.axes([0.65,0.65,0.2,0.2])        #子圖表
#=======垂直堆疊
fig=plt.figure()
ax1=fig.add_axes([0.1,0.5,0.8,0.4],xticklabels=[],ylim=(-1.2,1.2)) 
 #[左右間格,上下間格,,寬度,高度]                    y軸起始值  結束值
ax2=fig.add_axes([0.1,0.1,0.8,0.4],ylim=(-1.2,1.2))
x=np.linspace(0,10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x));

#=======plt.subplot:子圖表的簡單網格
#1
plt.figure()
for i in range(1,7):
    plt.subplot(3,2,i)
    plt.text(0.5,0.5,str((3,2,i)),fontsize=18,ha='center')
#2
fig=plt.figure()
fig.subplots_adjust(hspace=0.4,wspace=0.4)   #間隔
for i in range(1,7):
    ax=fig.add_subplot(2,3,i)
    ax.text(0.5,0.5,str((2,3,i)),fontsize=20,ha='center')
#3   plt.subplots一次準備整個網格
fig,ax=plt.subplots(2,3,sharex='col',sharey='row')   #共用X,y軸
for i in range(2):
    for j in range(3):
        ax[i,j].text(0.5,0.5,str((i,j)),fontsize=18,ha='center')
fig

#======plt.GridSpec  複雜排列
grid=plt.GridSpec(2,3,wspace=0.4,hspace=0.3)
plt.subplot(grid[0,0])   #左上   0,0
plt.subplot(grid[0,1:])  #右上  0,1~2
plt.subplot(grid[1,:2])  #左下  1,0~1
plt.subplot(grid[1,2])   #右下  1,2

#=========
mean=[0,0]
cov=[[1,1],[1,2]]   #變異數矩陣
x, y=np.random.multivariate_normal(mean,cov,3000).T
fig=plt.figure(figsize=(6,6))
grid=plt.GridSpec(4,4,hspace=0.2,wspace=0.2)
main_ax=fig.add_subplot(grid[:-1,1:])
y_hist=fig.add_subplot(grid[:-1,0],xticklabels=[],sharey=main_ax)
x_hist=fig.add_subplot(grid[-1,1:],yticklabels=[],sharex=main_ax)
main_ax.plot(x,y,'ok',markersize=3,alpha=0.2)

x_hist.hist(x,40,histtype='stepfilled',orientation='vertical',color='gray')
x_hist.invert_yaxis()   #Y軸上下反轉
y_hist.hist(y,40,histtype='stepfilled',orientation='horizontal',color='gray')
y_hist.invert_xaxis()   #X軸左右反轉

#%%文字+註解
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-whitegrid')
import numpy as np
import matplotlib as mpl

births=pd.read_csv('births.csv')
quartiles=np.percentile(births['births'],[25,50,75])
mu,sig=quartiles[1],0.74*(quartiles[2]-quartiles[0])
births=births.query('(births>@mu-5*@sig)&(births<@mu+5*@sig)')
births['day']=births['day'].astype(int)
births.index=pd.to_datetime(10000*births.year+100*births.month+births.day,format='%Y%m%d')
births_by_date=births.pivot_table('births',[births.index.month,births.index.day])    #依照月份、日期加總
births_by_date.index=[pd.datetime(2012,month,day)for(month,day) in births_by_date.index]
#繪圖
fig,ax=plt.subplots(figsize=(12,4))
births_by_date.plot(ax=ax)
#加上文字、註解  
style=dict(size=10,color='gray')
ax.text('2012-1-1',3950,"Nee Year's Day",**style)
ax.text('2012-7-4',4250,"Independence Day",**style)
ax.text('2012-12-25',3850,"Christmax",**style)

#加上標籤
ax.set(title='USA births by day',ylabel='average daily birth')
#月份標籤致中
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))

#=======註解位置表達方式
#依照資料位置
#ax.text(1,5,"註解",transform=ax.transData)       #資料座標
#依照圖表位置
#ax.text(0.5,0.1,"註解",transform=ax.transAxes)    #axes尺寸單位
#ax.text(0.5,0.1,"註解",transform=fig.transFigure)#figure尺寸單位

#%%  箭頭&註解
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
%matplotlib inline
import numpy as np

fig,ax=plt.subplots()
x=np.linspace(0,20,1000)
ax.plot(x,np.cos(x))
ax.axis('equal')
ax.annotate('local maximum',xy=(6.28,1),xytext=(10,4),arrowprops=dict(facecolor='black',shrink=0.05))
#xy=標示的點    xytext文字位置
ax.annotate('local minimum',xy=(5*np.pi,-1),xytext=(2,-6),arrowprops=dict(arrowstyle='->',connectionstyle="angle3,angleA=0,angleB=-90"))
#箭頭角度

#%%
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-whitegrid')
import numpy as np
import matplotlib as mpl

births=pd.read_csv('births.csv')
quartiles=np.percentile(births['births'],[25,50,75])
mu,sig=quartiles[1],0.74*(quartiles[2]-quartiles[0])
births=births.query('(births>@mu-5*@sig)&(births<@mu+5*@sig)')
births['day']=births['day'].astype(int)
births.index=pd.to_datetime(10000*births.year+100*births.month+births.day,format='%Y%m%d')
births_by_date=births.pivot_table('births',[births.index.month,births.index.day])    #依照月份、日期加總
births_by_date.index=[pd.datetime(2012,month,day)for(month,day) in births_by_date.index]

fig,ax=plt.subplots(figsize=(12,4))
births_by_date.plot(ax=ax)
#曲線箭頭標示
ax.annotate("New Years day",xy=('2012-1-1',4100),xycoords='data',xytext=(50,-30),
            textcoords='offset points',arrowprops=dict(arrowstyle="->",
                                                       connectionstyle="arc3,rad=-0.2"))
#直線箭頭+框框
ax.annotate("Independence Day",xy=('2012-7-4',4250),xycoords='data',bbox=dict(boxstyle="round",fc="none",ec="gray"),xytext=(10,-40),
            textcoords='offset points',ha='center',arrowprops=dict(arrowstyle="->"))

#標示+x軸區間
ax.annotate('Labor Day',xy=('2012-9-4',4850),xycoords='data',ha='center',xytext=(0,-20),textcoords='offset points')
ax.annotate('',xy=('2012-9-1',4850),xytext=('2012-9-7',4850),xycoords='data',textcoords='data',arrowprops={'arrowstyle':'|-|,widthA=0.2,widthB=0.2'})

#箭頭粗到細
ax.annotate('Halloween',xy=('2012-10-31',4600),xycoords='data',
            xytext=(-80,-40),textcoords='offset points',
            arrowprops=dict(arrowstyle="fancy",fc="0.6",ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"))
#黑框+箭頭
ax.annotate('Thanksgiving',xy=('2012-11-25', 4500), xycoords='data',
            xytext=(-120,-60),
            textcoords='offset points',
            bbox=dict(boxstyle="round4,pad=.5",fc="0.9"),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=80,rad=20"))
#對話框
ax.annotate('Christmas',xy=('2012-12-25',3850),xycoords='data',
            xytext=(-30,0),textcoords='offset points',
            size=13,ha='right',va="center",
            bbox=dict(boxstyle="round",alpha=0.1),
            arrowprops=dict(arrowstyle="wedge,tail_width=0.5",alpha=0.1))
#=====加上標題
ax.set(title='USA births by day of year(1969-1988)',ylabel='average daily births')

#=====月份座標軸
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())#日期坐標軸
ax.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())#格式
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%h'))
ax.set_ylim(3600,5400);
#%%自訂刻度
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('seaborn-whitegrid')
import numpy as np

ax=plt.axes(xscale='log',yscale='log')
print(ax.xaxis.get_major_locator())   #主要刻度
print(ax.xaxis.get_minor_locator())   #次要刻度
print(ax.xaxis.get_major_formatter())  #主要標籤
print(ax.xaxis.get_minor_formatter())  #次要標籤
#=====隱藏刻度、標籤
ax2=plt.axes()
ax2.plot(np.random.rand(50))
ax2.yaxis.set_major_locator(plt.NullLocator())
ax2.xaxis.set_major_formatter(plt.NullFormatter())
#-----減少刻度
fig,ax=plt.subplots(4,4,sharex=True,sharey=True)
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(5))
    axi.yaxis.set_major_locator(plt.MaxNLocator(5))
fig
#-----自訂刻度
fig,ax=plt.subplots()
x=np.linspace(0,3*np.pi,1000)
ax.plot(x,np.sin(x),lw=3,label='sin')
ax.plot(x,np.cos(x),lw=3,label='cos')
ax.grid(True)
ax.legend(frameon=False)
ax.axis('equal')
ax.set_xlim(0,3*np.pi)

ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi/4))
fig
#NullLocator  FixedLocator  IndexLocator  LogLocator  MultipleLocator MaxNLocator AutoLocator  AutoMinorLocator
#NullFormatter  IndexFormatter....

#%%手動自訂
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('classic')
import numpy as np

x=np.random.randn(1000)
#使用回色背景
ax=plt.axes(facecolor='#E6E6E6')
ax.set_axisbelow(True)
#白色實心線
plt.grid(color='w',linestyle='solid')
#隱藏軸小突起
for spine in ax.spines.values():
    spine.set_visible(False)
#隱藏上面與右側刻度
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
#淡化刻度標籤
ax.tick_params(color='gray',direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')
#直方圖
ax.hist(x,edgecolor='#E6E6E6',color='#BB7777');


#======預設rc參數
Ipython_default=plt.rcParams.copy()
from matplotlib import cycler
colors=cycler('color',
              ['#EE6666','#3388BB','#9988DD',
               '#EECC55','#88BB44','#FFBBBB'])
plt.rc('axes',facecolor='#E6E6E6',edgecolor='none',
       axisbelow=True,grid=True,prop_cycle=colors)
plt.rc('grid',color='w',linestyle='solid')
plt.rc('xtick',direction='out',color='gray')
plt.rc('ytick',direction='out',color='gray')
plt.rc('patch',edgecolor='E6E6E6')
plt.rc('lines',linewidth=2)
#畫圖
plt.hist(x);
#直線圖
plt.figure()
for i in range(4):
    plt.plot(np.random.rand(10))


#%%雷達圖
freshman=pd.Series({'會計學':70,'體育':78,'國文':73,'經濟學':68,'創意思考':86,'多元文化':81,'APPS程式設計':96,'英文':60,'國防':89,'公平交易法':76})

angle1=np.linspace(0,2*np.pi,len(freshman),endpoint=False)
angle=np.concatenate((angle1,[angle1[0]]))
data=np.concatenate((freshman,[freshman[0]]))

fig=plt.figure(dpi=800)
ax=fig.add_subplot(121,polar=True)
ax.plot(angle,data,'bo-',linewidth=1)
ax.set_thetagrids(angle1*180/np.pi,np.array(freshman.index))



























































