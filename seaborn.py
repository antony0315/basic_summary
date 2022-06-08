import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
%matplotlib inline
sns.set(rc={"figure.dpi":800, 'savefig.dpi':300})

sns.set()
data=np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)
data=pd.DataFrame(data,columns=['x','y'])
#密度圖
for col in 'xy':
    sns.kdeplot(data[col],shade=True)
#直方圖
sns.distplot(data['x'])
sns.distplot(data['y'])
#等高線圖
sns.kdeplot(data['x'],data['y'])
with sns.axes_style('white'):
    sns.jointplot("x","y",data,kind='kde')
#蜂巢圖
with sns.axes_style('white'):
    sns.jointplot("x","y",data,kind='hex')

#成對圖表
#分布圖
iris=sns.load_dataset('iris')
iris.head()
sns.pairplot(iris,hue='species',size=2.5)#hue=分類
#多面相直方圖
tips=sns.load_dataset('tips')
tips['tip_pct']=100*tips['tip']/tips['total_bill']
grid=sns.FacetGrid(tips,row='sex',col='time',margin_titles=True)
grid.map(plt.hist,"tip_pct",bins=np.linspace(0,40,15))
#boxplot因素圖
with sns.axes_style(style='ticks'):
    g=sns.factorplot('day','total_bill','sex',data=tips,kind='box')
    g.set_axis_labels('day','total_bill')
#聯合分佈
with sns.axes_style('white'):
    sns.jointplot('total_bill','tip',data=tips,kind='hex')

#回歸
with sns.axes_style('white'):
    sns.jointplot('total_bill','tip',data=tips,kind='reg')
#條狀圖
planets=sns.load_dataset('planets')
with sns.axes_style('white'):
    g=sns.factorplot('year',data=planets,aspect=2,
                     kind='count',color='steelblue')#count聚合計算
    g.set_xticklabels(step=5)

#分組條狀圖
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
with sns.axes_style('white'):
    g=sns.factorplot('year',data=planets,aspect=4,kind='count',
                     hue='method',order=range(2001,2015))
    g.set_ylabels('發現星星數')













