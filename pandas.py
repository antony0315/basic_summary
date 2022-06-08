import pandas as pd
#讀csv檔
pd.read_csv('.csv')
#刪除欄
data.drop('',1)
#====================series==============================
data=pd.Series([0.25,0.5,0.71,1])
print(data.index)
print(data[1])
data1=pd.Series([0.25,0.5,0.75,1],index=['a','b','c','d'])
print(data1['b'])
#series dictionary
population_dict={'california':38332521,
                 'texas':26448193,
                 'new york':19651127,
                 'florida':19552860}
population=pd.Series(population_dict)

#==================DataFrame===================================
population_dict={'california':38332521,
                 'texas':26448193,
                 'new york':19651127,
                 'florida':19552860}
population=pd.Series(population_dict)
area_dict={'california':423967,
                 'texas':170312,
                 'new york':141297,
                 'florida':695662}
area=pd.Series(area_dict)
states=pd.DataFrame({'population':population,
                     'area':area})
states.index
states.columns
states['area']
#建構DataFrame
#series->dataframe
pd.DataFrame(population,columns=['population'])

#list->dataframe
data2=[{'a':1,'b':2*i}for i in range(3)]
pd.DataFrame(data2)

pd.DataFrame([{'a':1,'b':2},{'b':3,'c':4}])

#numpy->dataframe
pd.DataFrame(np.random.rand(3,2),
             columns=['foo','bar'],
             index=['a','b','c'])
#===================Index物件  不可修改的陣列========================
ind=pd.Index([1,2,3,4,5])
ind[1]   #out 2
ind[::2] #out1,3,5
#集合
indA=pd.Index([1,3,5,7,9])
indB=pd.Index([2,3,5,7,11])
    #交集
    indA&indB
    #聯集
    indA|indB
    #對等差分
    indA^indB

#==================Series中選擇資料===============================
data3=pd.Series([0.25,0.5,0.75,1],
                index=['a','b','c','d'])
data['b']
#查詢索引
data3.keys()
#切片
data3['a':'c']
#隱含索引
data3[0:2]
#遮罩
data3[(data3>0.3)&(data3<0.8)]
#fancy索引
data3[['a','e']]
#======loc、iloc、ix=======
data4=pd.Series(['a','b','c'],index=[1,3,5])
#loc明確的索引
data4.loc[3]       #out b
data4.loc[1:3]     #out a b
#iloc python型態索引
data4.iloc[0]     #out a
data4.iloc[1:3]   #out b、c
#===============DataFrame中選擇資料================================
population=pd.Series({'california':38332521,
                 'texas':26448193,
                 'new york':19651127,
                 'florida':19552860})
area=pd.Series({'california':423967,
                 'texas':170312,
                 'new york':141297,
                 'florida':695662})
states=pd.DataFrame({'population':population,
                     'area':area})
#=========透過欄位名查詢
states['area']
states.area
#新增欄位
states['density']=states['population']/states['area']
#=========當作二微陣列查詢
states.values
#轉置
states.T
#index查詢
states.values[0]
#=========loc、iloc、ix
states.iloc[:3,:2]   #[欄,列],直欄橫列
states.loc[:'texas',:'area']
states.ix[:3,:'area']
#遮罩
states.loc[states.density>100,['population','area']]


#%%==========================pandas中操作資料=====================================
#====Series=======
A=pd.Series([2,4,6],index=[0,1,2])
B=pd.Series([1,3,5],index=[1,2,3])
#A+B
A.add(B,fill_value=0)
#====dataframe====
fill=A.stack().mean()
A.add(B,fill_value=fill)
#逐欄運算
df.substract(df['欄位名稱columns'],axis=0)
#+
add()
#-
sub(),substract()
#*
mul(),multiply()
#/
truediv(),div(),divide()
#//
floordiv()
#%
mod()
#**
pow()
#%%=====================處裡缺失===================================================
isnull()
notnull()
dropna()
fillna()
#============dropna()
#移除na 整列
data.dropna()
#移除na 整欄
data.dropna(axis='columns')
#整欄都是缺失值才移除
data.dropna(axis='columns',how='all')
#thresh=3,至少3個不是缺失值才保留
data.dropna(axis='columns',thresh=3)
#============fillna()
data.fillna(0)
#填入前一個值
data.fillna(method='ffill')
#填入下一個值
data.fillna(method='bfill')
#填入上一欄的值
data.fillna(method='ffill',axis=1)

#%%===================階層索引===========================================================
index=[('california',2000),('california',2010),('new york',2000),('new york',2010)]
population=[33871648,37253956,
            18976457,19378102]
#pandas multiIndex
index=pd.MultiIndex.from_tuples(index)
pop=pd.Series(population,index=index)
#多重索引 當作額外 維度
pop_df=pop.unstack()
#多維度 更改為 多重索引
pop_df.stack()
#新增維度
pop_df=pd.DataFrame({'total':pop,
                     'under18':[9267089,9284094,
                                4687374,4318033,
                                5906301,6879014]})
#建構MultiIndex的方法
#1
pd.MultiIndex.from_array([['a','a','b','b'],[1,2,1,2]])
#2
pd.MultiIndex.from_tuples([('a',1),('a',2),('b',1),('b',2)])
#3
pd.MultiIndex.from_product([['a','b'],[1,2]])
#4
pd.MultiIndex(levels=[['a','b'],[1,2]],
              labels=[[0,0,1,1],[0,1,0,1]])
#索引名稱
pop.index.names=['state','year']
#=========欄的multiIndex,
index=pd.MultiIndex.from_product([[2013,2014],[1,2]],names=['year','visit'])
columns=pd.MultiIndex.from_product(['A','B','C',['a','b']],names=['subject','type'])
#Series索引選取
data[['A','B']]
data.loc[]
data.iloc[]
#DataFrame選取
data.iloc[:2,:1]
data.loc[:,('A','B')]
#
idx=pd.IndexSlice
data.loc[idx[:,1],idx[:,'B']]

#=========索引排序
data=data.sort_index()
#stacking()、unstacking()
data.unstack(level=0)
data.unstack(level=1)
data.unstack().stack()
#========索引重設
data.reset_index(name='') #每層皆有所引
data.set_index(['',''])
#========資料聚合處理
data_mean=data.mean(level='')  #填入欄或列名稱
data.mean(axis=1,level='')

#%%=======================資料集合併Concat、Append===============================
#concat
pd.concat([series1,series2])    #上下合併
pd.concat([series1,series2],axis='col')    #左右合併  同axis=1
pd.concat([x,y],ignore_index=True)   #避免重複的值
pd.concat([df1,df1],join='inner')   #只合併交集
pd.concat([df1,df1],join_axes=[df1.columns])    #根據df1欄位合併
#append
df1.append(df2)
#======================資料集合併Merge、Join==================================
#合併相對應欄位
pd.merge(df1,df2)
#on 指定對應欄位
pd.merge(df1,df2,on='')  #需對應的欄位名稱
#left_on Right_on  欄位內容相同 但欄位名不同
pd.merge(df1,df2,left_on="",right_on="")
#拋棄相同欄位
pd.merge(df1,df2,left_on="",right_on="").drop('',axis=1)
#left_index right_index
pd.merge(df1,df2,left_index=True,right_index=True)
df1.join(df2)
pd.merge(df1,df2,left_index=True,right_on='colname')
#========join指定集合算術運算
#交集才合併(內部連接)
pd.merge(df1,df2,how='inner')
#聯集合併(外部連接)
pd.merge(df1,df2,how='outer')
#左連結
pd.merge(df1,df2,how='left')
#右連結
pd.merge(df1,df2,how='right')
#======合併時重複欄位增加標示
pd.merge(df1,df2,on='name',suffixes=["_L","_R"])


#%%-===========================聚合計算================================
#聚合計算列表
count()      #個數
first(),last()#第一項 最末項
mean(),median()
min(),max()
std(),var()
mad()#平均絕對差
prod()#所有資料項的積
sum()
#============
df.mean()
df.mean(axis='columns')
df.dropna().describe()
#===GroupBy:Split、Apply、Combine=====
df=pd.DataFrame({'key':['A','B','C','A','B','C'],'data':range(6)},columns=['key','data'])
#    key   data
#  0  A     0
#  1  B     1
#  2  C     2
#  3  A     3
#  4  B     4
#  5  C     5
df.groupby('key').sum()
#  key    data
#   A       3
#   B       5
#   C       7
#====聚合運算
df.groupby(['A','B'])['C'].sum()
df.groupby('key').aggregate({'min',np.median,max})
df.groupby('key').aggregate({'data1':'min',
                             'data2':'max'})
#====過濾
def filter_func(X):
    return x['data2'].std()>4
df.groupby('key').filter(filter_func)
#====轉換
df.groupby('key').transform(lambda x:x-x.mean())

def norm_by_data2(x):
    x['data1']/=x['data2'].sum
    return x
df.groupby.apply(norm_by_data2)     #data1除以同樣key的總和

#====指定分割
#list
L=[0,1,0,1,2,0]
df.groupby(L).sum()   #根據L的list分割
#字典
df2=df.set.index('key')
mapping={'A':'vowel','B':'consonant','C':'constant'}
df2.groupby(mapping).sum()
#依照文字、並改小寫
df.groupby(str.lower).mean()
#依照文字並合併索引
df.groupby([str.lower,mapping]).mean()

#=================================樞紐分析表
data.pivot_table('要計算的內容',index='目標索引',columns='目標欄位')

df_cut=pd.cut(titanic['age'][0,18,80])
data.pivot_table('survived',['sex',age],'class')

fare=pd.qcut(titanic['fare'],2)  #切成2等分
titanic.pivot_table('survived',['sex',age],[fare,'class'])

#======樞紐分析額外選項
Data.pivot_table(data,values=None,index=[],columns=[],aggfunc='mean',fill_vale=None,
                 maegin=True,dropna=True,margins_name='All')
#ex
titanic.pivot_table(index='sex',columns='class',aggfun={'survived':sum,'fare':'mean'})

titanic.pivot_table('survived',index='sex',columns='class',margins=True)

#%%===================================向量化字串==================================
name.str.capitalize()     #開頭大寫 後面小寫
match()            #傳回布林值
extract()          #傳回符合的群組
findall()       
replace()          #取代符合的字串
contains()         #re.search()傳回布林值
count()            #計算符合的數目
split()
rsplit()
#==ex==
name.str.extract('([A-Za-z]+)')
name.str.findall(r'^[^AEIOU].*[^aeiou]$')    #^[]字串開頭  []$ 字串結尾
#===============
get()
slice()
slice_replace()
cat()
repeat()
normalize()
pad()
wrap()
join()
get_dummies()
#==ex==
name.str[0:3]
name.str.split().str.get(-1)
#虛擬變數
full_monte=pd.DataFrame({'name':name,'info':['B|C|D','B|D','C|D']})
full_monte['info'].str.get_dummies('|')
#%%====================================Time Series===============================
import pandas as pd
from datetime import datetime
datetime(year=2015,month=7,day=4)

from dateutil import parser
date=parser.parse("4th of July, 2015")
date.strftime('%A')    #date是星期幾

import numpy as np
date=np.array('2015-07-04',dtype=np.datetime64)
date+np.arange(12)

np.datetime64('2015-07-04')
np.datetime64('2015-07-04 12:00')
np.datetime64('2015-07-04 12:59:59.5','ns')

date=pd.to_datetime("4th of July, 2015")

date+pd.to_timedelta(np.arange(12),'D')
#建構時間索引
index=pd.DatetimeIndex(['2014-07-04','2014-05-04','2015-07-04','2015-08-04'])
data=pd.Series([0,1,2,3],index=index)

dates=pd.to_datetime([datetime(2015, 7, 3),'4th if July, 2015','2015-Jul-6','07-07-2015',
                      '20150708'])
dates.to_period('D')
#計算區間
dates-dates[0]

#===========規則性的序列pd.date_range()
pd.date_range('2015-07-03','2015-07-10')
pd.date_range('2015-07-03',periods=8)
pd.date_range('2015-07-03',periods=8,freq='H')

pd.period_range('2015-07',periods=8,freq='M')

pd.timedelta_range(0,periods=10,freq='H')
pd.timedelta_range(0,periods=9,freq='2H30T')

from pandas.rtseries.offsets import BDay
pd.date_range('2015-07-01',periods=5,freq=BDay())

#%%=========================高效率:eval()、query()============================

result1=df[(df.A<8)&(df.B<9)]
result2=pd.eval('df[(df.A<8)&(df.B<9)]')
result3=df[df.eval('A<8&B<9')]
result4=df.query('A<8 and B<9')

np.allclose(reult1,result2)#True 兩者一樣

resultA=-df1*df2/(df3+df4)-df5
resultB=pd.eval('-df1*df2/(df3+df4)-df5')
np.allclose(resultA,resultB)
#dataframe中賦值計算
df.eval('D=(A+B)/C',inplace=True)

columns_mean=df.mean(1)
result1=df['A']+columns_mean
result2=df.eval('A+@columns_mean')  #@為標記變數名稱 非欄名
#result1=result2

result1=df[(df.A<0.5)&(df.B<0.5)]
result2=df.eval('df[(df.A<0.5)&(df.B<0.5)]')
result3=df.query('A<0.5 and B<0.5')

Cmean=df['C'].mean()
result1=df[(df.A<Cmean)&(df.B<Cmean)]
resul2=df.query('A<@Cmean and B<@Cmean')





























