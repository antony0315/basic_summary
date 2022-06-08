#%%numpy 資料型態
import numpy as np
import array
#List
np.array([1,4,2,5,3])
np.array([1,2,3,4],dtype='float32')#小數點
np.array([range(i,i+3)for i in[2,4,6]])
#快速建立陣列
np.zeros(10,dtype=int)
np.ones((3,5),dtype=float)
np.full((3,5),3.14)
np.arange(0,20,2)
np.linspace(0,1,5)#0~1 之間建立5個值
np.random.random((3,4))  #3列4欄  0~1
np.random.normal(0,1,(3,3))  #u=0 sd=1 3*3 normal dist random
np.random.randint(0,10,(3,3))#0~10 3*3 整數
np.eye(3)  #3*3單位矩陣
np.empty(3)  #為初始化陣列
#資料型態
# =============================================================================
# 布林:bool_ 
# 整數:int_ intc intp int8 int16 int32 int32 int64
# 正整數:uint8 uint16 uint32 uint64
# 小數點:float_ float16 float32 float64
# 複數:complex_ complex64 complex128
# =============================================================================
#%% numpy陣列基礎
import numpy as np
np.random.seed(123)  #指定依樣的亂數
x1=np.random.randint(10,size=6)      #1*6
x2=np.random.randint(10,size=(3,4))  #3*4
x3=np.random.randint(10,size=(3,4,5))#3個4*5
#=======陣列屬性
print('x3的維度數量:',x3.ndim)
print('x3的維度大小:',x3.shape)
print('x2陣列總大小:',x3.size)

print('x3的資料型態:',x3.dtype)

print('每個字串大小itemsize:',x3.itemsize,'bytes')
print('總陣列大小',x3.nbytes,'bytes')
#========單一陣列元素
x1
x1[4]
x1[-1]
x2[0,0]
#修改
x2[0,0]=12
#=========一維子陣列
x=np.arange(10)
#前5項
x[:5]
#第五項以後
x[5:]
#4~6項
x[4:7]
#間隔2
x[::2]
#第一項開始 間隔2
x[1::2]
#反轉元素
x[::-1]
#地5項以前 間隔2 反轉
x[5::-2]
#========多維子陣列=============
x2[:2,:3]   #0、1列   0、1、3欄
x2[:3,::2]   #0、1、2列    0、2欄
x2[::-1,::-1] #反轉欄列順序
#========copy陣列===========
x2_copy=x2[:2,:2].copy()
#x2 改變 x2_copy不會變
#=========陣列重塑============================================
grid=np.arange(1,10).reshape((3,3))

x=np.array([1,2,3])
x=x.reshape((1,3))     #列向量
x[:,np.newaxis]       #列向量
x[np.newaxis,:]       #欄向量
x.reshape((3,1))       #欄向量
#===========陣列的串接與分割==================================
x=np.array([1,2,3])
y=np.array([3,2,1])
z=np.array([4,5,6])
np.concatenate([x,y,z])


grid=np.array([[1,2,3],
               [4,5,6]])
#上下串接
np.concatenate([grid,grid])
#左右串接
np.concatenate([grid,grid],axis=1)

#========不同維度之間堆疊
x=np.array([1,2,3])
y=np.array([[99],
            [99]])
grid=np.array([[9,8,7],
               [6,5,4]])

np.vstack([x,grid])    #垂直堆疊
np.hstack([grid,y])    #水平堆疊
np.dstack([x,y])       #沿著第三軸堆疊

#========分割陣列
x=[1,2,3,99,98,3,2,1]
x1,x2,x3=np.split(x,[3,5])

grid=np.arange(16).reshape((4,4))
#上下切割
upper,lower=np.vsplit(grid,[3])
#左右切割
left,right=np.hsplit(grid,[2])
#%%numpy陣列中的計算
import numpy as np
np.arange(5)/np.arange(1,6)
x=np.arange(9).reshape((3,3))
2**x
x+2
x*2
x//2
#運算子
np.add(x,2)
np.subtrct(x,1)
np.negative(x)    #加負號
np.multiply(x,7)
np.divide(x,2)
np.floor_divide(x,2)  #商數
np.power(x,3)   #次方
np.mod(x,4)     #餘數
#絕對值
x=np.array([-2,-1,0,1,2])
abs(x)
np.abs(x)
#三角函數
theta=np.linspace(0,np.pi,3)
np.sin(theta)
np.cos(theta)
np.tan(theta)
#指數&對數
x=[1,2,3]
np.exp(x)     #e^x
np.exp2(x)    #2^x
np.power(x,3) #x^3

np.log(x)     #ln(x)
np.log2(x)    #log2(x)
np.log10(x)

x=[0.0001,0.00000001,0.00005]
np.expm1(x)    #增加精準度
np.log1p(x)
#特殊運算
from scipy import special
x=[1,5,10]
special.gamma(x)
special.gammaln(x)
special.beta(x,2)
special.erf(x)#高司積分 誤差函數
special.erfc(x)
special.erfinv(x)

#存在指定陣列
x=np.arange(5)
y=np.empty(5)
np.multiply(x,10,out=y)

x=np.zeros(10)
np.power(2,x,out=y[::2])

#聚合運算
x=np.arange(1,6)
np.add.reduce(x)   #加總

np.multiply.reduce(x)#相乘

np.add.accumulate(x)#加總過程

np.multiply.accumulate(x)#相乘過程
#外積
x=np.arange(1,6)
np.multiply.outer(x,x)

#%%聚合操作:Min Max
import numpy as np

L=np.random.random(100)
#加總
sum(L)
np.sum(L)

#min max
np.min(L)
np.max(L)
L.min()
L.max()

#多維度計算
M=np.random.random((3,4))
M.sum(axis=0)
M.max(axis=1)

np.sum    
np.prod   #成積
np.mean
np.std
np.vnp
np.min
np.max
np.argmin   #最小值索引
np.argmax   #最大值索引
np.median
np.percentile  #排名統計
np.any       #任一直為True
np.all       #所有值為True
#%%陣列上計算:Boardcasting
import numpy as np
a=np.array([0,1,2])
b=np.array([5,6,7])
c=np.ones((3,3))
a+c

d=np.arange(3)
e=np.arange(3)[:,np.newaxis]
d+e

M=np.ones((2,3))
a=np.arange(3,)
M+a

a=np.arange(3).reshape((3,1))
b=np.arange(3)
a+b

np.logaddexp(M,a[:,np.newaxis])

x=np.random.random((10,3))
xmean=x.mean(0)
x-xmean

rng=np.random.RandomState(0)      #set.seed(0)
x=rng.randint(10,size=(3,4))
np.count_nonzero(x<5)

np.sum(x<5)

np.any(x>9)

np.all(x<8,axis=1)

np.sum((x>2)&(x<8))

#%%Fancy索引
import numpy as np
rng=np.random.RandomState(42)
x=rng.randint(100,size=10)
print(x)
#索引1
print([x[3],x[7],x[9]])
#索引2
ind=[3,7,4]
print(x[ind])
#索引陣列
ind=np.array([[3,7],[4,5]])
x[ind]
#多維度索引
x=np.arange(12).reshape((3,4))
row=np.array([0,1,2])
col=np.array([2,1,3])
x[row,col]
x[row[:,np.newaxis],col]
#索引組合運用
x[2,[2,0,1]]
x[1:,[2,0,1]]

mask=np.array([1,0,1,0],dtype=bool)
x[row[:,np.newaxis],mask]
#%%範例:隨機抽樣本點
import matplotlib.pylab as plt
mean=[0,0]
cov=[[1,2],[2,5]]
x=rand.multivariate_normal(mean,cov,100)
x.shape
%matplotlib inline
plt.scatter(x[:,0],x[:,1])
#隨機抽點
indices=np.random.choice(x.shape[0],20,replace=False)
selection=x[indices]
plt.scatter(x[:,0],x[:,1],alpha=0.5)
plt.scatter(selection[:,0],selection[:,1],facecolor='red',s=20)
#%%Fancy修改值
x=np.arange(10)
for i in range(len(x)):
    x[i]-=1
print(x)

x=np.zeros(10)
j=[0,1,2,3,4]
np.add.at(x,j,1)
print(x)
#%%範例:資料裝箱
import numpy as np
import matplotlib.pylab as plt
np.random.seed(42)
x=np.random.randn(100)
bins=np.linspace(-5,5,20)
counts=np.zeros_like(bins)
i=np.searchsorted(bins,x)
np.add.at(counts,i,1)
plt.plot(bins,counts,linestyle='-',drawstyle='steps')
#%%排序陣列
x=np.random.randint(0,100,10)
print(np.sort(x))
x.sort()

#傳回排序後的索引值
i=np.argsort(x)
print(x[i])
#選擇欄或列排序
x=np.random.randint(0,100,(4,6))
np.sort(x,axis=0)#欄排序
np.sort(x,axis=1)#列排序
#%%分區
#最小的K個值放左側
x=np.array([7,2,3,1,65,8,5,5])
np.partition(x,3)
x2=np.random.randint(0,100,(4,6))
np.partition(x2,2,axis=0)
#傳回索引值
np.argpartition(x2)
#%%numpy 隨機亂數
import numpy as np
np.random.seed(1234)
#常態分布
mu, sigma = 0, 0.1 
s = np.random.normal(mu, sigma, 1000)

np.random.uniform([0,1,20])





















