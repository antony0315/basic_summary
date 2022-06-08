#%%內建清單函數
n1=[1,2,3] 
n2=[4,5,6]
n3=[0,1,5]
country=['china','usa','uk']
continent=['asia','america','euro']

#逐行讀取索引與元素
for i,j in enumerate(country):
    print(i,j)

#排序
sorted([2,8,12,1,6,3])
sorted([2,8,12,1,6,3],reverse=True)

#配對list元素
z=zip(country,continent,n1)
for i in z:
    print(i)

#%%
append   #附加
clear    #清空
copy     #複製
extend   #追加多個值,不replace
index    #索引
insert(0,)  #插入(位置,元素)
pop        #刪除元素  replace

a=[1,2,3,4,5]
list(map(lambda x:x**2,a))


#%%
add
difference(n1,n2)
dicard

#%%遞迴

def addn(n):
    if n==1:
        return 1
    return n+addn(n-1)
addn(100)

#%%filter 篩選器  
list(filter(lambda x:x%2==0,range(20)))

#%%map
list(map(lambda x:x*2,[5,6,7]))
#%%class
class classname(object):
    def __init__(self,var1='x1',var2='x2'):
        self.var1=0
        self.var2=0
    def add(self,var1,var2):
        print(var1*var2)




