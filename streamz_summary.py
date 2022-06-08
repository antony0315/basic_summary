from streamz import Stream


#%%核心
def increment(x):
    return x + 1

source = Stream()
source.map(increment).sink(print)
source.emit(100)


#%%累積狀態
def add(x, y):
    return x + y

source2 = Stream()
source2.accumulate(add).sink(print)
source2.emit(5)
source2.emit(100)
source2.emit(11000)


#累積幾種動物
def num_distinct(state, new):
    state.add(new)
    return state, len(state)

source3 = Stream()
source3.accumulate(num_distinct,returns_state=True, start=set()).sink(print)
 
source3.emit('cat')
source3.emit('dog')
source3.emit('cat') 
 
#%%流量控制
source = Stream()
source.sliding_window(3, return_partial=False).sink(print) 
 
source.emit(1)
source.emit(2)
source.emit(5) 
 
#%%
 
def increment(x):
    return x + 1

def decrement(x):
    return x - 1

source = Stream()
a = source.map(increment).sink(print)
b = source.map(decrement).sink(print)
b.visualize(rankdir='LR')
 
 
 
 
 
 