import plotly 
import plotly.io as pio

import plotly.graph_objects as go
import pandas as pd
pio.renderers.default='browser' #'svg' or 'browser'
data=pd.read_csv('https://raw.githubusercontent.com/tpof314/plotly_examples/master/data/nz_weather.csv')
iris=pd.read_csv('https://raw.githubusercontent.com/tpof314/plotly_examples/master/data/iris.csv')
#%%線圖
line=go.Scatter(x=data['DATE'],y=data['Auckland'],name='Auckland')
line2=go.Scatter(x=data['DATE'],y=data['Wellington'],name='Wellington')
fig=go.Figure([line,line2])
fig.update_layout(
    title='NEW ZEALAND WEATHER',
    xaxis_title='DATE',
    yaxis_title='Weather')
fig.show()
#%%  條形圖
data_2010=data[(data['DATE']>='2010-01')&(data['DATE']<'2011-01')]
bar=go.Bar(x=data_2010['DATE'],y=data_2010['Auckland'],text=data_2010['Auckland']
           ,textposition='outside',name='Auckland')
bar2=go.Bar(x=data_2010['DATE'],y=data_2010['Wellington'],text=data_2010['Wellington']
           ,textposition='outside',name='Wellington')
fig=go.Figure([bar,bar2])
fig.show()

#%%Histogram
hist=go.Histogram(x=data['Auckland'],xbins={'size':10})
fig=go.Figure(hist)
fig.update_layout(bargap=0.1)#間隔
fig.show()
#%%點狀圖
#iris.groupby('Name').count().index
name_color={'Iris-setosa':0,
            'Iris-versicolor':1,
            'Iris-virginica':2
    }
iris['color']=iris['Name'].map(name_color)
points=go.Scatter(x=iris['SepalLength'],y=iris['SepalWidth']
                  ,mode='markers',marker={'color':iris['color']})
fig=go.Figure(points)
fig.show()


#%%plotly.express
import plotly.express as px
fig=px.scatter(iris,x='SepalLength',y='SepalWidth',color='Name')
fig.show()


#%% matrix

fig=px.scatter_matrix(iris,dimensions=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
                      ,color='Name')
fig.show()


#%%3D
import plotly.graph_objects as go
import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/tpof314/plotly_examples/master/data/3d-line1.csv')


#go 3D
line=go.Scatter3d(x=df['x'],y=df['y'],z=df['z'],mode='markers',marker={'size':3,'color':'red'})
fig=go.Figure(line)
fig.show()


#express 3D
import plotly.express as px
fig=px.scatter_3d(df,x='x',y='y',z='z',color='color')
fig.show()

#%%    3D曲面
import plotly.graph_objects as go
import pandas as pd
data=pd.read_csv('https://raw.githubusercontent.com/tpof314/plotly_examples/master/data/mt_bruno_elevation.csv')
del data['index']

height=data.values
surface=go.Surface(z=height)
fig=go.Figure(surface)
fig.show()


#公式  z=x**2+y**2
import numpy as np
x=np.arange(-5,6)
y=np.arange(-5,6)
xv, yv=np.meshgrid(x,y)
z=xv**2+yv**2
surface=go.Surface(x=xv,y=yv,z=z)
fig=go.Figure(surface)
fig.show()

#%%地圖
#https://plotly.com/python/mapbox-density-heatmaps/
# https://plotly.com/python/mapbox-layers/   style
import plotly.graph_objects as go
import pandas as pd
data=pd.read_csv('https://raw.githubusercontent.com/tpof314/plotly_examples/master/data/earthquakes.csv')
my_map=go.Densitymapbox(lat=data['Latitude'],lon=data['Longitude'],z=data['Magnitude'],radius=5)
fig=go.Figure(my_map)
fig.update_layout(mapbox_style="open-street-map")
fig.show()






