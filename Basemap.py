import os
os.environ["PROJ_LIB"] = "C:\\Utilities\\Python\\Anaconda\\Library\\share"; #fixr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(8,8))
m=Basemap(projection='lcc',resolution=None,width=8E4,height=8E4,lat_0=23.5,lon_0=125)
m.etopo(scale=1,alpha=0.5)


















