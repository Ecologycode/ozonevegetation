import numpy as np
import pandas as pd
from netCDF4 import Dataset
#from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta
import glob
import datetime as dt
from datetime import datetime, timedelta
import scipy.stats as stats
import scipy.optimize as opt
import statsmodels.stats.stattools as st
import pingouin as pg
from scipy.stats import linregress
from datetime import datetime, timedelta
from netCDF4 import Dataset
from matplotlib import cm
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import random
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,mean_absolute_error

def colormap_res11():
	cdict=['#3281C4','#52A7DD','#87CDF1','#B6E2ED','#DDF0F4','#FAF3D0','#F7D162','#F6A31F','#ED5D19','#E12A12']
	return matplotlib.colors.ListedColormap(cdict,'indexed')

def colormap_res111():
	cdict=['#52A7DD','#87CDF1','#B6E2ED','#DDF0F4','#FAF3D0','#F7D162','#F6A31F','#ED5D19']
	return matplotlib.colors.ListedColormap(cdict,'indexed')
		
def colormap_res1():
	cdict=['#1950A2','#3281C4','#52A7DD','#87CDF1','#B6E2ED','#DDF0F4','#FAF3D0','#F7D162','#F6A31F','#ED5D19','#E12A12','#C01922']
	return matplotlib.colors.ListedColormap(cdict,'indexed')

def colormap1():
	cdict=['#11346E','#1B5AA9','#4298D3','#72BEEA','#ABDCEB','#DCF0F4',
	'#F9F1BE','#F7C24B','#F0851B','#E83C17','#C41B1F','#9B1E23']
	return matplotlib.colors.ListedColormap(cdict,'indexed')
	
def phenology():
	data=pd.read_csv(r'./data/code/phenology.csv')
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')

	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['Date_Mid_Greenup_Phase_1'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=60,vmax=160,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())

	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(60,170,10))
	cb.ax.tick_params(labelsize=12)
	cb.set_label('[SOS]',fontdict={'family':'arial','weight':'normal','size':16,})
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')

	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['Date_Mid_Senescence_Phase_1'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=220,vmax=320,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())

	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(220,330,10))
	cb.ax.tick_params(labelsize=12)
	cb.set_label('[EOS]',fontdict={'family':'arial','weight':'normal','size':16,})
	
	fig = plt.figure()
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['EVI2_Onset_Greenness_Maximum_1'])
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	Date_Mid_Greenup_Phase_1=np.array(data['EVI2_Onset_Greenness_Maximum_1'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=0,vmax=0.8,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(0,1.2,0.4))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('EVI$_{max}$',fontdict={'family':'arial','weight':'normal','size':16,})
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['EVI2_Growing_Season_Area_1'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=0,vmax=80,cmap=colormap_res11(),transform=ccrs.PlateCarree())		
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())

	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(0,88,8))
	cb.ax.tick_params(labelsize=12)
	cb.set_label('[EVI$_{area}$]',fontdict={'family':'arial','weight':'normal','size':16,})
	plt.show()
		
def aot40():
	data=pd.read_csv(r'./data/code/aot40.csv')
	projections = [	ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0),]
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['AOT40_spring'])/10**3
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=0,vmax=20,cmap=colormap_res11(),transform=ccrs.PlateCarree())		
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(0,22,2))
	cb.ax.tick_params(labelsize=12)
	cb.set_label('[10${^3}$ ppbv$\cdot$hour]',fontdict={'family':'arial','weight':'normal','size':32,})
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['AOT40_autumn'])/10**3
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=0,vmax=20,cmap=colormap_res11(),transform=ccrs.PlateCarree())		
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(0,22,2))
	cb.ax.tick_params(labelsize=12)
	cb.set_label('[10${^3}$ ppbv$\cdot$hour]',fontdict={'family':'arial','weight':'normal','size':16,})

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['AOT40_summer'])/10**3
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=0,vmax=20,cmap=colormap_res11(),transform=ccrs.PlateCarree())		
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(0,22,2))
	cb.ax.tick_params(labelsize=12)
	cb.set_label('[10${^3}$ ppbv$\cdot$hour]',fontdict={'family':'arial','weight':'normal','size':16,})
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['AOT40_all'])/10**3
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=0,vmax=35,cmap=colormap_res11(),transform=ccrs.PlateCarree())		
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(0,40,5))
	cb.ax.tick_params(labelsize=12)
	cb.set_label('[10${^3}$ ppbv$\cdot$hour]',fontdict={'family':'arial','weight':'normal','size':16,})
	plt.show()

def o3():
	data=pd.read_csv(r'./data/code/o3.csv')
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0),]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['O3_spring'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=0,vmax=50,cmap=colormap_res11(),transform=ccrs.PlateCarree())		
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(0,55,5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('Ozone (ppbv)',fontdict={'family':'arial','weight':'normal','size':16,})

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['O3_autumn'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=0,vmax=50,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(0,55,5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('Ozone (ppbv)',fontdict={'family':'arial','weight':'normal','size':16,})

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['O3_summer'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=0,vmax=50,cmap=colormap_res11(),transform=ccrs.PlateCarree())		
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())  # Adjust the extent as needed
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(0,55,5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('Ozone (ppbv)',fontdict={'family':'arial','weight':'normal','size':16,})

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['O3_all'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=0,vmax=50,cmap=colormap_res11(),transform=ccrs.PlateCarree())		
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())

	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(0,55,5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('Ozone (ppbv)',fontdict={'family':'arial','weight':'normal','size':16,})
	plt.show()
	
def co2():
	data=pd.read_csv(r'./data/code/co2.csv')
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0),]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['co2_spring'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=2,vmin=395,vmax=425,cmap=colormap1(),transform=ccrs.PlateCarree())		
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(405,425,5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('CO${_2}$ (ppbm)',fontdict={'family':'arial','weight':'normal','size':16,})
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon_x'])
	lat=np.array(data['lat_x'])    
	Date_Mid_Greenup_Phase_1=np.array(data['co2_autumn'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=395,vmax=425,cmap=colormap1(),transform=ccrs.PlateCarree())		
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(395,430,5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('CO${_2}$ (ppbm)',fontdict={'family':'arial','weight':'normal','size':16,})

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['co2_summer'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=395,vmax=425,cmap=colormap1(),transform=ccrs.PlateCarree())		
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())  # Adjust the extent as needed
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(395,430,5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('CO${_2}$ (ppbm)',fontdict={'family':'arial','weight':'normal','size':16,})
	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	lon=np.array(data['lon'])
	lat=np.array(data['lat'])    
	Date_Mid_Greenup_Phase_1=np.array(data['co2_all'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1,s=5,vmin=395,vmax=425,cmap=colormap1(),transform=ccrs.PlateCarree())		
	ax.set_extent([-180, 180,18, 90], crs=ccrs.PlateCarree())  # Adjust the extent as needed
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(395,430,5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('CO${_2}$ (ppbm)',fontdict={'family':'arial','weight':'normal','size':16,})		
	plt.show()

def partial_year_plot_aot40():	
	area=['CN','US','EU']
	gas=['O3','AOT40']
	gas1=['O$_{3}$','AOT40']
	cat=['all_EVI2_Growing_Season_Area_1','autumn_Date_Mid_Senescence_Phase_1','spring_Date_Mid_Greenup_Phase_1','summer_EVI2_Onset_Greenness_Maximum_1']
	letters=['(a)','(b)','(c)','(d)']	
	data=pd.read_csv(r"./data/code/corr_aot40.csv")	
	cn_r=np.array(data['CN_r'])
	cn_p=np.array(data['CN_p'])
	eu_r=np.array(data['EU_r'])
	eu_p=np.array(data['EU_p'])
	us_r=np.array(data['US_r'])
	us_p=np.array(data['US_p'])
	cn_std=np.array(data['CN_std'])
	eu_std=np.array(data['EU_std'])
	us_std=np.array(data['US_std'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.bar(np.arange(0,len(cn_r),1)-0.15,us_r,yerr=np.abs(us_std),width = 0.15,color='#E63946',label='U.S.',error_kw=dict(capsize=3))
	ax.bar(np.arange(0,len(cn_r),1),eu_r,yerr=np.abs(eu_std),width = 0.15,color='#AAE5F4',label='Europe',error_kw=dict(capsize=3))
	ax.bar(np.arange(0,len(cn_r),1)+0.15,cn_r,yerr=np.abs(cn_std),width = 0.15,color='orange',label='China',error_kw=dict(capsize=3))
	for l, value in enumerate(cn_r):
		if value>0:
			if cn_p[l]<0.05:
				ax.text(l+0.15, value + value*0.25- 0.02, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
		if value<0:
			if cn_p[l]<0.05:
				ax.text(l+0.15, value + value*0.25- 0.08, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
	for l, value in enumerate(us_r):
		if value>0:
			if us_p[l]<0.05:
				ax.text(l-0.15, value + value*0.25- 0.02, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
		if value<0:
			if us_p[l]<0.05:
				ax.text(l-0.15, value + value*0.25- 0.08, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
	for l, value in enumerate(eu_r):
		if value>0:
			if eu_p[l]<0.05:
				ax.text(l, value + value*0.25- 0.02, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
		if value<0:
			if eu_p[l]<0.05:
				ax.text(l, value + value*0.25- 0.08, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})						
	ax.axhline(y=0, color='black', linewidth=1)
	ax.set_xticks(np.arange(0,len(cn_r),1))
	ax.set_xticklabels(['SOS','EOS','EVI$_{max}$','EVI$_{area}$'],fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_yticks(np.arange(-0.5,1.25,0.25))
	ax.set_yticklabels([-0.5,-0.25,0,0.25,0.5,0.75,1.0], fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('${_R}$', fontdict={'family':'arial','weight':'normal','size':36,})
	ax.legend(prop={'family':'arial','weight':'normal','size':26,},loc='lower left',ncol=1,frameon=False)
	plt.show()
	
def partial_year_plot_o3():			
	data=pd.read_csv(r"./data/code/corr_o3.csv")
	cn_r=np.array(data['CN_r'])
	cn_p=np.array(data['CN_p'])
	eu_r=np.array(data['EU_r'])
	eu_p=np.array(data['EU_p'])
	us_r=np.array(data['US_r'])
	us_p=np.array(data['US_p'])
	cn_std=np.array(data['CN_std'])
	us_std=np.array(data['US_std'])
	eu_std=np.array(data['EU_std'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.bar(np.arange(0,len(cn_r),1)-0.15,us_r,yerr=np.abs(us_std),width = 0.15,color='#E63946',label='U.S.',error_kw=dict(capsize=3))
	ax.bar(np.arange(0,len(cn_r),1),eu_r,yerr=np.abs(eu_std),width = 0.15,color='#AAE5F4',label='Europe',error_kw=dict(capsize=3))
	ax.bar(np.arange(0,len(cn_r),1)+0.15,cn_r,yerr=np.abs(cn_std),width = 0.15,color='orange',label='China',error_kw=dict(capsize=3))
	for l, value in enumerate(cn_r):
		if value>0:
			if cn_p[l]<0.05:
				ax.text(l+0.15, value + value*0.25- 0.02, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
		if value<0:
			if cn_p[l]<0.05:
				ax.text(l+0.15, value + value*0.25- 0.08, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})

	for l, value in enumerate(us_r):
		if value>0:
			if us_p[l]<0.05:
				ax.text(l-0.15, value + value*0.25- 0.02, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
		if value<0:
			if us_p[l]<0.05:
				ax.text(l-0.15, value + value*0.25- 0.08, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})

	for l, value in enumerate(eu_r):
		if value>0:
			if eu_p[l]<0.05:
				ax.text(l, value + value*0.25- 0.02, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
		if value<0:
			if eu_p[l]<0.05:
				ax.text(l, value + value*0.25- 0.08, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})	
	ax.axhline(y=0, color='black', linewidth=1)
	ax.set_xticks(np.arange(0,len(cn_r),1))
	ax.set_xticklabels(['SOS','EOS','EVI$_{max}$','EVI$_{area}$'],fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(-0.5,0.75,0.25))
	ax.set_yticklabels([-0.5,-0.25,0,0.25,0.5], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylabel('${_R}$', fontdict={'family':'arial','weight':'normal','size':50,})
	ax.legend(prop={'family':'arial','weight':'normal','size':26,},loc='upper left',ncol=3)
	plt.show()
	
def partial_year_plot_temp():		
	data=pd.read_csv(r"./data/code/corr_temp.csv")			
	cn_r=np.array(data['CN_r'])
	cn_p=np.array(data['CN_p'])
	eu_r=np.array(data['EU_r'])
	eu_p=np.array(data['EU_p'])
	us_r=np.array(data['US_r'])
	us_p=np.array(data['US_p'])
	cn_std=np.array(data['CN_std'])
	us_std=np.array(data['US_std'])
	eu_std=np.array(data['EU_std'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.bar(np.arange(0,len(cn_r),1)-0.15,us_r,yerr=np.abs(us_std),width = 0.15,color='#E63946',label='U.S.',error_kw=dict(capsize=3))
	ax.bar(np.arange(0,len(cn_r),1),eu_r,yerr=np.abs(eu_std),width = 0.15,color='#AAE5F4',label='Europe',error_kw=dict(capsize=3))
	ax.bar(np.arange(0,len(cn_r),1)+0.15,cn_r,yerr=np.abs(cn_std),width = 0.15,color='orange',label='China',error_kw=dict(capsize=3))
	for l, value in enumerate(cn_r):
		if value>0:
			if cn_p[l]<0.05:
				ax.text(l+0.15, value + value*0.25- 0.02, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
		if value<0:
			if cn_p[l]<0.05:
				ax.text(l+0.15, value + value*0.25- 0.08, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})

	for l, value in enumerate(us_r):
		if value>0:
			if us_p[l]<0.05:
				ax.text(l-0.15, value + value*0.25- 0.02, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
		if value<0:
			if us_p[l]<0.05:
				ax.text(l-0.15, value + value*0.25- 0.08, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})

	for l, value in enumerate(eu_r):
		if value>0:
			if eu_p[l]<0.05:
				ax.text(l, value + value*0.25- 0.02, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
		if value<0:
			if eu_p[l]<0.05:
				ax.text(l, value + value*0.25- 0.08, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
	ax.axhline(y=0, color='black', linewidth=1)
	ax.set_xticks(np.arange(0,len(cn_r),1))
	ax.set_xticklabels(['SOS','EOS','EVI$_{max}$','EVI$_{area}$'],fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(-0.5,0.75,0.25))
	ax.set_yticklabels([-0.5,-0.25,0,0.25,0.5], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylabel('${_R}$', fontdict={'family':'arial','weight':'normal','size':50,})
	ax.legend(prop={'family':'arial','weight':'normal','size':26,},loc='upper left',ncol=3)
	plt.show()
	
def partial_year_plot_co2():	
	data=pd.read_csv(r"./data/code/corr_co2.csv")			
	cn_r=np.array(data['CN_r'])
	cn_p=np.array(data['CN_p'])
	eu_r=np.array(data['EU_r'])
	eu_p=np.array(data['EU_p'])
	us_r=np.array(data['US_r'])
	us_p=np.array(data['US_p'])
	cn_std=np.array(data['CN_std'])
	us_std=np.array(data['US_std'])
	eu_std=np.array(data['EU_std'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.bar(np.arange(0,len(cn_r),1)-0.15,us_r,yerr=np.abs(us_std),width = 0.15,color='#E63946',label='U.S.',error_kw=dict(capsize=3))
	ax.bar(np.arange(0,len(cn_r),1),eu_r,yerr=np.abs(eu_std),width = 0.15,color='#AAE5F4',label='Europe',error_kw=dict(capsize=3))
	ax.bar(np.arange(0,len(cn_r),1)+0.15,cn_r,yerr=np.abs(cn_std),width = 0.15,color='orange',label='China',error_kw=dict(capsize=3))
	for l, value in enumerate(cn_r):
		if value>0:
			if cn_p[l]<0.05:
				ax.text(l+0.15, value + value*0.25- 0.02, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
		if value<0:
			if cn_p[l]<0.05:
				ax.text(l+0.15, value + value*0.25- 0.08, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})

	for l, value in enumerate(us_r):
		if value>0:
			if us_p[l]<0.05:
				ax.text(l-0.15, value + value*0.25- 0.02, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
		if value<0:
			if us_p[l]<0.05:
				ax.text(l-0.15, value + value*0.25- 0.08, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})

	for l, value in enumerate(eu_r):
		if value>0:
			if eu_p[l]<0.05:
				ax.text(l, value + value*0.25- 0.02, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})
		if value<0:
			if eu_p[l]<0.05:
				ax.text(l, value + value*0.25- 0.08, '*', ha='center', va='bottom',fontdict={'family':'arial','weight':'normal','size':20,})	
	ax.axhline(y=0, color='black', linewidth=1)
	ax.set_xticks(np.arange(0,len(cn_r),1))
	ax.set_xticklabels(['EVI$_{max}$','EVI$_{area}$'],fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(-0.5,1.0,0.25))
	ax.set_yticklabels([-0.5,-0.25,0,0.25,0.5,0.75], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylabel('${_R}$', fontdict={'family':'arial','weight':'normal','size':50,})
	ax.legend(prop={'family':'arial','weight':'normal','size':26,},loc='upper left',ncol=3)
	plt.show()

def temporal_all_map():
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/aot40/spring_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['Date_Mid_Greenup_Phase_1_aot40_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{AOT40-SOS}$',fontdict={'family':'arial','weight':'normal','size':16,})

	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')	
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/aot40/autumn_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['Date_Mid_Senescence_Phase_1_aot40_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())
	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{AOT40-EOS}$',fontdict={'family':'arial','weight':'normal','size':16,})

	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/aot40/summer_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Onset_Greenness_Maximum_1_aot40_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())
	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{AOT40-EVI_{max}}$',fontdict={'family':'arial','weight':'normal','size':16,})
	
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')	
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/aot40/all_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Growing_Season_Area_1_aot40_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())
	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{AOT40-EVI_{area}}$',fontdict={'family':'arial','weight':'normal','size':16,})
	
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/o3/spring_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['Date_Mid_Greenup_Phase_1_o3_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{O_{3}-SOS}$',fontdict={'family':'arial','weight':'normal','size':16,})

	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/o3/autumn_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['Date_Mid_Senescence_Phase_1_o3_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{O_{3}-EOS}$',fontdict={'family':'arial','weight':'normal','size':16,})

	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/o3/summer_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Onset_Greenness_Maximum_1_o3_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())
	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{O_{3}-EVI_{max}}$',fontdict={'family':'arial','weight':'normal','size':16,})
	
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/o3/all_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])		
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Growing_Season_Area_1_o3_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{O_{3}-EVI_{area}}$',fontdict={'family':'arial','weight':'normal','size':16,})

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/temp/spring_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['Date_Mid_Greenup_Phase_1_temp_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())
	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{T-SOS}$',fontdict={'family':'arial','weight':'normal','size':16,})
	
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')	
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/temp/autumn_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['Date_Mid_Senescence_Phase_1_temp_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())
	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{T-EOS}$',fontdict={'family':'arial','weight':'normal','size':16,})
	
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/temp/summer_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Onset_Greenness_Maximum_1_temp_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())
	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{T-EVI_{max}}$',fontdict={'family':'arial','weight':'normal','size':16,})

	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')

	data11=pd.read_csv(r"./data/code/temporal/temp/all_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Growing_Season_Area_1_temp_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{T-EVI_{area}}$',fontdict={'family':'arial','weight':'normal','size':16,})
	
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/co2/summer_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Onset_Greenness_Maximum_1_co2_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{CO_{2}-EVI_{max}}$',fontdict={'family':'arial','weight':'normal','size':16,})

	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	data11=pd.read_csv(r"./data/code/temporal/co2/all_temporal.csv")
	lat=np.array(data11['lat'])
	lon=np.array(data11['lon'])	
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Growing_Season_Area_1_co2_r'])
	cs=ax.scatter(lon,lat,c=Date_Mid_Greenup_Phase_1_r,s=6,vmin=-1,vmax=1,cmap=colormap_res11(),transform=ccrs.PlateCarree())
	
	fig.subplots_adjust()
	l=0.19
	b=0.075
	w=0.655
	h=0.0155
	rect=[l,b,w,h]
	cbar_ax=fig.add_axes(rect)
	cb=fig.colorbar(cs,cax=cbar_ax,extend='both',orientation='horizontal')
	cb.set_ticks(np.arange(-1,1.5,0.5))
	cb.ax.tick_params(labelsize=16)
	cb.set_label('R$_{CO_{2}-EVI_{area}}$',fontdict={'family':'arial','weight':'normal','size':16,})
	plt.show()

def gaussian(x, mean, std_dev):
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    
def temporal_all_map_histogram():
	data11=pd.read_csv(r"./data/code/temporal/aot40/spring_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['Date_Mid_Greenup_Phase_1_aot40_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, patches =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')	
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a+0.01,1.1,str(round(a,2)), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	
	data11=pd.read_csv(r"./data/code/temporal/aot40/autumn_temporal.csv")
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, _ =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a-0.5,1.1,str(round(a,2)), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	
	data11=pd.read_csv(r"./data/code/temporal/aot40/summer_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Onset_Greenness_Maximum_1_aot40_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, _ =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a-0.5,1.1,str(round(a,2)), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	
	data11=pd.read_csv(r"./data/code/temporal/aot40/all_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Growing_Season_Area_1_aot40_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, _ =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a-0.5,1.1,str(round(a,2)), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	
	data11=pd.read_csv(r"./data/code/temporal/o3/spring_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['Date_Mid_Greenup_Phase_1_o3_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, patches =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')	
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a+0.01,1.1,str(round(a,2)), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	
	data11=pd.read_csv(r"./data/code/temporal/o3/autumn_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['Date_Mid_Senescence_Phase_1_o3_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, _ =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a-0.55,1.1,str(round(a,2)), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	
	data11=pd.read_csv(r"./data/code/temporal/o3/summer_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Onset_Greenness_Maximum_1_o3_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, _ =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a-0.5,1.1,str(round(a,2)), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	
	data11=pd.read_csv(r"./data/code/temporal/o3/all_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Growing_Season_Area_1_o3_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, _ =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a-0.5,1.1,str(round(a,2)), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})

	data11=pd.read_csv(r"./data/code/temporal/temp/spring_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['Date_Mid_Greenup_Phase_1_temp_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, _ =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a-0.38,1.1,str(round(a,2)), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})

	data11=pd.read_csv(r"./data/code/temporal/temp/autumn_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['Date_Mid_Senescence_Phase_1_temp_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, _ =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a+0.01,1.1,str(round(a,2)), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	
	data11=pd.read_csv(r"./data/code/temporal/temp/summer_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Onset_Greenness_Maximum_1_temp_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, _ =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a+0.01,1.1,str(round(a,2)), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	
	data11=pd.read_csv(r"./data/code/temporal/temp/all_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Growing_Season_Area_1_temp_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, _ =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a+0.01,1.1,str(round(a,2)), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	
	data11=pd.read_csv(r"./temporal/co2/summer_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Onset_Greenness_Maximum_1_co2_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, _ =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a+0.01,1.1,'0.12', fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	
	data11=pd.read_csv(r"./temporal/co2/all_temporal.csv")
	Date_Mid_Greenup_Phase_1_r=np.array(data11['EVI2_Growing_Season_Area_1_co2_r'])
	a=np.nanmean(Date_Mid_Greenup_Phase_1_r)
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	hist, bins, _ =ax.hist(Date_Mid_Greenup_Phase_1_r, bins=50, density=True,edgecolor='white',color='silver')
	params, _ = curve_fit(gaussian, bins[:-1], hist)
	mean, std_dev = params
	x_range = np.linspace(min(bins), max(bins), 100)
	fitted_curve = gaussian(x_range, mean, std_dev)
	ax.plot(x_range, fitted_curve, 'r-', label='Fitted Curve',linewidth=3)
	ax.axvline(0,color='black', linestyle='--',linewidth=3)
	ax.axvline(a,color='red', linestyle='--',linewidth=3)
	ax.text(a+0.01,1.1,'0.17', fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_ylim(0,1.225)
	ax.set_xticks(np.arange(-1,1.5,0.5))
	ax.set_xticklabels([-1,-0.5,-0,0.5,1], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_yticks(np.arange(0,1.5,0.5))
	ax.set_yticklabels([0,0.5,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	plt.show()	

def pvalue():
	projections = [ccrs.AzimuthalEquidistant(central_longitude=0.0, central_latitude=90.0)]
	data=pd.read_csv(r"./data/code/temporal/pvalue.csv")
	Date_Mid_Greenup_Phase_1_o3_p=np.array(data['Date_Mid_Greenup_Phase_1_aot40_p'])
	Date_Mid_Senescence_Phase_1_o3_p=np.array(data['Date_Mid_Senescence_Phase_1_aot40_p'])
	EVI2_Onset_Greenness_Maximum_1_o3_p=np.array(data['EVI2_Onset_Greenness_Maximum_1_aot40_p'])
	EVI2_Growing_Season_Area_1_o3_p=np.array(data['EVI2_Growing_Season_Area_1_aot40_p'])
	latpostivesig=lat[(Date_Mid_Greenup_Phase_1_aot40_p<=0.05)&(Date_Mid_Greenup_Phase_1_aot40_r>0)]
	lonpostivesig=lon[(Date_Mid_Greenup_Phase_1_aot40_p<=0.05)&(Date_Mid_Greenup_Phase_1_aot40_r>0)]
	latnegativesig=lat[(Date_Mid_Greenup_Phase_1_aot40_p<=0.05)&(Date_Mid_Greenup_Phase_1_aot40_r<0)]
	lonnegativesig=lon[(Date_Mid_Greenup_Phase_1_aot40_p<=0.05)&(Date_Mid_Greenup_Phase_1_aot40_r<0)]
	latpostivenosig=lat[(Date_Mid_Greenup_Phase_1_aot40_p>0.05)&(Date_Mid_Greenup_Phase_1_aot40_r>0)]
	lonpostivenosig=lon[(Date_Mid_Greenup_Phase_1_aot40_p>0.05)&(Date_Mid_Greenup_Phase_1_aot40_r>0)]
	latnegativenosig=lat[(Date_Mid_Greenup_Phase_1_aot40_p>0.05)&(Date_Mid_Greenup_Phase_1_aot40_r<0)]
	lonnegativenosig=lon[(Date_Mid_Greenup_Phase_1_aot40_p>0.05)&(Date_Mid_Greenup_Phase_1_aot40_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())
	
	latpostivesig=lat[(Date_Mid_Senescence_Phase_1_aot40_p<=0.05)&(Date_Mid_Senescence_Phase_1_aot40_r>0)]
	lonpostivesig=lon[(Date_Mid_Senescence_Phase_1_aot40_p<=0.05)&(Date_Mid_Senescence_Phase_1_aot40_r>0)]
	latnegativesig=lat[(Date_Mid_Senescence_Phase_1_aot40_p<=0.05)&(Date_Mid_Senescence_Phase_1_aot40_r<0)]
	lonnegativesig=lon[(Date_Mid_Senescence_Phase_1_aot40_p<=0.05)&(Date_Mid_Senescence_Phase_1_aot40_r<0)]
	latpostivenosig=lat[(Date_Mid_Senescence_Phase_1_aot40_p>0.05)&(Date_Mid_Senescence_Phase_1_aot40_r>0)]
	lonpostivenosig=lon[(Date_Mid_Senescence_Phase_1_aot40_p>0.05)&(Date_Mid_Senescence_Phase_1_aot40_r>0)]
	latnegativenosig=lat[(Date_Mid_Senescence_Phase_1_aot40_p>0.05)&(Date_Mid_Senescence_Phase_1_aot40_r<0)]
	lonnegativenosig=lon[(Date_Mid_Senescence_Phase_1_aot40_p>0.05)&(Date_Mid_Senescence_Phase_1_aot40_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())

	latpostivesig=lat[(EVI2_Onset_Greenness_Maximum_1_aot40_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r>0)]
	lonpostivesig=lon[(EVI2_Onset_Greenness_Maximum_1_aot40_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r>0)]
	latnegativesig=lat[(EVI2_Onset_Greenness_Maximum_1_aot40_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r<0)]
	lonnegativesig=lon[(EVI2_Onset_Greenness_Maximum_1_aot40_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r<0)]
	latpostivenosig=lat[(EVI2_Onset_Greenness_Maximum_1_aot40_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r>0)]
	lonpostivenosig=lon[(EVI2_Onset_Greenness_Maximum_1_aot40_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r>0)]
	latnegativenosig=lat[(EVI2_Onset_Greenness_Maximum_1_aot40_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r<0)]
	lonnegativenosig=lon[(EVI2_Onset_Greenness_Maximum_1_aot40_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())

	latpostivesig=lat[(EVI2_Growing_Season_Area_1_aot40_p<=0.05)&(EVI2_Growing_Season_Area_1_aot40_r>0)]
	lonpostivesig=lon[(EVI2_Growing_Season_Area_1_aot40_p<=0.05)&(EVI2_Growing_Season_Area_1_aot40_r>0)]
	latnegativesig=lat[(EVI2_Growing_Season_Area_1_aot40_p<=0.05)&(EVI2_Growing_Season_Area_1_aot40_r<0)]
	lonnegativesig=lon[(EVI2_Growing_Season_Area_1_aot40_p<=0.05)&(EVI2_Growing_Season_Area_1_aot40_r<0)]
	latpostivenosig=lat[(EVI2_Growing_Season_Area_1_aot40_p>0.05)&(EVI2_Growing_Season_Area_1_aot40_r>0)]
	lonpostivenosig=lon[(EVI2_Growing_Season_Area_1_aot40_p>0.05)&(EVI2_Growing_Season_Area_1_aot40_r>0)]
	latnegativenosig=lat[(EVI2_Growing_Season_Area_1_aot40_p>0.05)&(EVI2_Growing_Season_Area_1_aot40_r<0)]
	lonnegativenosig=lon[(EVI2_Growing_Season_Area_1_aot40_p>0.05)&(EVI2_Growing_Season_Area_1_aot40_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())

	Date_Mid_Greenup_Phase_1_o3_p=np.array(data['Date_Mid_Greenup_Phase_1_o3_p'])
	Date_Mid_Senescence_Phase_1_o3_p=np.array(data['Date_Mid_Senescence_Phase_1_o3_p'])
	EVI2_Onset_Greenness_Maximum_1_o3_p=np.array(data['EVI2_Onset_Greenness_Maximum_1_o3_p'])
	EVI2_Growing_Season_Area_1_o3_p=np.array(data['EVI2_Growing_Season_Area_1_o3_p'])
	latpostivesig=lat[(Date_Mid_Greenup_Phase_1_o3_p<=0.05)&(Date_Mid_Greenup_Phase_1_o3_r>0)]
	lonpostivesig=lon[(Date_Mid_Greenup_Phase_1_o3_p<=0.05)&(Date_Mid_Greenup_Phase_1_o3_r>0)]
	latnegativesig=lat[(Date_Mid_Greenup_Phase_1_o3_p<=0.05)&(Date_Mid_Greenup_Phase_1_o3_r<0)]
	lonnegativesig=lon[(Date_Mid_Greenup_Phase_1_o3_p<=0.05)&(Date_Mid_Greenup_Phase_1_o3_r<0)]
	latpostivenosig=lat[(Date_Mid_Greenup_Phase_1_o3_p>0.05)&(Date_Mid_Greenup_Phase_1_o3_r>0)]
	lonpostivenosig=lon[(Date_Mid_Greenup_Phase_1_o3_p>0.05)&(Date_Mid_Greenup_Phase_1_o3_r>0)]
	latnegativenosig=lat[(Date_Mid_Greenup_Phase_1_o3_p>0.05)&(Date_Mid_Greenup_Phase_1_o3_r<0)]
	lonnegativenosig=lon[(Date_Mid_Greenup_Phase_1_o3_p>0.05)&(Date_Mid_Greenup_Phase_1_o3_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())

	latpostivesig=lat[(Date_Mid_Senescence_Phase_1_o3_p<=0.05)&(Date_Mid_Senescence_Phase_1_o3_r>0)]
	lonpostivesig=lon[(Date_Mid_Senescence_Phase_1_o3_p<=0.05)&(Date_Mid_Senescence_Phase_1_o3_r>0)]
	latnegativesig=lat[(Date_Mid_Senescence_Phase_1_o3_p<=0.05)&(Date_Mid_Senescence_Phase_1_o3_r<0)]
	lonnegativesig=lon[(Date_Mid_Senescence_Phase_1_o3_p<=0.05)&(Date_Mid_Senescence_Phase_1_o3_r<0)]
	latpostivenosig=lat[(Date_Mid_Senescence_Phase_1_o3_p>0.05)&(Date_Mid_Senescence_Phase_1_o3_r>0)]
	lonpostivenosig=lon[(Date_Mid_Senescence_Phase_1_o3_p>0.05)&(Date_Mid_Senescence_Phase_1_o3_r>0)]
	latnegativenosig=lat[(Date_Mid_Senescence_Phase_1_o3_p>0.05)&(Date_Mid_Senescence_Phase_1_o3_r<0)]
	lonnegativenosig=lon[(Date_Mid_Senescence_Phase_1_o3_p>0.05)&(Date_Mid_Senescence_Phase_1_o3_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())

	latpostivesig=lat[(EVI2_Onset_Greenness_Maximum_1_o3_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r>0)]
	lonpostivesig=lon[(EVI2_Onset_Greenness_Maximum_1_o3_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r>0)]
	latnegativesig=lat[(EVI2_Onset_Greenness_Maximum_1_o3_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r<0)]
	lonnegativesig=lon[(EVI2_Onset_Greenness_Maximum_1_o3_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r<0)]
	latpostivenosig=lat[(EVI2_Onset_Greenness_Maximum_1_o3_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r>0)]
	lonpostivenosig=lon[(EVI2_Onset_Greenness_Maximum_1_o3_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r>0)]
	latnegativenosig=lat[(EVI2_Onset_Greenness_Maximum_1_o3_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r<0)]
	lonnegativenosig=lon[(EVI2_Onset_Greenness_Maximum_1_o3_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())

	latpostivesig=lat[(EVI2_Growing_Season_Area_1_o3_p<=0.05)&(EVI2_Growing_Season_Area_1_o3_r>0)]
	lonpostivesig=lon[(EVI2_Growing_Season_Area_1_o3_p<=0.05)&(EVI2_Growing_Season_Area_1_o3_r>0)]
	latnegativesig=lat[(EVI2_Growing_Season_Area_1_o3_p<=0.05)&(EVI2_Growing_Season_Area_1_o3_r<0)]
	lonnegativesig=lon[(EVI2_Growing_Season_Area_1_o3_p<=0.05)&(EVI2_Growing_Season_Area_1_o3_r<0)]
	latpostivenosig=lat[(EVI2_Growing_Season_Area_1_o3_p>0.05)&(EVI2_Growing_Season_Area_1_o3_r>0)]
	lonpostivenosig=lon[(EVI2_Growing_Season_Area_1_o3_p>0.05)&(EVI2_Growing_Season_Area_1_o3_r>0)]
	latnegativenosig=lat[(EVI2_Growing_Season_Area_1_o3_p>0.05)&(EVI2_Growing_Season_Area_1_o3_r<0)]
	lonnegativenosig=lon[(EVI2_Growing_Season_Area_1_o3_p>0.05)&(EVI2_Growing_Season_Area_1_o3_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())

	Date_Mid_Greenup_Phase_1_temp_p=np.array(data['Date_Mid_Greenup_Phase_1_temp_p'])
	Date_Mid_Senescence_Phase_1_temp_p=np.array(data['Date_Mid_Senescence_Phase_1_temp_p'])
	EVI2_Onset_Greenness_Maximum_1_temp_p=np.array(data['EVI2_Onset_Greenness_Maximum_1_temp_p'])
	EVI2_Growing_Season_Area_1_temp_p=np.array(data['EVI2_Growing_Season_Area_1_temp_p'])
	latpostivesig=lat[(Date_Mid_Greenup_Phase_1_temp_p<=0.05)&(Date_Mid_Greenup_Phase_1_temp_r>0)]
	lonpostivesig=lon[(Date_Mid_Greenup_Phase_1_temp_p<=0.05)&(Date_Mid_Greenup_Phase_1_temp_r>0)]
	latnegativesig=lat[(Date_Mid_Greenup_Phase_1_temp_p<=0.05)&(Date_Mid_Greenup_Phase_1_temp_r<0)]
	lonnegativesig=lon[(Date_Mid_Greenup_Phase_1_temp_p<=0.05)&(Date_Mid_Greenup_Phase_1_temp_r<0)]
	latpostivenosig=lat[(Date_Mid_Greenup_Phase_1_temp_p>0.05)&(Date_Mid_Greenup_Phase_1_temp_r>0)]
	lonpostivenosig=lon[(Date_Mid_Greenup_Phase_1_temp_p>0.05)&(Date_Mid_Greenup_Phase_1_temp_r>0)]
	latnegativenosig=lat[(Date_Mid_Greenup_Phase_1_temp_p>0.05)&(Date_Mid_Greenup_Phase_1_temp_r<0)]
	lonnegativenosig=lon[(Date_Mid_Greenup_Phase_1_temp_p>0.05)&(Date_Mid_Greenup_Phase_1_temp_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())

	latpostivesig=lat[(Date_Mid_Senescence_Phase_1_temp_p<=0.05)&(Date_Mid_Senescence_Phase_1_temp_r>0)]
	lonpostivesig=lon[(Date_Mid_Senescence_Phase_1_temp_p<=0.05)&(Date_Mid_Senescence_Phase_1_temp_r>0)]
	latnegativesig=lat[(Date_Mid_Senescence_Phase_1_temp_p<=0.05)&(Date_Mid_Senescence_Phase_1_temp_r<0)]
	lonnegativesig=lon[(Date_Mid_Senescence_Phase_1_temp_p<=0.05)&(Date_Mid_Senescence_Phase_1_temp_r<0)]
	latpostivenosig=lat[(Date_Mid_Senescence_Phase_1_temp_p>0.05)&(Date_Mid_Senescence_Phase_1_temp_r>0)]
	lonpostivenosig=lon[(Date_Mid_Senescence_Phase_1_temp_p>0.05)&(Date_Mid_Senescence_Phase_1_temp_r>0)]
	latnegativenosig=lat[(Date_Mid_Senescence_Phase_1_temp_p>0.05)&(Date_Mid_Senescence_Phase_1_temp_r<0)]
	lonnegativenosig=lon[(Date_Mid_Senescence_Phase_1_temp_p>0.05)&(Date_Mid_Senescence_Phase_1_temp_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())

	latpostivesig=lat[(EVI2_Onset_Greenness_Maximum_1_temp_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r>0)]
	lonpostivesig=lon[(EVI2_Onset_Greenness_Maximum_1_temp_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r>0)]
	latnegativesig=lat[(EVI2_Onset_Greenness_Maximum_1_temp_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r<0)]
	lonnegativesig=lon[(EVI2_Onset_Greenness_Maximum_1_temp_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r<0)]
	latpostivenosig=lat[(EVI2_Onset_Greenness_Maximum_1_temp_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r>0)]
	lonpostivenosig=lon[(EVI2_Onset_Greenness_Maximum_1_temp_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r>0)]
	latnegativenosig=lat[(EVI2_Onset_Greenness_Maximum_1_temp_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r<0)]
	lonnegativenosig=lon[(EVI2_Onset_Greenness_Maximum_1_temp_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')

	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())

	latpostivesig=lat[(EVI2_Growing_Season_Area_1_temp_p<=0.05)&(EVI2_Growing_Season_Area_1_temp_r>0)]
	lonpostivesig=lon[(EVI2_Growing_Season_Area_1_temp_p<=0.05)&(EVI2_Growing_Season_Area_1_temp_r>0)]
	latnegativesig=lat[(EVI2_Growing_Season_Area_1_temp_p<=0.05)&(EVI2_Growing_Season_Area_1_temp_r<0)]
	lonnegativesig=lon[(EVI2_Growing_Season_Area_1_temp_p<=0.05)&(EVI2_Growing_Season_Area_1_temp_r<0)]
	latpostivenosig=lat[(EVI2_Growing_Season_Area_1_temp_p>0.05)&(EVI2_Growing_Season_Area_1_temp_r>0)]
	lonpostivenosig=lon[(EVI2_Growing_Season_Area_1_temp_p>0.05)&(EVI2_Growing_Season_Area_1_temp_r>0)]
	latnegativenosig=lat[(EVI2_Growing_Season_Area_1_temp_p>0.05)&(EVI2_Growing_Season_Area_1_temp_r<0)]
	lonnegativenosig=lon[(EVI2_Growing_Season_Area_1_temp_p>0.05)&(EVI2_Growing_Season_Area_1_temp_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())

	EVI2_Onset_Greenness_Maximum_1_co2_p=np.array(data['EVI2_Onset_Greenness_Maximum_1_co2_p'])
	EVI2_Growing_Season_Area_1_co2_p=np.array(data['EVI2_Growing_Season_Area_1_co2_p'])
	latpostivesig=lat[(EVI2_Onset_Greenness_Maximum_1_co2_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r>0)]
	lonpostivesig=lon[(EVI2_Onset_Greenness_Maximum_1_co2_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r>0)]
	latnegativesig=lat[(EVI2_Onset_Greenness_Maximum_1_co2_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r<0)]
	lonnegativesig=lon[(EVI2_Onset_Greenness_Maximum_1_co2_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r<0)]
	latpostivenosig=lat[(EVI2_Onset_Greenness_Maximum_1_co2_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r>0)]
	lonpostivenosig=lon[(EVI2_Onset_Greenness_Maximum_1_co2_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r>0)]
	latnegativenosig=lat[(EVI2_Onset_Greenness_Maximum_1_co2_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r<0)]
	lonnegativenosig=lon[(EVI2_Onset_Greenness_Maximum_1_co2_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())
	
	latpostivesig=lat[(EVI2_Growing_Season_Area_1_co2_p<=0.05)&(EVI2_Growing_Season_Area_1_co2_r>0)]
	lonpostivesig=lon[(EVI2_Growing_Season_Area_1_co2_p<=0.05)&(EVI2_Growing_Season_Area_1_co2_r>0)]
	latnegativesig=lat[(EVI2_Growing_Season_Area_1_co2_p<=0.05)&(EVI2_Growing_Season_Area_1_co2_r<0)]
	lonnegativesig=lon[(EVI2_Growing_Season_Area_1_co2_p<=0.05)&(EVI2_Growing_Season_Area_1_co2_r<0)]
	latpostivenosig=lat[(EVI2_Growing_Season_Area_1_co2_p>0.05)&(EVI2_Growing_Season_Area_1_co2_r>0)]
	lonpostivenosig=lon[(EVI2_Growing_Season_Area_1_co2_p>0.05)&(EVI2_Growing_Season_Area_1_co2_r>0)]
	latnegativenosig=lat[(EVI2_Growing_Season_Area_1_co2_p>0.05)&(EVI2_Growing_Season_Area_1_co2_r<0)]
	lonnegativenosig=lon[(EVI2_Growing_Season_Area_1_co2_p>0.05)&(EVI2_Growing_Season_Area_1_co2_r<0)]	
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection=projections[0])
	extents = [-180, 180, 15, 60]
	ax.set_extent(extents, crs=ccrs.PlateCarree())
	ax.coastlines()
	ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')	
	theta = np.linspace(0, 2*np.pi, 100)
	center, radius = [0.5, 0.5], 0.5
	verts = np.vstack([np.sin(theta), np.cos(theta)]).T
	circle = mpath.Path(verts * radius + center)
	ax.set_boundary(circle, transform=ax.transAxes)
	ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black',color='floralwhite')
	ax.scatter(lonpostivenosig,latpostivenosig,s=6,color='blue',label='Postive',transform=ccrs.PlateCarree())	
	ax.scatter(lonnegativenosig,latnegativenosig,s=6,color='cyan',label='Negative',transform=ccrs.PlateCarree())
	ax.scatter(lonnegativesig,latnegativesig,s=6,color='green',label='Negative${^*}$',transform=ccrs.PlateCarree())
	ax.scatter(lonpostivesig,latpostivesig,s=6,color='red',label='Postive${^*}$',transform=ccrs.PlateCarree())
	plt.show()
		
def pvalue_hist():
	data=pd.read_csv(r"./data/code/temporal/pvalue.csv")
	Date_Mid_Greenup_Phase_1_o3_p=np.array(data['Date_Mid_Greenup_Phase_1_aot40_p'])
	Date_Mid_Senescence_Phase_1_o3_p=np.array(data['Date_Mid_Senescence_Phase_1_aot40_p'])
	EVI2_Onset_Greenness_Maximum_1_o3_p=np.array(data['EVI2_Onset_Greenness_Maximum_1_aot40_p'])
	EVI2_Growing_Season_Area_1_o3_p=np.array(data['EVI2_Growing_Season_Area_1_aot40_p'])
	
	latpostivesig=lat[(Date_Mid_Greenup_Phase_1_aot40_p<=0.05)&(Date_Mid_Greenup_Phase_1_aot40_r>0)]
	lonpostivesig=lon[(Date_Mid_Greenup_Phase_1_aot40_p<=0.05)&(Date_Mid_Greenup_Phase_1_aot40_r>0)]
	latnegativesig=lat[(Date_Mid_Greenup_Phase_1_aot40_p<=0.05)&(Date_Mid_Greenup_Phase_1_aot40_r<0)]
	lonnegativesig=lon[(Date_Mid_Greenup_Phase_1_aot40_p<=0.05)&(Date_Mid_Greenup_Phase_1_aot40_r<0)]
	latpostivenosig=lat[(Date_Mid_Greenup_Phase_1_aot40_p>0.05)&(Date_Mid_Greenup_Phase_1_aot40_r>0)]
	lonpostivenosig=lon[(Date_Mid_Greenup_Phase_1_aot40_p>0.05)&(Date_Mid_Greenup_Phase_1_aot40_r>0)]
	latnegativenosig=lat[(Date_Mid_Greenup_Phase_1_aot40_p>0.05)&(Date_Mid_Greenup_Phase_1_aot40_r<0)]
	lonnegativenosig=lon[(Date_Mid_Greenup_Phase_1_aot40_p>0.05)&(Date_Mid_Greenup_Phase_1_aot40_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})

	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})
	
	latpostivesig=lat[(Date_Mid_Senescence_Phase_1_aot40_p<=0.05)&(Date_Mid_Senescence_Phase_1_aot40_r>0)]
	lonpostivesig=lon[(Date_Mid_Senescence_Phase_1_aot40_p<=0.05)&(Date_Mid_Senescence_Phase_1_aot40_r>0)]
	latnegativesig=lat[(Date_Mid_Senescence_Phase_1_aot40_p<=0.05)&(Date_Mid_Senescence_Phase_1_aot40_r<0)]
	lonnegativesig=lon[(Date_Mid_Senescence_Phase_1_aot40_p<=0.05)&(Date_Mid_Senescence_Phase_1_aot40_r<0)]
	latpostivenosig=lat[(Date_Mid_Senescence_Phase_1_aot40_p>0.05)&(Date_Mid_Senescence_Phase_1_aot40_r>0)]
	lonpostivenosig=lon[(Date_Mid_Senescence_Phase_1_aot40_p>0.05)&(Date_Mid_Senescence_Phase_1_aot40_r>0)]
	latnegativenosig=lat[(Date_Mid_Senescence_Phase_1_aot40_p>0.05)&(Date_Mid_Senescence_Phase_1_aot40_r<0)]
	lonnegativenosig=lon[(Date_Mid_Senescence_Phase_1_aot40_p>0.05)&(Date_Mid_Senescence_Phase_1_aot40_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})

	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})
		
	latpostivesig=lat[(EVI2_Onset_Greenness_Maximum_1_aot40_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r>0)]
	lonpostivesig=lon[(EVI2_Onset_Greenness_Maximum_1_aot40_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r>0)]
	latnegativesig=lat[(EVI2_Onset_Greenness_Maximum_1_aot40_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r<0)]
	lonnegativesig=lon[(EVI2_Onset_Greenness_Maximum_1_aot40_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r<0)]
	latpostivenosig=lat[(EVI2_Onset_Greenness_Maximum_1_aot40_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r>0)]
	lonpostivenosig=lon[(EVI2_Onset_Greenness_Maximum_1_aot40_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r>0)]
	latnegativenosig=lat[(EVI2_Onset_Greenness_Maximum_1_aot40_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r<0)]
	lonnegativenosig=lon[(EVI2_Onset_Greenness_Maximum_1_aot40_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_aot40_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})

	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})
		
	latpostivesig=lat[(EVI2_Growing_Season_Area_1_aot40_p<=0.05)&(EVI2_Growing_Season_Area_1_aot40_r>0)]
	lonpostivesig=lon[(EVI2_Growing_Season_Area_1_aot40_p<=0.05)&(EVI2_Growing_Season_Area_1_aot40_r>0)]
	latnegativesig=lat[(EVI2_Growing_Season_Area_1_aot40_p<=0.05)&(EVI2_Growing_Season_Area_1_aot40_r<0)]
	lonnegativesig=lon[(EVI2_Growing_Season_Area_1_aot40_p<=0.05)&(EVI2_Growing_Season_Area_1_aot40_r<0)]
	latpostivenosig=lat[(EVI2_Growing_Season_Area_1_aot40_p>0.05)&(EVI2_Growing_Season_Area_1_aot40_r>0)]
	lonpostivenosig=lon[(EVI2_Growing_Season_Area_1_aot40_p>0.05)&(EVI2_Growing_Season_Area_1_aot40_r>0)]
	latnegativenosig=lat[(EVI2_Growing_Season_Area_1_aot40_p>0.05)&(EVI2_Growing_Season_Area_1_aot40_r<0)]
	lonnegativenosig=lon[(EVI2_Growing_Season_Area_1_aot40_p>0.05)&(EVI2_Growing_Season_Area_1_aot40_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})

	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})

	Date_Mid_Greenup_Phase_1_o3_p=np.array(data['Date_Mid_Greenup_Phase_1_o3_p'])
	Date_Mid_Senescence_Phase_1_o3_p=np.array(data['Date_Mid_Senescence_Phase_1_o3_p'])
	EVI2_Onset_Greenness_Maximum_1_o3_p=np.array(data['EVI2_Onset_Greenness_Maximum_1_o3_p'])
	EVI2_Growing_Season_Area_1_o3_p=np.array(data['EVI2_Growing_Season_Area_1_o3_p'])
	
	latpostivesig=lat[(Date_Mid_Greenup_Phase_1_o3_p<=0.05)&(Date_Mid_Greenup_Phase_1_o3_r>0)]
	lonpostivesig=lon[(Date_Mid_Greenup_Phase_1_o3_p<=0.05)&(Date_Mid_Greenup_Phase_1_o3_r>0)]
	latnegativesig=lat[(Date_Mid_Greenup_Phase_1_o3_p<=0.05)&(Date_Mid_Greenup_Phase_1_o3_r<0)]
	lonnegativesig=lon[(Date_Mid_Greenup_Phase_1_o3_p<=0.05)&(Date_Mid_Greenup_Phase_1_o3_r<0)]
	latpostivenosig=lat[(Date_Mid_Greenup_Phase_1_o3_p>0.05)&(Date_Mid_Greenup_Phase_1_o3_r>0)]
	lonpostivenosig=lon[(Date_Mid_Greenup_Phase_1_o3_p>0.05)&(Date_Mid_Greenup_Phase_1_o3_r>0)]
	latnegativenosig=lat[(Date_Mid_Greenup_Phase_1_o3_p>0.05)&(Date_Mid_Greenup_Phase_1_o3_r<0)]
	lonnegativenosig=lon[(Date_Mid_Greenup_Phase_1_o3_p>0.05)&(Date_Mid_Greenup_Phase_1_o3_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})

	latpostivesig=lat[(Date_Mid_Senescence_Phase_1_o3_p<=0.05)&(Date_Mid_Senescence_Phase_1_o3_r>0)]
	lonpostivesig=lon[(Date_Mid_Senescence_Phase_1_o3_p<=0.05)&(Date_Mid_Senescence_Phase_1_o3_r>0)]
	latnegativesig=lat[(Date_Mid_Senescence_Phase_1_o3_p<=0.05)&(Date_Mid_Senescence_Phase_1_o3_r<0)]
	lonnegativesig=lon[(Date_Mid_Senescence_Phase_1_o3_p<=0.05)&(Date_Mid_Senescence_Phase_1_o3_r<0)]
	latpostivenosig=lat[(Date_Mid_Senescence_Phase_1_o3_p>0.05)&(Date_Mid_Senescence_Phase_1_o3_r>0)]
	lonpostivenosig=lon[(Date_Mid_Senescence_Phase_1_o3_p>0.05)&(Date_Mid_Senescence_Phase_1_o3_r>0)]
	latnegativenosig=lat[(Date_Mid_Senescence_Phase_1_o3_p>0.05)&(Date_Mid_Senescence_Phase_1_o3_r<0)]
	lonnegativenosig=lon[(Date_Mid_Senescence_Phase_1_o3_p>0.05)&(Date_Mid_Senescence_Phase_1_o3_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})
		
	latpostivesig=lat[(EVI2_Onset_Greenness_Maximum_1_o3_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r>0)]
	lonpostivesig=lon[(EVI2_Onset_Greenness_Maximum_1_o3_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r>0)]
	latnegativesig=lat[(EVI2_Onset_Greenness_Maximum_1_o3_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r<0)]
	lonnegativesig=lon[(EVI2_Onset_Greenness_Maximum_1_o3_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r<0)]
	latpostivenosig=lat[(EVI2_Onset_Greenness_Maximum_1_o3_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r>0)]
	lonpostivenosig=lon[(EVI2_Onset_Greenness_Maximum_1_o3_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r>0)]
	latnegativenosig=lat[(EVI2_Onset_Greenness_Maximum_1_o3_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r<0)]
	lonnegativenosig=lon[(EVI2_Onset_Greenness_Maximum_1_o3_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_o3_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})
		
	latpostivesig=lat[(EVI2_Growing_Season_Area_1_o3_p<=0.05)&(EVI2_Growing_Season_Area_1_o3_r>0)]
	lonpostivesig=lon[(EVI2_Growing_Season_Area_1_o3_p<=0.05)&(EVI2_Growing_Season_Area_1_o3_r>0)]
	latnegativesig=lat[(EVI2_Growing_Season_Area_1_o3_p<=0.05)&(EVI2_Growing_Season_Area_1_o3_r<0)]
	lonnegativesig=lon[(EVI2_Growing_Season_Area_1_o3_p<=0.05)&(EVI2_Growing_Season_Area_1_o3_r<0)]
	latpostivenosig=lat[(EVI2_Growing_Season_Area_1_o3_p>0.05)&(EVI2_Growing_Season_Area_1_o3_r>0)]
	lonpostivenosig=lon[(EVI2_Growing_Season_Area_1_o3_p>0.05)&(EVI2_Growing_Season_Area_1_o3_r>0)]
	latnegativenosig=lat[(EVI2_Growing_Season_Area_1_o3_p>0.05)&(EVI2_Growing_Season_Area_1_o3_r<0)]
	lonnegativenosig=lon[(EVI2_Growing_Season_Area_1_o3_p>0.05)&(EVI2_Growing_Season_Area_1_o3_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})
		
	Date_Mid_Greenup_Phase_1_temp_p=np.array(data['Date_Mid_Greenup_Phase_1_temp_p'])
	Date_Mid_Senescence_Phase_1_temp_p=np.array(data['Date_Mid_Senescence_Phase_1_temp_p'])
	EVI2_Onset_Greenness_Maximum_1_temp_p=np.array(data['EVI2_Onset_Greenness_Maximum_1_temp_p'])
	EVI2_Growing_Season_Area_1_temp_p=np.array(data['EVI2_Growing_Season_Area_1_temp_p'])
	
	latpostivesig=lat[(Date_Mid_Greenup_Phase_1_temp_p<=0.05)&(Date_Mid_Greenup_Phase_1_temp_r>0)]
	lonpostivesig=lon[(Date_Mid_Greenup_Phase_1_temp_p<=0.05)&(Date_Mid_Greenup_Phase_1_temp_r>0)]
	latnegativesig=lat[(Date_Mid_Greenup_Phase_1_temp_p<=0.05)&(Date_Mid_Greenup_Phase_1_temp_r<0)]
	lonnegativesig=lon[(Date_Mid_Greenup_Phase_1_temp_p<=0.05)&(Date_Mid_Greenup_Phase_1_temp_r<0)]
	latpostivenosig=lat[(Date_Mid_Greenup_Phase_1_temp_p>0.05)&(Date_Mid_Greenup_Phase_1_temp_r>0)]
	lonpostivenosig=lon[(Date_Mid_Greenup_Phase_1_temp_p>0.05)&(Date_Mid_Greenup_Phase_1_temp_r>0)]
	latnegativenosig=lat[(Date_Mid_Greenup_Phase_1_temp_p>0.05)&(Date_Mid_Greenup_Phase_1_temp_r<0)]
	lonnegativenosig=lon[(Date_Mid_Greenup_Phase_1_temp_p>0.05)&(Date_Mid_Greenup_Phase_1_temp_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})
	
	latpostivesig=lat[(Date_Mid_Senescence_Phase_1_temp_p<=0.05)&(Date_Mid_Senescence_Phase_1_temp_r>0)]
	lonpostivesig=lon[(Date_Mid_Senescence_Phase_1_temp_p<=0.05)&(Date_Mid_Senescence_Phase_1_temp_r>0)]
	latnegativesig=lat[(Date_Mid_Senescence_Phase_1_temp_p<=0.05)&(Date_Mid_Senescence_Phase_1_temp_r<0)]
	lonnegativesig=lon[(Date_Mid_Senescence_Phase_1_temp_p<=0.05)&(Date_Mid_Senescence_Phase_1_temp_r<0)]
	latpostivenosig=lat[(Date_Mid_Senescence_Phase_1_temp_p>0.05)&(Date_Mid_Senescence_Phase_1_temp_r>0)]
	lonpostivenosig=lon[(Date_Mid_Senescence_Phase_1_temp_p>0.05)&(Date_Mid_Senescence_Phase_1_temp_r>0)]
	latnegativenosig=lat[(Date_Mid_Senescence_Phase_1_temp_p>0.05)&(Date_Mid_Senescence_Phase_1_temp_r<0)]
	lonnegativenosig=lon[(Date_Mid_Senescence_Phase_1_temp_p>0.05)&(Date_Mid_Senescence_Phase_1_temp_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})

	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})
		
	latpostivesig=lat[(EVI2_Onset_Greenness_Maximum_1_temp_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r>0)]
	lonpostivesig=lon[(EVI2_Onset_Greenness_Maximum_1_temp_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r>0)]
	latnegativesig=lat[(EVI2_Onset_Greenness_Maximum_1_temp_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r<0)]
	lonnegativesig=lon[(EVI2_Onset_Greenness_Maximum_1_temp_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r<0)]
	latpostivenosig=lat[(EVI2_Onset_Greenness_Maximum_1_temp_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r>0)]
	lonpostivenosig=lon[(EVI2_Onset_Greenness_Maximum_1_temp_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r>0)]
	latnegativenosig=lat[(EVI2_Onset_Greenness_Maximum_1_temp_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r<0)]
	lonnegativenosig=lon[(EVI2_Onset_Greenness_Maximum_1_temp_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_temp_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})

	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})
		
	latpostivesig=lat[(EVI2_Growing_Season_Area_1_temp_p<=0.05)&(EVI2_Growing_Season_Area_1_temp_r>0)]
	lonpostivesig=lon[(EVI2_Growing_Season_Area_1_temp_p<=0.05)&(EVI2_Growing_Season_Area_1_temp_r>0)]
	latnegativesig=lat[(EVI2_Growing_Season_Area_1_temp_p<=0.05)&(EVI2_Growing_Season_Area_1_temp_r<0)]
	lonnegativesig=lon[(EVI2_Growing_Season_Area_1_temp_p<=0.05)&(EVI2_Growing_Season_Area_1_temp_r<0)]
	latpostivenosig=lat[(EVI2_Growing_Season_Area_1_temp_p>0.05)&(EVI2_Growing_Season_Area_1_temp_r>0)]
	lonpostivenosig=lon[(EVI2_Growing_Season_Area_1_temp_p>0.05)&(EVI2_Growing_Season_Area_1_temp_r>0)]
	latnegativenosig=lat[(EVI2_Growing_Season_Area_1_temp_p>0.05)&(EVI2_Growing_Season_Area_1_temp_r<0)]
	lonnegativenosig=lon[(EVI2_Growing_Season_Area_1_temp_p>0.05)&(EVI2_Growing_Season_Area_1_temp_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})

	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})

	EVI2_Onset_Greenness_Maximum_1_co2_p=np.array(data['EVI2_Onset_Greenness_Maximum_1_co2_p'])
	EVI2_Growing_Season_Area_1_co2_p=np.array(data['EVI2_Growing_Season_Area_1_co2_p'])
	latpostivesig=lat[(EVI2_Onset_Greenness_Maximum_1_co2_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r>0)]
	lonpostivesig=lon[(EVI2_Onset_Greenness_Maximum_1_co2_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r>0)]
	latnegativesig=lat[(EVI2_Onset_Greenness_Maximum_1_co2_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r<0)]
	lonnegativesig=lon[(EVI2_Onset_Greenness_Maximum_1_co2_p<=0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r<0)]
	latpostivenosig=lat[(EVI2_Onset_Greenness_Maximum_1_co2_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r>0)]
	lonpostivenosig=lon[(EVI2_Onset_Greenness_Maximum_1_co2_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r>0)]
	latnegativenosig=lat[(EVI2_Onset_Greenness_Maximum_1_co2_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r<0)]
	lonnegativenosig=lon[(EVI2_Onset_Greenness_Maximum_1_co2_p>0.05)&(EVI2_Onset_Greenness_Maximum_1_co2_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})
	
	latpostivesig=lat[(EVI2_Growing_Season_Area_1_co2_p<=0.05)&(EVI2_Growing_Season_Area_1_co2_r>0)]
	lonpostivesig=lon[(EVI2_Growing_Season_Area_1_co2_p<=0.05)&(EVI2_Growing_Season_Area_1_co2_r>0)]
	latnegativesig=lat[(EVI2_Growing_Season_Area_1_co2_p<=0.05)&(EVI2_Growing_Season_Area_1_co2_r<0)]
	lonnegativesig=lon[(EVI2_Growing_Season_Area_1_co2_p<=0.05)&(EVI2_Growing_Season_Area_1_co2_r<0)]
	latpostivenosig=lat[(EVI2_Growing_Season_Area_1_co2_p>0.05)&(EVI2_Growing_Season_Area_1_co2_r>0)]
	lonpostivenosig=lon[(EVI2_Growing_Season_Area_1_co2_p>0.05)&(EVI2_Growing_Season_Area_1_co2_r>0)]
	latnegativenosig=lat[(EVI2_Growing_Season_Area_1_co2_p>0.05)&(EVI2_Growing_Season_Area_1_co2_r<0)]
	lonnegativenosig=lon[(EVI2_Growing_Season_Area_1_co2_p>0.05)&(EVI2_Growing_Season_Area_1_co2_r<0)]	
	a=len(latpostivesig)/len(lat)*100
	b=len(latnegativesig)/len(lat)*100
	c=len(latpostivenosig)/len(lat)*100
	d=len(latnegativenosig)/len(lat)*100
	
	count=[a,b,c,d]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.bar(np.arange(0,len(count)/2,0.5),count,color=['red','green','blue','cyan'],label=['Positive${^*}$','Negative${^*}$','Postive','Negative'],width=0.25)
	ax.set_xticks(np.arange(0,len(count)/2,0.5))
	ax.set_xticklabels(['Pos.${^*}$','Neg.${^*}$','Pos.','Neg.'],fontdict={'family':'arial','weight':'normal','size':26,})

	ax.set_yticks(np.arange(0,60,10))
	ax.set_yticklabels(np.arange(0,60,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_ylabel('Percentage (%)', fontdict={'family':'arial','weight':'normal','size':26,})	
	plt.show()

def plot_future_region():
	validate=pd.read_csv('./code/predict/region/CN_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse4=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	data=pd.read_csv('./code/predict/region/CN_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,springincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,springincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,springnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,springdecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,springdecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, springincrease5 - rmse1, springincrease5 + rmse1, alpha=0.2,color='darkred')
	ax.fill_between(year, springincrease3 - rmse1, springincrease3 + rmse1, alpha=0.2,color='red')
	ax.fill_between(year,springnochange - rmse1, springnochange + rmse1, alpha=0.2,color='black')
	ax.fill_between(year, springdecrease3 - rmse1, springdecrease3 + rmse1, alpha=0.2,color='blue')
	ax.fill_between(year, springdecrease5 - rmse1, springdecrease5 + rmse1, alpha=0.2,color='deepskyblue')
	ax.legend(prop={'family':'arial','weight':'normal','size':26,},loc='upper left',bbox_to_anchor=(1.05,1.12),borderaxespad = 0.,ncol=5,frameon=False)
	ax.set_ylim(115,130)
	ax.set_yticks(np.arange(115,135,5))
	ax.set_yticklabels(np.arange(115,135,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,autumnincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,autumnincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,autumnnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,autumndecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,autumndecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, autumnincrease5 - rmse2, autumnincrease5 + rmse2, alpha=0.2,color='darkred')
	ax.fill_between(year, autumnincrease3 - rmse2, autumnincrease3 + rmse2, alpha=0.2,color='red')
	ax.fill_between(year,autumnnochange - rmse2, autumnnochange + rmse2, alpha=0.2,color='black')
	ax.fill_between(year, autumndecrease3 - rmse2, autumndecrease3 + rmse2, alpha=0.2,color='blue')
	ax.fill_between(year, autumndecrease5 - rmse2, autumndecrease5 + rmse2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(270,300,5))
	ax.set_yticklabels(np.arange(270,300,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,summerincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,summerincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,summernochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,summerdecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,summerdecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, summerincrease5 - rmse3, summerincrease5 + rmse3, alpha=0.2,color='darkred')
	ax.fill_between(year, summerincrease3 - rmse3, summerincrease3 + rmse3, alpha=0.2,color='red')
	ax.fill_between(year,summernochange - rmse3, summernochange + rmse3, alpha=0.2,color='black')
	ax.fill_between(year, summerdecrease3 - rmse3, summerdecrease3 + rmse3, alpha=0.2,color='blue')
	ax.fill_between(year, summerdecrease5 - rmse3, summerdecrease5 + rmse3, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(0.2,0.7,0.1))
	ax.set_yticklabels([0.2,0.3,0.4,0.5,0.6], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,allincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,allincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,allnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,alldecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,alldecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, allincrease5 - rmse4, allincrease5 + rmse4, alpha=0.2,color='darkred')
	ax.fill_between(year, allincrease3 - rmse4, allincrease3 + rmse4, alpha=0.2,color='red')
	ax.fill_between(year,allnochange - rmse4, allnochange + rmse4, alpha=0.2,color='black')
	ax.fill_between(year, alldecrease3 - rmse4, alldecrease3 + rmse4, alpha=0.2,color='blue')
	ax.fill_between(year, alldecrease5 - rmse4, alldecrease5 + rmse4, alpha=0.2,color='deepskyblue')
	ax.set_ylim(16,60)
	ax.set_yticks(np.arange(20,70,10))
	ax.set_yticklabels(np.arange(20,70,10), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks(np.arange(2020,2060,10))
	ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})

	validate=pd.read_csv('./code/predict/region/EU_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse4=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	data=pd.read_csv('./code/predict/region/EU_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,springincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,springincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,springnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,springdecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,springdecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, springincrease5 - rmse1*2, springincrease5 + rmse1*2, alpha=0.2,color='darkred')
	ax.fill_between(year, springincrease3 - rmse1*2, springincrease3 + rmse1*2, alpha=0.2,color='red')
	ax.fill_between(year,springnochange - rmse1*2, springnochange + rmse1*2, alpha=0.2,color='black')
	ax.fill_between(year, springdecrease3 - rmse1*2, springdecrease3 + rmse1*2, alpha=0.2,color='blue')
	ax.fill_between(year, springdecrease5 - rmse1*2, springdecrease5 + rmse1*2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(90,150,10))
	ax.set_yticklabels(np.arange(90,150,10), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,autumnincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,autumnincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,autumnnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,autumndecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,autumndecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, autumnincrease5 - rmse2, autumnincrease5 + rmse2, alpha=0.2,color='darkred')
	ax.fill_between(year, autumnincrease3 - rmse2, autumnincrease3 + rmse2, alpha=0.2,color='red')
	ax.fill_between(year,autumnnochange - rmse2, autumnnochange + rmse2, alpha=0.2,color='black')
	ax.fill_between(year, autumndecrease3 - rmse2, autumndecrease3 + rmse2, alpha=0.2,color='blue')
	ax.fill_between(year, autumndecrease5 - rmse2, autumndecrease5 + rmse2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(265,290,5))
	ax.set_yticklabels(np.arange(265,290,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,summerincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,summerincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,summernochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,summerdecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,summerdecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, summerincrease5 - rmse3, summerincrease5 + rmse3, alpha=0.2,color='darkred')
	ax.fill_between(year, summerincrease3 - rmse3, summerincrease3 + rmse3, alpha=0.2,color='red')
	ax.fill_between(year,summernochange - rmse3, summernochange + rmse3, alpha=0.2,color='black')
	ax.fill_between(year, summerdecrease3 - rmse3, summerdecrease3 + rmse3, alpha=0.2,color='blue')
	ax.fill_between(year, summerdecrease5 - rmse3, summerdecrease5 + rmse3, alpha=0.2,color='deepskyblue')
	ax.set_yticks([0.35,0.37,0.39,0.41,0.43,0.45])
	ax.set_yticklabels([0.35,0.37,0.39,0.41,0.43,0.45], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,allincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,allincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,allnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,alldecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,alldecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, allincrease5 - rmse4*2, allincrease5 + rmse4*2, alpha=0.2,color='darkred')
	ax.fill_between(year, allincrease3 - rmse4*2, allincrease3 + rmse4*2, alpha=0.2,color='red')
	ax.fill_between(year,allnochange - rmse4*2, allnochange + rmse4*2, alpha=0.2,color='black')
	ax.fill_between(year, alldecrease3 - rmse4*2, alldecrease3 + rmse4*2, alpha=0.2,color='blue')
	ax.fill_between(year, alldecrease5 - rmse4*2, alldecrease5 + rmse4*2, alpha=0.2,color='deepskyblue')
	ax.set_ylim(36,80)
	ax.set_yticks(np.arange(40,140,20))
	ax.set_yticklabels(np.arange(40,140,20), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks(np.arange(2020,2060,10))
	ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})
	
	validate=pd.read_csv('./code/predict/region/US_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse4=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	data=pd.read_csv('./code/predict/region/US_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,springincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,springincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,springnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,springdecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,springdecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, springincrease5 - rmse1, springincrease5 + rmse1, alpha=0.2,color='darkred')
	ax.fill_between(year, springincrease3 - rmse1, springincrease3 + rmse1, alpha=0.2,color='red')
	ax.fill_between(year,springnochange - rmse1, springnochange + rmse1, alpha=0.2,color='black')
	ax.fill_between(year, springdecrease3 - rmse1, springdecrease3 + rmse1, alpha=0.2,color='blue')
	ax.fill_between(year, springdecrease5 - rmse1, springdecrease5 + rmse1, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(90,150,10))
	ax.set_yticklabels(np.arange(90,150,10), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])

	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,autumnincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,autumnincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,autumnnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,autumndecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,autumndecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, autumnincrease5 - rmse2, autumnincrease5 + rmse2, alpha=0.2,color='darkred')
	ax.fill_between(year, autumnincrease3 - rmse2, autumnincrease3 + rmse2, alpha=0.2,color='red')
	ax.fill_between(year,autumnnochange - rmse2, autumnnochange + rmse2, alpha=0.2,color='black')
	ax.fill_between(year, autumndecrease3 - rmse2, autumndecrease3 + rmse2, alpha=0.2,color='blue')
	ax.fill_between(year, autumndecrease5 - rmse2, autumndecrease5 + rmse2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(240,270,5))
	ax.set_yticklabels(np.arange(240,270,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])

	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,summerincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,summerincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,summernochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,summerdecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,summerdecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, summerincrease5 - rmse3/2, summerincrease5 + rmse3/2, alpha=0.2,color='darkred')
	ax.fill_between(year, summerincrease3 - rmse3/2, summerincrease3 + rmse3/2, alpha=0.2,color='red')
	ax.fill_between(year,summernochange - rmse3/2, summernochange + rmse3/2, alpha=0.2,color='black')
	ax.fill_between(year, summerdecrease3 - rmse3/2, summerdecrease3 + rmse3/2, alpha=0.2,color='blue')
	ax.fill_between(year, summerdecrease5 - rmse3/2, summerdecrease5 + rmse3/2, alpha=0.2,color='deepskyblue')
	ax.set_yticks([0.48,0.49,0.50,0.51,0.52])
	ax.set_yticklabels([0.48,0.49,0.50,0.51,0.52], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])

	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,allincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,allincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,allnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,alldecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,alldecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, allincrease5 - rmse4, allincrease5 + rmse4, alpha=0.2,color='darkred')
	ax.fill_between(year, allincrease3 - rmse4, allincrease3 + rmse4, alpha=0.2,color='red')
	ax.fill_between(year,allnochange - rmse4, allnochange + rmse4, alpha=0.2,color='black')
	ax.fill_between(year, alldecrease3 - rmse4, alldecrease3 + rmse4, alpha=0.2,color='blue')
	ax.fill_between(year, alldecrease5 - rmse4, alldecrease5 + rmse4, alpha=0.2,color='deepskyblue')
	ax.set_ylim(36,60)
	ax.set_yticks(np.arange(40,65,5))
	ax.set_yticklabels(np.arange(40,65,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks(np.arange(2020,2060,10))
	ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})
	plt.show()

def plot_future_site():
	validate=pd.read_csv('./code/predict/site/CN_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse4=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	data=pd.read_csv('./code/predict/site/CN_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,springincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,springincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,springnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,springdecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,springdecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, springincrease5 - rmse1, springincrease5 + rmse1, alpha=0.2,color='darkred')
	ax.fill_between(year, springincrease3 - rmse1, springincrease3 + rmse1, alpha=0.2,color='red')
	ax.fill_between(year,springnochange - rmse1, springnochange + rmse1, alpha=0.2,color='black')
	ax.fill_between(year, springdecrease3 - rmse1, springdecrease3 + rmse1, alpha=0.2,color='blue')
	ax.fill_between(year, springdecrease5 - rmse1, springdecrease5 + rmse1, alpha=0.2,color='deepskyblue')
	ax.set_ylim(80,180)
	ax.set_yticks(np.arange(80,200,20))
	ax.set_yticklabels(np.arange(80,200,20), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,autumnincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,autumnincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,autumnnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,autumndecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,autumndecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, autumnincrease5 - rmse2, autumnincrease5 + rmse2, alpha=0.2,color='darkred')
	ax.fill_between(year, autumnincrease3 - rmse2, autumnincrease3 + rmse2, alpha=0.2,color='red')
	ax.fill_between(year,autumnnochange - rmse2, autumnnochange + rmse2, alpha=0.2,color='black')
	ax.fill_between(year, autumndecrease3 - rmse2, autumndecrease3 + rmse2, alpha=0.2,color='blue')
	ax.fill_between(year, autumndecrease5 - rmse2, autumndecrease5 + rmse2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(260,360,20))
	ax.set_yticklabels(np.arange(260,360,20), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])

	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,summerincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,summerincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,summernochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,summerdecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,summerdecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, summerincrease5 - rmse3, summerincrease5 + rmse3, alpha=0.2,color='darkred')
	ax.fill_between(year, summerincrease3 - rmse3, summerincrease3 + rmse3, alpha=0.2,color='red')
	ax.fill_between(year,summernochange - rmse3, summernochange + rmse3, alpha=0.2,color='black')
	ax.fill_between(year, summerdecrease3 - rmse3, summerdecrease3 + rmse3, alpha=0.2,color='blue')
	ax.fill_between(year, summerdecrease5 - rmse3, summerdecrease5 + rmse3, alpha=0.2,color='deepskyblue')
	ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
	ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])

	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,allincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,allincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,allnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,alldecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,alldecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, allincrease5 - rmse4, allincrease5 + rmse4, alpha=0.2,color='darkred')
	ax.fill_between(year, allincrease3 - rmse4, allincrease3 + rmse4, alpha=0.2,color='red')
	ax.fill_between(year,allnochange - rmse4, allnochange + rmse4, alpha=0.2,color='black')
	ax.fill_between(year, alldecrease3 - rmse4, alldecrease3 + rmse4, alpha=0.2,color='blue')
	ax.fill_between(year, alldecrease5 - rmse4, alldecrease5 + rmse4, alpha=0.2,color='deepskyblue')
	ax.set_ylim(16,100)
	ax.set_yticks(np.arange(20,120,20))
	ax.set_yticklabels(np.arange(20,120,20), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks(np.arange(2020,2060,10))
	ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})

	validate=pd.read_csv('./code/predict/site/EU_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse4=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	data=pd.read_csv('./code/predict/site/EU_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,springincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,springincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,springnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,springdecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,springdecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, springincrease5 - rmse1*2, springincrease5 + rmse1*2, alpha=0.2,color='darkred')
	ax.fill_between(year, springincrease3 - rmse1*2, springincrease3 + rmse1*2, alpha=0.2,color='red')
	ax.fill_between(year,springnochange - rmse1*2, springnochange + rmse1*2, alpha=0.2,color='black')
	ax.fill_between(year, springdecrease3 - rmse1*2, springdecrease3 + rmse1*2, alpha=0.2,color='blue')
	ax.fill_between(year, springdecrease5 - rmse1*2, springdecrease5 + rmse1*2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(100,180,20))
	ax.set_yticklabels(np.arange(100,180,20), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,autumnincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,autumnincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,autumnnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,autumndecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,autumndecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, autumnincrease5 - rmse2, autumnincrease5 + rmse2, alpha=0.2,color='darkred')
	ax.fill_between(year, autumnincrease3 - rmse2, autumnincrease3 + rmse2, alpha=0.2,color='red')
	ax.fill_between(year,autumnnochange - rmse2, autumnnochange + rmse2, alpha=0.2,color='black')
	ax.fill_between(year, autumndecrease3 - rmse2, autumndecrease3 + rmse2, alpha=0.2,color='blue')
	ax.fill_between(year, autumndecrease5 - rmse2, autumndecrease5 + rmse2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(200,320,20))
	ax.set_yticklabels(np.arange(200,320,20), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])

	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,summerincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,summerincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,summernochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,summerdecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,summerdecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, summerincrease5 - rmse3, summerincrease5 + rmse3, alpha=0.2,color='darkred')
	ax.fill_between(year, summerincrease3 - rmse3, summerincrease3 + rmse3, alpha=0.2,color='red')
	ax.fill_between(year,summernochange - rmse3, summernochange + rmse3, alpha=0.2,color='black')
	ax.fill_between(year, summerdecrease3 - rmse3, summerdecrease3 + rmse3, alpha=0.2,color='blue')
	ax.fill_between(year, summerdecrease5 - rmse3, summerdecrease5 + rmse3, alpha=0.2,color='deepskyblue')
	ax.set_yticks([0.3,0.4,0.5,0.6])
	ax.set_yticklabels([0.3,0.4,0.5,0.6], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,allincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,allincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,allnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,alldecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,alldecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, allincrease5 - rmse4*2, allincrease5 + rmse4*2, alpha=0.2,color='darkred')
	ax.fill_between(year, allincrease3 - rmse4*2, allincrease3 + rmse4*2, alpha=0.2,color='red')
	ax.fill_between(year,allnochange - rmse4*2, allnochange + rmse4*2, alpha=0.2,color='black')
	ax.fill_between(year, alldecrease3 - rmse4*2, alldecrease3 + rmse4*2, alpha=0.2,color='blue')
	ax.fill_between(year, alldecrease5 - rmse4*2, alldecrease5 + rmse4*2, alpha=0.2,color='deepskyblue')
	ax.set_ylim(26,90)
	ax.set_yticks(np.arange(30,100,10))
	ax.set_yticklabels(np.arange(30,100,10), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks(np.arange(2020,2060,10))
	ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})
	
	validate=pd.read_csv('./code/predict/site/US_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse4=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	data=pd.read_csv('./code/predict/site/US_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,springincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,springincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,springnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,springdecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,springdecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, springincrease5 - rmse1, springincrease5 + rmse1, alpha=0.2,color='darkred')
	ax.fill_between(year, springincrease3 - rmse1, springincrease3 + rmse1, alpha=0.2,color='red')
	ax.fill_between(year,springnochange - rmse1, springnochange + rmse1, alpha=0.2,color='black')
	ax.fill_between(year, springdecrease3 - rmse1, springdecrease3 + rmse1, alpha=0.2,color='blue')
	ax.fill_between(year, springdecrease5 - rmse1, springdecrease5 + rmse1, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(100,160,10))
	ax.set_yticklabels(np.arange(100,160,10), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])

	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,autumnincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,autumnincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,autumnnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,autumndecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,autumndecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, autumnincrease5 - rmse2, autumnincrease5 + rmse2, alpha=0.2,color='darkred')
	ax.fill_between(year, autumnincrease3 - rmse2, autumnincrease3 + rmse2, alpha=0.2,color='red')
	ax.fill_between(year,autumnnochange - rmse2, autumnnochange + rmse2, alpha=0.2,color='black')
	ax.fill_between(year, autumndecrease3 - rmse2, autumndecrease3 + rmse2, alpha=0.2,color='blue')
	ax.fill_between(year, autumndecrease5 - rmse2, autumndecrease5 + rmse2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(240,340,20))
	ax.set_yticklabels(np.arange(240,340,20), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])

	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,summerincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,summerincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,summernochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,summerdecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,summerdecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, summerincrease5 - rmse3, summerincrease5 + rmse3, alpha=0.2,color='darkred')
	ax.fill_between(year, summerincrease3 - rmse3, summerincrease3 + rmse3, alpha=0.2,color='red')
	ax.fill_between(year,summernochange - rmse3, summernochange + rmse3, alpha=0.2,color='black')
	ax.fill_between(year, summerdecrease3 - rmse3, summerdecrease3 + rmse3, alpha=0.2,color='blue')
	ax.fill_between(year, summerdecrease5 - rmse3, summerdecrease5 + rmse3, alpha=0.2,color='deepskyblue')
	ax.set_yticks([0.3,0.4,0.5,0.6])
	ax.set_yticklabels([0.3,0.4,0.5,0.6], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])

	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,allincrease5,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,allincrease3,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,allnochange,fmt='o-',color='black',label='No change')
	ax.errorbar(year,alldecrease3,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,alldecrease5,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, allincrease5 - rmse4, allincrease5 + rmse4, alpha=0.2,color='darkred')
	ax.fill_between(year, allincrease3 - rmse4, allincrease3 + rmse4, alpha=0.2,color='red')
	ax.fill_between(year,allnochange - rmse4, allnochange + rmse4, alpha=0.2,color='black')
	ax.fill_between(year, alldecrease3 - rmse4, alldecrease3 + rmse4, alpha=0.2,color='blue')
	ax.fill_between(year, alldecrease5 - rmse4, alldecrease5 + rmse4, alpha=0.2,color='deepskyblue')
	ax.set_ylim(16,100)
	ax.set_yticks(np.arange(20,120,20))
	ax.set_yticklabels(np.arange(20,120,20), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks(np.arange(2020,2060,10))
	ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})
	plt.show()
	
def delta_new_site():
	validate=pd.read_csv('./code/predict/site/US_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	validate=pd.read_csv('./code/predict/site/EU_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	validate=pd.read_csv('./code/predict/site/CN_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	data=pd.read_csv('./code/predict/site/US_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])

	delta5=springincrease5[-1]-springnochange[-1]
	delta3=springincrease3[-1]-springnochange[-1]
	delta_3=springdecrease3[-1]-springnochange[-1]
	delta_5=springdecrease5[-1]-springnochange[-1]

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.bar(0.75,delta5,yerr=rmse1,width = 0.15,color='darkred',label='+3%')
	ax.bar(0.9,delta3,yerr=rmse1,width = 0.15,color='red',label='+1.5%')
	ax.bar(1.05,delta_3,yerr=rmse1,width = 0.15,color='blue',label='-1.5%')
	ax.bar(1.2,delta_5,yerr=rmse1,width = 0.15,color='deepskyblue',label='-3%')

	data=pd.read_csv('./code/predict/site/EU_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	
	delta5=springincrease5[-1]-springnochange[-1]
	delta3=springincrease3[-1]-springnochange[-1]
	delta_3=springdecrease3[-1]-springnochange[-1]
	delta_5=springdecrease5[-1]-springnochange[-1]
	
	ax.bar(1.75,delta5,yerr=rmse2,width = 0.15,color='darkred')
	ax.bar(1.9,delta3,yerr=rmse2,width = 0.15,color='red')
	ax.bar(2.05,delta_3,yerr=rmse2,width = 0.15,color='blue')
	ax.bar(2.2,delta_5,yerr=rmse2,width = 0.15,color='deepskyblue')
	
	data=pd.read_csv('./code/predict/site/CN_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	
	delta5=springincrease5[-1]-springnochange[-1]
	delta3=springincrease3[-1]-springnochange[-1]
	delta_3=springdecrease3[-1]-springnochange[-1]
	delta_5=springdecrease5[-1]-springnochange[-1]
	
	ax.bar(2.75,delta5,yerr=rmse3,width = 0.15,color='darkred')
	ax.bar(2.9,delta3,yerr=rmse3,width = 0.15,color='red')
	ax.bar(3.05,delta_3,yerr=rmse3,width = 0.15,color='blue')
	ax.bar(3.2,delta_5,yerr=rmse3,width = 0.15,color='deepskyblue')
	ax.axhline(y=0, color='black', linewidth=1)
	ax.legend(prop={'family':'arial','weight':'normal','size':28,},loc='upper left',bbox_to_anchor=(0,1.12),borderaxespad = 0.,ncol=4,frameon=False)
	ax.set_yticks(np.arange(-10,15,5))
	ax.set_yticklabels(np.arange(-10,15,5), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_xticks(np.arange(1,4,1))
	ax.set_xticklabels(['U.S.','Europe','China'], fontdict={'family':'arial','weight':'normal','size':26,})
	
	validate=pd.read_csv('./code/predict/site/US_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	validate=pd.read_csv('./code/predict/site/EU_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	validate=pd.read_csv('./code/predict/site/CN_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	data=pd.read_csv('./code/predict/site/US_predict.csv')
	year=np.array(data['year'])
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])

	delta5=autumnincrease5[-1]-autumnnochange[-1]
	delta3=autumnincrease3[-1]-autumnnochange[-1]
	delta_3=autumndecrease3[-1]-autumnnochange[-1]
	delta_5=autumndecrease5[-1]-autumnnochange[-1]
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.bar(0.75,delta5,yerr=rmse1,width = 0.15,color='darkred',label='+3%')
	ax.bar(0.9,delta3,yerr=rmse1,width = 0.15,color='red',label='+1.5%')
	ax.bar(1.05,delta_3,yerr=rmse1,width = 0.15,color='blue',label='-1.5%')
	ax.bar(1.2,delta_5,yerr=rmse1,width = 0.15,color='deepskyblue',label='-3%')

	data=pd.read_csv('./code/predict/site/EU_predict.csv')
	year=np.array(data['year'])
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])
	
	delta5=autumnincrease5[-1]-autumnnochange[-1]
	delta3=autumnincrease3[-1]-autumnnochange[-1]
	delta_3=autumndecrease3[-1]-autumnnochange[-1]
	delta_5=autumndecrease5[-1]-autumnnochange[-1]
	
	ax.bar(1.75,delta5,yerr=rmse2,width = 0.15,color='darkred')
	ax.bar(1.9,delta3,yerr=rmse2,width = 0.15,color='red')
	ax.bar(2.05,delta_3,yerr=rmse2,width = 0.15,color='blue')
	ax.bar(2.2,delta_5,yerr=rmse2,width = 0.15,color='deepskyblue')

	data=pd.read_csv('./code/predict/site/CN_predict.csv')
	year=np.array(data['year'])
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])
	
	delta5=autumnincrease5[-1]-autumnnochange[-1]
	delta3=autumnincrease3[-1]-autumnnochange[-1]
	delta_3=autumndecrease3[-1]-autumnnochange[-1]
	delta_5=autumndecrease5[-1]-autumnnochange[-1]
		
	ax.bar(2.75,delta5,yerr=rmse3,width = 0.15,color='darkred')
	ax.bar(2.9,delta3,yerr=rmse3,width = 0.15,color='red')
	ax.bar(3.05,delta_3,yerr=rmse3,width = 0.15,color='blue')
	ax.bar(3.2,delta_5,yerr=rmse3,width = 0.15,color='deepskyblue')
	ax.axhline(y=0, color='black', linewidth=1)
	ax.set_yticks(np.arange(-20,30,10))
	ax.set_yticklabels(np.arange(-20,30,10), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_xticks(np.arange(1,4,1))
	ax.set_xticklabels(['U.S.','Europe','China'], fontdict={'family':'arial','weight':'normal','size':26,})
	
	validate=pd.read_csv('./code/predict/site/US_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	validate=pd.read_csv('./code/predict/site/EU_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	validate=pd.read_csv('./code/predict/site/CN_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	data=pd.read_csv('./code/predict/site/US_predict.csv')
	year=np.array(data['year'])
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])

	delta5=summerincrease5[-1]-summernochange[-1]
	delta3=summerincrease3[-1]-summernochange[-1]
	delta_3=summerdecrease3[-1]-summernochange[-1]
	delta_5=summerdecrease5[-1]-summernochange[-1]
		
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.bar(0.75,delta5,yerr=rmse1,width = 0.15,color='darkred',label='+3%')
	ax.bar(0.9,delta3,yerr=rmse1,width = 0.15,color='red',label='+1.5%')
	ax.bar(1.05,delta_3,yerr=rmse1,width = 0.15,color='blue',label='-1.5%')
	ax.bar(1.2,delta_5,yerr=rmse1,width = 0.15,color='deepskyblue',label='-3%')

	data=pd.read_csv('./code/predict/site/EU_predict.csv')
	year=np.array(data['year'])
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])
	
	delta5=summerincrease5[-1]-summernochange[-1]
	delta3=summerincrease3[-1]-summernochange[-1]
	delta_3=summerdecrease3[-1]-summernochange[-1]
	delta_5=summerdecrease5[-1]-summernochange[-1]
	
	ax.bar(1.75,delta5,yerr=rmse2,width = 0.15,color='darkred')
	ax.bar(1.9,delta3,yerr=rmse2,width = 0.15,color='red')
	ax.bar(2.05,delta_3,yerr=rmse2,width = 0.15,color='blue')
	ax.bar(2.2,delta_5,yerr=rmse2,width = 0.15,color='deepskyblue')

	data=pd.read_csv('./code/predict/site/CN_predict.csv')
	year=np.array(data['year'])
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])
	
	delta5=summerincrease5[-1]-summernochange[-1]
	delta3=summerincrease3[-1]-summernochange[-1]
	delta_3=summerdecrease3[-1]-summernochange[-1]
	delta_5=summerdecrease5[-1]-summernochange[-1]
		
	ax.bar(2.75,delta5,yerr=rmse3,width = 0.15,color='darkred')
	ax.bar(2.9,delta3,yerr=rmse3,width = 0.15,color='red')
	ax.bar(3.05,delta_3,yerr=rmse3,width = 0.15,color='blue')
	ax.bar(3.2,delta_5,yerr=rmse3,width = 0.15,color='deepskyblue')
	ax.axhline(y=0, color='black', linewidth=1)
	ax.set_yticks([-0.05,0,0.05])
	ax.set_yticklabels([-0.05,0,0.05], fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_xticks(np.arange(1,4,1))
	ax.set_xticklabels(['U.S.','Europe','China'], fontdict={'family':'arial','weight':'normal','size':26,})

	validate=pd.read_csv('./code/predict/site/US_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	validate=pd.read_csv('./code/predict/site/EU_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	validate=pd.read_csv('./code/predict/site/CN_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
		
	data=pd.read_csv('./code/predict/site/US_predict.csv')
	year=np.array(data['year'])
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])

	delta5=allincrease5[-1]-allnochange[-1]
	delta3=allincrease3[-1]-allnochange[-1]
	delta_3=alldecrease3[-1]-allnochange[-1]
	delta_5=alldecrease5[-1]-allnochange[-1]
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.bar(0.75,delta5,yerr=rmse1,width = 0.15,color='darkred',label='+3%')
	ax.bar(0.9,delta3,yerr=rmse1,width = 0.15,color='red',label='+1.5%')
	ax.bar(1.05,delta_3,yerr=rmse1,width = 0.15,color='blue',label='-1.5%')
	ax.bar(1.2,delta_5,yerr=rmse1,width = 0.15,color='deepskyblue',label='-3%')

	data=pd.read_csv('./code/predict/site/EU_predict.csv')
	year=np.array(data['year'])
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])
	
	delta5=allincrease5[-1]-allnochange[-1]
	delta3=allincrease3[-1]-allnochange[-1]
	delta_3=alldecrease3[-1]-allnochange[-1]
	delta_5=alldecrease5[-1]-allnochange[-1]
	
	ax.bar(1.75,delta5,yerr=rmse2,width = 0.15,color='darkred')
	ax.bar(1.9,delta3,yerr=rmse2,width = 0.15,color='red')
	ax.bar(2.05,delta_3,yerr=rmse2,width = 0.15,color='blue')
	ax.bar(2.2,delta_5,yerr=rmse2,width = 0.15,color='deepskyblue')

	data=pd.read_csv('./code/predict/site/CN_predict.csv')
	year=np.array(data['year'])
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])
	
	delta5=allincrease5[-1]-allnochange[-1]
	delta3=allincrease3[-1]-allnochange[-1]
	delta_3=alldecrease3[-1]-allnochange[-1]
	delta_5=alldecrease5[-1]-allnochange[-1]
		
	ax.bar(2.75,delta5,yerr=rmse3,width = 0.15,color='darkred')
	ax.bar(2.9,delta3,yerr=rmse3,width = 0.15,color='red')
	ax.bar(3.05,delta_3,yerr=rmse3,width = 0.15,color='blue')
	ax.bar(3.2,delta_5,yerr=rmse3,width = 0.15,color='deepskyblue')
	ax.axhline(y=0, color='black', linewidth=1)
	ax.set_yticks(np.arange(-10,15,5))
	ax.set_yticklabels(np.arange(-10,15,5), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_xticks(np.arange(1,4,1))
	ax.set_xticklabels(['U.S.','Europe','China'], fontdict={'family':'arial','weight':'normal','size':26,})
	
def delta_new_region():
	validate=pd.read_csv('./code/predict/region/US_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	validate=pd.read_csv('./code/predict/region/EU_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	validate=pd.read_csv('./code/predict/region/CN_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	data=pd.read_csv('./code/predict/region/US_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])

	delta5=springincrease5[-1]-springnochange[-1]
	delta3=springincrease3[-1]-springnochange[-1]
	delta_3=springdecrease3[-1]-springnochange[-1]
	delta_5=springdecrease5[-1]-springnochange[-1]

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.bar(0.75,delta5,yerr=rmse1,width = 0.15,color='darkred',label='+3%')
	ax.bar(0.9,delta3,yerr=rmse1,width = 0.15,color='red',label='+1.5%')
	ax.bar(1.05,delta_3,yerr=rmse1,width = 0.15,color='blue',label='-1.5%')
	ax.bar(1.2,delta_5,yerr=rmse1,width = 0.15,color='deepskyblue',label='-3%')

	data=pd.read_csv('./code/predict/region/EU_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	
	delta5=springincrease5[-1]-springnochange[-1]
	delta3=springincrease3[-1]-springnochange[-1]
	delta_3=springdecrease3[-1]-springnochange[-1]
	delta_5=springdecrease5[-1]-springnochange[-1]
	
	ax.bar(1.75,delta5,yerr=rmse2,width = 0.15,color='darkred')
	ax.bar(1.9,delta3,yerr=rmse2,width = 0.15,color='red')
	ax.bar(2.05,delta_3,yerr=rmse2,width = 0.15,color='blue')
	ax.bar(2.2,delta_5,yerr=rmse2,width = 0.15,color='deepskyblue')
	
	data=pd.read_csv('./code/predict/region/CN_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	
	delta5=springincrease5[-1]-springnochange[-1]
	delta3=springincrease3[-1]-springnochange[-1]
	delta_3=springdecrease3[-1]-springnochange[-1]
	delta_5=springdecrease5[-1]-springnochange[-1]
	
	ax.bar(2.75,delta5,yerr=rmse3,width = 0.15,color='darkred')
	ax.bar(2.9,delta3,yerr=rmse3,width = 0.15,color='red')
	ax.bar(3.05,delta_3,yerr=rmse3,width = 0.15,color='blue')
	ax.bar(3.2,delta_5,yerr=rmse3,width = 0.15,color='deepskyblue')
	ax.axhline(y=0, color='black', linewidth=1)
	ax.legend(prop={'family':'arial','weight':'normal','size':26,},loc='upper left',bbox_to_anchor=(0,1.12),borderaxespad = 0.,ncol=5,frameon=False)
	ax.set_ylim(-10,10)
	ax.set_yticks(np.arange(-10,15,5))
	ax.set_yticklabels(np.arange(-10,15,5), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_xticks(np.arange(1,4,1))
	ax.set_xticklabels(['U.S.','Europe','China'], fontdict={'family':'arial','weight':'normal','size':26,})
	
	validate=pd.read_csv('./code/predict/region/US_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	validate=pd.read_csv('./code/predict/region/EU_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	validate=pd.read_csv('./code/predict/region/CN_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	data=pd.read_csv('./code/predict/region/US_predict.csv')
	year=np.array(data['year'])
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])

	delta5=autumnincrease5[-1]-autumnnochange[-1]
	delta3=autumnincrease3[-1]-autumnnochange[-1]
	delta_3=autumndecrease3[-1]-autumnnochange[-1]
	delta_5=autumndecrease5[-1]-autumnnochange[-1]
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.bar(0.75,delta5,yerr=rmse1,width = 0.15,color='darkred',label='+3%')
	ax.bar(0.9,delta3,yerr=rmse1,width = 0.15,color='red',label='+1.5%')
	ax.bar(1.05,delta_3,yerr=rmse1,width = 0.15,color='blue',label='-1.5%')
	ax.bar(1.2,delta_5,yerr=rmse1,width = 0.15,color='deepskyblue',label='-3%')

	data=pd.read_csv('./code/predict/region/EU_predict.csv')
	year=np.array(data['year'])
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])
	
	delta5=autumnincrease5[-1]-autumnnochange[-1]
	delta3=autumnincrease3[-1]-autumnnochange[-1]
	delta_3=autumndecrease3[-1]-autumnnochange[-1]
	delta_5=autumndecrease5[-1]-autumnnochange[-1]
	
	ax.bar(1.75,delta5,yerr=rmse2,width = 0.15,color='darkred')
	ax.bar(1.9,delta3,yerr=rmse2,width = 0.15,color='red')
	ax.bar(2.05,delta_3,yerr=rmse2,width = 0.15,color='blue')
	ax.bar(2.2,delta_5,yerr=rmse2,width = 0.15,color='deepskyblue')

	data=pd.read_csv('./code/predict/region/CN_predict.csv')
	year=np.array(data['year'])
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])
	
	delta5=autumnincrease5[-1]-autumnnochange[-1]
	delta3=autumnincrease3[-1]-autumnnochange[-1]
	delta_3=autumndecrease3[-1]-autumnnochange[-1]
	delta_5=autumndecrease5[-1]-autumnnochange[-1]
		
	ax.bar(2.75,delta5,yerr=rmse3,width = 0.15,color='darkred')
	ax.bar(2.9,delta3,yerr=rmse3,width = 0.15,color='red')
	ax.bar(3.05,delta_3,yerr=rmse3,width = 0.15,color='blue')
	ax.bar(3.2,delta_5,yerr=rmse3,width = 0.15,color='deepskyblue')
	ax.axhline(y=0, color='black', linewidth=1)
	ax.set_yticks(np.arange(-10,15,5))
	ax.set_yticklabels(np.arange(-10,15,5), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_xticks(np.arange(1,4,1))
	ax.set_xticklabels(['U.S.','Europe','China'], fontdict={'family':'arial','weight':'normal','size':26,})

	validate=pd.read_csv('./code/predict/region/US_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	validate=pd.read_csv('./code/predict/region/EU_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	validate=pd.read_csv('./code/predict/region/CN_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	data=pd.read_csv('./code/predict/region/US_predict.csv')
	year=np.array(data['year'])
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])

	delta5=summerincrease5[-1]-summernochange[-1]
	delta3=summerincrease3[-1]-summernochange[-1]
	delta_3=summerdecrease3[-1]-summernochange[-1]
	delta_5=summerdecrease5[-1]-summernochange[-1]
		
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.bar(0.75,delta5,yerr=rmse1,width = 0.15,color='darkred',label='+3%')
	ax.bar(0.9,delta3,yerr=rmse1,width = 0.15,color='red',label='+1.5%')
	ax.bar(1.05,delta_3,yerr=rmse1,width = 0.15,color='blue',label='-1.5%')
	ax.bar(1.2,delta_5,yerr=rmse1,width = 0.15,color='deepskyblue',label='-3%')

	data=pd.read_csv('./code/predict/EU_predict.csv')
	year=np.array(data['year'])
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])
	
	delta5=summerincrease5[-1]-summernochange[-1]
	delta3=summerincrease3[-1]-summernochange[-1]
	delta_3=summerdecrease3[-1]-summernochange[-1]
	delta_5=summerdecrease5[-1]-summernochange[-1]
	
	ax.bar(1.75,delta5,yerr=rmse2,width = 0.15,color='darkred')
	ax.bar(1.9,delta3,yerr=rmse2,width = 0.15,color='red')
	ax.bar(2.05,delta_3,yerr=rmse2,width = 0.15,color='blue')
	ax.bar(2.2,delta_5,yerr=rmse2,width = 0.15,color='deepskyblue')

	data=pd.read_csv('./code/predict/region/CN_predict.csv')
	year=np.array(data['year'])
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])
	
	delta5=summerincrease5[-1]-summernochange[-1]
	delta3=summerincrease3[-1]-summernochange[-1]
	delta_3=summerdecrease3[-1]-summernochange[-1]
	delta_5=summerdecrease5[-1]-summernochange[-1]
		
	ax.bar(2.75,delta5,yerr=rmse3,width = 0.15,color='darkred')
	ax.bar(2.9,delta3,yerr=rmse3,width = 0.15,color='red')
	ax.bar(3.05,delta_3,yerr=rmse3,width = 0.15,color='blue')
	ax.bar(3.2,delta_5,yerr=rmse3,width = 0.15,color='deepskyblue')
	ax.axhline(y=0, color='black', linewidth=1)
	ax.set_yticks([-0.05,0,0.05])
	ax.set_yticklabels([-0.05,0,0.05], fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_xticks(np.arange(1,4,1))
	ax.set_xticklabels(['U.S.','Europe','China'], fontdict={'family':'arial','weight':'normal','size':26,})

	validate=pd.read_csv('./code/predict/region/US_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	validate=pd.read_csv('./code/predict/region/EU_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	validate=pd.read_csv('./code/predict/region/CN_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
		
	data=pd.read_csv('./code/predict/region/US_predict.csv')
	year=np.array(data['year'])
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])

	delta5=allincrease5[-1]-allnochange[-1]
	delta3=allincrease3[-1]-allnochange[-1]
	delta_3=alldecrease3[-1]-allnochange[-1]
	delta_5=alldecrease5[-1]-allnochange[-1]
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.bar(0.75,delta5,yerr=rmse1,width = 0.15,color='darkred',label='+3%')
	ax.bar(0.9,delta3,yerr=rmse1,width = 0.15,color='red',label='+1.5%')
	ax.bar(1.05,delta_3,yerr=rmse1,width = 0.15,color='blue',label='-1.5%')
	ax.bar(1.2,delta_5,yerr=rmse1,width = 0.15,color='deepskyblue',label='-3%')

	data=pd.read_csv('./code/predict/region/EU_predict.csv')
	year=np.array(data['year'])
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])
	
	delta5=allincrease5[-1]-allnochange[-1]
	delta3=allincrease3[-1]-allnochange[-1]
	delta_3=alldecrease3[-1]-allnochange[-1]
	delta_5=alldecrease5[-1]-allnochange[-1]
	
	ax.bar(1.75,delta5,yerr=rmse2,width = 0.15,color='darkred')
	ax.bar(1.9,delta3,yerr=rmse2,width = 0.15,color='red')
	ax.bar(2.05,delta_3,yerr=rmse2,width = 0.15,color='blue')
	ax.bar(2.2,delta_5,yerr=rmse2,width = 0.15,color='deepskyblue')

	data=pd.read_csv('./data/new/co2/filled/linear/data_values/region/CN_predict.csv')
	year=np.array(data['year'])
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])
	
	delta5=allincrease5[-1]-allnochange[-1]
	delta3=allincrease3[-1]-allnochange[-1]
	delta_3=alldecrease3[-1]-allnochange[-1]
	delta_5=alldecrease5[-1]-allnochange[-1]
		
	ax.bar(2.75,delta5,yerr=rmse3,width = 0.15,color='darkred')
	ax.bar(2.9,delta3,yerr=rmse3,width = 0.15,color='red')
	ax.bar(3.05,delta_3,yerr=rmse3,width = 0.15,color='blue')
	ax.bar(3.2,delta_5,yerr=rmse3,width = 0.15,color='deepskyblue')
	ax.axhline(y=0, color='black', linewidth=1)
	ax.set_yticks(np.arange(-10,15,5))
	ax.set_yticklabels(np.arange(-10,15,5), fontdict={'family':'arial','weight':'normal','size':26,})
	ax.set_xticks(np.arange(1,4,1))
	ax.set_xticklabels(['U.S.','Europe','China'], fontdict={'family':'arial','weight':'normal','size':26,})
	plt.show()
	
def future_minus_site():
	validate=pd.read_csv('./data/new/co2/filled/linear/data_values/site/CN_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))

	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse4=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	data=pd.read_csv('./data/new/co2/filled/linear/data_values/site/CN_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,springincrease5-springnochange,fmt='o-',color='darkred',label='Increased 5%')
	ax.errorbar(year,springincrease3-springnochange,fmt='o-',color='red',label='Increased 3%')
	ax.errorbar(year,springdecrease3-springnochange,fmt='o-',color='blue',label='Decreased 3%')
	ax.errorbar(year,springdecrease5-springnochange,fmt='o-',color='deepskyblue',label='Decreased 5%')
	ax.fill_between(year, springincrease5-springnochange - rmse1, springincrease5-springnochange + rmse1, alpha=0.2,color='darkred')
	ax.fill_between(year, springincrease3-springnochange - rmse1, springincrease3-springnochange + rmse1, alpha=0.2,color='red')
	ax.fill_between(year, springdecrease3-springnochange - rmse1, springdecrease3-springnochange + rmse1, alpha=0.2,color='blue')
	ax.fill_between(year, springdecrease5-springnochange - rmse1, springdecrease5-springnochange + rmse1, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-10,15,5))
	ax.set_yticklabels(np.arange(-10,15,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,autumnincrease5-autumnnochange,fmt='o-',color='darkred',label='Increased 5%')
	ax.errorbar(year,autumnincrease3-autumnnochange,fmt='o-',color='red',label='Increased 3%')
	ax.errorbar(year,autumndecrease3-autumnnochange,fmt='o-',color='blue',label='Decreased 3%')
	ax.errorbar(year,autumndecrease5-autumnnochange,fmt='o-',color='deepskyblue',label='Decreased 5%')
	ax.fill_between(year, autumnincrease5-autumnnochange - rmse2, autumnincrease5-autumnnochange + rmse2, alpha=0.2,color='darkred')
	ax.fill_between(year, autumnincrease3-autumnnochange - rmse2, autumnincrease3-autumnnochange + rmse2, alpha=0.2,color='red')
	ax.fill_between(year, autumndecrease3-autumnnochange - rmse2, autumndecrease3-autumnnochange + rmse2, alpha=0.2,color='blue')
	ax.fill_between(year, autumndecrease5-autumnnochange - rmse2, autumndecrease5-autumnnochange + rmse2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-15,20,5))
	ax.set_yticklabels(np.arange(-15,20,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,summerincrease5-summernochange,fmt='o-',color='darkred',label='Increased 5%')
	ax.errorbar(year,summerincrease3-summernochange,fmt='o-',color='red',label='Increased 3%')
	ax.errorbar(year,summerdecrease3-summernochange,fmt='o-',color='blue',label='Decreased 3%')
	ax.errorbar(year,summerdecrease5-summernochange,fmt='o-',color='deepskyblue',label='Decreased 5%')
	ax.fill_between(year, summerincrease5-summernochange - rmse3, summerincrease5-summernochange + rmse3, alpha=0.2,color='darkred')
	ax.fill_between(year, summerincrease3-summernochange - rmse3, summerincrease3-summernochange + rmse3, alpha=0.2,color='red')
	ax.fill_between(year, summerdecrease3-summernochange - rmse3, summerdecrease3-summernochange + rmse3, alpha=0.2,color='blue')
	ax.fill_between(year, summerdecrease5-summernochange - rmse3, summerdecrease5-summernochange + rmse3, alpha=0.2,color='deepskyblue')
	ax.set_yticks([-0.06,-0.04,-0.02,0,0.02,0.04,0.06])
	ax.set_yticklabels([-0.06,-0.04,-0.02,0,0.02,0.04,0.06], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])

	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,allincrease5-allnochange,fmt='o-',color='darkred',label='Increased 5%')
	ax.errorbar(year,allincrease3-allnochange,fmt='o-',color='red',label='Increased 3%')
	ax.errorbar(year,alldecrease3-allnochange,fmt='o-',color='blue',label='Decreased 3%')
	ax.errorbar(year,alldecrease5-allnochange,fmt='o-',color='deepskyblue',label='Decreased 5%')
	ax.fill_between(year, allincrease5-allnochange - rmse4, allincrease5-allnochange + rmse4, alpha=0.2,color='darkred')
	ax.fill_between(year, allincrease3-allnochange - rmse4, allincrease3-allnochange + rmse4, alpha=0.2,color='red')
	ax.fill_between(year, alldecrease3-allnochange - rmse4, alldecrease3-allnochange + rmse4, alpha=0.2,color='blue')
	ax.fill_between(year, alldecrease5-allnochange - rmse4, alldecrease5-allnochange + rmse4, alpha=0.2,color='deepskyblue')
	ax.set_ylim(-4.5,4)
	ax.set_yticks(np.arange(-4,6,2))
	ax.set_yticklabels(np.arange(-4,6,2), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks(np.arange(2020,2060,10))
	ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})

	validate=pd.read_csv('./data/new/co2/filled/linear/data_values/site/EU_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse4=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	data=pd.read_csv('./data/new/co2/filled/linear/data_values/site/EU_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,springincrease5-springnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,springincrease3-springnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,springdecrease3-springnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,springdecrease5-springnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, springincrease5-springnochange - rmse1, springincrease5-springnochange + rmse1, alpha=0.2,color='darkred')
	ax.fill_between(year, springincrease3-springnochange - rmse1, springincrease3-springnochange + rmse1, alpha=0.2,color='red')
	ax.fill_between(year, springdecrease3-springnochange - rmse1, springdecrease3-springnochange + rmse1, alpha=0.2,color='blue')
	ax.fill_between(year, springdecrease5-springnochange - rmse1, springdecrease5-springnochange + rmse1, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-6,9,3))
	ax.set_yticklabels(np.arange(-6,9,3), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])

	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,autumnincrease5-autumnnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,autumnincrease3-autumnnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,autumndecrease3-autumnnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,autumndecrease5-autumnnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, autumnincrease5-autumnnochange - rmse2, autumnincrease5-autumnnochange + rmse2, alpha=0.2,color='darkred')
	ax.fill_between(year, autumnincrease3-autumnnochange - rmse2, autumnincrease3-autumnnochange + rmse2, alpha=0.2,color='red')
	ax.fill_between(year, autumndecrease3-autumnnochange - rmse2, autumndecrease3-autumnnochange + rmse2, alpha=0.2,color='blue')
	ax.fill_between(year, autumndecrease5-autumnnochange - rmse2, autumndecrease5-autumnnochange + rmse2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-15,20,5))
	ax.set_yticklabels(np.arange(-15,20,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,summerincrease5-summernochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,summerincrease3-summernochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,summerdecrease3-summernochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,summerdecrease5-summernochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, summerincrease5-summernochange - rmse3, summerincrease5-summernochange + rmse3, alpha=0.2,color='darkred')
	ax.fill_between(year, summerincrease3-summernochange - rmse3, summerincrease3-summernochange + rmse3, alpha=0.2,color='red')
	ax.fill_between(year, summerdecrease3-summernochange - rmse3, summerdecrease3-summernochange + rmse3, alpha=0.2,color='blue')
	ax.fill_between(year, summerdecrease5-summernochange - rmse3, summerdecrease5-summernochange + rmse3, alpha=0.2,color='deepskyblue')
	ax.set_yticks([-0.03,-0.02,-0.01,0,0.01,0.02,0.03])
	ax.set_yticklabels([-0.03,-0.02,-0.01,0,0.01,0.02,0.03], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])

	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,allincrease5-allnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,allincrease3-allnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,alldecrease3-allnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,alldecrease5-allnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, allincrease5-allnochange - rmse4, allincrease5-allnochange + rmse4, alpha=0.2,color='darkred')
	ax.fill_between(year, allincrease3-allnochange - rmse4, allincrease3-allnochange + rmse4, alpha=0.2,color='red')
	ax.fill_between(year, alldecrease3-allnochange - rmse4, alldecrease3-allnochange + rmse4, alpha=0.2,color='blue')
	ax.fill_between(year, alldecrease5-allnochange - rmse4, alldecrease5-allnochange + rmse4, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-10,15,5))
	ax.set_yticklabels(np.arange(-10,15,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks(np.arange(2020,2060,10))
	ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})
	
	validate=pd.read_csv('./data/new/co2/filled/linear/data_values/site/US_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse4=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	
	data=pd.read_csv('./data/new/co2/filled/linear/data_values/site/US_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,springincrease5-springnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,springincrease3-springnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,springdecrease3-springnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,springdecrease5-springnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, springincrease5-springnochange - rmse1, springincrease5-springnochange + rmse1, alpha=0.2,color='darkred')
	ax.fill_between(year, springincrease3-springnochange - rmse1, springincrease3-springnochange + rmse1, alpha=0.2,color='red')
	ax.fill_between(year, springdecrease3-springnochange - rmse1, springdecrease3-springnochange + rmse1, alpha=0.2,color='blue')
	ax.fill_between(year, springdecrease5-springnochange - rmse1, springdecrease5-springnochange + rmse1, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-6,8,2))
	ax.set_yticklabels(np.arange(-6,8,2), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])

	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,autumnincrease5-autumnnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,autumnincrease3-autumnnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,autumndecrease3-autumnnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,autumndecrease5-autumnnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, autumnincrease5-autumnnochange - rmse2, autumnincrease5-autumnnochange + rmse2, alpha=0.2,color='darkred')
	ax.fill_between(year, autumnincrease3-autumnnochange - rmse2, autumnincrease3-autumnnochange + rmse2, alpha=0.2,color='red')
	ax.fill_between(year, autumndecrease3-autumnnochange - rmse2, autumndecrease3-autumnnochange + rmse2, alpha=0.2,color='blue')
	ax.fill_between(year, autumndecrease5-autumnnochange - rmse2, autumndecrease5-autumnnochange + rmse2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-15,20,5))
	ax.set_yticklabels(np.arange(-15,20,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,summerincrease5-summernochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,summerincrease3-summernochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,summerdecrease3-summernochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,summerdecrease5-summernochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, summerincrease5-summernochange - rmse3, summerincrease5-summernochange + rmse3, alpha=0.2,color='darkred')
	ax.fill_between(year, summerincrease3-summernochange - rmse3, summerincrease3-summernochange + rmse3, alpha=0.2,color='red')
	ax.fill_between(year, summerdecrease3-summernochange - rmse3, summerdecrease3-summernochange + rmse3, alpha=0.2,color='blue')
	ax.fill_between(year, summerdecrease5-summernochange - rmse3, summerdecrease5-summernochange + rmse3, alpha=0.2,color='deepskyblue')
	ax.set_yticks([-0.03,-0.02,-0.01,0,0.01,0.02,0.03])
	ax.set_yticklabels([-0.03,-0.02,-0.01,0,0.01,0.02,0.03], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,allincrease5-allnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,allincrease3-allnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,alldecrease3-allnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,alldecrease5-allnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, allincrease5-allnochange - rmse4, allincrease5-allnochange + rmse4, alpha=0.2,color='darkred')
	ax.fill_between(year, allincrease3-allnochange - rmse4, allincrease3-allnochange + rmse4, alpha=0.2,color='red')
	ax.fill_between(year, alldecrease3-allnochange - rmse4, alldecrease3-allnochange + rmse4, alpha=0.2,color='blue')
	ax.fill_between(year, alldecrease5-allnochange - rmse4, alldecrease5-allnochange + rmse4, alpha=0.2,color='deepskyblue')
	ax.set_ylim(-12,10)
	ax.set_yticks(np.arange(-10,15,5))
	ax.set_yticklabels(np.arange(-10,15,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks(np.arange(2020,2060,10))
	ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})
	plt.show()

def future_minus_region():
	validate=pd.read_csv('./code/predict/region/CN_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse4=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	data=pd.read_csv('./code/predict/region/CN_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,springincrease5-springnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,springincrease3-springnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,springdecrease3-springnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,springdecrease5-springnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, springincrease5-springnochange - rmse1, springincrease5-springnochange + rmse1, alpha=0.2,color='darkred')
	ax.fill_between(year, springincrease3-springnochange - rmse1, springincrease3-springnochange + rmse1, alpha=0.2,color='red')
	ax.fill_between(year, springdecrease3-springnochange - rmse1, springdecrease3-springnochange + rmse1, alpha=0.2,color='blue')
	ax.fill_between(year, springdecrease5-springnochange - rmse1, springdecrease5-springnochange + rmse1, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-6,9,3))
	ax.set_yticklabels(np.arange(-6,9,3),fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,autumnincrease5-autumnnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,autumnincrease3-autumnnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,autumndecrease3-autumnnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,autumndecrease5-autumnnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, autumnincrease5-autumnnochange - rmse2, autumnincrease5-autumnnochange + rmse2, alpha=0.2,color='darkred')
	ax.fill_between(year, autumnincrease3-autumnnochange - rmse2, autumnincrease3-autumnnochange + rmse2, alpha=0.2,color='red')
	ax.fill_between(year, autumndecrease3-autumnnochange - rmse2, autumndecrease3-autumnnochange + rmse2, alpha=0.2,color='blue')
	ax.fill_between(year, autumndecrease5-autumnnochange - rmse2, autumndecrease5-autumnnochange + rmse2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-6,9,3))
	ax.set_yticklabels(np.arange(-6,9,3), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,summerincrease5-summernochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,summerincrease3-summernochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,summerdecrease3-summernochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,summerdecrease5-summernochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, summerincrease5-summernochange - rmse3, summerincrease5-summernochange + rmse3, alpha=0.2,color='darkred')
	ax.fill_between(year, summerincrease3-summernochange - rmse3, summerincrease3-summernochange + rmse3, alpha=0.2,color='red')
	ax.fill_between(year, summerdecrease3-summernochange - rmse3, summerdecrease3-summernochange + rmse3, alpha=0.2,color='blue')
	ax.fill_between(year, summerdecrease5-summernochange - rmse3, summerdecrease5-summernochange + rmse3, alpha=0.2,color='deepskyblue')
	ax.set_yticks([-0.06,-0.04,-0.02,0,0.02,0.04,0.06])
	ax.set_yticklabels([-0.06,-0.04,-0.02,0,0.02,0.04,0.06], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,allincrease5-allnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,allincrease3-allnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,alldecrease3-allnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,alldecrease5-allnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, allincrease5-allnochange - rmse4, allincrease5-allnochange + rmse4, alpha=0.2,color='darkred')
	ax.fill_between(year, allincrease3-allnochange - rmse4, allincrease3-allnochange + rmse4, alpha=0.2,color='red')
	ax.fill_between(year, alldecrease3-allnochange - rmse4, alldecrease3-allnochange + rmse4, alpha=0.2,color='blue')
	ax.fill_between(year, alldecrease5-allnochange - rmse4, alldecrease5-allnochange + rmse4, alpha=0.2,color='deepskyblue')
	ax.set_ylim(-4.5,4)
	ax.set_yticks(np.arange(-4,6,2))
	ax.set_yticklabels(np.arange(-4,6,2), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks(np.arange(2020,2060,10))
	ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})

	validate=pd.read_csv('./code/predict/region/EU_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse4=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	data=pd.read_csv('./code/predict/region/EU_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,springincrease5-springnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,springincrease3-springnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,springdecrease3-springnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,springdecrease5-springnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, springincrease5-springnochange - rmse1, springincrease5-springnochange + rmse1, alpha=0.2,color='darkred')
	ax.fill_between(year, springincrease3-springnochange - rmse1, springincrease3-springnochange + rmse1, alpha=0.2,color='red')
	ax.fill_between(year, springdecrease3-springnochange - rmse1, springdecrease3-springnochange + rmse1, alpha=0.2,color='blue')
	ax.fill_between(year, springdecrease5-springnochange - rmse1, springdecrease5-springnochange + rmse1, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-6,9,3))
	ax.set_yticklabels(np.arange(-6,9,3), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,autumnincrease5-autumnnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,autumnincrease3-autumnnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,autumndecrease3-autumnnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,autumndecrease5-autumnnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, autumnincrease5-autumnnochange - rmse2, autumnincrease5-autumnnochange + rmse2, alpha=0.2,color='darkred')
	ax.fill_between(year, autumnincrease3-autumnnochange - rmse2, autumnincrease3-autumnnochange + rmse2, alpha=0.2,color='red')
	ax.fill_between(year, autumndecrease3-autumnnochange - rmse2, autumndecrease3-autumnnochange + rmse2, alpha=0.2,color='blue')
	ax.fill_between(year, autumndecrease5-autumnnochange - rmse2, autumndecrease5-autumnnochange + rmse2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-6,9,3))
	ax.set_yticklabels(np.arange(-6,9,3), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,summerincrease5-summernochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,summerincrease3-summernochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,summerdecrease3-summernochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,summerdecrease5-summernochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, summerincrease5-summernochange - rmse3, summerincrease5-summernochange + rmse3, alpha=0.2,color='darkred')
	ax.fill_between(year, summerincrease3-summernochange - rmse3, summerincrease3-summernochange + rmse3, alpha=0.2,color='red')
	ax.fill_between(year, summerdecrease3-summernochange - rmse3, summerdecrease3-summernochange + rmse3, alpha=0.2,color='blue')
	ax.fill_between(year, summerdecrease5-summernochange - rmse3, summerdecrease5-summernochange + rmse3, alpha=0.2,color='deepskyblue')
	ax.set_yticks([-0.03,-0.02,-0.01,0,0.01,0.02,0.03])
	ax.set_yticklabels([-0.03,-0.02,-0.01,0,0.01,0.02,0.03], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,allincrease5-allnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,allincrease3-allnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,alldecrease3-allnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,alldecrease5-allnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, allincrease5-allnochange - rmse4, allincrease5-allnochange + rmse4, alpha=0.2,color='darkred')
	ax.fill_between(year, allincrease3-allnochange - rmse4, allincrease3-allnochange + rmse4, alpha=0.2,color='red')
	ax.fill_between(year, alldecrease3-allnochange - rmse4, alldecrease3-allnochange + rmse4, alpha=0.2,color='blue')
	ax.fill_between(year, alldecrease5-allnochange - rmse4, alldecrease5-allnochange + rmse4, alpha=0.2,color='deepskyblue')
	ax.set_ylim(-11,10)
	ax.set_yticks(np.arange(-10,15,5))
	ax.set_yticklabels(np.arange(-10,15,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks(np.arange(2020,2060,10))
	ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})
	
	validate=pd.read_csv('./code/predict/region/US_validation.csv')
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Greenup_Phase_1'])
	spring=np.array(validate['spring'])
	rmse1=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['Date_Mid_Senescence_Phase_1'])
	spring=np.array(validate['autumn'])
	rmse2=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Onset_Greenness_Maximum_1'])
	spring=np.array(validate['summer'])
	rmse3=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	Date_Mid_Greenup_Phase_1=np.array(validate['EVI2_Growing_Season_Area_1'])
	spring=np.array(validate['all'])
	rmse4=np.sqrt(mean_squared_error(spring,Date_Mid_Greenup_Phase_1))
	data=pd.read_csv('./code/predict/region/US_predict.csv')
	year=np.array(data['year'])
	springincrease5=np.array(data['springincrease5'])
	springincrease3=np.array(data['springincrease3'])
	springnochange=np.array(data['springnochange'])
	springdecrease5=np.array(data['springdecrease5'])
	springdecrease3=np.array(data['springdecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,springincrease5-springnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,springincrease3-springnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,springdecrease3-springnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,springdecrease5-springnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, springincrease5-springnochange - rmse1, springincrease5-springnochange + rmse1, alpha=0.2,color='darkred')
	ax.fill_between(year, springincrease3-springnochange - rmse1, springincrease3-springnochange + rmse1, alpha=0.2,color='red')
	ax.fill_between(year, springdecrease3-springnochange - rmse1, springdecrease3-springnochange + rmse1, alpha=0.2,color='blue')
	ax.fill_between(year, springdecrease5-springnochange - rmse1, springdecrease5-springnochange + rmse1, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-10,15,5))
	ax.set_yticklabels(np.arange(-10,15,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	autumnincrease5=np.array(data['autumnincrease5'])
	autumnincrease3=np.array(data['autumnincrease3'])
	autumnnochange=np.array(data['autumnnochange'])
	autumndecrease5=np.array(data['autumndecrease5'])
	autumndecrease3=np.array(data['autumndecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,autumnincrease5-autumnnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,autumnincrease3-autumnnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,autumndecrease3-autumnnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,autumndecrease5-autumnnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, autumnincrease5-autumnnochange - rmse2, autumnincrease5-autumnnochange + rmse2, alpha=0.2,color='darkred')
	ax.fill_between(year, autumnincrease3-autumnnochange - rmse2, autumnincrease3-autumnnochange + rmse2, alpha=0.2,color='red')
	ax.fill_between(year, autumndecrease3-autumnnochange - rmse2, autumndecrease3-autumnnochange + rmse2, alpha=0.2,color='blue')
	ax.fill_between(year, autumndecrease5-autumnnochange - rmse2, autumndecrease5-autumnnochange + rmse2, alpha=0.2,color='deepskyblue')
	ax.set_yticks(np.arange(-10,15,5))
	ax.set_yticklabels(np.arange(-10,15,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	summerincrease5=np.array(data['summerincrease5'])
	summerincrease3=np.array(data['summerincrease3'])
	summernochange=np.array(data['summernochange'])
	summerdecrease5=np.array(data['summerdecrease5'])
	summerdecrease3=np.array(data['summerdecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,summerincrease5-summernochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,summerincrease3-summernochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,summerdecrease3-summernochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,summerdecrease5-summernochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, summerincrease5-summernochange - rmse3, summerincrease5-summernochange + rmse3, alpha=0.2,color='darkred')
	ax.fill_between(year, summerincrease3-summernochange - rmse3, summerincrease3-summernochange + rmse3, alpha=0.2,color='red')
	ax.fill_between(year, summerdecrease3-summernochange - rmse3, summerdecrease3-summernochange + rmse3, alpha=0.2,color='blue')
	ax.fill_between(year, summerdecrease5-summernochange - rmse3, summerdecrease5-summernochange + rmse3, alpha=0.2,color='deepskyblue')
	ax.set_yticks([-0.01,0,0.01])
	ax.set_yticklabels([-0.01,0,0.01], fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks([])
	allincrease5=np.array(data['allincrease5'])
	allincrease3=np.array(data['allincrease3'])
	allnochange=np.array(data['allnochange'])
	alldecrease5=np.array(data['alldecrease5'])
	alldecrease3=np.array(data['alldecrease3'])
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)			
	ax.errorbar(year,allincrease5-allnochange,fmt='o-',color='darkred',label='+3%')
	ax.errorbar(year,allincrease3-allnochange,fmt='o-',color='red',label='+1.5%')
	ax.errorbar(year,alldecrease3-allnochange,fmt='o-',color='blue',label='-1.5%')
	ax.errorbar(year,alldecrease5-allnochange,fmt='o-',color='deepskyblue',label='-3%')
	ax.fill_between(year, allincrease5-allnochange - rmse4, allincrease5-allnochange + rmse4, alpha=0.2,color='darkred')
	ax.fill_between(year, allincrease3-allnochange - rmse4, allincrease3-allnochange + rmse4, alpha=0.2,color='red')
	ax.fill_between(year, alldecrease3-allnochange - rmse4, alldecrease3-allnochange + rmse4, alpha=0.2,color='blue')
	ax.fill_between(year, alldecrease5-allnochange - rmse4, alldecrease5-allnochange + rmse4, alpha=0.2,color='deepskyblue')
	ax.set_ylim(-12,10)
	ax.set_yticks(np.arange(-10,15,5))
	ax.set_yticklabels(np.arange(-10,15,5), fontdict={'family':'arial','weight':'normal','size':36,})
	ax.set_xticks(np.arange(2020,2060,10))
	ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})
	plt.show()
			
def future_ozone_region():
	sea=['all']
	area=['US','EU','CN']
	area1=['U.S.','EU','CN']
	for i in range(len(area)):
		data=pd.read_csv('./code/predict/region/'+area[i]+'_aot40.csv')	
		for j in range(len(sea)):
			increase5=np.array(data['AOT40_'+sea[j]+'increase3%'])
			increase3=np.array(data['AOT40_'+sea[j]+'increase1.5%'])
			nochange=np.array(data['AOT40_'+sea[j]])
			decrease3=np.array(data['AOT40_'+sea[j]+'decrease1.5%'])
			decrease5=np.array(data['AOT40_'+sea[j]+'decrease3%'])	
			year=np.array(data['year'])
			fig=plt.figure()
			ax=fig.add_subplot(1,1,1)	
			ax.errorbar(year,increase5,fmt='o-',color='darkred',label='+3%')
			ax.errorbar(year,increase3,fmt='o-',color='red',label='+1.5%')
			ax.errorbar(year,nochange,fmt='o-',color='black',label='No change')
			ax.errorbar(year,decrease3,fmt='o-',color='blue',label='-1.5%')
			ax.errorbar(year,decrease5,fmt='o-',color='deepskyblue',label='-3%')	
			if i==0:
				ax.set_ylim(-1000,25000)
				ax.set_yticks(np.arange(0,30000,5000))
				ax.set_yticklabels(np.arange(0,30000,5000), fontdict={'family':'arial','weight':'normal','size':36,})
			elif i==1:
				print(1)
				ax.set_ylim(-1000,25000)
				ax.set_yticks(np.arange(0,30000,5000))
				ax.set_yticklabels(np.arange(0,30000,5000), fontdict={'family':'arial','weight':'normal','size':36,})
			else:
				print(1)
				ax.set_ylim(-2000,100000)
				ax.set_yticks(np.arange(0,120000,20000))
				ax.set_yticklabels(np.arange(0,120000,20000), fontdict={'family':'arial','weight':'normal','size':36,})				
			ax.set_xticks(np.arange(2020,2060,10))
			ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})
			ax.set_title(area1[i], fontdict={'family':'arial','weight':'normal','size':36,})
			ax.set_xlabel('year',fontdict={'family':'arial','weight':'normal','size':36,})		
			ax.set_ylabel('AOT40 [ppbv$\cdot$hour]',fontdict={'family':'arial','weight':'normal','size':36,})		
	plt.show()

def future_ozone_site():
	area=['US','EU','CN']
	sea=['spring','autumn','summer','all']
	area1=['U.S.','EU','CN']
	for i in range(len(area)):
		data=pd.read_csv('./code/predict/site/'+area[i]+'_aot40.csv')	
		increase5=np.array(data['AOT40_allincrease3%'])
		increase3=np.array(data['AOT40_allincrease1.5%'])
		nochange=np.array(data['AOT40_all'])
		decrease3=np.array(data['AOT40_alldecrease1.5%'])
		decrease5=np.array(data['AOT40_alldecrease3%'])
		year=np.array(data['year'])
		fig=plt.figure()
		ax=fig.add_subplot(1,1,1)	
		ax.errorbar(year,increase5,fmt='o-',color='darkred',label='+3%')
		ax.errorbar(year,increase3,fmt='o-',color='red',label='+1.5%')
		ax.errorbar(year,nochange,fmt='o-',color='black',label='No change')
		ax.errorbar(year,decrease3,fmt='o-',color='blue',label='-1.5%')
		ax.errorbar(year,decrease5,fmt='o-',color='deepskyblue',label='-3%')
		if i==0:
			ax.set_ylim(-1000,20000)
			ax.set_yticks(np.arange(0,25000,5000))
			ax.set_yticklabels(np.arange(0,25000,5000), fontdict={'family':'arial','weight':'normal','size':36,})
		elif i==1:
			ax.set_ylim(-1000,18000)
			ax.set_yticks(np.arange(0,21000,3000))
			ax.set_yticklabels(np.arange(0,21000,3000), fontdict={'family':'arial','weight':'normal','size':36,})
		else:
			ax.set_ylim(-2000,120000)
			ax.set_yticks(np.arange(0,160000,40000))
			ax.set_yticklabels(np.arange(0,160000,40000), fontdict={'family':'arial','weight':'normal','size':36,})				
		ax.set_xticks(np.arange(2020,2060,10))
		ax.set_xticklabels(np.arange(2020,2060,10), fontdict={'family':'arial','weight':'normal','size':36,})
		ax.set_title(area1[i], fontdict={'family':'arial','weight':'normal','size':36,})
		ax.set_xlabel('year',fontdict={'family':'arial','weight':'normal','size':36,})		
		ax.set_ylabel('AOT40 [ppbv$\cdot$hour]',fontdict={'family':'arial','weight':'normal','size':36,})	
	plt.show()

def main(args):
	#phenology()
	#aot40()
	#o3()
	#co2()
	#partial_year_plot_aot40()
	#partial_year_plot_o3()
	#partial_year_plot_temp()
	#partial_year_plot_co2()
	#temporal_all_map()
	#temporal_all_map_histogram()
	#pvalue()
	#pvalue_hist()
	#plot_future_region()
	#plot_future_site()
	#delta_new_region()
	#delta_new_site()
	#future_minus_region()
	#future_minus_site()
	#future_ozone_region()
	#future_ozone_site()
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
