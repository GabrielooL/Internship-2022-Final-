# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:28:59 2022

@author: 14242
"""

import sunpy 
import astropy.units as u
from sunpy.net import Fido, attrs as a 
import astropy.time
from astropy.io import fits as astro_fits  
from astropy.io.fits import getdata

import copy
import glob

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import matplotlib as mpl 
from matplotlib.collections import LineCollection
import numpy as np
from IPython.display import HTML

import astropy.table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import AsymmetricPercentileInterval, ImageNormalize, LogStretch 

#These packages are for Using TrackPy
import matplotlib.pyplot as plt 
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp 
from numpy import sqrt  


List_2 = glob.glob(r'C:\Users\14242\2022 Summer Internship\Milos bigger data for the sun\*')


data_average = [] #Averages subtracted data
data_median = [] #Median subtracted data
for Polar in List_2: 
    Polardata = getdata(Polar) 
    data_average.append(Polardata - np.average(Polardata)) 
    data_median.append(Polardata - np.median(Polardata))  
Arr=np.array(data_average)#Converts Arr list into numpy array


Diff = [] #Consequtive Differences
for i in range (1,len(Arr)-1): 
    Diff.append(Arr[i+1]-Arr[i]) 
Diff = np.array(Diff)


#locates Gaussian-like blobs of Sun
f = tp.locate(Arr[1], 5, invert=True ,minmass = 500) 
tp.annotate(f,Arr[1]);#Trackpy graph of f


r = tp.batch(Arr,5, minmass =500);  


t = tp.link(r, 5, memory=2) 


t1 = tp.filter_stubs(t,7) #Filters out Spurious traj 
('Before:', t['particle'].nunique())
('After:', t1['particle'].nunique())


#Filters our particls from t1 
t2 = t1[((t1['mass'] <2000 ) & (t1['size'] > 1.2) &
         (t1['ecc'] < 1))] 


R = 1625.691406  


iloc_valid = []

for i in range(len(t2)):
    r = np.sqrt((t2['x'].iloc[i]- 2070.828369)**2 + (t2['y'].iloc[i] - 2008.760620)**2)
    if r<R: 
        (r, t2['particle'].iloc[i])
        iloc_valid.append(i)
        

Bull=[]
for i in range(10): 
    s= t2.iloc[iloc_valid].loc[i].sum(axis=0).loc['signal']  
    Bull+=[s]
    
    
x = np.linspace(0, 9, num=10)
y = np.array(Bull)

plt.title('Light Curve')
plt.plot(x, y, color="red")

plt.show()