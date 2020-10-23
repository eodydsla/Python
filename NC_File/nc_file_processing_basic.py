
# coding: utf-8

# In[4]:


import os
import sys
import numpy as np
import matplotlib
import xarray as xr
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# In[145]:


# Tempature 데이터 로드
# 8760
years = ['2015']
months = ['%02d' % i for i in range(1,13)]

DIR = './data/'

result_arr = np.empty(shape=(0,49,65),dtype='object') 
f_list = ["2m_temperature","boundary_layer_height","k_index","relative_humidity+975","surface_pressure",
     "total_precipitation","u_component_of_wind+950","v_component_of_wind+950"]

result_all = dict()

for f in f_list:  
    for y in years: 
        for m in months:
            print(m)
            fn = '%s@%s@%s.nc' % (f, y, m) 
            temp = Dataset(DIR+fn)
            v = next(reversed(temp.variables))
            temp = temp[v][:]
            temp = np.array(temp)
            result_arr = np.concatenate([result_arr,temp],axis=0)           
            result_all[f] = result_arr          


# In[142]:


# 분석 (변수별) 내용 파악하기
f = "boundary_layer_height"
fn = '%s@%s@%s.nc' % (f, y, m) 
temp = Dataset(DIR+fn)
temp.variables

