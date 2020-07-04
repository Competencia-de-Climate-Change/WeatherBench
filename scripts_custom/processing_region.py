#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
from pathlib import Path
import shutil
import xarray as xr


# Plotting Libraries
import matplotlib.pyplot as plt

from holoviews.operation import contours
import geoviews as gv
import geoviews.feature as gf

# Plot Options
gv.extension('matplotlib')
gv.output(size=200)

from IPython.core.display import HTML

# Center matplotlib's figures
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")


DATA_DIR = Path('../data')

# Author: Victor Faraggi
# Date: June 22nd 2020


# # Preprocesamiento: Selección de Región 
# 
# Se busca seleccionar una región que contenga a Chile para así estudiar el pronóstico meteórologico del país. 
# 
# Esto se realiza por las siguientes razones:
# 
# - Acercar el proyecto, desde un punto de vista geógrafico y "personal"
# - Disminuir tamaño dataset
# - Agilizar aplicación de modelos de ML

# In[2]:


# Si no desea guardar el nuevo dataset cambie el siguiente parametro
save_new_dataset = True


# In[3]:


all_files = Path(DATA_DIR.parent).glob(pattern=DATA_DIR.name + '/*[!0]/*.nc')

era5_5deg = xr.open_mfdataset(all_files, combine='by_coords')
# era5_5deg = xr.open_zarr('data/new_region/dataset.zarr')
print('Size in GB:', era5_5deg.nbytes / 1e9)


# ## Cómo se decide "hasta donde" seleccionar la Región
# 
# Los factores más importantes que controlan el clima en Chile son:
# 
# - el anticiclón del Pacífico Sur
# - el área de baja presión circumpolar austral 
# - la fría corriente de Humboldt 
# - la cordillera de los Andes. 
# 
# A pesar de la longitud de las costas chilenas, algunas zonas del interior pueden experimentar amplias oscilaciones de temperatura, y ciudades como San Pedro de Atacama, pueden experimentar incluso un clima de tipo continental. 
# 
# En los extremos noreste y sureste, las zonas fronterizas se internan en el altiplano y en las llanuras de la Patagonia chilena, dando a estas regiones patrones climáticos similares a los de Bolivia y Argentina, respectivamente.
# 
# Entre los numerosos efectos del clima presente en este país, destaca su influencia en la flora de Chile. 
# 
# 
# ![climate](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Diagrama_clim%C3%A1tico_de_Chile.svg/400px-Diagrama_clim%C3%A1tico_de_Chile.svg.png "climate")
# 
# 
# 
# Source: Wikipedia

# In[4]:


roi_lon = slice(44, 58)
roi_lat = slice(3, 15)

clipped_region = era5_5deg.isel(lon=roi_lon, lat=roi_lat)
print('Size in GB:', clipped_region.nbytes / 1e9)
clipped_region


# ## Visualizaciones

# In[5]:


def cartopy_world_plot(region_da, time_sel=0, level_sel=False, center=(-75, -20)):
    import cartopy.crs as ccrs
    
    if type(level_sel) is int:
        da = region_da.isel(level=level_sel)
    else:
        da = region_da
        
    plt.figure(figsize=(15,7))
    
    ax = plt.axes(projection=ccrs.Orthographic(*center))

    da.isel(time=time_sel).plot.contourf(ax=ax, transform=ccrs.PlateCarree());

    ax.set_global(); ax.coastlines();
    plt.show()

def geoview_flat_plot(region_da, time_sel=0, level_sel=False, wide_chile=True):
    
    if type(level_sel) is int:
        da = region_da.isel(level=level_sel)
    else:
        da = region_da
    
    gv_ds = gv.Dataset(da.isel(time=time_sel))
    plot = gv_ds.to(gv.Image,  ['lon', 'lat'], 't', 'time').opts(cmap='viridis', 
                                                                colorbar=True)
    extras = []
    if wide_chile:
        easter_island_text = gv.Text(-104.360481, -24.104671, 'Easter Island').opts(color='white')
        easter_island_point = gv.Points([(-109.360481, -27.104671)]).opts(color='red')

        easter = easter_island_point * easter_island_text

        falkland_islands_text = gv.Text(-49.563412, -56.820557, 'Falklands').opts(color='white')
        falkland_islands_point = gv.Points([(-51.563412, -59.820557)]).opts(color='red')

        falkland = falkland_islands_point * falkland_islands_text
        
        extras.append(easter * falkland)
    
    plot = contours(plot, filled=True, overlaid=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        final_plot = plot * gf.coastline
        for extra in extras:
            final_plot *= extra
        gv.output(final_plot)


# In[6]:


cartopy_world_plot(region_da=clipped_region.t, level_sel=0)

geoview_flat_plot(region_da=clipped_region.t, level_sel=0)


# ## Objetivo cumplido
# 
# SI

# ## Guardando nuevo Dataset
# 
# NOTE: Because we are saving as zarr dataset, chunks must be uniform

# In[7]:


# checking if chunks are uniform
print(len(set(clipped_region.chunks['time'])) == 1 ) 

# finding correct chunk_size for time dim
print(sum(clipped_region.chunks['time']) / 40)

# checking chunks agains
print(len(set(clipped_region.chunk({'time': 8766}).chunks['time'])) == 1)

rechunked_region = clipped_region.chunk({'time': 8766})


# In[8]:

# 
#from dask.distributed import Client
#client = Client(processes=False)
#client


# In[12]:


# Might take some time and should be ran as script to avoid tljh crash
if save_new_dataset:
    output_dir = DATA_DIR / 'new_region' 
#    shutil.rmtree(output_dir, ignore_errors=True) # remove if exists
    
    print('Exporting...')
    rechunked_region.to_netcdf(output_dir / 'dataset.nc')

#     rechunked_region.to_zarr(output_dir)


# In[10]:


# !jupyter nbconvert --to script Processing.ipynb


# In[ ]:




