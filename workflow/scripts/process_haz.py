import itertools
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio 
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.mask
from pyproj import CRS
import os
os.environ['USE_PYGEOS'] = '0'

from util.files import *
from util.const import *

# FIPS will be passed in as an argument, one day...
FIPS = '34007'
# STATE ABBR and NATION will be derived from FIPS, one day...
STATEABBR = 'NJ'
NATION = 'US'

# Link NSI with depth grids
# I want to reproject other files to the hazard CRS because
# this is the data we want to maintain spatial accuracy with the most
# I might want to clip this to the GC clip boundary since it can
# potentially speed up some code for doing 
# point in raster, etc. 
# For my first pass linking up, I also want to include
# the 5th and 95th percentile grids and just use
# a heuristic approach for estimating the standard deviation
# for a normal distribution
# Get this standard deviation parameter and then use the median
# value as the mean 
# That's all we get from the link NSI with hazard step...
# Then in the ensemble merge step, we sample from
# the spatially varying distribution across all RPs

# To start, let's reproject the NSI to the HAZ_CRS
# Then prepare the coordinates for point in raster checks
nsi = gpd.read_file(join(EXP_DIR_I, FIPS, 'nsi_sf.gpkg'))
nsi_reproj = nsi.to_crs(HAZ_CRS)

# For each depth grid, we will sample from the grid
# by way of a list of coordinates from the reprojected
# nsi geodataframe (this is the fastest way I know to do it)
coords = zip(nsi_reproj['geometry'].x, nsi_reproj['geometry'].y)
coord_list = [(x, y) for x, y in coords]
print('Store NSI coordinates in list')

# We'll store series of fd_id/depth pairs for each rp_pctile
# in a list and concat this into a df after iterating
depth_list = []

# Dictionary to store the depth grids
dg_dict = {}

# Loop through RPs and the percentiles
# Probably should rename directories accordingly
# since there the boostrapped percentile
# is useful information 
for rp, pctile in itertools.product(RET_PERS, HAZ_DIRS):
    pct = pctile.split('_')[-1]
    dg = read_dg(rp, pctile)
    print('Read in ' + rp + ' RP depth grid for ' 
          + pct + ' percentile')

    # Sample from the depth grid based on structure locations
    # I did some ground truthing in qgis
    # It appears that the sampled values align correctly
    sampled_depths = [x[0] for x in dg.sample(coord_list)]
    print('Sampled depths from grid')

    # Store the series 
    depths = pd.Series(sampled_depths,
                       index=nsi_reproj['fd_id'],
                       name='_'.join([rp, pct]))
    # Add the series to the list of series
    depth_list.append(depths)
    print('Added depths to list\n')


# Concat to dataframe
depth_df = pd.concat(depth_list, axis=1)

# Replace nodata values with 0
depth_df[depth_df == dg.nodata] = 0

# Retain only structures with some flood exposure
depth_df_f = depth_df[depth_df.sum(axis=1) > 0]

# Multiply by MTR_TO_FT to convert to feet
depth_df_f = depth_df_f*MTR_TO_FT

# Write out dataframe that links fd_id to depths
# with columns corresponding to RETPER_PCTILE (i.e. 500_Mid)
nsi_depths_out = join(EXP_DIR_I, FIPS, 'nsi_depths.pqt')
# Round to nearest hundredth foot
# Depth-damage functions don't have nearly the precision
# to make use of inches differences, but some precision
# is needed for subtracting first floor elevation before rounding
depth_df_f.round(2).reset_index().to_parquet(nsi_depths_out)