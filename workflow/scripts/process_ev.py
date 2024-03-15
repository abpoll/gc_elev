import json
import glob
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import shape
import rasterio 
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio.mask
from pyproj import CRS
import os
import shutil
os.environ['USE_PYGEOS'] = '0'

from util.files import *
from util.const import *
from util.ddfs import *

# FIPS will be passed in as an argument, one day...
FIPS = '34007'
# STATE ABBR and NATION will be derived from FIPS, one day...
STATEABBR = 'NJ'
NATION = 'US'

# For our case study, we are going to focus on Gloucester City, NJ
# Our config.yaml loads in a county indexed clip file
# so that we can restrict all our data to the GC boundaries

# Read in the data we downloaded from the county's REST API server
clip_filep = join(REF_DIR_R, FIPS, 'clip.json')
with open(clip_filep) as f:
    clip_data = json.load(f)

# Use pandas to get the data in a form that is easier
# to turn into a geodataframe for clipping
clip_df = pd.json_normalize(clip_data['features'])
# We want to make a polygon out of the geometry coordinates
# We can access that from the original json object
clip_geo = [shape(i['geometry']) for i in clip_data['features']]
# We can create a geodataframe of clip_df by adding clip_geo
# as its geometry column
clip_gdf = gpd.GeoDataFrame(clip_df,
                            crs=CLIP_CRS,
                            geometry=clip_geo)

# We can clean up the gdf by removing the
# type, id, geometry.type and geometry.coordinates columns
drop_col = ['type', 'id', 'geometry.type', 'geometry.coordinates']
clip_gdf = clip_gdf.drop(columns=drop_col)

# Write the file out to interim
clip_out_filep = join(FI, 'ref', FIPS, 'clip.gpkg')
prepare_saving(clip_out_filep)
clip_gdf.to_file(clip_out_filep,
                 driver='GPKG')

# The NSI comes with all the data necessary for performing a standard 
# flood risk assessment. It is still useful to process the raw data.
# Here, we subset to residential properties with 1 to 2 stories
# and save as a geodataframe. These are the types of residences we have
# multiple depth-damage functions for and a literature base to draw 
# from to introduce uncertainty in these loss estimates
# Read NSI
nsi_filep = join(EXP_DIR_R, FIPS, 'nsi.json')
with open(nsi_filep, 'r') as fp:
    nsi_full = json.load(fp)

# json normalize 
nsi_df = pd.json_normalize(nsi_full['features'])

# Convert to gdf
# This is useful for some spatial joins we need to perform
# Convert to geodataframe
geometry = gpd.points_from_xy(nsi_df['properties.x'],
                              nsi_df['properties.y'])
nsi_gdf = gpd.GeoDataFrame(nsi_df, geometry=geometry,
                           crs=NSI_CRS)

# Drop the following columns
drop_cols = ['type', 'geometry.type', 'geometry.coordinates']
nsi_gdf = nsi_gdf.drop(columns=drop_cols)

# Remove "properties" from columns
col_updates = [x.replace("properties.", "") for x in nsi_gdf.columns]
nsi_gdf.columns = col_updates

# Subset to residential properties and update
# RES 1 - single family
# RES 2 - manufactured home
# RES 3 - multifamily (but could fit into a depth-damage function
# archetype depending on # stories)
# We are going to use RES1 for this case-study
# It is the only occtype with naccs
# DDFs and has less ambiguous classification

# occtype category for easier use in loss estimation steps

# Get residential structures
nsi_res = nsi_gdf.loc[nsi_gdf['occtype'].str[:4] == 'RES1']

# For this case-study, don't use any building with more 
# than 2 stories
res1_3s_ind = nsi_res['num_story'] > 2
# Final residential dataframe
res_f = nsi_res.loc[~res1_3s_ind]

# Subset to relevant columns
cols = ['fd_id', 'occtype', 'found_type', 'cbfips', 'bldgtype',
        'ftprntsrc', 'found_ht', 'val_struct', 'sqft',
        'val_cont', 'source', 'firmzone', 'ground_elv_m',
        'geometry']

res_out = res_f.loc[:,cols]

# Clip to our clip boundary
# They are in the same CRS
nsi_clip_out = gpd.clip(res_out, clip_gdf)

# Write out to interim/exposure/FIPS/
# Single family homes -- sf
EXP_OUT_FILEP = join(EXP_DIR_I, FIPS, 'nsi_sf.gpkg')
prepare_saving(EXP_OUT_FILEP)
# Limit to sqft <= 99th percentile
# Arbitrary cutoff. The max value from the steps above
# is 400858 which is way too large
# There are other large values that are dropped with this
# arbitrary cutoff
# For GC case study, this value is 2696.41999
sqft_clip = nsi_clip_out['sqft'].quantile(.99)
nsi_clip_out[nsi_clip_out['sqft'] <= sqft_clip].to_file(EXP_OUT_FILEP,
                                                        driver='GPKG')

# Processing the NACCS DDFs
# Read depth damage functions
ddf_filedir = join(VULN_DIR_UZ, "physical", NATION)
naccs = pd.read_csv(join(ddf_filedir, "naccs_ddfs.csv"))

# For NACCS, we have the RES 1 DDFs
# First, subset to the relevant Occupancy types
# We want to end up with ddf ids 1swb, open, etc.
# don't need to keep the RES1- part for this case study
naccs['res_type'] = naccs['Occupancy'].str.split('-').str[0]
naccs['bld_type'] = naccs['Occupancy'].str.split('-').str[1]
occ_types = ['1SWB', '2SWB', '1SNB', '2SNB']
naccs_res = naccs.loc[(naccs['bld_type'].isin(occ_types)) &
                      ((naccs['res_type'] == 'RES1') |
                       (naccs['res_type'] == 'RES'))]

# Next, drop columns we don't need
drop_cols = ['Description', 'Source', 'Occupancy', 'res_type']
naccs_res = naccs_res.drop(columns=drop_cols)

# Rename DamageCategory
naccs_res = naccs_res.rename(columns={'DamageCategory': 'dam_cat',
                                      'bld_type': 'ddf_id'})

# Now get the melted dataframe
idvars = ['ddf_id', 'dam_cat']
naccs_melt = tidy_ddfs(naccs_res, idvars)

# Drop columns we don't need
drop_cols = ['depth_str', 'pct_dam']
naccs_f = naccs_melt.drop(columns=drop_cols)

# We want to pivot the dataframe so that Min/ML/Max are our columns
naccs_piv = naccs_f.pivot(index=['ddf_id', 'depth_ft'],
                          columns='dam_cat')['rel_dam'].reset_index()


# We do the interpolation again
df_int_list = []
for ddf_id, df in naccs_piv.groupby('ddf_id'):
    # This creates the duplicate rows
    ddf_int = df.loc[np.repeat(df.index, 10)].reset_index(drop=True)
    # Now we have to make them nulls by finding
    # the "original" indexed rows
    ddf_int.loc[ddf_int.index % 10 != 0,
                ['depth_ft', 'ML', 'Max', 'Min']] = np.nan
    # Now we interpolate
    ddf_int = ddf_int.interpolate().round(2)
    # Drop duplicate rows (this happens for the max depth values)
    ddf_int = ddf_int.drop_duplicates()
    # And append
    df_int_list.append(ddf_int)
naccs_ddfs = pd.concat(df_int_list, axis=0)

# We want to obtain our 'params' column
# same as above
p_cols = ['Min', 'ML', 'Max']
tri_params = naccs_ddfs[p_cols].values
# Drop the p_cols
naccs_out = naccs_ddfs.drop(columns=p_cols)
naccs_out = naccs_out.assign(params=tri_params.tolist())

# Get out dict of max depths
NACCS_MAX_DICT = ddf_max_depth_dict(naccs_out.reset_index(drop=True),
                                    'params')

# Main directory
ddf_out_dir = join(VULN_DIR_I, 'physical')
# Main ddf files
naccs_out_filep = join(ddf_out_dir, 'naccs_ddfs.pqt')
# Dictionaries - save as .json for simplicity
naccs_max_filep = join(ddf_out_dir, 'naccs.json')

# Only need to call this for one of the files
# since they share the same parent directory
prepare_saving(naccs_out_filep)

# Save as parquet files since
# these will directly read in the
# DDF params as a list, not as a string
naccs_out.to_parquet(naccs_out_filep)

# Save the json files
with open(naccs_max_filep, 'w') as fp:
    json.dump(NACCS_MAX_DICT, fp)

# Process ref data
# For each .shp file in our unzipped ref directory
# we are going to reproject & clip, then write out
for path in Path(REF_DIR_UZ).rglob('*.shp'):
    # Read in the file
    ref_shp = gpd.read_file(path)
    
    # Process the filename to figure out what 
    # reference data this is
    # the files are written out in the form of
    # tl_2022_34_tract.shp, for example
    # so we split the string on '_', take the
    # last element of the array, and ignore
    # the last 4 characters
    ref_name = path.name.split('_')[-1][:-4]
    # Replace the ref name with our ref_name dict values
    ref_name_out = REF_NAMES_DICT[ref_name]

    # Reproject and clip our reference shapefile
    ref_reproj = ref_shp.to_crs(clip_gdf.crs)
    ref_clipped = gpd.clip(ref_reproj, clip_gdf)
    
    # Write file
    ref_out_filep = join(REF_DIR_I, FIPS, ref_name_out + ".gpkg")
    prepare_saving(ref_out_filep)
    ref_clipped.to_file(ref_out_filep,
                        driver='GPKG')

    # Helpful message to track progress
    print("Saved Ref: " + ref_name_out)

# Process social vulnerability data
# Load relevant spatial data (tract, block group)
tract_filep = join(REF_DIR_I, FIPS, 'tract.gpkg')
bg_filep = join(REF_DIR_I, FIPS, 'bg.gpkg')
tract_geo = gpd.read_file(tract_filep)
bg_geo = gpd.read_file(bg_filep)

# CEJST data
ce_filep = join(VULN_DIR_R, 'social', NATION, 'cejst.csv')
cejst = pd.read_csv(ce_filep, dtype={'Census tract 2010 ID': 'str'})

# Columns to keep
# Identified as disadvantaged
# Census tract 2010 ID
keep_cols = ['Census tract 2010 ID', 'Identified as disadvantaged']
cejst_sub = cejst[keep_cols]
# Rename columns
cejst_sub.columns = ['GEOID', 'disadvantaged']

# Merge with tract_geo
cejst_f = tract_geo[['GEOID', 'geometry']].merge(cejst_sub,
                                                 on='GEOID',
                                                 how='inner')

# Retain only the disadvantaged 
cejst_f = cejst_f[cejst_f['disadvantaged'] == True].drop(columns='disadvantaged')

# Write file
cejst_out_filep = join(VULN_DIR_I, 'social', FIPS, 'cejst.gpkg')
prepare_saving(cejst_out_filep)
cejst_f.to_file(cejst_out_filep, driver='GPKG')

# NJ overburdened data

# Read data
ovb_filep = join(VULN_DIR_UZ, 'social', STATEABBR,
                 'Govt_census_group_2022_EJ.gdb')
ovb = gpd.read_file(ovb_filep)

# Rename some columns
ovb = ovb.rename(columns={'OVERBURDENED_COMMUNITY_CRITERI': 'ovb_crit'})

# Keep a subset of columns
ovb_f = ovb[['GEOID', 'ovb_crit', 'geometry']]

# The data already is limited to overburdened categories

# Subset to our study area
ovb_reproj = ovb_f.to_crs(clip_gdf.crs)
ovb_clipped = gpd.clip(ovb_reproj, clip_gdf)

# Write file
ovb_out_filep = join(VULN_DIR_I, 'social', FIPS, 'ovb.gpkg')
ovb_clipped.to_file(ovb_out_filep, driver='GPKG')

# CDC SVI data
svi_filename = 'svi.csv'
svi_filep = join(VULN_DIR_R, 'social', NATION, svi_filename)
svi = pd.read_csv(svi_filep)

# Subset columns
# The overall summary ranking variable is RPL_THEMES
# From https://www.atsdr.cdc.gov/placeandhealth/svi/
# documentation/SVI_documentation_2020.html
keep_cols = ['FIPS', 'RPL_THEMES']
svi_high = svi[keep_cols]

# Rename FIPS to GEOID
# Rename RPL_THEMES to sovi
# GEOID needs to be a str, 11 characters long
svi_high = svi_high.rename(columns={'FIPS': 'GEOID',
                                    'RPL_THEMES': 'sovi'})
svi_high['GEOID'] = svi_high['GEOID'].astype(str).str.zfill(11)

# Subset to tracts in our study area (using the tract_geo geometries)
svi_f = tract_geo[['GEOID', 'geometry']].merge(svi_high,
                                               on='GEOID',
                                               how='inner')

# Write out file
sovi_out_filep = join(VULN_DIR_I, 'social', FIPS, 'sovi.gpkg')
svi_f.to_file(sovi_out_filep, driver='GPKG')

# LMI data
# Read data
lmi_filename = 'ACS_2015_lowmod_blockgroup_all.xlsx'
lmi_filep = join(VULN_DIR_R, 'social', NATION, lmi_filename)
lmi = pd.read_excel(lmi_filep, engine='openpyxl')
# Get GEOID for merge (last 12 characters is the bg id)
lmi['GEOID'] = lmi['GEOID'].str[-12:]

# Retain GEOID and Lowmod_pct
keep_cols = ['GEOID', 'Lowmod_pct']
lmi_f = bg_geo[['GEOID', 'geometry']].merge(lmi[keep_cols],
                                            on='GEOID',
                                            how='inner')

# Write file
lmi_out_filep = join(VULN_DIR_I, 'social', FIPS, 'lmi.gpkg')
lmi_f.to_file(lmi_out_filep, driver='GPKG')

# Link attributes to nsi

# For zcta, tract, bg, and block
# we want to do spatial joins to link
# up fd_id in the NSI with the ref
# We will use config data to do this
# since other references may be brought in 
# down the line
# We are going to store fd_id/ref_id links in a dataframe
ref_df_list = []
for ref_name, ref_id in REF_ID_NAMES_DICT.items():
    ref_filep = join(REF_DIR_I, FIPS, ref_name + ".gpkg")

    # Load in the ref file
    ref_geo = gpd.read_file(ref_filep)

    # Limit the geodataframe to our ref id and 'geometry' column
    keep_col = [ref_id, 'geometry']
    ref_geo_sub = ref_geo[keep_col]

    # Limit the NSI to our fd_id and geometry column
    keep_col_nsi = ['fd_id', 'geometry']
    nsi_sub = nsi_clip_out[keep_col_nsi]

    # Reproj nsi_sub to the reference crs
    nsi_reproj = nsi_sub.to_crs(ref_geo.crs)

    # Do a spatial join
    nsi_ref = gpd.sjoin(nsi_reproj, ref_geo_sub, predicate='within')

    # Set index to fd_id and just keep the ref_id
    # Rename that column to our ref_name + '_id'
    # Append this to our ref_df_list
    nsi_ref_f = nsi_ref.set_index('fd_id')[[ref_id]]
    nsi_ref_f = nsi_ref_f.rename(columns={ref_id: ref_name + '_id'})
    ref_df_list.append(nsi_ref_f)

    # Helpful message
    print('Linked reference to NSI: ' + ref_name + '_id')

# Can concat and write
nsi_refs = pd.concat(ref_df_list, axis=1).reset_index()
ref_filep = join(EXP_DIR_I,  FIPS, 'nsi_ref.pqt')
prepare_saving(ref_filep)
nsi_refs.to_parquet(ref_filep)

# Read in processed sovi data
# Loop through the community boundary data
# Get links to the single family home data
# Store in single dataframe
# Write out

sovi_dir = join(VULN_DIR_I, 'social', FIPS)
filenames = ['lmi', 'sovi', 'ovb', 'cejst']

sovi_df_list = []
for fn in filenames:
    # Read in each gpkg
    fp = join(sovi_dir, fn + '.gpkg')
    sovi_geo = gpd.read_file(fp)

    # Subset sovi_geo based on thresholds
    # For cejst and ovb this is already done
    # For lmi and ovb need to do the filter as follows
    if fn == 'lmi':
        # See https://www.hudoig.gov/reports-publications/
        # report/cdbg-dr-program-generally-
        # met-low-and-moderate-income-requirements
        # The statutory hreshold is 50%, so retain those
        sovi_sub = sovi_geo[sovi_geo['Lowmod_pct'] > .5]
    elif fn == 'sovi':
        # Subset to threshhold for FMA/BRIC (from 2022 NOFO)
        sovi_sub = sovi_geo[sovi_geo['sovi'] > .6]
    else:
        sovi_sub = sovi_geo

    # Only need the geometry for sovi_sub
    sovi_sub = sovi_sub[['geometry']]
    
    # Limit the NSI to our fd_id and geometry column
    keep_col_nsi = ['fd_id', 'geometry']
    nsi_sub = nsi_clip_out[keep_col_nsi]

    # Reproj nsi_sub to the reference crs
    nsi_reproj = nsi_sub.to_crs(sovi_geo.crs)

    # Do a spatial join
    nsi_sovi = gpd.sjoin(nsi_reproj, sovi_sub, predicate='within')

    # Add indicator column
    nsi_sovi[fn] = True

    # Append this to our sovi_df_list
    sovi_df_list.append(nsi_sovi[['fd_id', fn]].set_index('fd_id'))

    # Helpful message
    print('Linked vulnerability to NSI: ' + fn)

sovi_df_f = pd.concat(sovi_df_list, axis=1).fillna(False)
sovi_out_filepath = join(sovi_dir, 'c_indicators.pqt')
sovi_df_f.to_parquet(sovi_out_filepath)