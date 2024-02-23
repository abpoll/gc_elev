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
# It is the only occtype with hazus and naccs
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

# Generally, we will process these DDFs the same way since they
# are written in mostly the same format
# However, there are a few preprocessing steps necessary for the hazus
# ddfs. Also, there are some differences for NACCS vs. HAZUS
# shallow uncertainty representation
# Read depth damage functions
ddf_filedir = join(VULN_DIR_UZ, "physical", NATION)
naccs = pd.read_csv(join(ddf_filedir, "naccs_ddfs.csv"))
hazus = pd.read_csv(join(ddf_filedir, "haz_fl_dept.csv"))

# First, preprocessing for hazus ddfs
# For basements, use FIA (MOD.) which does one and two floors by
# A and V zones
# For no basements, use USACE - IWR
# which does one and two floor, no flood zone specified
# 106: FIA (MOD.) 1S WB A zone
# 114: "" V zone
# 108: FIA (MOD.) 1S WB A zone
# 116: "" V zone
# 129: USACE - IWR 1S NB
# 130: USCAE - IWR 2S+ NB
# For elevating homes, we can use Pile foundation DDFs
# from USACE - Wilmington
# 178 - 1S Pile Foundation
# 183 - 2S Pile Foundation
# These are no basement homes, so to speak
# The USACE New Orleans DDFs have some pier foundation
# DDFs with fresh & salt water and long & short duration
# but this does not appear to apply to out study area
# Subset to DmgFnId in the codes above
dmg_ids = [106, 108, 114, 116, 129, 130, 178, 183]
hazus_res = hazus[(hazus['DmgFnId'].isin(dmg_ids)) & 
                  (hazus['Occupancy'] == 'RES1')]

# Make occtype column in the same form that the NSI has
# e.g. RES1-1SNB
# Add column for A or V zone
# Note: outside SFHA basement homes will take A zone
# What other option do we have? 

# Split Description by comma. 
# The split[0] element tells us stories (but description sometimes
# says floors instead of story...)
# Can get around this issue by looking at first word
# The split[1] element
# tells us w/ basement or no basement. Use this to create occtype
desc = hazus_res['Description'].str.split(',')
s_type = desc.str[0].str.split(' ').str[0]
s_type = s_type.str.replace('one', '1').str.replace('two', '2')
b_type = desc.str[1].str.strip()
# Below, we are just trying to get archetypes like
# 1SNB, 2SWB, 1SPL -- for pile foundation
occtype = np.where(b_type == 'w/ basement',
                   s_type + 'SWB',
                   s_type + 'SNB')
occtype = np.where(b_type == 'Pile foundation',
                   s_type + 'SPL',
                   occtype)
# Some of these HAZUS DDFs require us to keep track of the
# flood zone they're in
# I don't think this matters for our case study since
# there are no high wave coastsal zones
# This line is designed to work specifically 
# with the way the descriptions
# are written out for the DDFs used in this case study
fz = desc.str[-1].str.lower().str.replace('structure', '').str.strip()

# Need occtype, flood zone, depth_ft, and rel_dam columns
# Follow steps from naccs processing to get depth_ft and rel_dam
# First, drop unecessary columns
# Don't need Source_Table, Occupy_Class, Cover_Class, empty columns
# Description, Source, DmgFnId, Occupancy and first col (Unnamed: 0)
# because index was written out
# Don't need all na columns either (just for automobiles, apparently)
hazus_res = hazus_res.loc[:,[col for col in hazus_res.columns if 'ft' in col]]
hazus_res = hazus_res.dropna(axis=1, how='all')
# Add the occtype and fld_zone columns
hazus_res = hazus_res.assign(occtype=occtype,
                             fld_zone=fz.str[0])

# Then, occtype and fld_zone as index and melt rest of columns. 
idvars = ['occtype', 'fld_zone']

# Get a tidy ddf back
hazus_melt = tidy_ddfs(hazus_res, idvars)

# Delete depth_str and pctdam and standardize
# column names
# Since we just have the building types, call this
# bld_type instead of occtype
dropcols = ['depth_str', 'pct_dam', 'occtype', 'fld_zone']

# We create an "id" col for the ddfs
# Our key for HAZUS is bld_type & fld_zone
ddf_id = np.where(hazus_melt['fld_zone'].notnull(),
                  hazus_melt['occtype'] + '_' + hazus_melt['fld_zone'],
                  hazus_melt['occtype'])

# Add this to our dataframe so that we can drop bld_type & fld_zone
# Easier to have the flood zone as a capital letter
# It's lower case because of earlier code to do
# some processing
hazus_melt = hazus_melt.assign(ddf_id=pd.Series(ddf_id).str.upper())
# Drop columns
hazus = hazus_melt.drop(columns=dropcols)

# We need to interpolate between the values of the DDF that we
# are given. Generally speaking, this introduces artificial spread
# in the relative damage distribution since the interpolation is
# actually a combo of measurement & modeling uncertainty that
# the DDF bounds yield. But, linear interpolation between DDF points
# is so common that we will not depart from that before a paper
# makes the rigorous case that the approach is not needed, once
# you use DDFs w/ uncertainty bounds

# To do this interpolation, we loop through each ddf_id, 
# and then we will just sample 10 points and create nan rows
# (besides ddf_id). Then we interpolate, store in a list
# and concat at the end
df_int_list = []
for ddf_id, df in hazus.groupby('ddf_id'):
    # This creates the duplicate rows
    ddf_int = df.loc[np.repeat(df.index, 10)].reset_index(drop=True)
    # Now we have to make them nulls by finding
    # the "original" indexed rows
    ddf_int.loc[ddf_int.index % 10 != 0, ['depth_ft', 'rel_dam']] = np.nan
    # Now we interpolate
    ddf_int = ddf_int.interpolate()
    # Drop duplicate rows (this happens for the max depth values)
    ddf_int = ddf_int.drop_duplicates()
    # And append
    df_int_list.append(ddf_int)
hazus_ddfs = pd.concat(df_int_list, axis=0)

# Now we're going to process this tidy dataframe into a dictionary
# for easier ensemble generation

# After we get this new column, we are going to create two
# new columns based on the +/- .3*pt_estimate (30% uncertainty) 
# assumption from Maggie's paper 
# (https://www.nature.com/articles/s41467-020-19188-9)
# We will take the ddf_id, depth_ft, and these two columns
# to do the same thing as before for the dict of dicts
# We need to use max(0, ) and min(1, ) to make sure the +/- .3
# doesn't lead to negative losses, greater than 100% losses
# Since Maggie's paper, though, there have been studies
# suggesting that the damage distribution at each depth
# follows more of a long upper tailed Beta distribution. 
# While we don't have parameters for this, we can at least 
# represent a wider upper tail. So, -.3 and +.5 can better
# represent this
# A key assumption is that
# we can round depths to the nearest value in the
# dictionary to estimate their loss. There is no guidance in the
# use of DDFs about interpolating between values given on the DDF
# NFIP assessed damages data (recently released with the new v2 of
# the NFIP claims) only provides depth in feet, rounded to the
# nearest foot. So, any uncertainty surrounding the depth-damage
# relationship for any foot should include some component of 
# measurement error in representing some non rounded depth value
# to the rounded value and estimating a relationship
# To implement this, we will round all depths to the nearest foot
# before we check for whether they are inside the bounds for
# estimating losses with a particular depth-damage function
# Because of this, rounding the parameters to the nearest
# hundredth is a much lower order concern
dam_low = np.maximum(0,
                     hazus_ddfs['rel_dam'] - .3*hazus_ddfs['rel_dam']).round(2)
dam_high = np.minimum(1,
                      hazus_ddfs['rel_dam'] + .5*hazus_ddfs['rel_dam']).round(2)

# Add these columns into our dataframe
hazus_ddfs = hazus_ddfs.assign(low=dam_low,
                               high=dam_high)

# For reasons that will become more obvious later,
# it is really helpful to store our params as a list
# Get param cols
uni_params = ['low', 'high']

# Get df of ddf_id, depth_ft, rel_dam
hazus_f = hazus_ddfs[['ddf_id', 'depth_ft', 'rel_dam']]
# Now store params as a list
hazus_f = hazus_f.assign(params=hazus_ddfs[uni_params].values.tolist())

# We are going to write out hazus_f 
# In generating the ensemble for losses
# we are going to merge this dataframe
# with our structure ensemble - merged with
# depths. So, on haz_depth & depth_ft from hazus_f
# plus the structure archetype, we can get
# the rel_dam parameters. We will draw from this
# and get the rel_dam realization for this
# state of the world
# But, he way this data is stored requires a few assumptions
# about loss estimation
# First, any depths below that lowest depth have 0 loss
# Second, any depths above the highest depth have the same
# loss as the highest depth 
# To implement this, we will check depths (after drawing from their
# distribution at each location) for whether they are inside
# the range of the particular DDF which can be defined with 
# conastants. If below, loss is 0. If above, swap with
# the upper bound
# This is why it's very helpful to have the params stored as 
# a list, because now we can get unique key/value pairs
# for the ddf_id and the params
# We need two dicts for HAZUS
# One is with the params list
# One is just ddf_id to rel_dam (for benchmark loss calculations
# when uncertainty is not considered)

# We can call our helper function to get our dictionaries
HAZUS_MAX_DICT = ddf_max_depth_dict(hazus_f.reset_index(drop=True),
                                    'params')
HAZUS_MAX_NOUNC_DICT = ddf_max_depth_dict(hazus, 'rel_dam')

# For NACCS, we have the RES 1 DDFs
# For elevation, we have RES-OPEN and RES-ENC
# These are very similar in terms of damages so
# only need to retain RES-OPEN for simplicity for
# our current case study
# NACCS need some preprocessing as well
# First, subset to the relevant Occupancy types
# We want to end up with ddf ids 1swb, open, etc.
# don't need to keep the RES1- part for this case study
naccs['res_type'] = naccs['Occupancy'].str.split('-').str[0]
naccs['bld_type'] = naccs['Occupancy'].str.split('-').str[1]
occ_types = ['1SWB', '2SWB', '1SNB', '2SNB', 'OPEN']
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

# We need one hazus file with params for 
# uncertainty and one w/ just rel_dam
hazus_unc = hazus_f[['ddf_id', 'depth_ft', 'params']]
hazus_nounc = hazus_f[['ddf_id', 'depth_ft', 'rel_dam']]

# Main directory
ddf_out_dir = join(VULN_DIR_I, 'physical')
# Main ddf files
hazus_out_filep = join(ddf_out_dir, 'hazus_ddfs.pqt')
hazus_nounc_out_filep = join(ddf_out_dir, 'hazus_ddfs_nounc.pqt')
naccs_out_filep = join(ddf_out_dir, 'naccs_ddfs.pqt')
# Dictionaries - save as .json for simplicity
naccs_max_filep = join(ddf_out_dir, 'naccs.json')
hazus_max_filep = join(ddf_out_dir, 'hazus.json')
hazus_max_nounc_filep = join(ddf_out_dir, 'hazus_nounc.json')

# Only need to call this for one of the files
# since they share the same parent directory
prepare_saving(hazus_out_filep)

# Save as parquet files since
# these will directly read in the
# DDF params as a list, not as a string
hazus_unc.to_parquet(hazus_out_filep)
hazus_nounc.to_parquet(hazus_nounc_out_filep)
naccs_out.to_parquet(naccs_out_filep)

# Save the json files
with open(naccs_max_filep, 'w') as fp:
    json.dump(NACCS_MAX_DICT, fp)

with open(hazus_max_filep, 'w') as fp:
    json.dump(HAZUS_MAX_DICT, fp)

with open(hazus_max_nounc_filep, 'w') as fp:
    json.dump(HAZUS_MAX_NOUNC_DICT, fp)

# We want S_FLD_HAZ_AR from the National Flood Hazard Layer
fld_haz_fp = join(POL_DIR_UZ, FIPS, 'S_FLD_HAZ_AR.shp')
nfhl = gpd.read_file(fld_haz_fp)

# Keep FLD_ZONE, FLD_AR_ID, STATIC_BFE, geometry
keep_cols = ['FLD_ZONE', 'FLD_AR_ID', 'STATIC_BFE', 'ZONE_SUBTY',
             'geometry']
nfhl_f = nfhl.loc[:,keep_cols]

# Adjust .2 pct X zones to X_500
nfhl_f.loc[nfhl_f['ZONE_SUBTY'] == '0.2 PCT ANNUAL CHANCE FLOOD HAZARD',
           'FLD_ZONE'] = nfhl_f['FLD_ZONE'] + '_500'

# Update column names
# Lower case
nfhl_f.columns = [x.lower() for x in nfhl_f.columns]

# Drop ZONE_SUBTY
nfhl_f = nfhl_f.drop(columns=['zone_subty'])

# Clip flood zones to our study area
clip_out_filep = join(FI, 'ref', FIPS, 'clip.gpkg')
clip_gdf = gpd.read_file(clip_out_filep)

# Reproj flood zones
nfhl_reproj = nfhl_f.to_crs(clip_gdf.crs)

# Clip
nfhl_clip = gpd.clip(nfhl_reproj, clip_gdf)

# Reproject back
nfhl_clip_out = nfhl_clip.to_crs(nfhl_f.crs)

# Write file
nfhl_out_filep = join(POL_DIR_I, FIPS, 'fld_zones.gpkg')
prepare_saving(nfhl_out_filep)
nfhl_clip_out.to_file(nfhl_out_filep,
                      driver='GPKG')

# This is optional: delete the nfhl directory to reduce
# the file storage burden
if RM_NFHL:
    # Get directory name
    nfhl_dir = join(POL_DIR_UZ, FIPS)
    
    # Try to remove the tree; if it fails,
    # throw an error using try...except.
    try:
        shutil.rmtree(nfhl_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

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
ovb_filep = join(VULN_DIR_R, 'social', STATEABBR, 'overburdened.gpkg')
ovb = gpd.read_file(ovb_filep)

# Remove "properties" from columns
col_updates = [x.replace("properties.", "") for x in ovb.columns]
ovb.columns = col_updates

# Rename some columns
ovb = ovb.rename(columns={'OVERBURDENED_COMMUNITY_CRITERI': 'ovb_crit'})

# Keep a subset of columns
ovb_f = ovb[['GEOID', 'ovb_crit', 'geometry']]

# The data already is limited to overburdened categories

# Write file
ovb_out_filep = join(VULN_DIR_I, 'social', FIPS, 'ovb.gpkg')
ovb_f.to_file(ovb_out_filep, driver='GPKG')

# NOAA SOVI data
sovi_suffix = 'SoVI2010_' + STATEABBR
sovi_filename = 'SoVI0610_' + STATEABBR + '.shp'
sovi_filep = join(VULN_DIR_UZ, 'social', STATEABBR,
                  sovi_suffix, sovi_filename)
sovi = gpd.read_file(sovi_filep)

# Subset columns
keep_cols = ['GEOID10', 'SOVI0610_1', 'SOVI0610_2',
             'SOVI0610' + STATEABBR]
sovi_high = sovi[keep_cols]

# Rename GEOID10 to GEOID
sovi_high = sovi_high.rename(columns={'GEOID10': 'GEOID'})

# Subset to tracts in our study area (using the tract_geo geometries)
sovi_f = tract_geo[['GEOID', 'geometry']].merge(sovi_high,
                                                on='GEOID',
                                                how='inner')

# Write out file
sovi_out_filep = join(VULN_DIR_I, 'social', FIPS, 'sovi.gpkg')
sovi_f.to_file(sovi_out_filep, driver='GPKG')

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
# Project nsi to flood zone crs
nsi_fz = nsi_clip_out.to_crs(nfhl_clip_out.crs)

# Spatial join, retaining flood zone cols
# Only need the id and geom from nsi for this
fz_m = gpd.sjoin(nsi_fz[['fd_id', 'geometry']],
                 nfhl_clip_out,
                 predicate='within')

# I checked for issues like overlapping flood zones
# resulting in NSI structures in multiple polygons
# and did not find any. That's good, but chances
# are there will be counties where this happens
# and we will need code to handle these consistently

# Write out fd_id/fld_ar_id/fld_zone/static_bfe
keep_cols = ['fd_id', 'fld_zone', 'fld_ar_id', 'static_bfe']
fz_m_out = fz_m[keep_cols]

nsi_fz_filep = join(EXP_DIR_I, FIPS, 'nsi_fz.pqt')
prepare_saving(nsi_fz_filep)
fz_m_out.to_parquet(nsi_fz_filep)

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
        # Subset to threshhold for FMA (from 2022 NOFO)
        sovi_sub = sovi_geo[sovi_geo['SOVI0610' + STATEABBR] > .75]
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