import os
import json
from pathlib import Path
from os.path import join
os.environ["USE_PYGEOS"] = "0"
import geopandas as gpd
import pandas as pd
import numpy as np

from util.files import *
from util.const import *
from util.ddfs import *

# FIPS will be passed in as an argument, one day...
FIPS = '34007'
# STATE ABBR and NATION will be derived from FIPS, one day...
STATEABBR = 'NJ'
NATION = 'US'

# Generate structure ensemble
# Merge hazard data in
# Sample from the depth grids
# Add our vulnerability uncertainty
# (it's conditioned on the depth value in 
# a particular state of the world)

# Load the single family homes,
# the fd_id/reference file
# the fd_id/depths file
# the fd_id flood zone file
nsi_struct = gpd.read_file(join(EXP_DIR_I, FIPS, 'nsi_sf.gpkg'))
nsi_ref = pd.read_parquet(join(EXP_DIR_I, FIPS, 'nsi_ref.pqt'))
nsi_depths = pd.read_parquet(join(EXP_DIR_I, FIPS, 'nsi_depths.pqt'))
nsi_fz = pd.read_parquet(join(EXP_DIR_I, FIPS, 'nsi_fz.pqt'))

# Filter to properties with > 0 
nsi_depths = nsi_depths[nsi_depths.iloc[:,1:].sum(axis=1) > 0]

# We need to melt our dataframe
# Split return periods and scenarios
# then pivot with fd_id and scenarios as our id vars
nsi_d_melt = nsi_depths.melt(id_vars='fd_id', value_name='depth_ft')
nsi_d_melt['rp'] = nsi_d_melt['variable'].str.split('_').str[0]
nsi_d_melt['scen'] = nsi_d_melt['variable'].str.split('_').str[1]
depths_df = nsi_d_melt.pivot(index=['fd_id', 'scen'], columns=['rp'],
                             values='depth_ft').reset_index()

# Need foundation type, number stories, structure value
# for our ensemble. Structure value will be the center of 
# the distribution and will be passed to the loss estimation
# function. Foundation type will be drawn from the implicit
# distribution in the NSI data. For each census block, 
# we are going to get the multinomial probabilities of 
# a building having a certain foundation type & number of stories
# Ideally, we would do this conditioned on prefirm but the
# building year column is based on median year built from ACS
# data
# From the foundation type that is drawn from the multinomial in 
# the ensemble, we will get the FFE from the distribution 
# defined in the code for the Wing et al. 2022 paper
# The point estimate version will just use default values

# Start by retaining only relevant columns in nsi_struct
# Then subset this and nsi_ref to the fd_id in nsi_depths
# We do need sqft for elevation cost or floodproof estimates

# Normally we would only keep the below, but I'm commenting those out
# because we also want to keep found_ht
# keep_cols = ['fd_id', 'occtype', 'val_struct']
keep_cols = ['fd_id', 'occtype', 'val_struct', 'bldgtype',
             'found_type', 'found_ht', 'sqft']
nsi_res = nsi_struct[keep_cols]

# Let's merge in refs into nsi_res
nsi_res = nsi_res.merge(nsi_ref, on='fd_id')

# We're also going to merge in fzs
nsi_res = nsi_res.merge(nsi_fz[['fd_id', 'fld_zone']], on='fd_id')

# Retain only the rows that correspond to structures
# that are exposed to flood depths
## For this case study, we don't need to merge depths in
# at this stage
full_df = nsi_res[nsi_res['fd_id'].isin(nsi_depths['fd_id'])]

# Let's get the fld_zone column processed for the way it needs
# to be done for using hazus ddfs
# Get the first character of the flood zone and only retain it
# if it's a V zone. We are going to use A zone for A and outside
# (if any) flood zone depth exposures
ve_zone = np.where(full_df['fld_zone'].str[0] == 'V',
                   'V',
                   'A')
full_df = full_df.assign(fz_ddf = ve_zone)

# Load DDFs
naccs_ddfs = pd.read_parquet(join(VULN_DIR_I, 'physical', 'naccs_ddfs.pqt'))
hazus_ddfs = pd.read_parquet(join(VULN_DIR_I, 'physical', 'hazus_ddfs.pqt'))
hazus_nounc = pd.read_parquet(join(VULN_DIR_I, 'physical', 'hazus_ddfs_nounc.pqt'))

# Load helper dictionaries
with open(join(VULN_DIR_I, 'physical', 'hazus.json'), 'r') as fp:
    HAZUS_MAX_DICT = json.load(fp)

with open(join(VULN_DIR_I, 'physical', 'hazus_nounc.json'), 'r') as fp:
    HAZUS_MAX_NOUNC_DICT = json.load(fp)

with open(join(VULN_DIR_I, 'physical', 'naccs.json'), 'r') as fp:
    NACCS_MAX_DICT = json.load(fp)

# We need a randon number generator
rng = np.random.default_rng()

# Need to create a dataframe w/ N_SOW rows for each fd_id
# From full_df, keep fd_id, val_struct, bg_id, and the
# depth columns. 
# The way I usually do this is with
# df.loc[np.repeat(df.index, N)].reset_index(drop=True)
# With this approach, we can do everything in a vectorized
# form by passing array_like data of size N*len(df)
# to different rng() calls to get all the draws from
# distributions that we need
drop_cols = ['block_id', 'fld_zone']
ens_df = full_df.drop(columns=drop_cols)
ens_df = ens_df.loc[np.repeat(ens_df.index, N_SOW)].reset_index(drop=True)
print('Created Index for Ensemble')

# Values
# Draw from the structure value distribution for each property
# normal(val_struct, val_struct*CF_DET) where these are array_like
# I also want to treat this as truncated
# on the lower end since there is a risk of drawing impossibly
# low numbers (like negative) with this approach
# https://github.com/kieranrcampbell/blog-notebooks/blob/master/
# Fast%20vectorized%20sampling%20from%20truncated%
# 20normal%20distributions%20in%20python.ipynb
# outlines an approach to use numpy to do a truncated sample
# TODO move this to a util file
def truncnorm_rvs_recursive(x, sigma, lower_clip):
    rng = np.random.default_rng()
    q = rng.normal(x, sigma)
    if np.any(q < lower_clip):
        # Adjustment to the code provided to index the sigma vector
        q[q < lower_clip] = truncnorm_rvs_recursive(x[q < lower_clip],
                                                    sigma[q < lower_clip],
                                                    lower_clip)

    return q
# Using 20000 as an artificial, arbitrary lower bound on value
ens_df['val_s'] = truncnorm_rvs_recursive(ens_df['val_struct'],
                                          ens_df['val_struct']*COEF_VARIATION,
                                          20000)

print('Draw values')

# For this case study, use the below code
# This drops the "RES1-" part of the occtype column
# and keeps 1SNB, 2SNB, etc.
ens_df['bld_types'] = ens_df['occtype'].str.split('-').str[1]

# In theory, bld_type is naccs_ddf_type. No need to 
# take this storage up in practice... just refer to bld_type
# when needed
# For WB homes, hazus_ddf_type is bld_types + '_' + ens_df['fz_ddf']
# For NB homes, it's bld_types
# It makes practical sense to create a new series for this
ens_df['hazus_types'] = np.where(ens_df['bld_types'].str[-2:] == 'WB',
                                 ens_df['bld_types'] + '_' + ens_df['fz_ddf'],
                                 ens_df['bld_types'])


# We are going to use the fnd_type to draw from the
# FFE distribution
# Need to use np.stack to get the array of floats
tri_params = np.stack(ens_df['found_type'].map(FFE_DICT))

# Can use [:] to access like a matrix and directly input to 
# rng.triangular
# 0, 1, and 2 are column indices corresponding to left,
# mode, and right
# We round this to the nearest foot
ffes = np.round(rng.triangular(tri_params[:,0],
                               tri_params[:,1],
                               tri_params[:,2]))
ens_df['ffe'] = ffes

print('Generated Structure Characteristics')

## For this case study, we're using depths as scenarios
# Loop through each of Lower, Mid, Upper in the depths_df
# and merge this depth_df into ens_df
# This is hard coded (i.e. Lower/Mid) and (500) which isn't ideal
# and I should replace these with values in the config file
# Store this in a dictionary - it's a little easier
ens_dfs = {}
# Also helps to have a dictionary for the depths adjusted
# by first floor elevation
depth_ffes = {}
for scen in ['Lower', 'Mid', 'Upper']:
    print('Scenario: ' + scen)
    # We subset to the scenario
    depth_df = depths_df[depths_df['scen'] == scen].drop(columns=['scen'])
    # We only need to keep properties with depth[500] > 0
    keep_rows = depth_df['500'] > 0
    depth_df = depth_df.loc[keep_rows]
    # Replace 0 values with na
    depth_df[depth_df == 0] = np.nan
    # Let's do an inner merge so that we don't have
    # to keep the ensemble members that correspond to 
    # 0 losses under this scenario
    ens_dfs[scen] = ens_df.merge(depth_df, how='inner', on='fd_id')
    # Dataframe for adjusted depths
    # depth_df and ens_dfs
    depth_ffes[scen] = ens_dfs[scen][RET_PERS].subtract(ens_dfs[scen]['ffe'],
                                                        axis=0).round(1) 
    print('Adjuted depths by FFE\n')

# Now, we are going to loop through each return period
# and estimate losses for NACCS and HAZUS using our helper
# functions for each of these

# We do this for each of the ens_df in ens_dfs
for scen, ens_df in ens_dfs.items():
    print('Scenario: ' + scen)
    # Get the depth_ffe_df
    depth_ffe_df = depth_ffes[scen]
    
    # We will store these in dictionaries with return period keys
    hazus_loss = {}
    naccs_loss = {}
    
    for rp in RET_PERS:
        naccs_loss[rp] = est_naccs_loss(ens_df['bld_types'],
                                        depth_ffe_df[rp],
                                        naccs_ddfs,
                                        NACCS_MAX_DICT)
        hazus_loss[rp] = est_hazus_loss(ens_df['hazus_types'],
                                        depth_ffe_df[rp],
                                        hazus_ddfs,
                                        HAZUS_MAX_DICT)
    
        print('Estimate Losses for NACCS & Hazus, RP: ' + rp)
    
    # Then, we convert these to dataframes
    hazus_df = pd.DataFrame.from_dict(hazus_loss)
    naccs_df = pd.DataFrame.from_dict(naccs_loss)

    # We define the losses_df by concatenating the hazus & naccs
    # data frames along their columns, after fixing their column
    # names
    hazus_df.columns = ['haz_rel_dam_' + x for x in hazus_df.columns]
    naccs_df.columns = ['naccs_rel_dam_' + x for x in naccs_df.columns]
    losses_df = pd.concat([hazus_df, naccs_df], axis=1)

    # Now we concat these with ens_df, stories, fnd_type,
    # ffe, structure value, and depth_ffe_df
    depth_ffe = pd.DataFrame.from_dict(depth_ffe_df)
    
    # Add clearer column names
    depth_ffe.columns = ['depth_ffe_' + x for x in depth_ffe.columns]

    # For our case study, ens_df contains occtype & 
    # found_ht, so don't need to add structure characteristics
    # back in
    ens_df = pd.concat([ens_df, losses_df, depth_ffe],
                       axis=1)
    
    # Get relative damage columns
    rel_cols = [x for x in ens_df.columns if 'rel_dam' in x]
    # For each relative damage column, scale by val_s, the structure
    # value realization
    # We need to do this for naccs & hazus prefixes
    for col in rel_cols:
        prefix = col.split('_')[0]
        rp = col.split('_')[-1]
        ens_df[prefix + '_loss_' + rp] = ens_df[col]*ens_df['val_s']
    
    print('Obtained Full Ensemble')

    # Now we calculate EAL
    # We will use trapezoidal approximation for this
    # Using trapezoid method and adding bin of lowest probability
    # events to obtain expected annual 
    
    # We make a list of our loss columns
    # This is easier to do splitting by prefix
    hazus_loss_list = ['hazus_loss_' + x for x in RET_PERS]
    naccs_loss_list = ['naccs_loss_' + x for x in RET_PERS]
    # As well as the corresponding probabilities
    p_rp_list = [round(1/int(x), 4) for x in RET_PERS]
    
    # Then we create an empty series
    # Two, for hazus & naccs loss estimates
    eal_hazus = pd.Series(index=ens_df.index).fillna(0)
    eal_naccs = pd.Series(index=ens_df.index).fillna(0)
    
    # We loop through our loss list and apply the 
    # trapezoidal approximation
    for i in range(len(hazus_loss_list) - 1):
        loss1_hazus = ens_df[hazus_loss_list[i]]
        loss2_hazus = ens_df[hazus_loss_list[i+1]]
        loss1_naccs = ens_df[naccs_loss_list[i]]
        loss2_naccs = ens_df[naccs_loss_list[i+1]]
        rp1 = p_rp_list[i]
        rp2 = p_rp_list[i+1]
        # We add each approximation
        eal_hazus += (loss1_hazus + loss2_hazus)*(rp1-rp2)/2
        eal_naccs += (loss1_naccs + loss2_naccs)*(rp1-rp2)/2
    # This is the final trapezoid to add in
    final_eal_hazus = eal_hazus + ens_df[hazus_loss_list[-1]]*p_rp_list[-1]
    final_eal_naccs = eal_naccs + ens_df[naccs_loss_list[-1]]*p_rp_list[-1]
    print('Calculated EAL')
    # Add eal columns to our dataframe
    ens_df = pd.concat([ens_df, pd.Series(final_eal_hazus, name='hazus_eal')],
                       axis=1)
    ens_df = pd.concat([ens_df, pd.Series(final_eal_naccs, name='naccs_eal')],
                       axis=1)
    
    # Let's also get the SOW index - start at 0
    sow_ind = np.arange(len(ens_df))%N_SOW
    ens_df = pd.concat([ens_df, pd.Series(sow_ind, name='sow_ind')], axis=1)

    # Put this back in ens_dfs[scen]
    ens_dfs[scen] = ens_df
    print('Stored in dictionary\n')

# Write out our ensemble dfs
ens_out_filep = join(FO, 'ensemble.pqt')
prepare_saving(ens_out_filep)
ens_dfs['Lower'].to_parquet(join(FO, 'ensemble_Lower.pqt'))
ens_dfs['Mid'].to_parquet(join(FO, 'ensemble_Mid.pqt'))
ens_dfs['Upper'].to_parquet(join(FO, 'ensemble_Upper.pqt'))