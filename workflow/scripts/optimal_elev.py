import os
import json
from pathlib import Path
from os.path import join
import pandas as pd
import numpy as np
import numpy.ma as ma

from util.files import *
from util.const import *
from util.ddfs import *

# FIPS will be passed in as an argument, one day...
FIPS = '34007'
# STATE ABBR and NATION will be derived from FIPS, one day...
STATEABBR = 'NJ'
NATION = 'US'

# I think it also could make sense to pass in scenario and
# ddf type as arguments. For main results
# we're using 'mid' and 'naccs' but for generating
# our sensitivity analysis results we will need to pass
# in the other scenarios and 'hazus'
scenarios = ['Lower', 'Mid', 'Upper']
ddf_types = ['naccs', 'hazus']

# Everything that we do here is based on the ensemble values
# That means we take the ffe variable in our ensemble df
# and adjust it by the heightening amount, re-estimate losses
# across all return periods, and re-estimate eal
# In fact, since depths are "fixed" in our case study
# we don't have to adjust the ffe variable, and can instead
# adjust the depth_ffe_* columns

# These are shared columns for subsetting
# We need found_type because it is used in
# elevation cost estimation
# We need sqft because it's a key variable for
# elevation cost estimation
# We need bldgtype for elevation cost estimation, too
sub_cols = ['fd_id', 'found_type', 'sqft', 'bldgtype',
            'bld_types', 'hazus_types', 'naccs_eal',
            'haz_eal', 'val_s', 'sow_ind']
# We need to add depth_ffe_* columns 
depth_ffe_cols = ['depth_ffe_' + x for x in RET_PERS]
sub_cols = sub_cols + depth_ffe_cols

# Load the scenario data
# There are only a few columns we need
# Store them in a dict for each ensemble dataset
ens_dfs = {}
for scen in scenarios:
    ens_filep = join(FO, 'ensemble_' + scen + '.pqt')
    ens_df = pd.read_parquet(ens_filep, columns=sub_cols)
    print('Load data: ' + scen)
    ens_dfs[scen] = ens_df

# We'll need DDFs for estimating benefits
# Load DDFs
naccs_ddfs = pd.read_parquet(join(VULN_DIR_I, 'physical', 'naccs_ddfs.pqt'))
hazus_ddfs = pd.read_parquet(join(VULN_DIR_I, 'physical', 'hazus_ddfs.pqt'))

# Load helper dictionaries
with open(join(VULN_DIR_I, 'physical', 'hazus.json'), 'r') as fp:
    HAZUS_MAX_DICT = json.load(fp)

with open(join(VULN_DIR_I, 'physical', 'naccs.json'), 'r') as fp:
    NACCS_MAX_DICT = json.load(fp)

# Load and prepare discount rates and house lifetime

# Download discount rate chains from external source
# The rows correspond to house lifetime, indexed at 0
# The columns correspond to states of the world, indexed at 0
dr_chains = pd.read_csv(join(FE, 'dr_chains.csv'),
                        header=None)

# Following https://www.journals.uchicago.edu/doi/10.1086/718145
# and https://doi.org/10.1162/rest_a_01109, replace values < 0 with 0
# The economic argument is that descriptive discount rates will not
# be less than 0 for long. Bauer and Rudebusch (the latter link)
# have ~ 3 paragraphs addressing the mechanisms behind this
# which I have paraphrased badly here. Some of the intuition is that
# when nominal rates are low and inflation is high, households
# can hold cash and reduce spending, bringing inflation down
# and real rates back up. They offer a more complex, comprehsneive,
# and convincing argument. I have had some conversations about
# scrutinizing this assumption in a future paper. 
dr_chains[dr_chains < 0] = 0

# Need to turn these into discount factors, following
# Maggie's code
# We need the rates as percentages, then we take the cumulative
# sum of these such that the discount factor in year t
# is the sum of all rates leading to that
# Then we take e^- of that value
dr_factors = np.exp(-(dr_chains/100).cumsum())

# Generate house lifetime draws from the weibull distribution
# following https://www.nature.com/articles/s41467-020-19188-9
# Weibull with shape and scale parameters of 2.8 and 73.5
# In numpy, you generate draws from a 1 parameter Weibull
# using the shape parameter, and multiply these draws from
# the scale parameter
# It's likely that house lifetime distributions are different for
# elevated and non-elevated properties exposed to flooding, but
# we don't have this information. While the method can be improved,
# Maggie's paper demonstrates the importance of accounting for
# house lifetime uncertainty in estimating project benefits. In
# particular, when discount rates in the future are low and project
# lifetimes are long, the net benefits will be higher than
# under the standard procedure (moderate discount rate and
# 30 year lifetime). The standard procedure -- which also requires
# that the project BCR > 1 -- will tend to under-prefer
# investment in lower valued structures which need a longer time
# to exceed the BCR > 1 threshold. So again, while future work
# should improve on this, accounting for house lifetime uncertainty
# matters when you're dealing with a BCR > 1 rule, as we are. 
# Since this is about the interaction with the discount rate, and
# we don't have the ability to associate the parameters with
# housing characteristics, I will use the same draws for each house
# and interpret this as uncertainty in the house lifetime parameter
# whose benchmark value is 30. 

rng = np.random.default_rng()
lifetime = rng.weibull(W_SHAPE, N_SOW)*W_SCALE

# From the config file take the inflation values, heightening values, 
# and fixed cost values. For heightening, we need to linearly interpolate
# for our structure specific heightening cost estimates. For the others, 
# we need to generate N_SOW length realizations. We can pre-populate
# a cost dataframe with this information and for each 
# foundation type, heightening combo, we will have
# the SOW specific cost estimate to apply to the structure. 
# The costs are applied against the expected annual losses
# to figure out the optimal heightening. We do not need to include
# discount rates at this step since these are uniformly applied
# in our case study since all the elevations are assumed to occur
# at the same time. Time-based elevations that account for changing
# cost estimates and discount rates is an extension of this work. 

# Do the interpolation on the elev costs
# To get basement/bldgtype as multiindex from our dict
elev_cost_df = pd.DataFrame.from_dict(ELEV_COST_DICT).stack().to_frame()
# To break out the lists into columns
# Column names as the foot value (as int) 
e_c_df = pd.DataFrame(elev_cost_df[0].values.tolist(),
                      index=elev_cost_df.index,
                      columns=[2, 4, 8]).reset_index()

# Melt and rename to get ready for linear interpolation between feet
e_c_df = e_c_df.melt(id_vars=['level_0', 'level_1'], value_vars=[2, 4, 8])
e_c_df.columns = ['fnd_type', 'bldgtype', 'elev_ht', 'cost_sqft']

# Loop through fnd_type, bldgtype groups
# Add missing foot values and interpolate using
# spline of order 1 to get values filled
# past the 8 foot value and up to 10
# Store each interpolated dataframe in a list and concat at the end
elev_dfs = []
for fnd_bldg, df_sub in e_c_df.groupby(['fnd_type', 'bldgtype']):
    # keep track of foundation type and bldgtype
    fnd = fnd_bldg[0]
    bld = fnd_bldg[1]

    # use elev ht as index and get a series of costs
    elevs = df_sub.set_index('elev_ht')['cost_sqft']
    # get the elevations from 2 to 3 feet that we are missing
    missing_elevs = [x for x in np.arange(2, 11) if x not in elevs.index]
    # combine elevs and missing elevs
    elevs_f = pd.concat([elevs, pd.DataFrame(index=pd.Index(missing_elevs))])
    # sort index and interpolate
    elevs_f = elevs_f.sort_index().interpolate('spline', order=1).round(1)

    # We consider elevation from 3 to 10 feet only
    elevs_f = elevs_f.loc[3:10]

    # Reset index and rename columns
    elevs_f = elevs_f.reset_index()
    elevs_f.columns = ['elev_ht', 'cost_sqft']
    # Add back fnd_type and bldgtype
    # using first character as capital letter
    elevs_f['fnd_type'] = fnd[0].upper()
    elevs_f['bldgtype'] = bld[0].upper()

    elev_dfs.append(elevs_f)
# Final cost per sqft dataframe
elev_costs = pd.concat(elev_dfs, axis=0).reset_index(drop=True)

# Sample N_SOW from uniform(CPI_LOW, CPI_HIGH)
# Sample N_SOW from uniform (ELEV_FIX_LOW, ELEV_FIX_HIGH)
rng = np.random.default_rng()
construction_infl = rng.uniform(CPI_LOW, CPI_HIGH, N_SOW)
fixed = rng.uniform(ELEV_FIX_LOW, ELEV_FIX_HIGH, N_SOW)

# Get the cost dataframe for each sow_ind 
# Columns are sow_ind, bldgtype, heightening, cost
# We need to multiply each cost_sqft value in elev_costs
# by each element of construction_infl and make sure this is
# indexed by the sow. Then, we need to add the fixed cost
# corresponding to that sow

# Repeat the elev_costs df so that each entry (ht, cst, types) has
# N_SOW rows
e_c_ens = elev_costs.loc[np.repeat(elev_costs.index, N_SOW)]
e_c_ens = e_c_ens.reset_index(drop=True)
# Then repeat the construction_infl and fixed series len(elev_costs)
# times. Do this via tiling (i.e. repeat the whole array not 
# the elements) 
c_infl_full = np.tile(construction_infl, len(elev_costs))
fixed_full = np.tile(fixed, len(elev_costs))

# Now create new column in e_c_ens for
# cost_sqft*c_infl_full and fixed_full
e_c_ens['cost_sqft_unc'] = e_c_ens['cost_sqft']*c_infl_full
e_c_ens['cost_fix_unc'] = fixed_full

# Then get the sow_ind for the e_c_ens dataframe
sow_ind = np.arange(len(e_c_ens))%N_SOW
e_c_ens = pd.concat([e_c_ens, pd.Series(sow_ind, name='sow_ind')], axis=1)

# Write out the elevation cost ensemble
elev_ens_filep = join(EXP_DIR_I, FIPS, 'elev_ens.pqt')
prepare_saving(elev_ens_filep)
e_c_ens.to_parquet(elev_ens_filep)

# We will also do the cost_sqft_unc*sqft + cost_fix_unc, indexed
# on sow_ind for each structure across SOWs, but will only write out
# the costs associated with the optimal elevation
# Since the elevation cost ensemble is written out, it's always
# accessible to inspect elevation costs for any 
# eligible heightening for homes in/across SOWs

# Loop through possible heightenings of 3 through 10 feet, inclusive
# For each of these, add that value to each column in depth_ffe_*
# Then, go through the procedures from benchmark_ensemble
# to calculate losses per return period and ultimately the eal
# Then, we compare this eal to the non-elevated eal, which is 
# stored in the reference "eal_col"
# We then take the subset of 
# fd_id, sow_ind, fnd_type, bldgtype, sqft, reduced_eal
# and merge it with e_c_ens. (sow_ind, fnd_type, bldgtype)
# Take reduced_eal/(sqft*cost_sqft_unc + cost_fix_unc) and store
# it as bcr. Groupby on fd_id and calculate the mean bcr.
# Store this as a series with name corresponding to the amount
# of heightening and index corresponding to fd_id,
# and store that series in a list
# I think we also want to store the reduced_eal and
# the costs for each sow. 
# After looping through all of these, we can concat our list
# into a dataframe and figure out which column corresponds
# to the highest bcr for each fd_id. I think we can do this
# using df.idxmax(axis="columns") if we concat on columns
# Finally, we use the corresponding
# heightening value to match up each fd_id to its
# avoided loss, heightening, and expected costs. The BCR needs to 
# be adjusted in the allocate funding procedure later on by different
# discount rate projections. I think we need to discount the 
# reduced_eal in each SOW AND divide each of those by costs again
# to get the correct expected BCR. But you don't need to do 
# discounting to find the optimal elevation. 

for scen, ens_df in ens_dfs.items():
    print('Scenario: ' + scen)
    # Get fnd_type variable for ens_df that corresponds to B and S
    # where crawl space (C) from found_type gets classified as B
    # This is needed for a future step
    ens_df['fnd_type'] = np.where(ens_df['found_type'] == 'S',
                                  'S', 
                                  'B')

    # We're going to make a lifetime_mask and dr_matrix 
    # to calculate present values of potential heightenings - the avoided
    # losses as well as the residual risk
    # Prepare lifetime mask and matrix for discount factors
    
    # discount factor matrix
    dr_matrix = np.tile(dr_factors, (1, len(ens_df['fd_id'].unique())))
    
    # Use the lifetime series to create a mask
    # Can adapt this code
    # https://stackoverflow.com/questions/55190295/
    # create-a-2-d-mask-from-a-1-d-numpy-array
    # This code is complex, so I want to explain what is happening. You
    # can also look at the stackoverflow link which provides helpful
    # information. 
    # So, let's start from the inside out. The first command 
    # is np.less.outer(lifetime, np.arange(100))
    # This takes the lifetime array, which is N_SOW in length
    # and broadcasts that with outer into a N_SOW*100 shape 2d array
    # 100 is the max lifetime we consider since discount rates are
    # projected through 2100. We're comparing the values in lifetime
    # to the values in the np.arange(100) array, and when the lifetime
    # value is less, the element is assigned True. This creates
    # a mask of True/False values which we need to match up to
    # our 100x2390000 matrix of eal_avoid values. We do this first by
    # transposing, then tiling along the columns the same number
    # of times as we have structures in our sample. 
    lifetime_mask = np.tile(np.less.outer(lifetime, np.arange(100)).T,
                            (1, len(ens_df['fd_id'].unique())))

    # Write out the lifetime mask
    lifetime_filename = 'lifetime_mask' + '_' + scen + '.npy'
    lifetime_filep = join(EXP_DIR_I, FIPS, lifetime_filename)
    with open(lifetime_filep, 'wb') as f:
        np.save(f, lifetime_mask)

    # Also loop through ddf types
    for ddf_type in ddf_types:
        print('DDF: ' + ddf_type)
        eal_col = 'naccs_eal' if ddf_type == 'naccs' else 'haz_eal'
        # List for series of mean bcr at each heightening
        h_list = []
        
        for h in np.arange(3, 11):
            # Adjust depth_ffe_* columns by h
            # We substract h because these are 
            # depths relative to first floor
            # and now the first floor is higher
            depth_ffe_df = ens_df.loc[:,depth_ffe_cols] - h
            # Remove 'depth_ffe_' part from the column
            depth_ffe_df.columns = [x.split('_')[-1]
                                    for x in depth_ffe_df.columns]
            
            # We will store losses in dictionaries
            # with return period keys
            elev_losses = {}

            # The reason I'm looping through ddf_type is because
            # I think this way accommodates more flexibility in the
            # future if there are other
            # damage functions employed
            for rp in RET_PERS:
                if ddf_type == 'naccs':
                    elev_losses[rp] = est_naccs_loss(ens_df['bld_types'],
                                                     depth_ffe_df[rp],
                                                     naccs_ddfs,
                                                     NACCS_MAX_DICT)
                else:
                    elev_losses[rp] = est_hazus_loss(ens_df['hazus_types'],
                                                     depth_ffe_df[rp],
                                                     hazus_ddfs,
                                                     HAZUS_MAX_DICT)
            
                print('Estimate Losses for Elevated Home, RP: ' + rp)
            
            # Then, we convert these to dataframes
            loss_df = pd.DataFrame.from_dict(elev_losses)
            
            # For each relative damage column, 
            # scale by val_s
            # loss_df and ens_df are index aligned, so this works
            for col in loss_df.columns:
                loss_df['loss_' + col] = loss_df[col]*ens_df['val_s']
            
            # We make a list of our loss columns
            loss_list = ['loss_' + x for x in RET_PERS]
            # As well as the corresponding probabilities
            p_rp_list = [round(1/int(x), 4) for x in RET_PERS]
            
            # Then we create an empty series
            eal_elev = pd.Series(index=loss_df.index).fillna(0)
            
            # We loop through our loss list and apply the 
            # trapezoidal approximation
            for i in range(len(loss_list) - 1):
                loss1 = loss_df[loss_list[i]]
                loss2 = loss_df[loss_list[i+1]]
                rp1 = p_rp_list[i]
                rp2 = p_rp_list[i+1]
                # We add each approximation
                eal_elev += (loss1 + loss2)*(rp1-rp2)/2
            # This is the final trapezoid to add in
            final_eal = eal_elev + loss_df[loss_list[-1]]*p_rp_list[-1]
            print('Calculated EAL')
        
            # Calculate avoided losses and add to ens_df
            # Cannot be less than 0
            eal_avoid_temp = ens_df[eal_col] - final_eal
            eal_avoid_temp[eal_avoid_temp < 0] = 0
            ens_df['eal_avoid_' + str(h)] = eal_avoid_temp
            
            # Present value - avoided losses
            eal_avoid = np.tile(ens_df['eal_avoid_' + str(h)], (100, 1))
            # Apply the lifetime_mask to eal_avoid
            eal_av_life = ma.masked_array(eal_avoid,
                                          mask=lifetime_mask,
                                          fill_value=0)
            # present value 
            pv_avoided = (eal_av_life*dr_matrix).sum(axis=0)
            # Add back into ens_df
            ens_df['pv_avoid_' + str(h)] = pv_avoided.data
            # Also get the relative avoided
            ens_df['avoid_rel_eal_' + str(h)] = (ens_df['eal_avoid_' + str(h)]
                                                 /ens_df['val_s'])
            
            # Merge e_c_ens on subset of ens_df columns to figure out
            # the elevation cost and get this into ens_df
            ens_sub = ens_df[['fd_id', 'sow_ind', 'fnd_type',
                              'bldgtype', 'sqft']].copy()
            
            # Also subset e_c_ens for the correct heightening
            # Don't need cost_sqft for this either
            e_c_ens_sub = e_c_ens[e_c_ens['elev_ht'] == h].drop(columns=['elev_ht',
                                                                         'cost_sqft'])
            
            # Merge on sow, fnd_type, bldgtype
            e_c_merge = ens_sub.merge(e_c_ens_sub,
                                      on=['fnd_type', 'bldgtype', 'sow_ind'])
            
            # Get upfront costs
            invsts = (e_c_merge['sqft']*e_c_merge['cost_sqft_unc']
                     + e_c_merge['cost_fix_unc'])
            ens_df['elev_invst_' + str(h)] = invsts
        
            # Present value - residual risk (our final_eal column)
            eal_resid = np.tile(final_eal, (100, 1))
            # Apply the lifetime_mask to eal_avoid
            eal_r_life = ma.masked_array(eal_resid,
                                         mask=lifetime_mask,
                                         fill_value=0)
            # present value 
            pv_resid = (eal_r_life*dr_matrix).sum(axis=0)
            # Add back into ens_df
            ens_df['pv_resid_' + str(h)] = pv_resid.data
            # Also get the relative resid
            ens_df['resid_rel_eal_' + str(h)] = final_eal/ens_df['val_s']
        
            # Get costs
            # Add present value of residual risk to upfront cost
            ens_df['elev_cost_' + str(h)] = (ens_df['pv_resid_' + str(h)]
                                             + ens_df['elev_invst_' + str(h)])
            
            # Now we have the avoided loss and elev cost for this level of
            # heightening stored in ens_df
            # It also helps to do some side calculations to save some time
            # later in obtaining the optimal level of heightening
            # Get the ratio of eal_avoid_str(h) to elev_cost_str(h)
            # Groupby on fd_id and take the mean
            # Store this as a series with name corresponding to the amount
            # of heightening and index corresponding to fd_id,
            # and store that series in a list
            ens_df['npv_' + str(h)] = (ens_df['pv_avoid_' + str(h)]
                                       - ens_df['elev_cost_' + str(h)])
            
            npvs = ens_df.groupby(['fd_id'])['npv_' + str(h)].mean()
            h_list.append(npvs)
        
            print('Calculations done for heightening by ' + str(h) + ' feet\n')
        
        # Get the dataframe of mean bcr across SOWs for each structure
        # for each heightening. Find the heightening for each structure
        # that leads to the max mean bcr, and write this out to a file. 
        # When we do the full bcr estimation later, we will loop through
        # each value in this series, subset the ens_df based on
        # fd_id with that bcr_part_h as their max mean bcr, and then
        # do the full discounting and recalculation of BCR. You need
        # to use the discount rate chain that corresponds to each SOW
        # to discount the avoided losses correctly
        # and you need to divide that by the costs in that SOW
        # THEN you can calculate our objectives like net benefits and
        # check conditions like BCR > 1. 
        opt_elev = pd.concat(h_list, axis=1)
        opt_elev['opt_elev'] = opt_elev.idxmax('columns')
        npv_out_filename = 'opt_height_' + ddf_type + '_' + scen + '.pqt'
        npv_out_filep = join(EXP_DIR_I, FIPS, npv_out_filename)
        prepare_saving(npv_out_filep)
        opt_elev.to_parquet(npv_out_filep) 

        # Write out ens_df columns related to optimal elevation
        # eal_avoid_h and elev_cost_h
        # Only need to do this for the heightening that corresponds
        # to the optimal level
        # Subset ens_df based on the information in opt_elev
        # Do this by looping through the values in opt_elev, getting the list
        # of fd_id that correspond to this, and then storing the ens_df
        # rows & columns (eal_avoid_h and elev_cost_h) in a list along
        # with the heightening amount
        # You will end up concatenating a dataframe that is
        # sow_ind, fd_id, eal_avoid_opt, elev_cost_opt, elev_h
        
        elev_df_l = []
        for elev_h in opt_elev['opt_elev'].unique():
            # Subset of fd_id that have this optimal heightening
            struct_sub = opt_elev[opt_elev['opt_elev'] == elev_h].index
            # elev value
            h = elev_h.split('_')[-1]
            # Corresponding columns
            ens_col_sub = ['pv_avoid_' + str(h), 'elev_cost_' + str(h),
                           'elev_invst_' + str(h), 'pv_resid_' + str(h),
                           'fd_id', 'sow_ind', 
                           'avoid_rel_eal_' + str(h), 'resid_rel_eal_' + str(h),
                            eal_col]
            # Corresponding rows and columns
            ens_sub = ens_df.loc[ens_df['fd_id'].isin(struct_sub),
                                 ens_col_sub]
            # Rename columns
            ens_sub.columns = ['pv_avoid', 'pv_cost', 
                               'elev_invst', 'pv_resid',
                               'fd_id', 'sow_ind', 
                               'avoid_rel_eal', 'resid_rel_eal', 'base_eal']
            # Add the heightening amount back in
            ens_sub['opt_elev'] = h
            
            elev_df_l.append(ens_sub)
            print('Processed rows with optimal elevation height of ' + str(h))
        
        elev_df_f = pd.concat(elev_df_l, axis=0).sort_index()
        
        # Should also calculate present value of the 'base' eal
        # and write out the lifetime data that was generated
        eal_base = np.tile(elev_df_f['base_eal'], (100, 1))
        # Apply the lifetime_mask to eal_avoid
        eal_life = ma.masked_array(eal_base,
                                   mask=lifetime_mask,
                                   fill_value=0)
        # present value 
        pv_base = (eal_life*dr_matrix).sum(axis=0)
        # Add back into ens_df
        elev_df_f['pv_base'] = pv_base.data

        # Write file in FIPS specific exp/ directory
        opt_elev_filename = 'ens_opt_elev_' + ddf_type + '_' + scen + '.pqt'
        opt_elev_filep = join(EXP_DIR_I, FIPS, opt_elev_filename)
        elev_df_f.to_parquet(opt_elev_filep)
        print('Wrote file: ' + opt_elev_filename + '\n')