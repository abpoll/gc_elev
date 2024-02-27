import os
from pathlib import Path
from os.path import join
import pandas as pd
import numpy as np
import json

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

# Load the ensemble data, along with the optimal
# elevation results
sort_dfs = {}

for scen in scenarios:
    ens_filep = join(FO, 'ensemble_' + scen + '.pqt')
    ens_df = pd.read_parquet(ens_filep)
    print('Load data: ' + scen)
    
    for ddf_type in ddf_types:
        opt_elev_filename = 'ens_opt_elev_' + ddf_type + '_' + scen + '.pqt'
        opt_elev_df = pd.read_parquet(join(EXP_DIR_I, FIPS, opt_elev_filename))
        print('Load opt elev: ' + ddf_type)

        # Merge on fd_id and sow_ind to get eal_avoid, elev_cost, and opt_elev
        # into the ensemble
        ens_df = ens_df.merge(opt_elev_df,
                              on=['fd_id', 'sow_ind'],
                              suffixes=['','_opt'])

        # Add metrics for objectives that we don't have yet
        ens_df['rel_eal'] = ens_df['base_eal']/ens_df['val_s']
        ens_df['npv_opt'] = ens_df['pv_avoid'] - ens_df['pv_cost']

        # We need to group by on fd_id and aggregate on our sorting columns
        sub_cols = ['pv_resid', 'npv_opt', 'fd_id', 'elev_invst',
                    'avoid_rel_eal', 'rel_eal', 'val_s']
        sort_df = ens_df.groupby('fd_id')[sub_cols].mean()
        
        sort_dfs[scen + '_' + ddf_type] = sort_df

        print('Store df for sorting\n')

# We also need to load in the links between structures and the
# social vulnerability data for sorting rules
sovi_filepath = join(VULN_DIR_I, 'social', FIPS, 'c_indicators.pqt')
sovi_df = pd.read_parquet(sovi_filepath)

# Merge these in later after we've aggregated ens_df on the
# column we're sorting on for these (npv_opt). The way the
# sorting will work for these is subset to communities that
# meet the indicator and then spend our budget based on
# npv_opt. 

# Now that we have these values we can start sorting! 

# We will sort until we expend our budget. We get these values
# from the hma projects dataset for elevation projects
# We will sample by 500K until 6 million (roughly 95th%ile)
# then by 1M until 15 million (roughly 99th%ile)
budgets_typ = np.arange(1e6, 6.1e6, 5e5)
budgets_high = np.arange(7e6, 15.1e6, 1e6)
budgets = np.append(budgets_typ, budgets_high)

# To calculate objectives
# We need to evaluate the objective in each SOW
# and then get our across SOW value
# For mean(npv), we take the sum of npv_opt (from elevated homes) 
# in each SOW and then take the mean of that
# For mean(up_cost), we take the sum of elev_invst(from elevated homes)
# in each SOW and then take the mean of that
# For mean(pv_resid), we take the sum of pv_resid (for elevated)
# and pv_base (for not elevated) in each SOW and then take the mean
# For mean(resid_rel_eal), we take the mean of resid_rel_eal
# (for elevated) and rel_eal (for not elevated) in each SOW and
# then take the mean
# For mean(slope_resid_rel_eal), we find the slope of the 
# resid_rel_eal & rel_eal (based on elevated homes) in each SOW
# and tehn take the mean
# Household based sorting

# Dict of sort keys to fd_id values
sort_dict = {}
# This is for community sorting, more explanation later
slack_dict = {}

# We also want to write out the ordering and
# the allocations
elev_dict = {}

# We also want to store the objective evaluations for
# individual SOWs
# The key is the policy
# and the values are the evaluations of objectives
# across all SOWs
objs_all_sows = {}

# Columns we sort from top to bottom
h_sort_desc = ['npv_opt', 'avoid_rel_eal',
               'rel_eal']

# We loop through each scen_ddftype combo
# to execute the code below so that we have a potentially
# different set of elevated homes for each policy
# for each scenario
for scen, sort_df in sort_dfs.items():
    print('Scenario: ' + scen)
    # Loop through ascending columns and sort, store in dict
    # We want to sort on the col, and give ties to lower
    # valued structures
    for col in h_sort_desc:
        sort_dict[col] = sort_df.sort_values([col, 'val_s'],
                                              ascending=[False, True]).index
    
    # Loop through descending columns and sort, store in dict
    # for col in h_sort_asc:
    #     sort_dict[col] = sort_df.sort_values([col, 'val_s'],
    #                                           ascending=[True, True]).index
    
    # Community based sorting
    sort_c_df = sort_df.join(sovi_df, how='inner')
    
    # Columns for community sorting
    c_sort_cols = ['lmi', 'ovb', 'cejst']
    
    # Loop through these to subset, sort by npv_opt
    # and follow the code from above
    # Loop through ascending columns and sort, store in dict
    # We add in the remaining observations in case we have
    # budget left over
    # TODO if the sort_col is in c_sort_cols,
    # we need to add a step where we ensure
    # the majority of benefits come from
    # the sort_pri. We will need to loop separately
    # from the remainder of the sort_dict.items() (or put
    # a switch on the loop) to do the processing separately
    # and then calculate objectives all the same
    # The thing that changes is subsetting df based on budget. 
    # We need to add an if/else where if sort_col is in c_sort_cols
    # there is some cross checking. It will help to have
    # a separate dict that stores the ids of sort_pri and sort_slack
    # for each of the columns in c_sort_cols. 
    for col in c_sort_cols:
        sort_temp = sort_c_df[sort_c_df[col] == True]
        sort_pri = sort_temp.sort_values(['npv_opt', 'val_s'],
                                         ascending=[False, True]).index
        sort_temp2 = sort_c_df[sort_c_df[col] == False]
        sort_slack = sort_temp2.sort_values(['npv_opt', 'val_s'],
                                             ascending=[False, True]).index
        sort_dict[col] = sort_pri.join(sort_slack, how='outer')
        # If we have extra budget, we can use it for
        # homes outside the community of interest
        slack_dict[col] = sort_slack
    
    # Loop through budgets and the keys in sort_dict
    # Calculate the elev_inst cumulative sum and subset to
    # the value just under the budget
    # Then calculate all of the objective values
    # Store in a dict of
    # sort_key_budget keys to objectives values
    obj_dict = {}
    for budget in budgets:
        for sort_col, fd_id in sort_dict.items():
            # Key for obj dict
            obj_key = scen + '_' + sort_col + '_' + str(budget)
            # Sort our df according to the rule at hand
            sorted_df = sort_df.reindex(fd_id)
    
            # Calculate the cumulative sum of elev_inst
            sorted_df['policy_cost'] = sorted_df['elev_invst'].cumsum()
    
            # Subset df based on budget
            # But also with some additional rules for
            # community based sorting
            if sort_col in c_sort_cols:
                # First, we get our primary df and our slack df
                slack_ids = slack_dict[sort_col]
                pri_df = sorted_df[~sorted_df['fd_id'].isin(slack_ids)]
                slack_df = sorted_df[sorted_df['fd_id'].isin(slack_ids)]
                # We need to recalculate policy costs
                pri_df['policy_cost'] = pri_df['elev_invst'].cumsum()
                slack_df['policy_cost'] = slack_df['elev_invst'].cumsum()
                
                # Now subset based on our budget
                elevated_sub = pri_df[pri_df['policy_cost'] <= budget]
                # Then we check if we have any budget leftover
                slack = budget - elevated_sub['policy_cost'].max()
                # From our slack dataframe, we'll check which
                # rows meet this slack
                slack_elev = slack_df[slack_df['policy_cost'] <= slack]
                # And we also have to subset based on the majority
                # of npv coming from our elevated_sub df
                slack_ben_max = elevated_sub['npv_opt'].sum()/2 
                slack_elev['npv_check'] = slack_elev['npv_opt'].cumsum()
                slack_elev_sub = slack_elev[(slack_elev['npv_check']
                                             <= slack_ben_max)]
                slack_elev_sub = slack_elev_sub.drop(columns='npv_check')
                
                # Now concat
                elevated = pd.concat([elevated_sub, slack_elev_sub], axis=0)
                
            # If not community sorting, you just go through the sorted
            # dataframe and subset subject to your budget
            else:
                elevated = sorted_df[sorted_df['policy_cost'] <= budget]
    
            # Calculated objectives
            # We do this by finding the subset in ens_df that are elevated
            # and the subset that are not. That's where we calculate
            # our objectives for within each SOW, and then we take
            # the expected values across the SOWs
            elev_ens = ens_df[ens_df['fd_id'].isin(elevated['fd_id'])]
            orig_ens = ens_df[~ens_df['fd_id'].isin(elevated['fd_id'])]
    
            # For npv, we calculate the sum of npv of elevated homes
            npvs = elev_ens.groupby('sow_ind')['npv_opt'].sum()
            # Our objective value for this policy is the mean of that
            npv = np.mean(npvs)
            
            # For up_cost, we calculate the sum of elev_invst 
            up_costs = elev_ens.groupby('sow_ind')['elev_invst'].sum()
            # Then we take the mean
            up_cost = np.mean(up_costs)
    
            # Get the pv resid based on the whole set
            # of homes with risk (there are benefits out of scope
            # of our npv calculation which could be associated
            # with lowering pv of residual risk, so we want
            # policies that balance the npv of elevation while
            # also not leaving more residual risk than needed)
            
            # We calculate the sum of pv_resid for elev_ens
            # And we calculate the sum of pv_base for orig_ens
            # Since these are indexed on sow_ind, we can add the
            # two series. That's our resids, then we take the
            # mean for our resid objective value for this policy
            resid_elev = elev_ens.groupby('sow_ind')['pv_resid'].sum()
            resid_orig = orig_ens.groupby('sow_ind')['pv_base'].sum()
            resids = resid_elev + resid_orig
            resid = np.mean(resids)
    
            # Slope between residual relative risk and structure
            # value. We want to do this for all of the houses
            # So, we need to go back to sort_df. For homes in sort_df
            # that are in elevated, we want to use their
            # reid_rel_eal for "y". For homes that are not elevated, 
            # we want to use rel_eal for "y".
            # We want to evaluate this for each SOW
            # and then take the average of the vaues
            x = np.log(ens_df['val_s'])
            y = np.where(ens_df['fd_id'].isin(elevated['fd_id']),
                         ens_df['resid_rel_eal'],
                         ens_df['rel_eal'])
            # So we need to take a copy of fd_id & sow_ind from ens_df
            ens_resid = ens_df[['fd_id', 'sow_ind']].copy()
            # We need to link our x/y for the policy
            ens_resid['resid_rel_eal'] = y
            ens_resid['val_s'] = x
            # We need to group on SOW index
            ens_r_gb = ens_resid.groupby('sow_ind')
            # Then we evaluate the correlation
            resid_eqs = ens_r_gb.apply(lambda x:
                                       x['val_s'].corr(x['resid_rel_eal']))
            # Take the absolute value
            resid_eqs = np.abs(resid_eqs)
            # Resid_eqs is across SOWs and that goes into the
            # objs_all_sows dict
            # Resid_eq is the mean of this, which is our objective value
            resid_eq = np.mean(resid_eqs)
    
            # The avoid_eq metric is the average residual eal
            # after elevation
            # We get this by grouping on sow_ind and calculating the
            # mean of resid_rel_eal (avoid_eqs) and then taking 
            # the mean of that. Like above, avoid_eqs goes
            # in the objs_all_sows dict
            avoid_eqs = ens_r_gb['resid_rel_eal'].mean()
            avoid_eq = np.mean(avoid_eqs)
    
            # Store objectives in dict
            obj_dict[obj_key] = (npv, resid, up_cost,
                                 avoid_eq,
                                 resid_eq)
    
            # Store objective evaluations for SOWs in the dict
            # First we can concatenate all of our series into
            # a dataframe, then have our policy key point to that
            objs_sows = pd.concat([npvs, resids, up_costs,
                                   avoid_eqs, resid_eqs], axis=1)
            objs_all_sows[obj_key] = objs_sows
            
    
            # Need to store the fd_id that end up in elevated in a dict
            elev_dict[obj_key] = elevated['fd_id'].astype(int).to_list()
    
            print('Calculate objective values for policy:\n'+
                  'Sort by ' + sort_col + '\nWith Budget of $M ' + str(budget))

# Get the dataframe of objectives
# Need to have scen_policy then split into scen & policy columns
objs = pd.DataFrame.from_dict(obj_dict).T.reset_index()
objs.columns =  ['scen_policy', 'npv', 'pv_resid', 'up_cost',
                 'avoid_eq', 'resid_eq']
objs['policy'] = objs['scen_policy'].str.split('_').str[2:].apply(lambda x: '_'.join(x))
objs['scen'] = objs['scen_policy'].str.split('_').str[:2].apply(lambda x: '_'.join(x))
objs['sort'] = objs['policy'].str.split('_').str[:-1].apply(lambda x: '_'.join(x))
objs['budget'] = objs['policy'].str.split('_').str[-1].astype(float).astype(int)

# Add a community vs. household indicator
objs.loc[objs['sort'].isin(c_sort_cols), 'res'] = 'community'
objs.loc[~objs['sort'].isin(c_sort_cols), 'res'] = 'household'

# We also want the dataframe of objective evaluations
# within each SOW
d = []
for k, v in objs_all_sows.items():
    # Get the sow_ind column back
    df = v.reset_index()
    df.columns = ['sow_ind', 'npv', 'pv_resid', 'up_cost',
                  'avoid_eq', 'resid_eq']
    # Add the policy as a column
    df['scen_policy'] = k
    df['policy'] = df['scen_policy'].str.split('_').str[2:].apply(lambda x: '_'.join(x))
    df['scen'] = df['scen_policy'].str.split('_').str[:2].apply(lambda x: '_'.join(x))
    df['sort'] = df['policy'].str.split('_').str[:-1].apply(lambda x: '_'.join(x))
    df['budget'] = df['policy'].str.split('_').str[-1].astype(float).astype(int)
    d.append(df)

objs_sows = pd.concat(d, axis=0)

# Write out the dataframe of objective values
# and the dictionary of policy to fd_ids that are
# elevated
obj_filep = join(FO, 'pol_obj_vals.pqt')
objs.to_parquet(obj_filep)

elev_ids_filep = join(FO, 'pol_elev_ids.json')
with open(elev_ids_filep, 'w') as fp:
    json.dump(elev_dict, fp)

# Write out the dataframe of objective evaluations
# for each SOW
objs_sows_filep = join(FO, 'objs_sows.pqt')
# Need to drop the bin column we made
objs_sows.to_parquet(objs_sows_filep)