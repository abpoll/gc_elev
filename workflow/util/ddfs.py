# Packages
import pandas

'''
Helpful functions for processing depth-damage functions
that are stored in different data structures
'''

# HAZUS and USACE DDFs can be
# formatted in tidy df format
# this way
def tidy_ddfs(raw_ddf, idvars):
    # For HAZUS we have occtype and fld_zone
    # For NACCS we have Occupancy and DamageCategory
    ddf_melt = raw_ddf.melt(id_vars=idvars,
                            var_name='depth_str',
                            value_name='pct_dam')
    # # Need to convert depth_ft into a number
    # Replace ft with empty character
    # If string ends with m, make negative number
    # Else, make positive number
    ddf_melt['depth_str'] = ddf_melt['depth_str'].str.replace('ft', '')
    negdepth = ddf_melt.loc[ddf_melt['depth_str'].str[-1] == 
                            'm']['depth_str'].str[:-1].astype(float)*-1
    posdepth = ddf_melt.loc[ddf_melt['depth_str'].str[-1] != 
                            'm']['depth_str'].astype(float)

    ddf_melt.loc[ddf_melt['depth_str'].str[-1] == 'm',
                'depth_ft'] = negdepth
    ddf_melt.loc[ddf_melt['depth_str'].str[-1] != 'm',
                'depth_ft'] = posdepth

    # Divide pct_dam by 100
    ddf_melt['rel_dam'] = ddf_melt['pct_dam']/100
    
    return ddf_melt

def ddf_max_depth_dict(tidy_ddf, dam_col):
    # We want all depths above max depths for the DDFs
    # to take the param values of the max depth DDF
    # First, we groupby bld type for naccs and get max depth for
    # each bld type
    max_ddf_depths = tidy_ddf.groupby(['ddf_id'])['depth_ft'].idxmax()
    # Locate these rows in the dataframe for the ddfs
    max_d_params = tidy_ddf.iloc[max_ddf_depths]
    # Can create a dict of bld_type to params
    # which will be called for any instance in loss estimation
    # where a depth value is not null, but the params value is
    # We will just use this dict to fill the param values
    # with those corresponding to the max depth for that same bld type
    DDF_DICT = dict(zip(max_d_params['ddf_id'], max_d_params[dam_col]))

    return DDF_DICT