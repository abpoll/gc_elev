'''
Constants that are widely used throughout the
framework
'''
import os
from os.path import join
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
import pandas as pd

# Absolute directory
# The root of the project directory 
# is obtained with the Path(os.path.realpath(__file__).parents[1]
# command. parents[1] would take us to workflow/
ABS_DIR = Path(os.path.realpath(__file__)).parents[2]

# From ABS_DIR, we can access our config.yaml file
# for the project
CONFIG_FILEP = join(ABS_DIR, 'config', 'config.yaml')

# Open the config file and load
with open(CONFIG_FILEP) as f:
    CONFIG = yaml.load(f, Loader=SafeLoader)

# Wildcards for urls
URL_WILDCARDS = CONFIG['url_wildcards']

# Get the file extensions for api endpoints
API_EXT = CONFIG['api_ext']

# Get the name of the external hazard directory
# Actually, there will be a dictionary
# of these which corresponds to different
# %iles of the waterlevel/precip distributions
# for each RP
HAZ_DIR_MID = CONFIG['haz_dir_mid']
HAZ_DIR_LOW = CONFIG['haz_dir_low']
HAZ_DIR_HIGH = CONFIG['haz_dir_high']
HAZ_DIR_SUB = CONFIG['haz_dir_sub']
HAZ_DIRS = [HAZ_DIR_LOW, HAZ_DIR_MID, HAZ_DIR_HIGH]

# Get the CRS constants
NSI_CRS = CONFIG['nsi_crs']
CLIP_CRS = CONFIG['clip_crs']

# Get hazard model variables
# Get Return Period list
RET_PERS = CONFIG['RPs']
HAZ_FILEP = CONFIG['haz_filename']
HAZ_CRS = CONFIG['haz_crs']

# Toggles for deleting zip directories
# that were downloaded after they
# have been processed
RM_NFHL = CONFIG['rm_nfhl']

# Dictionary of ref_names
REF_NAMES_DICT = CONFIG['ref_names']

# Dictionary of ref_id_names
REF_ID_NAMES_DICT = CONFIG['ref_id_names']

# Coefficient of variation
# for structure values
COEF_VARIATION = CONFIG['coef_var']

# First floor elevation dictionary
FFE_DICT = CONFIG['ffe_dict']

# MTR_TO_FT constant
MTR_TO_FT = 3.28084

# Elevation cost parameters
BLS_CPI = CONFIG['bls_cpi']
CB_CPI = CONFIG['cb_cpi']
ELEV_FIX_LOW = CONFIG['clara_elev_fixed']
ELEV_FIX_HIGH = CONFIG['usace_elev_fixed']
ELEV_COST_DICT = CONFIG['elev_cost_dict']

# Number of states of the world
N_SOW = CONFIG['sows']

# Get the files we need downloaded
# These are specified in the "download" key 
# in the config file
# We transpose because one of the utils
# needs to return a list of the output files
# TODO the logic here works for one county, but 
# will need rethinking for generalizability
# I think the way to do it would be to break it
# into DOWNLOAD_FIPS, DOWNLOAD_STATE, etc.
# and these are accessed differently
DOWNLOAD = pd.json_normalize(CONFIG['download'], sep='_').T