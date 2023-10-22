from zipfile import ZipFile
import zipfile_deflate64
import os
from pathlib import Path
from collections import Counter
from util.unzip import *

# Call zipped_downloads and unzipped_dirs
# from the util.unzip script
# This gives us a list
# of files to unzip, and the directories
# to unzip them to
to_unzip = zipped_downloads()
unzip_dirs = unzipped_dirs()

# We're going to loop through the files we need to unzip
# and extract them into the appropriate directories
# One thing that will help the directory structure stay organized
# is to keep track of what the destination parent directory is
# If the parent directory appears multiple times in unzip_dirs
# we should also use the name of the file (excluding extension)
# as a subdirectory
# We can use the Counter() class from collections for this...
count = Counter(unzip_dirs)
need_subdir = [k for k, v in count.items() if v > 1]

for i, filepath in enumerate(to_unzip):
    # Get path version of filepath tjat
    # we are unzipping
    path = Path(filepath)

    # If unzip_dirs[i] is in need_subdir
    # we are going to add a subdirectory
    # from str.split('/')[-1][:-4]
    # This gives us noaa from noaa.zip, for example

    # So first, we get our out_filedir
    out_filedir = unzip_dirs[i]
    # Then we add subdir if we need it
    if unzip_dirs[i] in need_subdir:
        subdir = filepath.split('/')[-1][:-4]
        out_filedir = join(out_filedir, subdir)
        
    # Extract to out_filedir
    with ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(out_filedir)
    
    #TODO helpful log message
    print('Unzipped: ' + str(path.name).split('.')[0])