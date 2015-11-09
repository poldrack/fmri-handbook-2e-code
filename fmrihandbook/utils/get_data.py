"""
load data for fmri-handbook-2e notebooks
"""

import os
from fmrihandbook.utils.download_file import DownloadFile

datadir=os.getenv('FMRIDATADIR')
assert os.path.exists(datadir)

urlbase='https://s3.amazonaws.com/openfmri/'

def get_data():
    data={}
    # required starting files
    base_files={'T1':'ds031/sub-01/ses-018/anat/sub-01_ses-018_run-001_T1w.nii.gz',
        'T2':'ds031/sub-01/ses-018/anat/sub-01_ses-018_run-001_T2w.nii.gz',
        'func':'ds031/sub-01/ses-014/func/sub-01_ses-014_task-nback_run-001_bold.nii.gz',
        'sbref':'ds031/sub-01/ses-014/func/sub-01_ses-014_task-nback_run-001_sbref.nii.gz',
        'fieldmap-mag':'ds031/sub-01/ses-014/fmap/sub-01_ses-014_magnitude1.nii.gz',
        'fieldmap-phasediff':'ds031/sub-01/ses-014/fmap/sub-01_ses-014_phasediff.nii.gz'}

    missing_base_files=[]

    for k in base_files.keys():
        f=os.path.join(datadir,base_files[k])
        if os.path.exists(f):
            data[k]=f
        else:
            print('getting %s'%base_files[k])
            DownloadFile(os.path.join(urlbase,base_files[k]),
	        os.path.join(datadir,base_files[k]))
            missing_base_files.append(base_files[k])


    return data
