"""
load data for fmri-handbook-2e notebooks
"""

import os

datadir=os.getenv('FMRIBOOKDATA')
assert os.path.exists(datadir)

def get_data():
    data={}
    # required starting files
    base_files={'T1':'sub018_t1.nii.gz',
        'T2':'sub018_t2.nii.gz',
        'func':'sub00001_ses014_task002_run001_bold.nii.gz',
        'meanfunc':'sub00001_ses014_task002_run001_bold_mean.nii.gz',
        'fieldmap-mag':'sub00001_ses014_001_magnitude.nii.gz',
        'fieldmap-phasediff':'sub00001_ses014_001_phasediff.nii.gz'}

    missing_base_files=[]

    for k in base_files.keys():
        f=os.path.join(datadir,base_files[k])
        if os.path.exists(f):
            data[k]=f
        else:
            missing_base_files.append(base_files[k])


    if len(missing_base_files)>0:
        print 'ERROR: missing base data files'
        print 'You need to download the base data'
        for m in missing_base_files:
            print m

    processed_files={'T1_bc':'sub018_t1_corrected.nii.gz',
        'T1_brain':'sub018_t1_corrected_brain.nii.gz',
        'T1_brainmask':'sub018_t1_corrected_brain_mask.nii.gz',
        'T1_wmseg':'sub018_t1_corrected_brain_wmseg.nii.gz',
        'fieldmap':'fieldmap.nii.gz',
        'motionpar':'motion.par',
        'func_ica':'sub00001_ses014_task002_run001_bold.ica',
        'T1_mni':'t1_to_mni_warp.nii.gz',
        'T1_mni_warp':'t1_to_mni_warp.h5',
        'T2_reg2T1':'t2_reg2t1.nii.gz',
        'func_reg_to_t1':'epi_to_t1_warped.nii.gz'}

    missing_processed_files=[]

    for k in processed_files.keys():
        f=os.path.join(datadir,processed_files[k])
        if os.path.exists(f):
            data[k]=f
        else:
            missing_processed_files.append(processed_files[k])


    if len(missing_processed_files)>0:
        print 'ERROR: missing processed data files'
        print 'You need to run the DataPreparation script first'
        for m in missing_processed_files:
            print m

    return data
