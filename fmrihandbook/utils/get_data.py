"""
load data for fmri-handbook-2e notebooks
"""

import os

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
        'func_reg_to_t1':'epi_to_t1_warped.nii.gz',
        'meanfunc_unwarped':'meanfunc_unwarped.nii.gz'}

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
