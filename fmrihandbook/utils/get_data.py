"""
load data for fmri-handbook-2e notebooks
"""

import os,pickle
from fmrihandbook.utils.download_file import DownloadFile
import tarfile

datadir=os.getenv('FMRIDATADIR')
if not os.path.exists(datadir):
    print('data dir %s does not exist - creating it'%datadir)
    os.mkdir(datadir)


urlbase='https://s3.amazonaws.com/openfmri/'
datadict=os.path.join(datadir,'data_dictionary.pkl')

def get_data(dataset=None,save_datadict=True):

    if  os.path.exists(datadict):
        data=pickle.load(open(datadict,'rb'))
    else:
        data={}

    if dataset=='ds031':
        if not 'ds031' in data:
            data['ds031']={}
            data['ds031']['datadir']=os.path.join(datadir,'ds031')
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
                data['ds031'][k]=f
            else:
                print('getting %s'%base_files[k])
                if not os.path.exists(os.path.join(datadir,base_files[k])):
                    DownloadFile(os.path.join(urlbase,base_files[k]),
    	               os.path.join(datadir,base_files[k]))
                data['ds031'][k]=f
    elif dataset=='ds005':
        if not os.path.exists(os.path.join(datadir,'ds005_R2.0.0')):
            if not os.path.exists(os.path.join(datadir,'ds005_BIDS.tgz')):
                print('downloading ds005 from AWS...')
                DownloadFile(os.path.join(urlbase,'tarballs/ds005_R2.0.0.tgz'),
                    os.path.join(datadir,'ds005_BIDS.tgz'))
            print('extracting ds005_BIDS.tgz')
            tf=tarfile.open(os.path.join(datadir,'ds005_BIDS.tgz'),'r:gz')
            tf.extractall(path=datadir)
        if not 'ds005' in data:
            data['ds005']={}
        data['ds005']['datadir']=os.path.join(datadir,'ds005_R2.0.0')

    if save_datadict:
        pickle.dump(data,open(datadict,'wb'))
    return data
