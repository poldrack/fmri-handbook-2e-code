"""
config setup for fmri-handbook
"""

import os,pickle
from fmrihandbook.utils.get_data import get_data
import distutils.spawn

class Config:
    datadir=os.getenv('FMRIDATADIR')
    if not datadir:
        raise Exception('You need to set the environment variable FMRIDATADIR')

    if not os.path.exists(datadir):
        raise Exception('FMRIDATADIR must exist: %s'%datadir)
    else:
        print('using base dir:',datadir)

    figuredir=os.getenv('FMRIFIGUREDIR')
    if figuredir is None:
        figuredir=os.path.join(os.path.dirname(datadir),'figures')

    if not os.path.exists(figuredir):
        print('figure dir %s does not exist - creating it'%figuredir)
        os.mkdir(figuredir)

    datadict=os.path.join(datadir,'data_dictionary.pkl')
    try:
        data=pickle.load(open(datadict,'rb'))
    except:
        print('no data dictionary found, checking data')
        data=get_data()

    orig_figuredir='https://web.stanford.edu/group/poldracklab/fmri-handbook-2e-data/figures-1e/'

    fsldir=os.getenv('FSLDIR')
    if not fsldir:
        raise Exception('You need to set the environment variable FSLDIR')

    if not distutils.spawn.find_executable('matlab'):
        spmdir=None
    else:
        spmdir=os.getenv('SPMDIR')
        if not spmdir:
            spmdir=None
            print('environment variable SPMDIR is not set - SPM functions will not work')
            #raise Exception('You need to set the environment variable SPMDIR')

    def __init__(self):
        self.set_email(os.getenv('ENTREZEMAIL'))
        self.set_img_format('pdf')

    def set_email(self,email):
        self.email=email
    def set_img_format(self,img_format):
        self.img_format=img_format
