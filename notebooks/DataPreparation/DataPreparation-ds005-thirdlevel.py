
# coding: utf-8

# This notebook generates the processed data for dataset ds005, which is used in some of the chapter-specific notebooks from Poldrack, Mumford, and Nichols' _Handbook of fMRI Data Analysis (2nd Edition)_.  This also provides an example of using the nipype workflow mechanism.

# In[1]:

import os, errno, sys,shutil
import json


from fmrihandbook.utils.config import Config

config=Config()

run_mixedfx=1


from nipype.interfaces import fsl, nipy, ants
import nibabel
import numpy
import nilearn.plotting
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
import fmrihandbook.utils
from fmrihandbook.utils.compute_fd_dvars import compute_fd,compute_dvars
import pickle
from fmrihandbook.utils.get_data import get_data

import nipype.interfaces.io as nio           # Data i/o
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.algorithms.modelgen as model   # model specification
from nipype.interfaces.base import Bunch
import glob
import nipype.interfaces.utility as niu
from nipype.interfaces.c3 import C3dAffineTool
from nipype.interfaces.utility import Merge, IdentityInterface


rerun_analyses=False  # set to true to force rerun of everything


# In[2]:

config.data=get_data('ds005')
print(config.data)

workdir=os.path.join(config.data['ds005']['datadir'],'nipype_workdir')

copeinfo = pe.Node(interface=niu.IdentityInterface(fields=['copenum']), name="copeinfo")

copeinfo.iterables = ('copenum',[1,2,3,4])

regtypes = pe.Node(interface=niu.IdentityInterface(fields=['regtype']), name="regtypes")

regtypes.iterables = ('regtype',['ants','affine'])

pickfirst = lambda x: x[0]

datasink = pe.Node(nio.DataSink(), name='datasink')

# Save the relevant data into an output directory
datasink.inputs.base_directory = os.path.join(config.data['ds005']['datadir'],'derivatives')

# ### Third-level model: across subjects

# In[ ]:

mixed_fx = pe.Workflow(name='mixedfx')
mixed_fx.base_dir = workdir

smoothlevels = pe.Node(interface=niu.IdentityInterface(fields=['fwhm']), name="smoothlevels")

smoothlevels.iterables = ('fwhm',[0,4,8,16,32])



datasource_cope = pe.Node(interface=nio.DataGrabber(infields=['copenum','regtype'],
                    outfields=['copes']),
                    name = 'datasource_cope')

datasource_cope.inputs.base_directory = config.data['ds005']['datadir']
#fixedfx/cope/_copenum_1/_regtype_affine/_subject_id_sub-03/_flameo0/cope1.nii.gz
datasource_cope.inputs.template = 'derivatives/fixedfx/cope/_copenum_%d/_regtype_%s/_subject_id_*/_flameo0/cope1.nii.gz'
datasource_cope.inputs.template_args = dict(copes=[['copenum','regtype']])
datasource_cope.inputs.sort_filelist = True

datasource_varcope = pe.Node(interface=nio.DataGrabber(infields=['copenum','regtype'],
                    outfields=['varcopes']),
                    name = 'datasource_varcope')

datasource_varcope.inputs.base_directory = config.data['ds005']['datadir']
datasource_varcope.inputs.template = 'derivatives/fixedfx/varcope/_copenum_%d/_regtype_%s/_subject_id_*/_flameo0/varcope1.nii.gz'
datasource_varcope.inputs.template_args = dict(varcopes=[['copenum','regtype']])
datasource_varcope.inputs.sort_filelist = True


copemerge    = pe.MapNode(interface=fsl.Merge(dimension='t'),
                          iterfield=['in_files'],
                          name="copemerge")

varcopemerge = pe.MapNode(interface=fsl.Merge(dimension='t'),
                       iterfield=['in_files'],
                       name="varcopemerge")

mixed_fx.connect(copeinfo,'copenum',datasource_cope,'copenum')
mixed_fx.connect(copeinfo,'copenum',datasource_varcope,'copenum')
mixed_fx.connect(regtypes,'regtype',datasource_cope,'regtype')
mixed_fx.connect(regtypes,'regtype',datasource_varcope,'regtype')



mixed_fx.connect(datasource_cope,'copes',copemerge,'in_files')
mixed_fx.connect(datasource_varcope,'varcopes',varcopemerge,'in_files')

level3model = pe.Node(interface=fsl.L2Model(),
                      name='l3model')
level3model.inputs.num_copes=16

smooth_cope=pe.Node(interface=fsl.utils.Smooth(), name="smooth_cope",iterfield=['copenum','fwhm','regtype'])
smooth_varcope=pe.Node(interface=fsl.utils.Smooth(), name="smooth_varcope",iterfield=['copenum','fwhm','regtype'])


mixed_fx.connect(copemerge,('merged_file',pickfirst),smooth_cope,'in_file')
mixed_fx.connect(varcopemerge,('merged_file',pickfirst),smooth_varcope,'in_file')
mixed_fx.connect(smoothlevels,'fwhm',smooth_cope,'fwhm')
mixed_fx.connect(smoothlevels,'fwhm',smooth_varcope,'fwhm')


flame1 = pe.MapNode(interface=fsl.FLAMEO(run_mode='flame1'), name="flame1",
                    iterfield=['cope_file','var_cope_file'])
flame1.inputs.mask_file=os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
#fixed_fx.connect(datasource_mask,'mask',flameo,'mask_file')




mixed_fx.connect(smooth_cope,'smoothed_file',flame1,'cope_file')
mixed_fx.connect(smooth_varcope,'smoothed_file',flame1,'var_cope_file')
mixed_fx.connect(smooth_cope,'smoothed_file',datasink,'smoothed_cope')


mixed_fx.connect(level3model,'design_mat',flame1,'design_file')
mixed_fx.connect(level3model,'design_con',flame1,'t_con_file')
mixed_fx.connect(level3model,'design_grp',flame1,'cov_split_file')


#mixed_fx.connect(flame1,'stats_dir',datasink,'stats')

# compute smoothness of residuals

smoothest=pe.Node(fsl.SmoothEstimate(),name='smoothest')
smoothest.inputs.dof=15
smoothest.inputs.mask_file=os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain_mask.nii.gz')

def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
        return files


mixed_fx.connect(flame1,('res4d',pickfirst),smoothest,'residual_fit_file')

# clustering with gaussian random field theory

cluster=pe.Node(fsl.Cluster(),name='cluster')
cluster.inputs.threshold = 3.0
cluster.inputs.pthreshold=0.05
cluster.inputs.out_threshold_file='thresh_zstat1.nii.gz'
cluster.inputs.out_localmax_txt_file='localmax.txt'

mixed_fx.connect(flame1,('zstats',pickfirst),cluster,'in_file')
mixed_fx.connect(smoothest,'dlh',cluster,'dlh')
mixed_fx.connect(smoothest,'volume',cluster,'volume')

mixed_fx.connect(flame1,('zstats',pickfirst),datasink,'mixedfx.zstat')

mixed_fx.connect(cluster,'threshold_file',datasink,'cluster.thresh_zstat')
mixed_fx.connect(cluster,'localmax_txt_file',datasink,'cluster.localmax')



if run_mixedfx:
    mixed_fx.run(plugin='MultiProc', plugin_args={'n_procs' : 16})


# ### Nonparametric correction using randomise
# 
# Run this outside of nipype because interface is currenly not working properly

# In[ ]:

randomise_output_dir=os.path.join(config.data['ds005']['datadir'],'derivatives/randomise')
if not os.path.exists(randomise_output_dir):
    os.mkdir(randomise_output_dir)

f=open('run_randomise.sh','w')

for cope in range(1,5):
    for fwhm in [0,4,8,16,32]:
        for regtype in ['affine','ants']:
            copefile=os.path.join(config.data['ds005']['datadir'],
                    'derivatives/smoothed_cope/_copenum_%d/_regtype_%s/_fwhm_%d/cope1_merged_smooth.nii.gz'%(cope,regtype,fwhm))
            print(copefile)
            assert os.path.exists(copefile)
            outdir=os.path.join(randomise_output_dir,'_copenum_%d/_regtype_%s/_fwhm_%d'%(cope,regtype,fwhm))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            nperms=2500
            cmd='randomise -i %s -o "%s/randomise_" -c 3.00 -C 3.00 -n %d -1 -T -v 10'%(copefile,outdir,nperms)
            print(cmd)
            f.write(cmd+'\n')
f.close()


