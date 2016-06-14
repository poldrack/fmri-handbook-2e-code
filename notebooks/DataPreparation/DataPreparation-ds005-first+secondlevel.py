
# This code generates the processed data for dataset ds005,
# which is used in some of the chapter-specific notebooks from
# Poldrack, Mumford, and Nichols' Handbook of fMRI Data Analysis
# (2nd Edition).  This also provides an example of using the
# nipype workflow mechanism.

import os, errno, sys,shutil
import json

try:
    subject_id=sys.argv[1]
except:
    raise Exception('specify subject_id as an argument')

from fmrihandbook.utils.config import Config

verbose=True
rerun_analyses=False  # set to true to force rerun of everything

config=Config()

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

config.data=get_data('ds005')
print(config.data)

# Map field names to individual subject runs.
info = dict(func=[['subject_id','subject_id','runcode']],
            anat=[['subject_id', 'subject_id']])

infosource = pe.Node(interface=niu.IdentityInterface(fields=['subject_id']), name="infosource")

# this builds off of example at http://www.mit.edu/~satra/nipype-nightly/users/examples/fmri_ants_openfmri.html

infosource.iterables = ('subject_id', [subject_id])

runinfo = pe.Node(interface=niu.IdentityInterface(fields=['runcode']), name="runinfo")

runinfo.iterables = ('runcode',['1','2','3'])


datasource_anat = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                    outfields=['anat']),
                    name = 'datasource_anat')

datasource_anat.inputs.base_directory = config.data['ds005']['datadir']

datasource_anat.inputs.template = '%s/anat/%s_T1w.nii.gz'


datasource_anat.inputs.template_args = dict(anat=[['subject_id','subject_id']])

datasource_anat.inputs.sort_filelist = True


datasource_func = pe.Node(interface=nio.DataGrabber(infields=['subject_id','runcode'],
                    outfields=['func']),
                    name = 'datasource_func')

datasource_func.inputs.base_directory = config.data['ds005']['datadir']

datasource_func.inputs.template = '%s/func/%s_task-mixedgamblestask_run-0%s_bold.nii.gz'

datasource_func.inputs.template_args = dict(func=[['subject_id','subject_id','runcode']])

datasource_func.inputs.sort_filelist = True


preprocessing = pe.Workflow(name="preprocessing")

try:
    workdir='/Users/poldrack/data_unsynced/fmri-handbook-2e-data/ds005/nipype_workdir'
    assert os.path.exists(workdir)
except:
    workdir=os.path.join(config.data['ds005']['datadir'],'nipype_workdir')
    if not os.path.exists(workdir):
        os.mkdir(workdir)

preprocessing.base_dir = workdir

preprocessing.connect(infosource,'subject_id',datasource_anat,'subject_id')

preprocessing.connect(infosource,'subject_id',datasource_func,'subject_id')
preprocessing.connect(runinfo,'runcode',datasource_func,'runcode')



# ### Bias field correction

# In[4]:

bfc = pe.Node(interface=ants.N4BiasFieldCorrection(), name="bfc")
bfc.inputs.dimension = 3
bfc.inputs.save_bias = True

preprocessing.connect(datasource_anat, 'anat', bfc, 'input_image')

datasink = pe.Node(nio.DataSink(), name='datasink')

# Save the relevant data into an output directory
datasink.inputs.base_directory = os.path.join(config.data['ds005']['datadir'],'derivatives')
if not os.path.exists(datasink.inputs.base_directory):
    os.mkdir(datasink.inputs.base_directory)


preprocessing.connect(bfc, 'bias_image', datasink, 'bfc.bias')
preprocessing.connect(bfc, 'output_image', datasink, 'bfc.output')


# ### Brain extraction using BET###

# In[5]:

bet_struct=pe.Node(interface=fsl.BET(), name="bet_struct")
bet_struct.inputs.reduce_bias=True
bet_struct.inputs.frac=0.4

preprocessing.connect(bfc,'output_image',bet_struct,'in_file')
preprocessing.connect(bet_struct, 'out_file', datasink, 'bet.output')
preprocessing.connect(bet_struct, 'mask_file', datasink, 'bet.mask')



# ### Segmentation using FAST
#
# Do this to obtain the white matter mask, which we need for BBR registration.

# In[6]:

fast=pe.Node(interface=fsl.FAST(), name="fast")

preprocessing.connect(bet_struct,'out_file',fast,'in_files')
preprocessing.connect(fast, 'partial_volume_files', datasink, 'fast.pvefiles')
preprocessing.connect(fast, 'tissue_class_map', datasink, 'fast.seg')

binarize = pe.Node(fsl.ImageMaths(op_string='-nan -thr 0.5 -bin'),
                   name='binarize')
pickindex = lambda x, i: x[i]
preprocessing.connect(fast, ('partial_volume_files', pickindex, 2),
                 binarize, 'in_file')
preprocessing.connect(binarize, 'out_file', datasink, 'fast.wmseg')



# ### Spatial normalization using ANTs

# In[7]:

antsreg=pe.Node(interface=ants.Registration(), name="antsreg")
antsreg.inputs.fixed_image = os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain.nii.gz')
antsreg.inputs.transforms = ['Translation', 'Rigid', 'Affine', 'SyN']
antsreg.inputs.transform_parameters = [(0.1,), (0.1,), (0.1,), (0.2, 3.0, 0.0)]
antsreg.inputs.number_of_iterations = ([[10, 10, 10]]*3 +
                [[1, 5, 3]])
antsreg.inputs.dimension = 3
antsreg.inputs.write_composite_transform = True
antsreg.inputs.metric = ['Mattes'] * 3 + [['Mattes', 'CC']]
antsreg.inputs.metric_weight = [1] * 3 + [[0.5, 0.5]]
antsreg.inputs.radius_or_number_of_bins = [32] * 3 + [[32, 4]]
antsreg.inputs.sampling_strategy = ['Regular'] * 3 + [[None, None]]
antsreg.inputs.sampling_percentage = [0.3] * 3 + [[None, None]]
antsreg.inputs.convergence_threshold = [1.e-8] * 3 + [-0.01]
antsreg.inputs.convergence_window_size = [20] * 3 + [5]
antsreg.inputs.smoothing_sigmas = [[4, 2, 1]] * 3 + [[1, 0.5, 0]]
antsreg.inputs.sigma_units = ['vox'] * 4
antsreg.inputs.shrink_factors = [[6, 4, 2]] + [[3, 2, 1]]*2 + [[4, 2, 1]]
antsreg.inputs.use_estimate_learning_rate_once = [True] * 4
antsreg.inputs.use_histogram_matching = [False] * 3 + [True]
antsreg.inputs.initial_moving_transform_com = True
antsreg.inputs.output_warped_image = True

preprocessing.connect(bet_struct,'out_file',antsreg,'moving_image')
preprocessing.connect(antsreg, 'warped_image', datasink, 'ants.warped_image')
preprocessing.connect(antsreg, 'composite_transform', datasink, 'ants.composite_transform')
preprocessing.connect(antsreg, 'inverse_composite_transform', datasink, 'ants.inverse_composite_transform')




# In[8]:

# get linear warp of anatomy to MNI space, for comparison to nonlinear
linearMNI=pe.Node(fsl.FLIRT(), name='linearMNI')
linearMNI.inputs.reference=os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain.nii.gz')
linearMNI.inputs.dof=12

preprocessing.connect(bet_struct,'out_file',linearMNI,'in_file')

preprocessing.connect(linearMNI,'out_file',datasink,'affine.out_file')
preprocessing.connect(linearMNI,'out_matrix_file',datasink,'affine.matrix')




# ## Functional preprocessing
#
# ### Motion correction using MCFLIRT

# This will take a few minutes.

# In[9]:

mcflirt=pe.Node(interface=fsl.MCFLIRT(), name="mcflirt")
mcflirt.inputs.save_plots=True
mcflirt.inputs.mean_vol=True

preprocessing.connect(datasource_func, 'func', mcflirt, 'in_file')

preprocessing.connect(mcflirt, 'out_file', datasink, 'mcflirt.out_file')
preprocessing.connect(mcflirt, 'par_file', datasink, 'mcflirt.par')
preprocessing.connect(mcflirt, 'mean_img', datasink, 'mcflirt.mean')




# Make links for the mean functional image and the motion parameters.

# ## Brain extraction
#
# Use FSL's BET to obtain the brain mask for the functional data

# In[10]:

bet_func=pe.Node(interface=fsl.BET(), name="bet_func")

bet_func.inputs.functional=True
bet_func.inputs.mask=True

preprocessing.connect(mcflirt, 'out_file', bet_func, 'in_file')

preprocessing.connect(bet_func, 'out_file', datasink, 'betfunc.out_file')
preprocessing.connect(bet_func, 'mask_file', datasink, 'betfunc.mask_file')

meanbetfunc=pe.Node(interface=fsl.MeanImage(), name="meanbetfunc")
preprocessing.connect(bet_func,'out_file',meanbetfunc,'in_file')
preprocessing.connect(meanbetfunc, 'out_file', datasink, 'betfunc.mean_file')



# #### BBR registration of functional to structural

# In[11]:

mean2anat = pe.Node(fsl.FLIRT(), name='mean2anat')
mean2anat.inputs.dof = 6
preprocessing.connect(meanbetfunc, 'out_file', mean2anat, 'in_file')
preprocessing.connect(bet_struct, 'out_file', mean2anat, 'reference')

mean2anatbbr = pe.Node(fsl.FLIRT(), name='mean2anatbbr')
mean2anatbbr.inputs.dof = 6
mean2anatbbr.inputs.cost = 'bbr'
mean2anatbbr.inputs.schedule = os.path.join(os.getenv('FSLDIR'),
                                            'etc/flirtsch/bbr.sch')

preprocessing.connect(meanbetfunc, 'out_file', mean2anatbbr, 'in_file')
preprocessing.connect(binarize, 'out_file', mean2anatbbr, 'wm_seg')
preprocessing.connect(bet_struct, 'out_file', mean2anatbbr, 'reference')
preprocessing.connect(mean2anat, 'out_matrix_file',
                 mean2anatbbr, 'in_matrix_file')

preprocessing.connect(mean2anat, 'out_matrix_file', datasink, 'mean2anat.out_matrix')

preprocessing.connect(mean2anatbbr, 'out_matrix_file', datasink, 'bbr.out_matrix')
preprocessing.connect(mean2anatbbr, 'out_file', datasink, 'bbr.out_file')


# convert BBR matrix to ITK for ANTS

convert2itk = pe.Node(C3dAffineTool(),
                      name='convert2itk')
convert2itk.inputs.fsl2ras = True
convert2itk.inputs.itk_transform = True
preprocessing.connect(mean2anatbbr, 'out_matrix_file', convert2itk, 'transform_file')
preprocessing.connect(meanbetfunc, 'out_file', convert2itk, 'source_file')
preprocessing.connect(bet_struct, 'out_file', convert2itk, 'reference_file')



# In[ ]:

# Concatenate the affine and ants transforms into a list

pickfirst = lambda x: x[0]

merge = pe.Node(Merge(2),  name='mergexfm')
preprocessing.connect(convert2itk, 'itk_transform', merge, 'in2')
preprocessing.connect(antsreg, 'composite_transform', merge, 'in1')

warpmean = pe.Node(ants.ApplyTransforms(), name='warpmean')
warpmean.inputs.input_image_type = 0
warpmean.inputs.interpolation = 'Linear'
warpmean.inputs.invert_transform_flags = [False] #[False, False]
warpmean.inputs.terminal_output = 'file'
warpmean.inputs.args = '--float'
warpmean.inputs.reference_image = os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain.nii.gz')


preprocessing.connect(mean2anatbbr, 'out_file', warpmean, 'input_image')
preprocessing.connect(antsreg, 'composite_transform', warpmean, 'transforms')
#preprocessing.connect(merge, 'out', warpmean, 'transforms')
preprocessing.connect(warpmean,'output_image',datasink,'ants.warped_mean')


# high pass filtering

hpfilt=pe.Node(interface=fsl.maths.TemporalFilter(),name='hpfilt')
TR=2.0
hpf_cutoff=80.

hpfilt.inputs.highpass_sigma = hpf_cutoff/(2*TR)
preprocessing.connect(bet_func, 'out_file',hpfilt,'in_file')


preprocessing.connect(hpfilt,'out_file',datasink,'hpfilt')

rescale = pe.Node(interface=fsl.maths.BinaryMaths(),name='rescale')
rescale.inputs.operation = "add"

preprocessing.connect(hpfilt,'out_file',rescale,'in_file')
preprocessing.connect(meanbetfunc, 'out_file',rescale,'operand_file')

preprocessing.connect(rescale,'out_file',datasink,'rescaled')

# spatial smoothing using FSL's susan or regular gaussian smoothing

use_SUSAN=False

if use_SUSAN:
    medianval = pe.MapNode(interface=fsl.ImageStats(op_string='-k %s -p 50'),
                           iterfield=['in_file','mask_file'],
                           name='medianval')
    preprocessing.connect(mcflirt, 'out_file', medianval, 'in_file')
    preprocessing.connect(bet_func, 'mask_file', medianval, 'mask_file')

    smooth = pe.MapNode(interface=fsl.SUSAN(),
                        iterfield=['in_file', 'brightness_threshold'],
                        name='smooth')
    def getbtthresh(medianvals):
        return [0.75 * val for val in medianvals]

    preprocessing.connect(medianval, ('out_stat', getbtthresh), smooth, 'brightness_threshold')
else:
    smooth=pe.Node(interface=fsl.utils.Smooth(), name="smooth")

smooth.inputs.fwhm=6

# Turn off smoothing for now - we will do it later on the constrast images
#preprocessing.connect(rescale,'out_file',smooth,'in_file')
#preprocessing.connect(smooth,'smoothed_file',datasink,'smooth')



graph=preprocessing.run(plugin='MultiProc', plugin_args={'n_procs' : 16})




# ## First-level modeling

# In[ ]:

firstlevel = pe.Workflow(name="firstlevel")


firstlevel.base_dir = workdir



datasource_func = pe.Node(interface=nio.DataGrabber(infields=['subject_id','runcode'],
                    outfields=['func']),
                    name = 'datasource_func')

datasource_func.inputs.base_directory = config.data['ds005']['datadir']

datasource_func.inputs.template = 'derivatives/rescaled/_runcode_%s/_subject_id_%s/%s_task-mixedgamblestask_run-0%s_bold_mcf_brain_filt_maths.nii.gz'

datasource_func.inputs.template_args = dict(func=[['runcode','subject_id','subject_id','runcode']])

datasource_func.inputs.sort_filelist = True


firstlevel.connect(infosource,'subject_id',datasource_anat,'subject_id')
firstlevel.connect(infosource,'subject_id',datasource_func,'subject_id')
firstlevel.connect(runinfo,'runcode',datasource_func,'runcode')




# In[ ]:

# creat a function that can read all of the onsets and motion files
# and generate the appropriate structure

def get_onsets(subject_id,runnum):
    from fmrihandbook.utils.config import Config
    from nipype.interfaces.base import Bunch
    import numpy
    import os,json


    config=Config()

    regressors=[]
    regressor_names=[]
    fdfile=os.path.join(config.data['ds005']['datadir'],'derivatives/mcflirt/par/_runcode_%s/_subject_id_%s/fd.txt'%(runnum,subject_id))
    regressors.append(numpy.loadtxt(fdfile))
    regressor_names.append('fd')

    dvarsfile=os.path.join(config.data['ds005']['datadir'],'derivatives/mcflirt/par/_runcode_%s/_subject_id_%s/dvars.txt'%(runnum,subject_id))
    regressors.append(numpy.loadtxt(dvarsfile))
    regressor_names.append('dvars')

    motfile=os.path.join(config.data['ds005']['datadir'],
                'derivatives/mcflirt/par/_runcode_%s/_subject_id_%s/%s_task-mixedgamblestask_run-0%s_bold_mcf.nii.gz.par'%(runnum,subject_id,subject_id,runnum))
    motpars=numpy.loadtxt(motfile)
    for i in range(motpars.shape[1]):
        regressors.append(motpars[:,i])
        regressor_names.append('motpar%d'%int(i+1))
        td=numpy.zeros(motpars.shape[0])
        td[1:]=motpars[1:,i] -motpars[:-1,i]
        regressors.append(td)
        regressor_names.append('disp%d'%int(i+1))


    onsfile=os.path.join(config.data['ds005']['datadir'],'ds005_onsets.json')
    with open(onsfile, 'r') as f:
        ons = json.load(f)
    conditions=['task','param-gain','param-loss','param-rt']
    info=ons[subject_id]['run-0%s'%runnum]
    onsets=[]
    durations=[]
    amplitudes=[]
    for c in conditions:
        onsets.append(info[c]['onset'])
        durations.append(info[c]['duration'])
        amplitudes.append(info[c]['weight'])

    info = [Bunch(conditions=conditions,
              onsets=onsets,
              durations=durations,
                amplitudes=amplitudes,
                 regressors=regressors,
                 regressor_names=regressor_names)]
    return info

getonsets = pe.Node(niu.Function(input_names=['subject_id','runnum'],
                output_names=['info'],
                function=get_onsets),
                name='getonsets')

firstlevel.connect(infosource,'subject_id',getonsets,'subject_id')
firstlevel.connect(runinfo,'runcode',getonsets,'runnum')

specifymodel=pe.Node(interface=model.SpecifyModel(),name='specifymodel')

specifymodel.inputs.input_units = 'secs'
#specifymodel.inputs.functional_runs = preprocessed_epi
specifymodel.inputs.time_repetition = 2.0
specifymodel.inputs.high_pass_filter_cutoff = hpf_cutoff
#s.inputs.subject_info = info
firstlevel.connect(getonsets,'info',specifymodel,'subject_info')
firstlevel.connect(datasource_func,'func',specifymodel,'functional_runs')



# In[ ]:

contrasts=[['task>Baseline','T',
            ['task'],[1]],
           ['param-gain','T',
            ['param-gain'],[1]],
           ['param-loss-neg','T',
            ['param-loss'],[-1]],
           ['param-rt','T',
            ['param-rt'],[1]]]

level1design = pe.Node(interface=fsl.model.Level1Design(),name='level1design')
level1design.inputs.interscan_interval =2.0
level1design.inputs.bases = {'dgamma':{'derivs': True}}
level1design.inputs.model_serial_correlations=True
level1design.inputs.contrasts=contrasts

firstlevel.connect(specifymodel,'session_info',level1design,'session_info')




# In[ ]:

modelgen = pe.Node(interface=fsl.model.FEATModel(),name='modelgen')
firstlevel.connect(level1design,'fsf_files',modelgen,'fsf_file')
firstlevel.connect(level1design,'ev_files',modelgen,'ev_files')

filmgls= pe.Node(interface=fsl.FILMGLS(),name='filmgls')

filmgls.inputs.autocorr_noestimate = True
firstlevel.connect(datasource_func,'func',filmgls,'in_file')
firstlevel.connect(modelgen,'design_file',filmgls,'design_file')
firstlevel.connect(modelgen,'con_file',filmgls,'tcon_file')

firstlevel.connect(filmgls,'param_estimates',datasink,'filmgls.param_estimates')
firstlevel.connect(filmgls,'sigmasquareds',datasink,'filmgls.sigmasquareds')
firstlevel.connect(filmgls,'copes',datasink,'filmgls.copes')
firstlevel.connect(filmgls,'varcopes',datasink,'filmgls.varcopes')
firstlevel.connect(filmgls,'dof_file',datasink,'filmgls.dof_file')
firstlevel.connect(filmgls,'tstats',datasink,'filmgls.tstats')
firstlevel.connect(filmgls,'zstats',datasink,'filmgls.zstats')


firstlevel.run()


# ## warp copes and varcopes to MNI space
#
# use suboptimal two-step procedure - first apply linear registration from BBR, then apply nonlinear from ANTS
#
# this is necesary because of errors getting FSL mat file into ANTS

# In[ ]:

datasource_stat = pe.Node(interface=nio.DataGrabber(infields=['subject_id','runcode'],
                    outfields=['stats']),
                    name = 'datasource_stat')

datasource_stat.inputs.base_directory = config.data['ds005']['datadir']
datasource_stat.inputs.template = 'derivatives/filmgls/*copes/_runcode_%s/_subject_id_%s/*.nii.gz'
datasource_stat.inputs.template_args = dict(stats=[['runcode','subject_id']])
datasource_stat.inputs.sort_filelist = True


datasource_bbrmat = pe.Node(interface=nio.DataGrabber(infields=['subject_id','runcode'],
                    outfields=['bbrmat']),
                    name = 'datasource_bbrmat')

datasource_bbrmat.inputs.base_directory = config.data['ds005']['datadir']
datasource_bbrmat.inputs.template = 'derivatives/bbr/out_matrix/_runcode_%s/_subject_id_%s/*.mat'
datasource_bbrmat.inputs.template_args = dict(bbrmat=[['runcode','subject_id']])
datasource_bbrmat.inputs.sort_filelist = True

datasource_antsreg = pe.Node(interface=nio.DataGrabber(infields=['subject_id','runcode'],
                    outfields=['composite_transform']),
                    name = 'datasource_antsreg')

datasource_antsreg.inputs.base_directory = config.data['ds005']['datadir']
datasource_antsreg.inputs.template = 'derivatives/ants/composite_transform/_subject_id_%s/transformComposite.h5'
datasource_antsreg.inputs.template_args = dict(composite_transform=[['subject_id']])
datasource_antsreg.inputs.sort_filelist = True

datasource_linearreg = pe.Node(interface=nio.DataGrabber(infields=['subject_id','runcode'],
                    outfields=['matrix']),
                    name = 'datasource_linearreg')

datasource_linearreg.inputs.base_directory = config.data['ds005']['datadir']
datasource_linearreg.inputs.template = 'derivatives/affine/matrix/_subject_id_%s/%s_T1w_corrected_brain_flirt.mat'
datasource_linearreg.inputs.template_args = dict(matrix=[['subject_id','subject_id']])
datasource_linearreg.inputs.sort_filelist = True




secondlevel = pe.Workflow(name='secondlevel')
secondlevel.base_dir = workdir


secondlevel.connect(infosource,'subject_id',datasource_bbrmat,'subject_id')
secondlevel.connect(infosource,'subject_id',datasource_stat,'subject_id')
secondlevel.connect(infosource,'subject_id',datasource_anat,'subject_id')
secondlevel.connect(infosource,'subject_id',datasource_antsreg,'subject_id')
secondlevel.connect(infosource,'subject_id',datasource_linearreg,'subject_id')
secondlevel.connect(infosource,'subject_id',datasource_func,'subject_id')




secondlevel.connect(runinfo,'runcode',datasource_stat,'runcode')
secondlevel.connect(runinfo,'runcode',datasource_bbrmat,'runcode')
secondlevel.connect(runinfo,'runcode',datasource_anat,'runcode')
secondlevel.connect(runinfo,'runcode',datasource_antsreg,'runcode')
secondlevel.connect(runinfo,'runcode',datasource_linearreg,'runcode')
secondlevel.connect(runinfo,'runcode',datasource_func,'runcode')


bbrstats=pe.MapNode(fsl.FLIRT(),name='bbrstats', iterfield=['in_file'])
bbrstats.inputs.apply_xfm=True


secondlevel.connect(datasource_stat, 'stats', bbrstats,'in_file')
secondlevel.connect(datasource_bbrmat, 'bbrmat', bbrstats, 'in_matrix_file')
secondlevel.connect(datasource_anat, 'anat', bbrstats,'reference')

secondlevel.connect(bbrstats,'out_file',datasink,'bbr.stats')



# In[ ]:

# warp stats maps using both ANTS and linear registration

warpstats = pe.MapNode(ants.ApplyTransforms(), name='warpstats', iterfield=['input_image'])
warpstats.inputs.input_image_type = 0
warpstats.inputs.interpolation = 'Linear'
warpstats.inputs.invert_transform_flags = [False] #[False, False]
warpstats.inputs.terminal_output = 'file'
warpstats.inputs.args = '--float'
warpstats.inputs.reference_image = os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain.nii.gz')

#secondlevel.connect(bbrstats, 'out_file', warpstats, 'subject_id')


secondlevel.connect(bbrstats, 'out_file', warpstats, 'input_image')
secondlevel.connect(datasource_antsreg, 'composite_transform', warpstats, 'transforms')
secondlevel.connect(warpstats,'output_image',datasink,'ants.warped_stats')

warpstats_linear=pe.MapNode(fsl.FLIRT(),name='warpstats_linear', iterfield=['in_file'])
warpstats_linear.inputs.apply_xfm=True
warpstats_linear.inputs.reference=os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain.nii.gz')


secondlevel.connect(bbrstats, 'out_file', warpstats_linear, 'in_file')
secondlevel.connect(datasource_linearreg, 'matrix', warpstats_linear, 'in_matrix_file')
secondlevel.connect(warpstats_linear,'out_file',datasink,'affine.warped_stats')


secondlevel.run()


# ### Second level analysis - fixed effects across runs

# In[ ]:

fixed_fx = pe.Workflow(name='fixedfx')
fixed_fx.base_dir = workdir

copeinfo = pe.Node(interface=niu.IdentityInterface(fields=['copenum']), name="copeinfo")

copeinfo.iterables = ('copenum',[1,2,3,4])

regtypes = pe.Node(interface=niu.IdentityInterface(fields=['regtype']), name="regtypes")

regtypes.iterables = ('regtype',['ants','affine'])


datasource_cope = pe.Node(interface=nio.DataGrabber(infields=['subject_id','copenum','regtype'],
                    outfields=['copes']),
                    name = 'datasource_cope')

datasource_cope.inputs.base_directory = config.data['ds005']['datadir']
datasource_cope.inputs.template = 'derivatives/%s/warped_stats/_runcode_*/_subject_id_%s/_warpstats*/cope%d_*.nii.gz'
datasource_cope.inputs.template_args = dict(copes=[['regtype','subject_id','copenum']])
datasource_cope.inputs.sort_filelist = True

datasource_varcope = pe.Node(interface=nio.DataGrabber(infields=['subject_id','copenum','regtype'],
                    outfields=['varcopes']),
                    name = 'datasource_varcope')

datasource_varcope.inputs.base_directory = config.data['ds005']['datadir']
datasource_varcope.inputs.template = 'derivatives/%s/warped_stats/_runcode_*/_subject_id_%s/_warpstats*/varcope%d_*.nii.gz'
datasource_varcope.inputs.template_args = dict(varcopes=[['regtype','subject_id','copenum']])
datasource_varcope.inputs.sort_filelist = True


fixed_fx.connect(infosource,'subject_id',datasource_cope,'subject_id')
fixed_fx.connect(infosource,'subject_id',datasource_varcope,'subject_id')
fixed_fx.connect(copeinfo,'copenum',datasource_cope,'copenum')
fixed_fx.connect(copeinfo,'copenum',datasource_varcope,'copenum')
fixed_fx.connect(regtypes,'regtype',datasource_cope,'regtype')
fixed_fx.connect(regtypes,'regtype',datasource_varcope,'regtype')



copemerge    = pe.MapNode(interface=fsl.Merge(dimension='t'),
                          iterfield=['in_files'],
                          name="copemerge")

varcopemerge = pe.MapNode(interface=fsl.Merge(dimension='t'),
                       iterfield=['in_files'],
                       name="varcopemerge")

fixed_fx.connect(datasource_cope,'copes',copemerge,'in_files')
fixed_fx.connect(datasource_varcope,'varcopes',varcopemerge,'in_files')

level2model = pe.Node(interface=fsl.L2Model(),
                      name='l2model')
level2model.inputs.num_copes=3


flameo = pe.MapNode(interface=fsl.FLAMEO(run_mode='fe'), name="flameo",
                    iterfield=['cope_file','var_cope_file'])
flameo.inputs.mask_file=os.path.join(os.getenv('FSLDIR'),'data/standard/MNI152_T1_2mm_brain_mask.nii.gz')


fixed_fx.connect(copemerge,'merged_file',flameo,'cope_file')
fixed_fx.connect(varcopemerge,'merged_file',flameo,'var_cope_file')

fixed_fx.connect(level2model,'design_mat',flameo,'design_file')
fixed_fx.connect(level2model,'design_con',flameo,'t_con_file')
fixed_fx.connect(level2model,'design_grp',flameo,'cov_split_file')

fixed_fx.connect(flameo,'copes',datasink,'fixedfx.cope')
fixed_fx.connect(flameo,'var_copes',datasink,'fixedfx.varcope')


fixed_fx.run()
