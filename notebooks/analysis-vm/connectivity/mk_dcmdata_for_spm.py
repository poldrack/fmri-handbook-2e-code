import numpy
import nibabel
from nipy.modalities.fmri.hemodynamic_models import spm_hrf,compute_regressor
import os

outdir='../results/dcmfiles'
if not os.path.exists(outdir):
    os.mkdir(outdir)

dcmdata=numpy.load('../results/dcmdata.npz')
data_conv=dcmdata['data']

u=numpy.convolve(dcmdata['u'],spm_hrf(0.01,oversampling=1))
u=u[range(0,data_conv.shape[0],int(1./0.01))]
ntp=u.shape[0]

data=data_conv[range(0,data_conv.shape[0],int(1./0.01))]

roi_locations=[[2,2,2],[4,4,4],[6,6,6],[8,8,8],[10,10,10]]
datamat=numpy.zeros((12,12,12,ntp))
datamat[2:11,2:11,2:11,:]+=100
for i in range(5):
    datamat[roi_locations[i][0],roi_locations[i][1],roi_locations[i][2],:]+=data[:,i]

img=nibabel.Nifti1Image(datamat,numpy.identity(4))
img.to_filename(os.path.join(outdir,'all.nii'))

for i in range(ntp):
    tmp=datamat[:,:,:,i]
    img=nibabel.Nifti1Image(tmp,numpy.identity(4))
    img.to_filename(os.path.join(outdir,'img%03d.nii'%i))
