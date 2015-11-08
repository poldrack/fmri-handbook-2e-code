% This batch script analyses the Attention to Visual Motion fMRI dataset
% available from the SPM website using DCM:
%   http://www.fil.ion.ucl.ac.uk/spm/data/attention/
% as described in the SPM manual:
%   http://www.fil.ion.ucl.ac.uk/spm/doc/spm12_manual.pdf#Chap:DCM_fmri
%__________________________________________________________________________
% Copyright (C) 2014 Wellcome Trust Centre for Neuroimaging

% Guillaume Flandin & Peter Zeidman
% $Id: dcm_spm12_batch.m 12 2014-09-29 19:58:09Z guillaume $


% Directory containing the attention data
%--------------------------------------------------------------------------
%data_path = fileparts(mfilename('fullpath'));
%if isempty(data_path), data_path = pwd; end
%fprintf('%-40s:', 'Downloading Attention dataset...');
%urlwrite('http://www.fil.ion.ucl.ac.uk/spm/download/data/attention/attention.zip','attention.zip');
%unzip(fullfile(data_path,'attention.zip'));
%data_path = fullfile(data_path,'attention');
%fprintf(' %30s\n', '...done');

% Initialise SPM
%--------------------------------------------------------------------------
spm('Defaults','fMRI');
spm_jobman('initcfg');
%spm_get_defaults('cmdline',1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GLM SPECIFICATION, ESTIMATION & INFERENCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%factors = load(fullfile(data_path,'factors.mat'));

%TBD need to make fake images
data_path='/Users/poldrack/code/fmri-analysis-vm/analysis/results/dcmfiles'
f = spm_select('FPList', data_path, '^*nii$');

clear matlabbatch

% OUTPUT DIRECTORY
%--------------------------------------------------------------------------
matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = cellstr(data_path);
matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'GLM';

% MODEL SPECIFICATION
%--------------------------------------------------------------------------
matlabbatch{2}.spm.stats.fmri_spec.dir = cellstr(fullfile(data_path,'GLM'));
matlabbatch{2}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{2}.spm.stats.fmri_spec.timing.RT    = 1;
matlabbatch{2}.spm.stats.fmri_spec.sess.scans            = cellstr(f);
matlabbatch{2}.spm.stats.fmri_spec.sess.cond(1).name     = 'Task';
matlabbatch{2}.spm.stats.fmri_spec.sess.cond(1).onset    = [20,60,100,140,180,220,260];
matlabbatch{2}.spm.stats.fmri_spec.sess.cond(1).duration = 20;

% MODEL ESTIMATION
%--------------------------------------------------------------------------
matlabbatch{3}.spm.stats.fmri_est.spmmat = cellstr(fullfile(data_path,'GLM','SPM.mat'));

% INFERENCE
%--------------------------------------------------------------------------
matlabbatch{4}.spm.stats.con.spmmat = cellstr(fullfile(data_path,'GLM','SPM.mat'));
matlabbatch{4}.spm.stats.con.consess{1}.fcon.name = 'Effects of Interest';
matlabbatch{4}.spm.stats.con.consess{1}.fcon.weights = eye(1);
matlabbatch{4}.spm.stats.con.consess{2}.tcon.name = 'Task';
matlabbatch{4}.spm.stats.con.consess{2}.tcon.weights = [1 0];

%spm_jobman('run',matlabbatch);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VOLUMES OF INTEREST
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear matlabbatch

% EXTRACTING TIME SERIES for each ROI
%--------------------------------------------------------------------------
matlabbatch{1}.spm.util.voi.spmmat = cellstr(fullfile(data_path,'GLM','SPM.mat'));
matlabbatch{1}.spm.util.voi.adjust = 1;  % "effects of interest" F-contrast
matlabbatch{1}.spm.util.voi.session = 1; % session 1
matlabbatch{1}.spm.util.voi.name = 'ROI0';
matlabbatch{1}.spm.util.voi.roi{1}.spm.spmmat = {''}; % using SPM.mat above
matlabbatch{1}.spm.util.voi.roi{1}.spm.contrast = 1;  % "Motion" T-contrast
matlabbatch{1}.spm.util.voi.roi{1}.spm.threshdesc = 'none';
matlabbatch{1}.spm.util.voi.roi{1}.spm.thresh = 1.0;
matlabbatch{1}.spm.util.voi.roi{1}.spm.extent = 0;
matlabbatch{1}.spm.util.voi.roi{2}.sphere.centre = [2 2 2];
matlabbatch{1}.spm.util.voi.roi{2}.sphere.radius = 1;
matlabbatch{1}.spm.util.voi.roi{2}.sphere.move.fixed = 1;
matlabbatch{1}.spm.util.voi.expression = 'i1 & i2';

matlabbatch{2}.spm.util.voi.spmmat = cellstr(fullfile(data_path,'GLM','SPM.mat'));
matlabbatch{2}.spm.util.voi.adjust = 1;  % "effects of interest" F-contrast
matlabbatch{2}.spm.util.voi.session = 1; % session 1
matlabbatch{2}.spm.util.voi.name = 'ROI1';
matlabbatch{2}.spm.util.voi.roi{1}.spm.spmmat = {''}; % using SPM.mat above
matlabbatch{2}.spm.util.voi.roi{1}.spm.contrast = 1;  % "Motion" T-contrast
matlabbatch{2}.spm.util.voi.roi{1}.spm.threshdesc = 'none';
matlabbatch{2}.spm.util.voi.roi{1}.spm.thresh = 1.0;
matlabbatch{2}.spm.util.voi.roi{1}.spm.extent = 0;
matlabbatch{2}.spm.util.voi.roi{2}.sphere.centre = [4 4 4];
matlabbatch{2}.spm.util.voi.roi{2}.sphere.radius = 1;
matlabbatch{2}.spm.util.voi.roi{2}.sphere.move.fixed = 1;
matlabbatch{2}.spm.util.voi.expression = 'i1 & i2';

matlabbatch{3}.spm.util.voi.spmmat = cellstr(fullfile(data_path,'GLM','SPM.mat'));
matlabbatch{3}.spm.util.voi.adjust = 1;  % "effects of interest" F-contrast
matlabbatch{3}.spm.util.voi.session = 1; % session 1
matlabbatch{3}.spm.util.voi.name = 'ROI2';
matlabbatch{3}.spm.util.voi.roi{1}.spm.spmmat = {''}; % using SPM.mat above
matlabbatch{3}.spm.util.voi.roi{1}.spm.contrast = 1;  % "Motion" T-contrast
matlabbatch{3}.spm.util.voi.roi{1}.spm.threshdesc = 'none';
matlabbatch{3}.spm.util.voi.roi{1}.spm.thresh = 1.0;
matlabbatch{3}.spm.util.voi.roi{1}.spm.extent = 0;
matlabbatch{3}.spm.util.voi.roi{2}.sphere.centre = [6 6 6];
matlabbatch{3}.spm.util.voi.roi{2}.sphere.radius = 1;
matlabbatch{3}.spm.util.voi.roi{2}.sphere.move.fixed = 1;
matlabbatch{3}.spm.util.voi.expression = 'i1 & i2';

matlabbatch{4}.spm.util.voi.spmmat = cellstr(fullfile(data_path,'GLM','SPM.mat'));
matlabbatch{4}.spm.util.voi.adjust = 1;  % "effects of interest" F-contrast
matlabbatch{4}.spm.util.voi.session = 1; % session 1
matlabbatch{4}.spm.util.voi.name = 'ROI3';
matlabbatch{4}.spm.util.voi.roi{1}.spm.spmmat = {''}; % using SPM.mat above
matlabbatch{4}.spm.util.voi.roi{1}.spm.contrast = 1;  % "Motion" T-contrast
matlabbatch{4}.spm.util.voi.roi{1}.spm.threshdesc = 'none';
matlabbatch{4}.spm.util.voi.roi{1}.spm.thresh = 1.0;
matlabbatch{4}.spm.util.voi.roi{1}.spm.extent = 0;
matlabbatch{4}.spm.util.voi.roi{2}.sphere.centre = [8 8 8];
matlabbatch{4}.spm.util.voi.roi{2}.sphere.radius = 1;
matlabbatch{4}.spm.util.voi.roi{2}.sphere.move.fixed = 1;
matlabbatch{4}.spm.util.voi.expression = 'i1 & i2';

matlabbatch{5}.spm.util.voi.spmmat = cellstr(fullfile(data_path,'GLM','SPM.mat'));
matlabbatch{5}.spm.util.voi.adjust = 1;  % "effects of interest" F-contrast
matlabbatch{5}.spm.util.voi.session = 1; % session 1
matlabbatch{5}.spm.util.voi.name = 'ROI4';
matlabbatch{5}.spm.util.voi.roi{1}.spm.spmmat = {''}; % using SPM.mat above
matlabbatch{5}.spm.util.voi.roi{1}.spm.contrast = 1;  % "Motion" T-contrast
matlabbatch{5}.spm.util.voi.roi{1}.spm.threshdesc = 'none';
matlabbatch{5}.spm.util.voi.roi{1}.spm.thresh = 1.0;
matlabbatch{5}.spm.util.voi.roi{1}.spm.extent = 0;
matlabbatch{5}.spm.util.voi.roi{2}.sphere.centre = [10 10 10];
matlabbatch{5}.spm.util.voi.roi{2}.sphere.radius = 1;
matlabbatch{5}.spm.util.voi.roi{2}.sphere.move.fixed = 1;
matlabbatch{5}.spm.util.voi.expression = 'i1 & i2';

%spm_jobman('run',matlabbatch);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DYNAMIC CAUSAL MODELLING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear DCM

% SPECIFICATION DCMs "attentional modulation of backward/forward connection"
%--------------------------------------------------------------------------
% To specify a DCM, you might want to create a template one using the GUI
% then use spm_dcm_U.m and spm_dcm_voi.m to insert new inputs and new
% regions. The following code creates a DCM file from scratch, which
% involves some technical subtleties and a deeper knowledge of the DCM
% structure.

load(fullfile(data_path,'GLM','SPM.mat'));

% Load regions of interest
%--------------------------------------------------------------------------
load(fullfile(data_path,'GLM','VOI_ROI0_1.mat'),'xY');
DCM.xY(1) = xY;
load(fullfile(data_path,'GLM','VOI_ROI1_1.mat'),'xY');
DCM.xY(2) = xY;
load(fullfile(data_path,'GLM','VOI_ROI2_1.mat'),'xY');
DCM.xY(3) = xY;
load(fullfile(data_path,'GLM','VOI_ROI3_1.mat'),'xY');
DCM.xY(4) = xY;
load(fullfile(data_path,'GLM','VOI_ROI4_1.mat'),'xY');
DCM.xY(5) = xY;


DCM.n = length(DCM.xY);      % number of regions
DCM.v = length(DCM.xY(1).u); % number of time points

% Time series
%--------------------------------------------------------------------------
DCM.Y.dt  = SPM.xY.RT;
DCM.Y.X0  = DCM.xY(1).X0;
for i = 1:DCM.n
    DCM.Y.y(:,i)  = DCM.xY(i).u;
    DCM.Y.name{i} = DCM.xY(i).name;
end

DCM.Y.Q    = spm_Ce(ones(1,DCM.n)*DCM.v);

% Experimental inputs
%--------------------------------------------------------------------------
DCM.U.dt   =  SPM.Sess.U(1).dt;
DCM.U.name = [SPM.Sess.U.name];
DCM.U.u    = [SPM.Sess.U(1).u(33:end,1) ...
              SPM.Sess.U(1).u(33:end,1) ...
              SPM.Sess.U(1).u(33:end,1)];

% DCM parameters and options
%--------------------------------------------------------------------------
DCM.delays = repmat(SPM.xY.RT/2,DCM.n,1);
DCM.TE     = 0.04;

DCM.options.nonlinear  = 0;
DCM.options.two_state  = 0;
DCM.options.stochastic = 0;
DCM.options.nograph    = 1;

% Connectivity matrices for null model with no PPI
%--------------------------------------------------------------------------
DCM.a = eye(5,5);
DCM.a(3,2)=1;
DCM.a(4,2)=1;
DCM.a(3,1)=1;
DCM.a(5,1)=1;

DCM.b = zeros(5,5);
DCM.c = [1 0 0 0 0];
DCM.d = zeros(5,5,0);

save(fullfile(data_path,'GLM','DCM_nullmodel.mat'),'DCM');

% add PPI
DCM.b(3,1)=1;
DCM.b(5,1)=1;

save(fullfile(data_path,'GLM','DCM_truemodel.mat'),'DCM');


% DCM Estimation
%--------------------------------------------------------------------------
clear matlabbatch

matlabbatch{1}.spm.dcm.fmri.estimate.dcmmat = {...
    fullfile(data_path,'GLM','DCM_truemodel.mat');};
     ...
    fullfile(data_path,'GLM','DCM_nullmodel.mat')};

spm_jobman('run',matlabbatch);

% Bayesian Model Comparison
%--------------------------------------------------------------------------
DCM_bwd = load('DCM_truemodel.mat','F');
DCM_fwd = load('DCM_nullmodel.mat','F');
fprintf('Model evidence: %f (bwd) vs %f (fwd)\n',DCM_bwd.F,DCM_fwd.F);
