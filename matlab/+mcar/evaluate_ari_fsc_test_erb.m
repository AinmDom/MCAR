function evaluate_ari_fsc_test_erb(configPath)
%EVALUATE_ARI_FSC_TEST_ERB Frozen AKerbError metrics on derived ARI test data.
projectRoot=fileparts(fileparts(fileparts(mfilename('fullpath'))));
config=jsondecode(fileread(char(configPath)));
assert(string(config.status)=="frozen_pre_test" && config.test_access_authorized);
access=jsondecode(fileread(fullfile(projectRoot,char(config.test_access_state))));
assert(string(access.status)=="completed" && access.test_subjects_read==22);
outputPath=fullfile(projectRoot,char(config.result_root),'erb_subject_level.csv');
assert(~isfile(outputPath),'Refusing to overwrite ARI locked-test ERB output.');
split=readtable(fullfile(projectRoot,char(config.split_csv)),'TextType','string');
subjects=split.subject_id(split.split=="test"); assert(numel(subjects)==22);
originalDir=pwd; restoreDirectory=onCleanup(@() cd(originalDir)); %#ok<NASGU>
cd(fullfile(projectRoot,'external','SUpDEq')); supdeq_start; cd(originalDir);
result=table();
for subjectIndex=1:numel(subjects)
    label=char(subjects(subjectIndex));
    feature=fullfile(projectRoot,char(config.feature_root),[label '.h5']);
    waveform=fullfile(projectRoot,char(config.prepared_waveform_root),[label '.h5']);
    prediction=fullfile(projectRoot,char(config.prediction_root),'subjects',label,'prediction.h5');
    assert(string(h5readatt(feature,'/','split'))=="test" && string(h5readatt(prediction,'/','split'))=="test");
    referenceDb=read_eardbf(feature,'/reference_logmag_db');
    mcaDb=read_eardbf(feature,'/mca_logmag_db');
    residual=read_eardbf(prediction,'/predicted_residual_db');
    directions=double(h5read(feature,'/direction_features')).';
    interpolation=logical(h5read(feature,'/interpolation_evaluation_mask')); interpolation=interpolation(:);
    referenceHrir=double(h5read(waveform,'/hrir'));
    assert(isequal(size(referenceHrir),[256,2,1550]) && sum(interpolation)==1524);
    referenceHrir=permute(referenceHrir,[1,3,2]);
    methods=["MCA","FSC"];
    spectra={mcaDb,mcaDb+residual};
    for methodIndex=1:2
        estimatedHrir=reconstruct_from_metadata(spectra{methodIndex},feature);
        values=erb_metrics(estimatedHrir,referenceHrir,directions,interpolation);
        result=[result; table(string(label),sscanf(label,'nh%d'),methods(methodIndex), ...
            "FullSphereERB","dB",values(1),'VariableNames', ...
            {'SubjectLabel','SubjectID','Method','Metric','Unit','Value'}); ...
            table(string(label),sscanf(label,'nh%d'),methods(methodIndex), ...
            "Contralateral25ERB","dB",values(2),'VariableNames', ...
            {'SubjectLabel','SubjectID','Method','Metric','Unit','Value'})]; %#ok<AGROW>
    end
    fprintf('ARI locked-test ERB [%d/22] %s\n',subjectIndex,label);
end
assert(height(result)==88 && all(isfinite(result.Value)));
writetable(result,outputPath);
end

function values=read_eardbf(path,dataset)
values=permute(double(h5read(path,dataset)),[3,2,1]);
end

function hrir=reconstruct_from_metadata(selectedDb,sourcePath)
phase=read_eardbf(sourcePath,'/strict_ild/mca_selected_phase_rad');
outsideReal=read_eardbf(sourcePath,'/strict_ild/mca_outside_real');
outsideImag=read_eardbf(sourcePath,'/strict_ild/mca_outside_imag');
selected=double(h5read(sourcePath,'/strict_ild/selected_bin_indices_zero_based'))+1;
outside=double(h5read(sourcePath,'/strict_ild/outside_bin_indices_zero_based'))+1;
spectrum=complex(zeros(2,1550,513));
spectrum(:,:,selected)=10.^(selectedDb/20).*exp(1i*phase);
spectrum(:,:,outside)=outsideReal+1i*outsideImag;
hrir=zeros(256,1550,2);
for ear=1:2
    fullSpectrum=AKsingle2bothSidedSpectrum(squeeze(spectrum(ear,:,:)).');
    time=real(ifft(fullSpectrum)); hrir(:,:,ear)=time(1:256,:);
end
end

function values=erb_metrics(estimate,reference,directions,mask)
weights=directions(:,6); azimuth=directions(:,1); elevation=directions(:,2);
left25=mask & great_circle_mask(azimuth,elevation,270,0,25);
right25=mask & great_circle_mask(azimuth,elevation,90,0,25);
[leftErb,~]=AKerbError(estimate(:,mask,1),reference(:,mask,1),[50,22050],44100);
rightErb=AKerbError(estimate(:,mask,2),reference(:,mask,2),[50,22050],44100);
fullWeights=normalized_weights(weights(mask));
leftWeights=normalized_weights(weights(left25)); rightWeights=normalized_weights(weights(right25));
values(1)=0.5*(mean(abs(leftErb)*fullWeights)+mean(abs(rightErb)*fullWeights));
values(2)=0.5*(mean(abs(leftErb(:,left25(mask)))*leftWeights)+ ...
    mean(abs(rightErb(:,right25(mask)))*rightWeights));
assert(all(isfinite(values)) && nnz(left25)>0 && nnz(right25)>0);
end

function mask=great_circle_mask(azimuth,elevation,centerAz,centerEl,radius)
dotProduct=sind(elevation).*sind(centerEl).*cosd(azimuth-centerAz)+cosd(elevation).*cosd(centerEl);
mask=acosd(min(1,max(-1,dotProduct)))<=radius+1e-10;
end

function values=normalized_weights(values)
values=values(:); values=values/sum(values);
end
