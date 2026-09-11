function prepare_ari_fsc_test_subject(preparedFile, outputFile)
%PREPARE_ARI_FSC_TEST_SUBJECT Create frozen MCA features from one derived test HRIR.
preparedFile = char(preparedFile); outputFile = char(outputFile);
assert(isfile(preparedFile) && ~isfile(outputFile), 'Invalid locked-test input/output.');
assert(string(h5readatt(preparedFile,'/','split')) == "test", 'Locked adapter requires test input.');
assert(double(h5readatt(preparedFile,'/','test_subjects_read')) == 1, ...
    'Raw test access provenance is missing.');
assert(double(h5readatt(preparedFile,'/','complete')) == 1, 'Prepared test input is incomplete.');

hrir = double(h5read(preparedFile,'/hrir'));
features = double(h5read(preparedFile,'/direction_features'));
q26 = double(h5read(preparedFile,'/q26_source_indices_zero_based')) + 1;
samplingRateHz = double(h5readatt(preparedFile,'/','sampling_rate_hz'));
assert(isequal(size(hrir),[256,2,1550]) && isequal(size(features),[6,1550]));
assert(numel(q26)==26 && numel(unique(q26))==26 && samplingRateHz==44100);
assert(all(isfinite(hrir),'all') && all(isfinite(features),'all'));

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
originalDir = pwd; restoreDirectory = onCleanup(@() cd(originalDir)); %#ok<NASGU>
cd(fullfile(projectRoot,'external','SUpDEq')); supdeq_start;
nfft=1024; singleSidedCount=513;
referenceSpectrum=fft(hrir,nfft,1);
referenceLeft=squeeze(referenceSpectrum(1:singleSidedCount,1,:)).';
referenceRight=squeeze(referenceSpectrum(1:singleSidedCount,2,:)).';
frequencyHz=(0:(singleSidedCount-1)).'*samplingRateHz/nfft;
frequencyMask=frequencyHz>=50 & frequencyHz<=20000;
assert(nnz(frequencyMask)==463);
referenceGrid=[features(1,:).',90-features(2,:).',features(6,:).'];
sparseHrtf=struct('HRTF_L',referenceLeft(q26,:),'HRTF_R',referenceRight(q26,:), ...
    'f',frequencyHz,'fs',samplingRateHz,'Nmax',3,'FFToversize',4, ...
    'samplingGrid',referenceGrid(q26,1:2));
mca=supdeq_interpHRTF(sparseHrtf,referenceGrid,'SUpDEq','SH',inf,0.09,1e-2,true,0,true,'fadeDown');
referenceSpectrumSelected=cat(3,referenceLeft(:,frequencyMask).',referenceRight(:,frequencyMask).');
mcaSpectrumSelected=cat(3,mca.HRTF_L(:,frequencyMask).',mca.HRTF_R(:,frequencyMask).');
correctionSpectrum=mca.p.corrFilt_lim(frequencyMask,:,:);
referenceDb=single(20*log10(max(abs(referenceSpectrumSelected),1e-10)));
mcaDb=single(20*log10(max(abs(mcaSpectrumSelected),1e-10)));
correctionDb=single(20*log10(max(abs(correctionSpectrum),1e-10)));
targetDb=referenceDb-mcaDb;
assert(isequal(size(mcaDb),[463,1550,2]) && all(isfinite(mcaDb),'all'));
fullMca=cat(3,mca.HRTF_L.',mca.HRTF_R.');
selectedPhase=single(angle(fullMca(frequencyMask,:,:)));
outside=fullMca(~frequencyMask,:,:);
referenceIld=single(10*log10(squeeze(sum(hrir(:,1,:).^2,1))./squeeze(sum(hrir(:,2,:).^2,1))));
referenceIld=referenceIld(:);

partial=[outputFile '.partial']; if isfile(partial), delete(partial); end
write_spectral(partial,'/reference_logmag_db',referenceDb);
write_spectral(partial,'/mca_logmag_db',mcaDb);
write_spectral(partial,'/correction_logmag_db',correctionDb);
write_spectral(partial,'/target_residual_db',targetDb);
h5create(partial,'/direction_features',size(features),'Datatype','double'); h5write(partial,'/direction_features',features);
h5create(partial,'/frequency_hz',size(frequencyHz(frequencyMask)),'Datatype','double'); h5write(partial,'/frequency_hz',frequencyHz(frequencyMask));
h5create(partial,'/sparse_direction_indices_zero_based',size(q26),'Datatype','int32'); h5write(partial,'/sparse_direction_indices_zero_based',int32(q26-1));
interpolation=true(1550,1); interpolation(q26)=false;
h5create(partial,'/interpolation_evaluation_mask',size(interpolation),'Datatype','uint8'); h5write(partial,'/interpolation_evaluation_mask',uint8(interpolation));
h5create(partial,'/strict_ild/mca_selected_phase_rad',size(selectedPhase),'Datatype','single'); h5write(partial,'/strict_ild/mca_selected_phase_rad',selectedPhase);
h5create(partial,'/strict_ild/mca_outside_real',size(outside),'Datatype','single'); h5write(partial,'/strict_ild/mca_outside_real',single(real(outside)));
h5create(partial,'/strict_ild/mca_outside_imag',size(outside),'Datatype','single'); h5write(partial,'/strict_ild/mca_outside_imag',single(imag(outside)));
selected=find(frequencyMask)-1; outsideIndices=find(~frequencyMask)-1;
h5create(partial,'/strict_ild/selected_bin_indices_zero_based',size(selected),'Datatype','int32'); h5write(partial,'/strict_ild/selected_bin_indices_zero_based',int32(selected));
h5create(partial,'/strict_ild/outside_bin_indices_zero_based',size(outsideIndices),'Datatype','int32'); h5write(partial,'/strict_ild/outside_bin_indices_zero_based',int32(outsideIndices));
h5create(partial,'/strict_ild/reference_ild_db',size(referenceIld),'Datatype','single'); h5write(partial,'/strict_ild/reference_ild_db',referenceIld);
h5writeatt(partial,'/strict_ild','single_sided_frequency_count',int32(513)); h5writeatt(partial,'/strict_ild','hrir_length',int32(256));
label=string(h5readatt(preparedFile,'/','subject_id'));
h5writeatt(partial,'/','subject_id',sscanf(char(label),'nh%d')); h5writeatt(partial,'/','subject_label',char(label));
h5writeatt(partial,'/','split','test'); h5writeatt(partial,'/','complete',int32(1));
h5writeatt(partial,'/','test_subjects_read',int32(1)); h5writeatt(partial,'/','tensor_layout','ear,direction,frequency');
movefile(partial,outputFile,'f');
end

function write_spectral(path,dataset,values)
h5create(path,dataset,size(values),'Datatype','single'); h5write(path,dataset,values);
end
