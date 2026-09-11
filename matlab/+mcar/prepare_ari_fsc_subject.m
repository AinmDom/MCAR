function prepare_ari_fsc_subject(preparedFile, outputFile)
%PREPARE_ARI_FSC_SUBJECT Create one frozen ARI MCA/residual HDF5 input.
% This isolated adapter accepts only a previously prepared train/validation
% waveform file. It intentionally refuses the test split before reading HRIR.

preparedFile = char(preparedFile);
outputFile = char(outputFile);
assert(isfile(preparedFile), 'Prepared waveform file not found: %s', preparedFile);
assert(~isfile(outputFile), 'Refusing to overwrite feature file: %s', outputFile);
split = string(h5readatt(preparedFile, '/', 'split'));
assert(any(split == ["train", "val"]), ...
    'TEST ACCESS REFUSED: ARI MCA feature generator accepts only train or val.');
assert(double(h5readatt(preparedFile, '/', 'test_subjects_read')) == 0, ...
    'Prepared input violates the frozen zero-test-access policy.');

hrir = double(h5read(preparedFile, '/hrir')); % MATLAB layout [time, ear, direction]
features = double(h5read(preparedFile, '/direction_features')); % [feature, direction]
q26 = double(h5read(preparedFile, '/q26_source_indices_zero_based')) + 1;
samplingRateHz = double(h5readatt(preparedFile, '/', 'sampling_rate_hz'));
assert(isequal(size(hrir), [256, 2, 1550]), 'Unexpected prepared ARI HRIR shape.');
assert(isequal(size(features), [6, 1550]), 'Unexpected ARI direction feature shape.');
assert(numel(q26) == 26 && numel(unique(q26)) == 26, 'Invalid frozen ARI Q26.');
assert(samplingRateHz == 44100 && all(isfinite(hrir), 'all') && all(isfinite(features), 'all'), ...
    'Invalid ARI prepared input.');

projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
supdeqDir = fullfile(projectRoot, 'external', 'SUpDEq');
originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir)); %#ok<NASGU>
cd(supdeqDir);
supdeq_start;

nfft = 1024;
fftOversize = 4;
singleSidedCount = nfft / 2 + 1;
referenceSpectrum = fft(hrir, nfft, 1);
referenceLeft = squeeze(referenceSpectrum(1:singleSidedCount, 1, :)).';
referenceRight = squeeze(referenceSpectrum(1:singleSidedCount, 2, :)).';
frequencyHz = (0:(singleSidedCount - 1)).' * samplingRateHz / nfft;
frequencyMask = frequencyHz >= 50 & frequencyHz <= 20000;
assert(nnz(frequencyMask) == 463, 'Frozen frequency representation must have 463 bins.');
referenceGrid = [features(1, :).', 90 - features(2, :).', features(6, :).'];
sparseHrtf = struct('HRTF_L', referenceLeft(q26, :), ...
    'HRTF_R', referenceRight(q26, :), 'f', frequencyHz, ...
    'fs', samplingRateHz, 'Nmax', 3, 'FFToversize', fftOversize, ...
    'samplingGrid', referenceGrid(q26, 1:2));
mca = supdeq_interpHRTF(sparseHrtf, referenceGrid, ...
    'SUpDEq', 'SH', inf, 0.09, 1e-2, true, 0, true, 'fadeDown');

referenceSpectrumSelected = cat(3, referenceLeft(:, frequencyMask).', ...
    referenceRight(:, frequencyMask).');
mcaSpectrumSelected = cat(3, mca.HRTF_L(:, frequencyMask).', ...
    mca.HRTF_R(:, frequencyMask).');
correctionSpectrum = mca.p.corrFilt_lim(frequencyMask, :, :);
referenceDb = single(20 * log10(max(abs(referenceSpectrumSelected), 1e-10)));
mcaDb = single(20 * log10(max(abs(mcaSpectrumSelected), 1e-10)));
correctionDb = single(20 * log10(max(abs(correctionSpectrum), 1e-10)));
targetDb = referenceDb - mcaDb;
assert(isequal(size(mcaDb), [463, 1550, 2]) && all(isfinite(mcaDb), 'all') && ...
    all(isfinite(referenceDb), 'all') && all(isfinite(correctionDb), 'all'), ...
    'Invalid MCA spectral tensors.');

fullMca = cat(3, mca.HRTF_L.', mca.HRTF_R.'); % [frequency, direction, ear]
selectedPhase = single(angle(fullMca(frequencyMask, :, :)));
outside = fullMca(~frequencyMask, :, :);
leftEnergy = squeeze(sum(hrir(:, 1, :).^2, 1));
rightEnergy = squeeze(sum(hrir(:, 2, :).^2, 1));
referenceIld = single(10 * log10(leftEnergy ./ rightEnergy));
referenceIld = referenceIld(:);
assert(isequal(size(referenceIld), [1550, 1]) && all(isfinite(referenceIld), 'all'), ...
    'Invalid reference HRIR ILD.');

partial = [outputFile '.partial'];
if isfile(partial), delete(partial); end
write_spectral(partial, '/mca_logmag_db', mcaDb);
write_spectral(partial, '/reference_logmag_db', referenceDb);
write_spectral(partial, '/correction_logmag_db', correctionDb);
write_spectral(partial, '/target_residual_db', targetDb);
h5create(partial, '/direction_features', size(single(features)), 'Datatype', 'single');
h5write(partial, '/direction_features', single(features));
h5create(partial, '/frequency_hz', size(single(frequencyHz(frequencyMask).')), 'Datatype', 'single');
h5write(partial, '/frequency_hz', single(frequencyHz(frequencyMask).'));
h5create(partial, '/sparse_direction_indices_zero_based', size(int32(q26 - 1)), 'Datatype', 'int32');
h5write(partial, '/sparse_direction_indices_zero_based', int32(q26 - 1));
mask = ones(1550, 1, 'uint8'); mask(q26) = 0;
h5create(partial, '/interpolation_evaluation_mask', size(mask), 'Datatype', 'uint8');
h5write(partial, '/interpolation_evaluation_mask', mask);
h5create(partial, '/strict_ild/mca_selected_phase_rad', size(selectedPhase), 'Datatype', 'single');
h5write(partial, '/strict_ild/mca_selected_phase_rad', selectedPhase);
h5create(partial, '/strict_ild/mca_outside_real', size(single(real(outside))), 'Datatype', 'single');
h5write(partial, '/strict_ild/mca_outside_real', single(real(outside)));
h5create(partial, '/strict_ild/mca_outside_imag', size(single(imag(outside))), 'Datatype', 'single');
h5write(partial, '/strict_ild/mca_outside_imag', single(imag(outside)));
h5create(partial, '/strict_ild/selected_bin_indices_zero_based', size(int32(find(frequencyMask) - 1)), 'Datatype', 'int32');
h5write(partial, '/strict_ild/selected_bin_indices_zero_based', int32(find(frequencyMask) - 1));
h5create(partial, '/strict_ild/outside_bin_indices_zero_based', size(int32(find(~frequencyMask) - 1)), 'Datatype', 'int32');
h5write(partial, '/strict_ild/outside_bin_indices_zero_based', int32(find(~frequencyMask) - 1));
h5create(partial, '/strict_ild/reference_ild_db', size(referenceIld), 'Datatype', 'single');
h5write(partial, '/strict_ild/reference_ild_db', referenceIld);
h5writeatt(partial, '/strict_ild', 'single_sided_frequency_count', int32(singleSidedCount));
h5writeatt(partial, '/strict_ild', 'hrir_length', int32(256));
subjectLabel = string(h5readatt(preparedFile, '/', 'subject_id'));
h5writeatt(partial, '/', 'complete', int32(1));
h5writeatt(partial, '/', 'subject_id', int32(str2double(extractAfter(subjectLabel, 2))));
h5writeatt(partial, '/', 'subject_label', char(subjectLabel));
h5writeatt(partial, '/', 'split', char(split));
h5writeatt(partial, '/', 'sparse_order', int32(3));
h5writeatt(partial, '/', 'sparse_direction_count', int32(26));
h5writeatt(partial, '/', 'strict_ild_metadata', int32(1));
h5writeatt(partial, '/', 'test_subjects_read', int32(0));
movefile(partial, outputFile, 'f');
end

function write_spectral(path, dataset, values)
h5create(path, dataset, size(values), 'Datatype', 'single');
h5write(path, dataset, values);
end
