function evaluate_fsc_matched_density_test_primary(outputRootOverride)
% Evaluate the four frozen FSC primary metrics on the authorized test split.
% This is a locked-test, diagonal-only evaluator for Q14/Q26/Q50.

if nargin < 1 || isempty(outputRootOverride)
    outputRootOverride = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
        'results', 'sonicom_fsc_matched_density_all_metrics_test_v1');
end
validateattributes(outputRootOverride, {'char','string'}, {'scalartext'});
outputRoot = char(outputRootOverride);
assert(~isfolder(outputRoot), 'Refusing to overwrite %s', outputRoot);

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
gridTable = readtable(fullfile(projectRoot, 'configs', 'data', ...
    'sonicom_nested_sparse_grid_q14_q26_q50_v1.csv'), 'TextType', 'string');
q50 = double(gridTable.source_index_zero_based(gridTable.direction_count == 50)) + 1;
fixedMask = true(793, 1); fixedMask(q50) = false;
assert(sum(fixedMask) == 743, 'Common evaluation mask must contain 743 directions');
splitTable = readtable(fullfile(projectRoot, 'configs', 'data', ...
    'sonicom_subject_split_v1.csv'), 'TextType', 'string');
subjects = splitTable.subject_id(splitTable.split == "test");
assert(numel(subjects) == 44, 'Expected 44 frozen test subjects');

counts = [14 26 50];
sourceRoots = { ...
    fullfile(projectRoot, 'data', 'processed', 'sonicom_fsc_q14_residual_test_v1'), ...
    fullfile(projectRoot, 'data', 'processed', 'sonicom_residual_q26_v1'), ...
    fullfile(projectRoot, 'data', 'processed', 'sonicom_fsc_q50_residual_test_v1')};
predictionRoots = { ...
    fullfile(projectRoot, 'artifacts', 'reconstruction', 'sonicom_fsc_q14_e190_ensemble_test'), ...
    fullfile(projectRoot, 'artifacts', 'reconstruction', 'sonicom_film_siren_spectral_cnn_final_e190_ensemble_test'), ...
    fullfile(projectRoot, 'artifacts', 'reconstruction', 'sonicom_fsc_q50_e190_ensemble_test')};
predictionNames = {"FSC-Q14@Q14", "FSC-Q26@Q26", "FSC-Q50@Q50"};
sofaRoot = fullfile(projectRoot, 'data', 'HRTF', 'sonicom_measured_ffcmp_minphase_44k1', 'subjects');

% Start SUpDEq so that AKerbError and AKsingle2bothSidedSpectrum are on path.
originalDir = pwd; cleanup = onCleanup(@() cd(originalDir)); %#ok<NASGU>
cd(fullfile(projectRoot, 'external', 'SUpDEq')); supdeq_start; cd(originalDir);

metricIds = ["FullSphereERB", "Contralateral25ERB", ...
    "ContralateralHighFrequency", "HorizontalILDMAE"];
metricLabels = ["Full-sphere ERB", "Contralateral 25-deg ERB", ...
    "Contralateral HF", "Horizontal ILD MAE"];
rows = table();
for subjectIndex = 1:numel(subjects)
    label = char(subjects(subjectIndex));
    for countIndex = 1:numel(counts)
        count = counts(countIndex);
        sourcePath = fullfile(sourceRoots{countIndex}, 'subjects', label, ...
            sprintf('q%d.h5', count));
        predictionPath = fullfile(predictionRoots{countIndex}, 'subjects', label, 'prediction.h5');
        assert(isfile(sourcePath) && isfile(predictionPath), 'Missing test artifact for %s Q%d', label, count);
        split = string(h5readatt(sourcePath, '/', 'split'));
        assert(split == "test", 'Non-test source encountered: %s', sourcePath);
        referenceDb = read_eardbf(sourcePath, '/reference_logmag_db');
        mcaDb = read_eardbf(sourcePath, '/mca_logmag_db');
        directions = double(h5read(sourcePath, '/direction_features')).';
        frequencyHz = double(h5read(sourcePath, '/frequency_hz')); frequencyHz = frequencyHz(:);
        interpolation = logical(h5read(sourcePath, '/interpolation_evaluation_mask')); interpolation = interpolation(:);
        sparseIndices = double(h5read(sourcePath, '/sparse_direction_indices_zero_based')); sparseIndices = sparseIndices(:) + 1;
        assert(isequal(size(referenceDb), [2 793 463]) && isequal(size(mcaDb), [2 793 463]));
        assert(isequal(size(directions), [793 6]) && numel(sparseIndices) == count);
        % The export keeps the frozen Q26 interpolation mask for metric layout;
        % the observed sparse set is recorded separately and must match Q.
        assert(sum(interpolation) == 767 && numel(sparseIndices) == count);
        assert(isequal(logical(h5readatt(sourcePath, '/', 'complete')), true));

        predictedResidual = read_eardbf(predictionPath, '/predicted_residual_db');
        assert(isequal(size(predictedResidual), [2 793 463]) && all(isfinite(predictedResidual), 'all'));
        assert(string(h5readatt(predictionPath, '/', 'split')) == "test");
        predictedDb = mcaDb + predictedResidual;
        predictedHrir = reconstruct_from_metadata(predictedDb, sourcePath);
        sofaPath = fullfile(sofaRoot, [label '_FreeFieldCompMinPhase_44kHz.sofa']);
        sofa = SOFAload(sofaPath);
        referenceHrir = permute(double(sofa.Data.IR), [3 1 2]);
        assert(isequal(size(referenceHrir), [256 793 2]));
        referenceIld = calculate_ild(referenceHrir(:,:,1), referenceHrir(:,:,2));
        values = strict_metrics(predictedDb, predictedHrir, referenceDb, referenceHrir, ...
            referenceIld, directions, fixedMask, frequencyHz);
        for metricIndex = 1:numel(metricIds)
            rows = [rows; table(string(label), sscanf(label, 'P%d'), count, ...
                predictionNames{countIndex}, metricIds(metricIndex), metricLabels(metricIndex), ...
                values(metricIndex), 'VariableNames', {'SubjectLabel','SubjectID','DirectionCount', ...
                'MethodLabel','Metric','MetricLabel','Value_dB'})]; %#ok<AGROW>
        end
        fprintf('FSC test primary [%d/%d] %s Q%d\n', ...
            (subjectIndex-1)*3+countIndex, numel(subjects)*3, label, count);
    end
end
assert(height(rows) == 44*3*4 && all(isfinite(rows.Value_dB)));
mkdir(outputRoot);
writetable(rows, fullfile(outputRoot, 'primary_subject_level.csv'));
aggregate = table();
for countIndex = 1:3
    count = counts(countIndex);
    for metricIndex = 1:numel(metricIds)
        values = rows.Value_dB(rows.DirectionCount == count & rows.Metric == metricIds(metricIndex));
        aggregate = [aggregate; table(predictionNames{countIndex}, count, metricIds(metricIndex), ...
            numel(values), mean(values), std(values), 'VariableNames', ...
            {'MethodLabel','DirectionCount','Metric','SubjectCount','Mean_dB','SD_dB'})]; %#ok<AGROW>
    end
end
writetable(aggregate, fullfile(outputRoot, 'primary_summary_mean_std.csv'));
summary = struct('status','completed','split','test','subject_count',44, ...
    'test_subject_count_read',44,'diagonal_counts',counts, ...
    'fixed_evaluation_direction_count',sum(fixedMask), ...
    'metric_row_count',height(rows),'aggregate_row_count',height(aggregate), ...
    'all_finite',all(isfinite(rows.Value_dB)), ...
    'metric_definition','Frozen MATLAB AKerbError/contralateral HF/horizontal ILD definitions');
fid = fopen(fullfile(outputRoot, 'primary_summary.json'), 'w'); fprintf(fid, '%s\n', jsonencode(summary)); fclose(fid);
fprintf('FSC test primary complete: %s\n', outputRoot);
end

function hrir = reconstruct_from_metadata(predictedDb, sourcePath)
    phase = read_eardbf(sourcePath, '/strict_ild/mca_selected_phase_rad');
    outsideReal = read_eardbf(sourcePath, '/strict_ild/mca_outside_real');
    outsideImag = read_eardbf(sourcePath, '/strict_ild/mca_outside_imag');
selected = double(h5read(sourcePath, '/strict_ild/selected_bin_indices_zero_based')) + 1;
outside = double(h5read(sourcePath, '/strict_ild/outside_bin_indices_zero_based')) + 1;
assert(isequal(size(phase), [2 793 463]) && numel(selected) == 463 && numel(outside) == 50);
spectrum = complex(zeros(2, 793, 513));
spectrum(:,:,selected) = 10 .^ (predictedDb / 20) .* exp(1i * phase);
spectrum(:,:,outside) = outsideReal + 1i * outsideImag;
hrir = zeros(256, 793, 2);
for ear = 1:2
    value = squeeze(spectrum(ear,:,:)).';
    fullSpectrum = AKsingle2bothSidedSpectrum(value);
    time = real(ifft(fullSpectrum));
    hrir(:,:,ear) = time(1:256,:);
end
end

function values = read_eardbf(path, dataset)
% HDF5 written by Python is [ear,direction,frequency]; MATLAB h5read exposes
% the reversed dimension order for these files.
raw = double(h5read(path, dataset));
values = permute(raw, [3 2 1]);
end

function values = strict_metrics(selectedDb, hrir, referenceDb, referenceHrir, referenceIld, directions, fixedMask, frequencyHz)
azimuth = directions(:,1); elevation = directions(:,2); y = directions(:,4); weights = directions(:,6);
left25 = fixedMask & great_circle_mask(azimuth, elevation, 270, 0, 25);
right25 = fixedMask & great_circle_mask(azimuth, elevation, 90, 0, 25);
leftHemisphere = fixedMask & y < -1e-12; rightHemisphere = fixedMask & y > 1e-12;
[leftErb, ~] = AKerbError(hrir(:,fixedMask,1), referenceHrir(:,fixedMask,1), [50, 22050], 44100);
rightErb = AKerbError(hrir(:,fixedMask,2), referenceHrir(:,fixedMask,2), [50, 22050], 44100);
fixedWeights = normalized_weights(weights(fixedMask));
left25Weights = normalized_weights(weights(left25)); right25Weights = normalized_weights(weights(right25));
values(1) = 0.5 * (mean(abs(leftErb) * fixedWeights) + mean(abs(rightErb) * fixedWeights));
values(2) = 0.5 * (mean(abs(leftErb(:,left25(fixedMask))) * left25Weights) + ...
    mean(abs(rightErb(:,right25(fixedMask))) * right25Weights));
values(3) = 0.5 * (high_frequency_error(squeeze(selectedDb(1,:,:)), squeeze(referenceDb(1,:,:)), frequencyHz, leftHemisphere, weights) + ...
    high_frequency_error(squeeze(selectedDb(2,:,:)), squeeze(referenceDb(2,:,:)), frequencyHz, rightHemisphere, weights));
ild = calculate_ild(hrir(:,:,1), hrir(:,:,2));
horizontal = fixedMask & abs(elevation) <= 1e-9;
values(4) = mean(abs(ild(horizontal)-referenceIld(horizontal)));
assert(all(isfinite(values)));
end

function value = high_frequency_error(estimate, reference, frequencyHz, directionMask, weights)
mask = frequencyHz > 10000 & frequencyHz <= 22050;
errorDb = abs(estimate(directionMask,mask)-reference(directionMask,mask));
w = normalized_weights(weights(directionMask));
value = mean(w.' * errorDb);
end

function mask = great_circle_mask(azimuth, elevation, centerAz, centerEl, radius)
dotProduct = sind(elevation).*sind(centerEl).*cosd(azimuth-centerAz) + ...
    cosd(elevation).*cosd(centerEl);
mask = acosd(min(1,max(-1,dotProduct))) <= radius + 1e-10;
end

function values = normalized_weights(values)
values = values(:); values = values / sum(values);
end

function ild = calculate_ild(leftHrir, rightHrir)
ild = 10*log10(sum(abs(leftHrir).^2,1)./sum(abs(rightHrir).^2,1)).';
end
