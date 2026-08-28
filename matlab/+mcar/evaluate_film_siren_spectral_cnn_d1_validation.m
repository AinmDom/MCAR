function evaluate_film_siren_spectral_cnn_d1_validation(subjectLimit, outputName)
%EVALUATE_FILM_SIREN_SPECTRAL_CNN_D1_VALIDATION Strict residual-method screen.
% Compares D1 hybrid, its single-member parent, corrected FiLM ensemble, and
% MCAR v3.5.1 on validation only. Differences are HYBRID minus baseline.

if nargin < 1 || isempty(subjectLimit), subjectLimit = inf; end
if nargin < 2 || isempty(outputName)
    outputName = 'sonicom_film_siren_spectral_cnn_d1_e40_validation';
end
validateattributes(subjectLimit, {'numeric'}, {'scalar', 'positive'});
validateattributes(outputName, {'char', 'string'}, {'scalartext'});

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
datasetRoot = fullfile(projectRoot, 'data', 'processed', 'sonicom_residual_q26_v1');
sofaRoot = fullfile(projectRoot, 'data', 'HRTF', ...
    'sonicom_measured_ffcmp_minphase_44k1', 'subjects');
splitFile = fullfile(projectRoot, 'configs', 'data', 'sonicom_subject_split_v1.csv');
supdeqDir = fullfile(projectRoot, 'external', 'SUpDEq');
methodIds = ["HYBRID", "PARENT", "FILMENS", "MCAR"];
methodLabels = ["FiLM-SIREN + spectral CNN D1 E40", ...
    "FiLM-SIREN seed 20260821 E130", ...
    "FiLM-SIREN corrected E130 ensemble", "MCAR v3.5.1"];
predictionNames = [ ...
    "sonicom_film_siren_spectral_cnn_d1_seed20260821_e40_best_validation", ...
    "sonicom_film_siren_gl_final_d1d2_notch_seed20260821_e130_validation", ...
    "sonicom_film_siren_gl_final_d1d2_notch_e130_ensemble_validation", ...
    "sonicom_q26_validation_v351_previous30_b70"];
metricIds = ["FullSphereERB", "Contralateral25ERB", ...
    "ContralateralHighFrequency", "HorizontalILDMAE"];

assert(isfolder(supdeqDir) && isfolder(datasetRoot) && isfolder(sofaRoot));
predictionRoots = strings(size(predictionNames));
for index = 1:numel(predictionNames)
    predictionRoots(index) = fullfile(projectRoot, 'artifacts', ...
        'reconstruction', predictionNames(index));
    assert(isfolder(predictionRoots(index)), ...
        'Prediction root not found: %s', predictionRoots(index));
end
outputRoot = fullfile(projectRoot, 'results', char(outputName));
assert(~isfolder(outputRoot), 'Refusing to overwrite %s', outputRoot);
mkdir(outputRoot);

originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir); supdeq_start; cd(projectRoot);
splitTable = readtable(splitFile, 'TextType', 'string');
subjects = splitTable(splitTable.split == "val", :);
assert(height(subjects) == 44, 'Expected exactly 44 validation subjects.');
subjects = subjects(1:min(height(subjects), floor(subjectLimit)), :);
metricLong = table();
qualityChecks = table();

for subjectIndex = 1:height(subjects)
    label = subjects.subject_id(subjectIndex);
    subjectId = sscanf(char(label), 'P%d');
    sourceFile = fullfile(datasetRoot, 'subjects', char(label), 'q26.h5');
    sofaFile = fullfile(sofaRoot, sprintf('%s_FreeFieldCompMinPhase_44kHz.sofa', label));
    assert(string(h5readatt(sourceFile, '/', 'split')) == "val");
    referenceDb = double(h5read(sourceFile, '/reference_logmag_db'));
    mcaDb = double(h5read(sourceFile, '/mca_logmag_db'));
    features = double(h5read(sourceFile, '/direction_features'));
    frequencyHz = double(h5read(sourceFile, '/frequency_hz')); frequencyHz = frequencyHz(:);
    evaluationMask = logical(h5read(sourceFile, '/interpolation_evaluation_mask'));
    evaluationMask = evaluationMask(:);
    phase = double(h5read(sourceFile, '/strict_ild/mca_selected_phase_rad'));
    outsideReal = double(h5read(sourceFile, '/strict_ild/mca_outside_real'));
    outsideImag = double(h5read(sourceFile, '/strict_ild/mca_outside_imag'));
    selectedIndices = double(h5read(sourceFile, ...
        '/strict_ild/selected_bin_indices_zero_based')) + 1; selectedIndices = selectedIndices(:);
    outsideIndices = double(h5read(sourceFile, ...
        '/strict_ild/outside_bin_indices_zero_based')) + 1; outsideIndices = outsideIndices(:);
    referenceIld = double(h5read(sourceFile, '/strict_ild/reference_ild_db'));
    referenceIld = referenceIld(:);
    hrirLength = double(h5readatt(sourceFile, '/strict_ild', 'hrir_length'));
    samplingRateHz = double(h5readatt(sourceFile, '/', 'sampling_rate_hz'));

    sofa = SOFAload(sofaFile);
    referenceLeft = squeeze(double(sofa.Data.IR(:, 1, :))).';
    referenceRight = squeeze(double(sofa.Data.IR(:, 2, :))).';
    rawReferenceIld = calculate_ild(referenceLeft, referenceRight);
    referenceIldError = max(abs(rawReferenceIld - referenceIld));
    assert(referenceIldError <= 2e-4, 'Reference ILD mismatch for %s', label);
    azimuth = features(1, :).'; elevation = features(2, :).';
    y = features(4, :).'; solidAngleWeight = features(6, :).';
    left25 = evaluationMask & great_circle_mask(azimuth, elevation, 270, 0, 25);
    right25 = evaluationMask & great_circle_mask(azimuth, elevation, 90, 0, 25);
    leftHemisphere = evaluationMask & y < -1e-12;
    rightHemisphere = evaluationMask & y > 1e-12;
    horizontal = evaluationMask & abs(elevation) <= 1e-9;

    for methodIndex = 1:numel(methodIds)
        predictionFile = fullfile(predictionRoots(methodIndex), 'subjects', ...
            char(label), 'prediction.h5');
        assert(isfile(predictionFile), 'Missing prediction: %s', predictionFile);
        assert(string(h5readatt(predictionFile, '/', 'split')) == "val");
        residualDb = double(h5read(predictionFile, '/predicted_residual_db'));
        assert(isequal(size(residualDb), size(referenceDb)) && ...
            all(isfinite(residualDb), 'all'), 'Invalid residual for %s', label);
        selectedDb = mcaDb + residualDb;
        leftHrir = reconstruct_hrir(selectedDb(:, :, 1), phase(:, :, 1), ...
            outsideReal(:, :, 1), outsideImag(:, :, 1), selectedIndices, ...
            outsideIndices, hrirLength);
        rightHrir = reconstruct_hrir(selectedDb(:, :, 2), phase(:, :, 2), ...
            outsideReal(:, :, 2), outsideImag(:, :, 2), selectedIndices, ...
            outsideIndices, hrirLength);
        values = strict_metrics(selectedDb, leftHrir, rightHrir, referenceDb, ...
            referenceLeft, referenceRight, referenceIld, frequencyHz, ...
            samplingRateHz, solidAngleWeight, evaluationMask, left25, right25, ...
            leftHemisphere, rightHemisphere, horizontal);
        for metricIndex = 1:numel(metricIds)
            metricLong = [metricLong; table(label, subjectId, ...
                methodIds(methodIndex), methodLabels(methodIndex), ...
                metricIds(metricIndex), values(metricIndex), ...
                'VariableNames', {'SubjectLabel', 'SubjectID', 'Method', ...
                'MethodLabel', 'Metric', 'Value_dB'})]; %#ok<AGROW>
        end
    end
    qualityChecks = [qualityChecks; table(label, subjectId, referenceIldError, ...
        sum(evaluationMask), sum(horizontal), 'VariableNames', ...
        {'SubjectLabel', 'SubjectID', 'ReferenceILDMetadataMaxError_dB', ...
        'InterpolationDirectionCount', 'HorizontalDirectionCount'})]; %#ok<AGROW>
    fprintf('Stage-D D1 strict validation [%d/%d]: %s\n', ...
        subjectIndex, height(subjects), label);
end

aggregate = aggregate_metrics(metricLong, methodIds, metricIds);
pairwise = paired_statistics(metricLong, methodIds(2:end), metricIds, 10000, 20260828);
writetable(metricLong, fullfile(outputRoot, 'metric_long.csv'));
writetable(aggregate, fullfile(outputRoot, 'aggregate_metrics.csv'));
writetable(pairwise, fullfile(outputRoot, 'paired_bootstrap_comparisons.csv'));
writetable(qualityChecks, fullfile(outputRoot, 'quality_checks.csv'));
summary = struct('schema_version', '1.0', 'status', 'completed', 'split', 'val', ...
    'subject_count', height(subjects), 'test_subject_count_read', 0, ...
    'methods', methodIds, 'method_labels', methodLabels, 'metrics', metricIds, ...
    'bootstrap_replicates', 10000, 'bootstrap_seed', 20260828, ...
    'difference_definition', 'HYBRID minus baseline; negative favors HYBRID', ...
    'aggregate', table2struct(aggregate), ...
    'pairwise_comparisons', table2struct(pairwise));
write_json(fullfile(outputRoot, 'summary.json'), summary);
fprintf('Stage-D D1 strict validation complete: %s\n', outputRoot);
clear restoreDirectory;
end

function values = strict_metrics(selectedDb, leftHrir, rightHrir, referenceDb, ...
        referenceLeft, referenceRight, referenceIld, frequencyHz, samplingRateHz, ...
        weights, fullMask, left25, right25, leftHemisphere, rightHemisphere, horizontal)
[leftErb, ~] = AKerbError(leftHrir(:, fullMask), referenceLeft(:, fullMask), ...
    [50, samplingRateHz / 2], samplingRateHz);
rightErb = AKerbError(rightHrir(:, fullMask), referenceRight(:, fullMask), ...
    [50, samplingRateHz / 2], samplingRateHz);
fullWeights = normalized_weights(weights(fullMask));
values(1) = 0.5 * (mean(abs(leftErb) * fullWeights) + ...
    mean(abs(rightErb) * fullWeights));
leftWeights = normalized_weights(weights(left25));
rightWeights = normalized_weights(weights(right25));
values(2) = 0.5 * (mean(abs(leftErb(:, left25(fullMask))) * leftWeights) + ...
    mean(abs(rightErb(:, right25(fullMask))) * rightWeights));
values(3) = 0.5 * (high_frequency_error(selectedDb(:, :, 1), ...
    referenceDb(:, :, 1), frequencyHz, leftHemisphere, weights) + ...
    high_frequency_error(selectedDb(:, :, 2), referenceDb(:, :, 2), ...
    frequencyHz, rightHemisphere, weights));
methodIld = calculate_ild(leftHrir, rightHrir);
values(4) = mean(abs(methodIld(horizontal) - referenceIld(horizontal)));
end

function aggregate = aggregate_metrics(metricLong, methodIds, metricIds)
aggregate = table();
for metric = metricIds
    for method = methodIds
        values = metricLong.Value_dB(metricLong.Metric == metric & metricLong.Method == method);
        aggregate = [aggregate; table(method, metric, numel(values), mean(values), ...
            std(values), 'VariableNames', {'Method', 'Metric', 'SubjectCount', ...
            'Mean_dB', 'Std_dB'})]; %#ok<AGROW>
    end
end
end

function pairwise = paired_statistics(metricLong, baselines, metricIds, replicates, seed)
rng(seed, 'twister');
pairwise = table();
for metric = metricIds
    hybrid = metricLong.Value_dB(metricLong.Metric == metric & metricLong.Method == "HYBRID");
    for baselineId = baselines
        baseline = metricLong.Value_dB(metricLong.Metric == metric & ...
            metricLong.Method == baselineId);
        difference = hybrid - baseline;
        indices = randi(numel(difference), numel(difference), replicates);
        bootstrapMeans = mean(difference(indices), 1);
        interval = prctile(bootstrapMeans, [2.5, 97.5]);
        pairwise = [pairwise; table(metric, baselineId, mean(difference), ...
            interval(1), interval(2), sum(difference < 0), sum(difference == 0), ...
            sum(difference > 0), 'VariableNames', {'Metric', 'Baseline', ...
            'HybridMinusBaselineMean_dB', 'Bootstrap95Lower_dB', ...
            'Bootstrap95Upper_dB', 'HybridWins', 'Ties', 'HybridLosses'})]; %#ok<AGROW>
    end
end
end

function hrir = reconstruct_hrir(selectedDb, phase, outsideReal, outsideImag, ...
        selectedIndices, outsideIndices, hrirLength)
spectrum = complex(zeros(numel(selectedIndices) + numel(outsideIndices), size(selectedDb, 2)));
spectrum(selectedIndices, :) = 10 .^ (selectedDb / 20) .* exp(1i * phase);
spectrum(outsideIndices, :) = outsideReal + 1i * outsideImag;
oversized = real(ifft(AKsingle2bothSidedSpectrum(spectrum)));
hrir = oversized(1:hrirLength, :);
end

function ild = calculate_ild(left, right)
ild = (10 * log10(sum(abs(left) .^ 2, 1) ./ sum(abs(right) .^ 2, 1))).';
end

function mask = great_circle_mask(azimuth, elevation, centerAzimuth, centerElevation, radius)
dotProduct = sind(elevation) .* sind(centerElevation) + ...
    cosd(elevation) .* cosd(centerElevation) .* cosd(azimuth - centerAzimuth);
mask = acosd(min(1, max(-1, dotProduct))) <= radius + 1e-10;
end

function weights = normalized_weights(weights)
weights = weights(:); weights = weights / sum(weights);
end

function value = high_frequency_error(estimateDb, referenceDb, frequencyHz, mask, weights)
frequencyMask = frequencyHz > 10000 & frequencyHz <= 20000;
value = mean(abs(estimateDb(frequencyMask, mask) - referenceDb(frequencyMask, mask)) ...
    * normalized_weights(weights(mask)));
end

function write_json(path, value)
encoded = jsonencode(value);
handle = fopen(path, 'w', 'n', 'UTF-8');
fprintf(handle, '%s\n', encoded); fclose(handle);
end
