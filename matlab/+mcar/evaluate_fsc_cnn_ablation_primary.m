function evaluate_fsc_cnn_ablation_primary(manifestPath)
%EVALUATE_FSC_CNN_ABLATION_PRIMARY Frozen Q26 validation ERB metrics.
%
% Evaluates only the six checkpoint-specific prediction roots frozen in the
% FSC CNN-ablation manifest.  The metric implementation is the same
% AKerbError/reconstruction path used by the formal Stage-D evaluator.

arguments
    manifestPath (1, 1) string
end

manifest = jsondecode(fileread(manifestPath));
assert(string(manifest.status) == "frozen");
assert(string(manifest.dataset.split) == "val");
assert(manifest.dataset.subject_count == 44);
assert(manifest.dataset.test_access_allowed == false);

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
datasetRoot = fullfile(projectRoot, manifest.dataset.root);
sofaRoot = fullfile(projectRoot, manifest.dataset.reference_sofa_root);
splitFile = fullfile(projectRoot, manifest.dataset.split_csv);
outputRoot = fullfile(projectRoot, manifest.output_root);
outputFile = fullfile(outputRoot, 'primary_subject_level.csv');
assert(~isfile(outputFile), 'Refusing to overwrite %s', outputFile);
if ~isfolder(outputRoot)
    mkdir(outputRoot);
end

supdeqDir = fullfile(projectRoot, 'external', 'SUpDEq');
originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir);
supdeq_start;
cd(projectRoot);

splitTable = readtable(splitFile, 'TextType', 'string');
subjects = splitTable(splitTable.split == "val", :);
assert(height(subjects) == 44, 'Expected exactly 44 validation subjects.');
methods = manifest.predictions;
assert(numel(methods) == 6, 'Expected three paired E130/E190 prediction pairs.');
metricIds = ["FullSphereERB", "Contralateral25ERB"];
rows = table();
quality = table();

for subjectIndex = 1:height(subjects)
    label = subjects.subject_id(subjectIndex);
    subjectId = sscanf(char(label), 'P%d');
    sourceFile = fullfile(datasetRoot, 'subjects', char(label), 'q26.h5');
    sofaFile = fullfile(sofaRoot, sprintf('%s_FreeFieldCompMinPhase_44kHz.sofa', label));
    assert(string(h5readatt(sourceFile, '/', 'split')) == "val");
    referenceDb = double(h5read(sourceFile, '/reference_logmag_db'));
    mcaDb = double(h5read(sourceFile, '/mca_logmag_db'));
    features = double(h5read(sourceFile, '/direction_features'));
    evaluationMask = logical(h5read(sourceFile, '/interpolation_evaluation_mask'));
    evaluationMask = evaluationMask(:);
    assert(sum(evaluationMask) == 767);
    phase = double(h5read(sourceFile, '/strict_ild/mca_selected_phase_rad'));
    outsideReal = double(h5read(sourceFile, '/strict_ild/mca_outside_real'));
    outsideImag = double(h5read(sourceFile, '/strict_ild/mca_outside_imag'));
    selectedIndices = double(h5read(sourceFile, ...
        '/strict_ild/selected_bin_indices_zero_based')) + 1;
    selectedIndices = selectedIndices(:);
    outsideIndices = double(h5read(sourceFile, ...
        '/strict_ild/outside_bin_indices_zero_based')) + 1;
    outsideIndices = outsideIndices(:);
    hrirLength = double(h5readatt(sourceFile, '/strict_ild', 'hrir_length'));
    samplingRateHz = double(h5readatt(sourceFile, '/', 'sampling_rate_hz'));

    sofa = SOFAload(sofaFile);
    referenceLeft = squeeze(double(sofa.Data.IR(:, 1, :))).';
    referenceRight = squeeze(double(sofa.Data.IR(:, 2, :))).';
    azimuth = features(1, :).';
    elevation = features(2, :).';
    solidAngleWeight = features(6, :).';
    left25 = evaluationMask & great_circle_mask(azimuth, elevation, 270, 0, 25);
    right25 = evaluationMask & great_circle_mask(azimuth, elevation, 90, 0, 25);

    for methodIndex = 1:numel(methods)
        method = methods(methodIndex);
        predictionFile = fullfile(projectRoot, method.root, 'subjects', ...
            char(label), 'prediction.h5');
        assert(isfile(predictionFile), 'Missing prediction: %s', predictionFile);
        assert(string(h5readatt(predictionFile, '/', 'split')) == "val");
        residualDb = double(h5read(predictionFile, '/predicted_residual_db'));
        assert(isequal(size(residualDb), size(referenceDb)) && ...
            all(isfinite(residualDb), 'all'));
        selectedDb = mcaDb + residualDb;
        leftHrir = reconstruct_hrir(selectedDb(:, :, 1), phase(:, :, 1), ...
            outsideReal(:, :, 1), outsideImag(:, :, 1), selectedIndices, ...
            outsideIndices, hrirLength);
        rightHrir = reconstruct_hrir(selectedDb(:, :, 2), phase(:, :, 2), ...
            outsideReal(:, :, 2), outsideImag(:, :, 2), selectedIndices, ...
            outsideIndices, hrirLength);
        values = erb_metrics(leftHrir, rightHrir, referenceLeft, referenceRight, ...
            samplingRateHz, solidAngleWeight, evaluationMask, left25, right25);
        for metricIndex = 1:numel(metricIds)
            rows = [rows; table(label, subjectId, double(method.seed), ...
                string(method.variant), metricIds(metricIndex), values(metricIndex), ...
                'VariableNames', {'SubjectLabel', 'SubjectID', 'Seed', ...
                'Variant', 'Metric', 'Value'})]; %#ok<AGROW>
        end
    end
    quality = [quality; table(label, subjectId, sum(evaluationMask), ...
        'VariableNames', {'SubjectLabel', 'SubjectID', ...
        'InterpolationDirectionCount'})]; %#ok<AGROW>
    fprintf('FSC CNN ablation primary [%d/44]: %s\n', subjectIndex, label);
end

assert(height(rows) == 44 * 6 * 2);
assert(all(isfinite(rows.Value)));
writetable(rows, outputFile);
writetable(quality, fullfile(outputRoot, 'primary_quality_checks.csv'));
summary = struct('schema_version', '1.0', 'status', 'completed', ...
    'split', 'val', 'subject_count', 44, 'seed_count', 3, ...
    'variant_count', 2, 'metric_count', 2, 'row_count', height(rows), ...
    'all_finite', true, 'test_subjects_read', 0, ...
    'metric_implementation', 'AKerbError via frozen Stage-D strict reconstruction');
write_json(fullfile(outputRoot, 'primary_summary.json'), summary);
clear restoreDirectory;
end

function values = erb_metrics(leftHrir, rightHrir, referenceLeft, referenceRight, ...
        samplingRateHz, weights, fullMask, left25, right25)
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
end

function hrir = reconstruct_hrir(selectedDb, phase, outsideReal, outsideImag, ...
        selectedIndices, outsideIndices, hrirLength)
spectrum = complex(zeros(numel(selectedIndices) + numel(outsideIndices), size(selectedDb, 2)));
spectrum(selectedIndices, :) = 10 .^ (selectedDb / 20) .* exp(1i * phase);
spectrum(outsideIndices, :) = outsideReal + 1i * outsideImag;
oversized = real(ifft(AKsingle2bothSidedSpectrum(spectrum)));
hrir = oversized(1:hrirLength, :);
end

function mask = great_circle_mask(azimuth, elevation, centerAzimuth, centerElevation, radius)
dotProduct = sind(elevation) .* sind(centerElevation) + ...
    cosd(elevation) .* cosd(centerElevation) .* cosd(azimuth - centerAzimuth);
mask = acosd(min(1, max(-1, dotProduct))) <= radius + 1e-10;
end

function weights = normalized_weights(weights)
weights = weights(:);
weights = weights / sum(weights);
end

function write_json(path, value)
handle = fopen(path, 'w', 'n', 'UTF-8');
fprintf(handle, '%s\n', jsonencode(value));
fclose(handle);
end
