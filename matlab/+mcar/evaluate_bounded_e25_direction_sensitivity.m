function evaluate_bounded_e25_direction_sensitivity(subjectLimit, outputName)
%EVALUATE_BOUNDED_E25_DIRECTION_SENSITIVITY Strict Q14/Q26/Q50 validation.

if nargin < 1 || isempty(subjectLimit), subjectLimit = inf; end
if nargin < 2 || isempty(outputName)
    outputName = 'sonicom_bounded_e25_input_direction_sensitivity_v1';
end
validateattributes(subjectLimit, {'numeric'}, {'scalar', 'positive'});
validateattributes(outputName, {'char', 'string'}, {'scalartext'});

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
configPath = fullfile(projectRoot, 'configs', 'experiments', ...
    'sonicom_bounded_e25_input_direction_sensitivity_v1.json');
config = jsondecode(fileread(configPath));
assert(string(config.dataset.split) == "val" && ...
    ~config.inference.test_access_allowed, 'Evaluation must remain validation-only.');
artifactRoot = fullfile(projectRoot, 'artifacts', 'sparsity', config.output_name);
inferenceReport = jsondecode(fileread(fullfile(artifactRoot, 'inference_report.json')));
assert(string(inferenceReport.status) == "completed" && ...
    inferenceReport.test_subject_count_read == 0 && ...
    inferenceReport.q26_reproduction_max_abs_error_db <= ...
        inferenceReport.q26_reproduction_tolerance_db, ...
    'A passed Q26 reproduction report is required.');

splitTable = readtable(fullfile(projectRoot, config.dataset.split_file), ...
    'TextType', 'string');
subjects = splitTable(splitTable.split == "val", :);
assert(height(subjects) == config.dataset.subject_count, ...
    'Expected 44 validation subjects.');
subjects = subjects(1:min(height(subjects), floor(subjectLimit)), :);
gridTable = readtable(fullfile(projectRoot, config.dataset.sparse_grid_file), ...
    'TextType', 'string');
counts = double(config.dataset.direction_counts(:).');
assert(isequal(counts, [14, 26, 50]), 'Frozen direction counts changed.');
q50 = double(gridTable.source_index_zero_based( ...
    gridTable.direction_count == 50)) + 1;
fixedMask = true(config.dataset.reference_direction_count, 1);
fixedMask(q50) = false;
assert(sum(fixedMask) == config.dataset.fixed_evaluation_direction_count, ...
    'Fixed Q50-excluded evaluation mask changed.');

supdeqDir = fullfile(projectRoot, 'external', 'SUpDEq');
originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir); supdeq_start; cd(projectRoot);
metricIds = ["FullSphereERB", "Contralateral25ERB", ...
    "ContralateralHighFrequency", "HorizontalILDMAE"];
metricLabels = ["Full-sphere ERB", "Contralateral 25-deg ERB", ...
    "Contralateral HF", "Horizontal ILD MAE"];
metricLong = table();
qualityChecks = table();

for subjectIndex = 1:height(subjects)
    subjectLabel = subjects.subject_id(subjectIndex);
    subjectId = sscanf(char(subjectLabel), 'P%d');
    for count = counts
        levelRoot = fullfile(artifactRoot, 'subjects', char(subjectLabel), ...
            sprintf('q%d', count));
        cachePath = fullfile(levelRoot, 'cache.mat');
        predictionPath = fullfile(levelRoot, 'bounded_prediction.h5');
        assert(isfile(cachePath) && isfile(predictionPath), ...
            'Missing Q%d artifact for %s.', count, subjectLabel);
        loaded = load(cachePath, 'cache'); cache = loaded.cache;
        assert(cache.directionCount == count && string(cache.split) == "val", ...
            'Cache provenance mismatch for %s Q%d.', subjectLabel, count);
        assert(string(h5readatt(predictionPath, '/', 'split')) == "val" && ...
            double(h5readatt(predictionPath, '/', 'sparse_direction_count')) == count, ...
            'Prediction provenance mismatch for %s Q%d.', subjectLabel, count);

        frequencyHz = double(cache.frequencyHz(cache.frequencyMask));
        referenceDb = selected_db(cache.referenceLeft, cache.referenceRight, ...
            cache.frequencyMask);
        referenceHrir = double(cache.referenceHrir);
        referenceIld = calculate_ild(referenceHrir(:, :, 1), ...
            referenceHrir(:, :, 2));
        directionFeatures = build_direction_features(cache.referenceGrid);
        [selectedDb, reconstructedHrir] = load_bounded(predictionPath, cache);
        values = strict_metrics(selectedDb, reconstructedHrir, referenceDb, ...
            referenceHrir, referenceIld, directionFeatures, fixedMask, ...
            frequencyHz, double(cache.samplingRateHz));
        for metricIndex = 1:numel(metricIds)
            metricLong = [metricLong; table(subjectLabel, subjectId, count, ...
                metricIds(metricIndex), metricLabels(metricIndex), ...
                values(metricIndex), 'VariableNames', {'SubjectLabel', ...
                'SubjectID', 'DirectionCount', 'Metric', 'MetricLabel', ...
                'Value_dB'})]; %#ok<AGROW>
        end
        qualityChecks = [qualityChecks; table(subjectLabel, subjectId, count, ...
            sum(fixedMask), sum(fixedMask & abs(directionFeatures(2, :).') <= 1e-9), ...
            all(isfinite(selectedDb), 'all'), ...
            'VariableNames', {'SubjectLabel', 'SubjectID', 'DirectionCount', ...
            'EvaluationDirectionCount', 'HorizontalDirectionCount', ...
            'AllFinite'})]; %#ok<AGROW>
        fprintf('Bounded E25 direction sensitivity [%d/%d] %s Q%d\n', ...
            (subjectIndex - 1) * numel(counts) + find(counts == count), ...
            height(subjects) * numel(counts), subjectLabel, count);
    end
end

replicates = double(config.statistics.bootstrap_replicates);
seed = double(config.statistics.bootstrap_seed);
aggregate = aggregate_metrics(metricLong, metricIds, counts, replicates, seed);
effects = direction_effects(metricLong, metricIds, replicates, seed);
outputRoot = fullfile(projectRoot, 'results', char(outputName));
assert(~isfolder(outputRoot), 'Refusing to overwrite %s', outputRoot);
mkdir(outputRoot); mkdir(fullfile(outputRoot, 'figures'));
writetable(metricLong, fullfile(outputRoot, 'metric_long.csv'));
writetable(aggregate, fullfile(outputRoot, 'aggregate_metrics.csv'));
writetable(effects, fullfile(outputRoot, 'paired_direction_effects.csv'));
writetable(qualityChecks, fullfile(outputRoot, 'quality_checks.csv'));
plot_sensitivity(aggregate, metricIds, metricLabels, ...
    fullfile(outputRoot, 'figures'));
summary = struct('schema_version', '1.0', 'status', 'completed', ...
    'split', 'val', 'subject_count', height(subjects), ...
    'direction_counts', counts, 'center_direction_count', 26, ...
    'fixed_evaluation_direction_count', sum(fixedMask), ...
    'fixed_evaluation_mask_policy', 'exclude all Q50 input directions', ...
    'bootstrap_replicates', replicates, 'bootstrap_seed', seed, ...
    'q26_reproduction_max_abs_error_db', ...
        inferenceReport.q26_reproduction_max_abs_error_db, ...
    'test_subject_count_read', 0, 'all_finite', ...
        all(isfinite(metricLong.Value_dB)) && all(qualityChecks.AllFinite), ...
    'aggregate', table2struct(aggregate), ...
    'paired_direction_effects', table2struct(effects));
write_json(fullfile(outputRoot, 'summary.json'), summary);
fprintf('Bounded E25 direction sensitivity complete: %s\n', outputRoot);
clear restoreDirectory;
end

function selectedDb = selected_db(left, right, mask)
spectra = cat(3, double(left), double(right));
selectedDb = 20 * log10(max(abs(permute( ...
    spectra(:, mask, :), [2, 1, 3])), 1e-10));
end

function features = build_direction_features(grid)
azimuth = double(grid(:, 1)); elevation = 90 - double(grid(:, 2));
features = [azimuth.'; elevation.'; ...
    (cosd(elevation) .* cosd(azimuth)).'; ...
    (cosd(elevation) .* sind(azimuth)).'; sind(elevation).'; ...
    double(grid(:, 3)).'];
end

function [selectedDb, hrir] = load_bounded(path, cache)
residualDb = double(h5read(path, '/predicted_residual_db'));
assert(isequal(size(residualDb), [sum(cache.frequencyMask), 793, 2]) && ...
    all(isfinite(residualDb), 'all'), 'Unexpected Bounded residual shape.');
spectra = cat(3, double(cache.mcaLeft), double(cache.mcaRight));
selected = permute(spectra(:, cache.frequencyMask, :), [2, 1, 3]);
phase = angle(selected);
mcaDb = 20 * log10(max(abs(selected), 1e-10));
selectedDb = mcaDb + residualDb;
for ear = 1:2
    earSpectrum = spectra(:, :, ear).';
    earSpectrum(cache.frequencyMask, :) = 10 .^ ...
        (selectedDb(:, :, ear) / 20) .* exp(1i * phase(:, :, ear));
    spectra(:, :, ear) = earSpectrum.';
end
hrir = spectra_to_hrir(spectra);
end

function hrir = spectra_to_hrir(spectra)
hrir = zeros(256, 793, 2);
for ear = 1:2
    bothSided = AKsingle2bothSidedSpectrum(spectra(:, :, ear).');
    oversized = real(ifft(bothSided));
    hrir(:, :, ear) = oversized(1:256, :);
end
end

function values = strict_metrics(selectedDb, hrir, referenceDb, ...
        referenceHrir, referenceIld, features, fixedMask, frequencyHz, samplingRateHz)
azimuth = features(1, :).'; elevation = features(2, :).';
y = features(4, :).'; weights = features(6, :).';
left25 = fixedMask & great_circle_mask(azimuth, elevation, 270, 0, 25);
right25 = fixedMask & great_circle_mask(azimuth, elevation, 90, 0, 25);
leftHemisphere = fixedMask & y < -1e-12;
rightHemisphere = fixedMask & y > 1e-12;
horizontal = fixedMask & abs(elevation) <= 1e-9;
[leftErb, ~] = AKerbError(hrir(:, fixedMask, 1), ...
    referenceHrir(:, fixedMask, 1), [50, samplingRateHz / 2], samplingRateHz);
rightErb = AKerbError(hrir(:, fixedMask, 2), ...
    referenceHrir(:, fixedMask, 2), [50, samplingRateHz / 2], samplingRateHz);
fullWeights = normalized_weights(weights(fixedMask));
values(1) = 0.5 * (mean(abs(leftErb) * fullWeights) + ...
    mean(abs(rightErb) * fullWeights));
values(2) = 0.5 * (mean(abs(leftErb(:, left25(fixedMask))) * ...
    normalized_weights(weights(left25))) + ...
    mean(abs(rightErb(:, right25(fixedMask))) * ...
    normalized_weights(weights(right25))));
values(3) = 0.5 * (high_frequency_error(selectedDb(:, :, 1), ...
    referenceDb(:, :, 1), frequencyHz, leftHemisphere, weights) + ...
    high_frequency_error(selectedDb(:, :, 2), referenceDb(:, :, 2), ...
    frequencyHz, rightHemisphere, weights));
methodIld = calculate_ild(hrir(:, :, 1), hrir(:, :, 2));
values(4) = mean(abs(methodIld(horizontal) - referenceIld(horizontal)));
end

function aggregate = aggregate_metrics(metricLong, metricIds, counts, replicates, seed)
rng(seed, 'twister'); aggregate = table();
for metric = metricIds
    for count = counts
        values = metricLong.Value_dB(metricLong.Metric == metric & ...
            metricLong.DirectionCount == count);
        indices = randi(numel(values), numel(values), replicates);
        means = mean(values(indices), 1); interval = prctile(means, [2.5, 97.5]);
        aggregate = [aggregate; table(count, metric, numel(values), mean(values), ...
            std(values), interval(1), interval(2), 'VariableNames', ...
            {'DirectionCount', 'Metric', 'SubjectCount', 'Mean_dB', 'Std_dB', ...
            'Bootstrap95Lower_dB', 'Bootstrap95Upper_dB'})]; %#ok<AGROW>
    end
end
end

function effects = direction_effects(metricLong, metricIds, replicates, seed)
rng(seed, 'twister'); effects = table();
for metric = metricIds
    center = sortrows(metricLong(metricLong.Metric == metric & ...
        metricLong.DirectionCount == 26, :), 'SubjectID');
    for count = [14, 50]
        current = sortrows(metricLong(metricLong.Metric == metric & ...
            metricLong.DirectionCount == count, :), 'SubjectID');
        assert(isequal(current.SubjectID, center.SubjectID), ...
            'Listener alignment failed.');
        difference = current.Value_dB - center.Value_dB;
        indices = randi(numel(difference), numel(difference), replicates);
        means = mean(difference(indices), 1); interval = prctile(means, [2.5, 97.5]);
        effects = [effects; table(count, 26, metric, mean(difference), ...
            interval(1), interval(2), sum(difference < 0), sum(difference == 0), ...
            sum(difference > 0), 'VariableNames', {'DirectionCount', ...
            'ReferenceDirectionCount', 'Metric', 'QMinusQ26Mean_dB', ...
            'Bootstrap95Lower_dB', 'Bootstrap95Upper_dB', 'QWins', 'Ties', ...
            'QLosses'})]; %#ok<AGROW>
    end
end
end

function plot_sensitivity(aggregate, metricIds, metricLabels, figuresRoot)
figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1200, 760]);
layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for index = 1:numel(metricIds)
    nexttile; rows = sortrows(aggregate(aggregate.Metric == metricIds(index), :), ...
        'DirectionCount');
    lower = rows.Mean_dB - rows.Bootstrap95Lower_dB;
    upper = rows.Bootstrap95Upper_dB - rows.Mean_dB;
    errorbar(rows.DirectionCount, rows.Mean_dB, lower, upper, '-o', ...
        'LineWidth', 1.6, 'MarkerFaceColor', [0.15, 0.42, 0.72], ...
        'Color', [0.15, 0.42, 0.72]); grid on; box on;
    xticks(rows.DirectionCount); xlabel('Observed directions'); ylabel('Error (dB)');
    title(metricLabels(index));
end
title(layout, 'Bounded E25 input-direction sensitivity (44 validation listeners)');
exportgraphics(figureHandle, fullfile(figuresRoot, ...
    'bounded_e25_direction_sensitivity.png'), 'Resolution', 300);
exportgraphics(figureHandle, fullfile(figuresRoot, ...
    'bounded_e25_direction_sensitivity.pdf'), 'ContentType', 'vector');
close(figureHandle);
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
value = mean(abs(estimateDb(frequencyMask, mask) - ...
    referenceDb(frequencyMask, mask)) * normalized_weights(weights(mask)));
end

function write_json(path, value)
encoded = jsonencode(value);
handle = fopen(path, 'w', 'n', 'UTF-8');
fprintf(handle, '%s\n', encoded); fclose(handle);
end
