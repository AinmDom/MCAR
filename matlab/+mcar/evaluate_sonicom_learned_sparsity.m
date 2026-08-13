function evaluate_sonicom_learned_sparsity(subjectLimit, outputName)
%EVALUATE_SONICOM_LEARNED_SPARSITY Compare MCAR, FSP-AE, and RANF over Q.

if nargin < 1 || isempty(subjectLimit)
    subjectLimit = inf;
end
if nargin < 2 || isempty(outputName)
    outputName = 'sonicom_learned_methods_sparsity_v1';
end
validateattributes(subjectLimit, {'numeric'}, {'scalar', 'positive'});
validateattributes(outputName, {'char', 'string'}, {'scalartext'});

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
configPath = fullfile(projectRoot, 'configs', 'experiments', ...
    'sonicom_learned_methods_sparsity_v1.json');
config = jsondecode(fileread(configPath));
assert(any(string(config.status) == ["running", "completed"]), ...
    'Formal evaluation requires a running or completed frozen protocol.');
splitTable = readtable(fullfile(projectRoot, config.dataset.split_file), ...
    'TextType', 'string');
subjects = splitTable(splitTable.split == "test", :);
assert(height(subjects) == 44, 'Expected 44 frozen SONICOM test subjects.');
subjects = subjects(1:min(height(subjects), floor(subjectLimit)), :);
referenceTable = readtable(fullfile(projectRoot, ...
    config.dataset.reference_grid), 'TextType', 'string');
assert(height(referenceTable) == 793, 'Frozen reference grid was modified.');
fixedMask = logical(referenceTable.is_interpolation_evaluation);
assert(sum(fixedMask) == 767, 'Fixed evaluation mask must contain 767 targets.');
q26Indices = double(config.sparse_grid_selection.q26_source_indices_zero_based) + 1;
assert(~any(fixedMask(q26Indices)), 'Fixed mask must exclude all Q26 inputs.');

artifactRoot = fullfile(projectRoot, 'artifacts', 'sparsity', config.output_name);
validationPath = fullfile(artifactRoot, 'validation_report.json');
adaptationCostPath = fullfile(artifactRoot, 'ranf_adaptation_cost.json');
assert(isfile(validationPath) && ...
    string(jsondecode(fileread(validationPath)).status) == "passed", ...
    'A passed full-artifact validation report is required.');
assert(isfile(adaptationCostPath), ...
    'RANF adaptation-cost report is required before evaluation.');
subjectRoot = fullfile(artifactRoot, 'subjects');
ranfRoots = containers.Map({'6', '14', '26'}, { ...
    fullfile(projectRoot, 'artifacts', 'reconstruction', ...
        'sonicom_ranf_learned_sparsity_q6'), ...
    fullfile(projectRoot, 'artifacts', 'reconstruction', ...
        'sonicom_ranf_learned_sparsity_q14'), ...
    fullfile(projectRoot, 'artifacts', 'reconstruction', ...
        'sonicom_ranf_q26_final_test')});
outputRoot = fullfile(projectRoot, 'results', char(outputName));
figuresRoot = fullfile(outputRoot, 'figures');
if ~isfolder(figuresRoot)
    mkdir(figuresRoot);
end
supdeqDir = fullfile(projectRoot, 'external', 'SUpDEq');
originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir);
supdeq_start;
cd(projectRoot);

counts = double(config.dataset.direction_counts(:).');
methodIds = ["MCARv32", "FSPAE", "RANF"];
methodLabels = ["MCAR v3.2", "FSP-AE", "RANF"];
metricIds = ["MeasuredDomainERB", "Contralateral25ERB", ...
    "ContralateralHemisphereHighFrequency", "HorizontalILDMAE"];
metricLong = table();
qualityChecks = table();

for subjectIndex = 1:height(subjects)
    subjectLabel = subjects.subject_id(subjectIndex);
    subjectId = sscanf(char(subjectLabel), 'P%d');
    for count = counts
        levelRoot = fullfile(subjectRoot, char(subjectLabel), sprintf('q%d', count));
        cachePath = fullfile(levelRoot, 'cache.mat');
        mcarPath = fullfile(levelRoot, 'mcar_prediction.h5');
        fspPath = fullfile(levelRoot, 'fspae_prediction.h5');
        ranfPath = fullfile(ranfRoots(num2str(count)), 'subjects', ...
            char(subjectLabel), 'prediction.sofa');
        assert(isfile(cachePath) && isfile(mcarPath) && isfile(fspPath), ...
            'Missing learned prediction for %s Q%d.', subjectLabel, count);
        assert(isfile(ranfPath), 'Missing RANF prediction: %s', ranfPath);
        loaded = load(cachePath, 'cache');
        cache = loaded.cache;
        assert(cache.directionCount == count && cache.split == "test", ...
            'Cache protocol mismatch for %s Q%d.', subjectLabel, count);
        frequencyHz = double(cache.frequencyHz(cache.frequencyMask));
        samplingRateHz = double(cache.samplingRateHz);
        referenceHrir = double(cache.referenceHrir);
        referenceDb = selected_db(cache.referenceLeft, cache.referenceRight, ...
            cache.frequencyMask);
        directionFeatures = build_direction_features(cache.referenceGrid);
        referenceIld = calculate_ild(referenceHrir(:, :, 1), ...
            referenceHrir(:, :, 2));

        [mcarDb, mcarHrir] = load_mcar(mcarPath, cache);
        [fspDb, fspHrir, fspFrequencyError] = load_fsp( ...
            fspPath, cache.frequencyMask, cache.frequencyHz);
        [ranfDb, ranfHrir, ranfGridError] = load_ranf( ...
            ranfPath, referenceTable, cache.frequencyMask);
        currentIndices = double(cache.sourceIndices(:));
        ranfObservedError = max(abs(ranfHrir(:, currentIndices, :) - ...
            referenceHrir(:, currentIndices, :)), [], 'all');
        % The cached reference HRIR is single precision; RANF copies the
        % original double-precision observations into its output SOFA.
        assert(ranfObservedError <= 2e-7, ...
            'RANF changed an observed HRIR for %s Q%d.', subjectLabel, count);

        dbValues = {mcarDb, fspDb, ranfDb};
        hrirValues = {mcarHrir, fspHrir, ranfHrir};
        for methodIndex = 1:numel(methodIds)
            values = strict_metrics(dbValues{methodIndex}, ...
                hrirValues{methodIndex}, referenceDb, referenceHrir, ...
                referenceIld, directionFeatures, fixedMask, frequencyHz, ...
                samplingRateHz);
            for metricIndex = 1:numel(metricIds)
                metricLong = [metricLong; table(subjectLabel, subjectId, count, ...
                    methodIds(methodIndex), methodLabels(methodIndex), ...
                    metricIds(metricIndex), values(metricIndex), ...
                    'VariableNames', {'SubjectLabel', 'SubjectID', ...
                    'DirectionCount', 'Method', 'MethodLabel', 'Metric', ...
                    'Value_dB'})]; %#ok<AGROW>
            end
        end
        qualityChecks = [qualityChecks; table(subjectLabel, subjectId, count, ...
            fspFrequencyError, ranfGridError, ranfObservedError, ...
            sum(fixedMask), 'VariableNames', {'SubjectLabel', 'SubjectID', ...
            'DirectionCount', 'FSPAEFrequencyMaxAbsError_Hz', ...
            'RANFGridMaxAbsError_deg', 'RANFObservedHRIRMaxAbsError', ...
            'EvaluationDirectionCount'})]; %#ok<AGROW>
        fprintf('Learned sparsity [%d/%d] %s Q%d\n', ...
            (subjectIndex - 1) * numel(counts) + find(counts == count), ...
            height(subjects) * numel(counts), subjectLabel, count);
    end
end

aggregate = aggregate_metrics(metricLong);
[robustness, pairwise] = robustness_statistics(metricLong, methodIds, ...
    methodLabels, metricIds, counts, 10000, 20260812);
writetable(metricLong, fullfile(outputRoot, 'metric_long.csv'));
writetable(aggregate, fullfile(outputRoot, 'aggregate_metrics.csv'));
writetable(robustness, fullfile(outputRoot, 'robustness_summary.csv'));
writetable(pairwise, fullfile(outputRoot, 'paired_bootstrap_comparisons.csv'));
writetable(qualityChecks, fullfile(outputRoot, 'quality_checks.csv'));
copyfile(validationPath, fullfile(outputRoot, 'artifact_validation_report.json'));
copyfile(adaptationCostPath, fullfile(outputRoot, 'ranf_adaptation_cost.json'));
plot_curves(aggregate, methodIds, methodLabels, metricIds, figuresRoot);

summary = struct('schema_version', '1.0', 'status', 'completed', ...
    'subject_count', height(subjects), 'direction_counts', counts, ...
    'fixed_evaluation_direction_count', sum(fixedMask), ...
    'evaluation_mask_policy', ...
        'Exclude the complete Q26 input set at every Q and for every method', ...
    'bootstrap_replicates', 10000, 'bootstrap_seed', 20260812, ...
    'aggregate', table2struct(aggregate), ...
    'robustness', table2struct(robustness), ...
    'pairwise_comparisons', table2struct(pairwise));
write_json(fullfile(outputRoot, 'summary.json'), summary);
write_report(fullfile(outputRoot, 'README.md'), robustness, pairwise, ...
    height(subjects));
fprintf('SONICOM learned-method sparsity evaluation complete: %s\n', outputRoot);
clear restoreDirectory;
end

function selectedDb = selected_db(left, right, mask)
spectra = cat(3, double(left), double(right));
selectedDb = 20 * log10(max(abs(permute( ...
    spectra(:, mask, :), [2, 1, 3])), 1e-10));
end

function features = build_direction_features(grid)
azimuth = double(grid(:, 1));
elevation = 90 - double(grid(:, 2));
features = [azimuth.'; elevation.'; ...
    (cosd(elevation) .* cosd(azimuth)).'; ...
    (cosd(elevation) .* sind(azimuth)).'; sind(elevation).'; ...
    double(grid(:, 3)).'];
end

function [selectedDb, hrir] = load_mcar(path, cache)
residualDb = double(h5read(path, '/predicted_residual_db'));
assert(isequal(size(residualDb), [sum(cache.frequencyMask), 793, 2]), ...
    'Unexpected MCAR residual shape.');
spectra = cat(3, double(cache.mcaLeft), double(cache.mcaRight));
phase = angle(permute(spectra(:, cache.frequencyMask, :), [2, 1, 3]));
mcaDb = 20 * log10(max(abs(permute( ...
    spectra(:, cache.frequencyMask, :), [2, 1, 3])), 1e-10));
selectedDb = mcaDb + residualDb;
for ear = 1:2
    earSpectrum = spectra(:, :, ear).';
    earSpectrum(cache.frequencyMask, :) = 10 .^ ...
        (selectedDb(:, :, ear) / 20) .* exp(1i * phase(:, :, ear));
    spectra(:, :, ear) = earSpectrum.';
end
hrir = spectra_to_hrir(spectra);
end

function [selectedDb, hrir, frequencyError] = load_fsp(path, mask, allFrequency)
magnitudeRaw = double(h5read(path, '/predicted_magnitude_db'));
hrirRaw = double(h5read(path, '/predicted_hrir'));
frequency = double(h5read(path, '/frequency_hz'));
assert(isequal(size(magnitudeRaw), [512, 2, 793]) && ...
    isequal(size(hrirRaw), [256, 2, 793]), 'Unexpected FSP-AE shape.');
selectedIndices = find(mask);
selectedDb = permute(magnitudeRaw(selectedIndices - 1, :, :), [1, 3, 2]);
hrir = permute(hrirRaw, [1, 3, 2]);
frequencyError = max(abs(frequency(:) - double(allFrequency(2:end))));
assert(frequencyError <= 1e-3 && all(isfinite(selectedDb), 'all') && ...
    all(isfinite(hrir), 'all'), 'Invalid FSP-AE strict representation.');
end

function [selectedDb, hrir, gridError] = load_ranf(path, referenceTable, mask)
sofa = SOFAload(path);
sourcePosition = double(sofa.SourcePosition);
azimuthError = abs(mod(sourcePosition(:, 1) - ...
    double(referenceTable.azimuth_deg) + 180, 360) - 180);
elevationError = abs(sourcePosition(:, 2) - ...
    double(referenceTable.elevation_deg));
gridError = max([azimuthError; elevationError]);
assert(gridError <= 1e-8, 'RANF direction order differs from reference grid.');
hrir = permute(double(sofa.Data.IR), [3, 1, 2]);
spectra = complex(zeros(793, 513, 2));
for ear = 1:2
    fullSpectrum = fft(hrir(:, :, ear), 1024, 1);
    spectra(:, :, ear) = fullSpectrum(1:513, :).';
end
selectedDb = 20 * log10(max(abs(permute( ...
    spectra(:, mask, :), [2, 1, 3])), 1e-10));
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
        referenceHrir, referenceIld, directionFeatures, fixedMask, ...
        frequencyHz, samplingRateHz)
azimuth = directionFeatures(1, :).';
elevation = directionFeatures(2, :).';
y = directionFeatures(4, :).';
weight = directionFeatures(6, :).';
left25 = fixedMask & great_circle_mask(azimuth, elevation, 270, 0, 25);
right25 = fixedMask & great_circle_mask(azimuth, elevation, 90, 0, 25);
leftHemisphere = fixedMask & y < -1e-12;
rightHemisphere = fixedMask & y > 1e-12;
horizontal = fixedMask & abs(elevation) <= 1e-9;
[erbLeft, ~] = AKerbError(hrir(:, fixedMask, 1), ...
    referenceHrir(:, fixedMask, 1), [50, samplingRateHz / 2], samplingRateHz);
erbRight = AKerbError(hrir(:, fixedMask, 2), ...
    referenceHrir(:, fixedMask, 2), [50, samplingRateHz / 2], samplingRateHz);
fullWeight = normalized_weights(weight(fixedMask));
values(1) = 0.5 * (mean(abs(erbLeft) * fullWeight) + ...
    mean(abs(erbRight) * fullWeight));
leftWithin = left25(fixedMask);
rightWithin = right25(fixedMask);
values(2) = 0.5 * (mean(abs(erbLeft(:, leftWithin)) * ...
    normalized_weights(weight(left25))) + ...
    mean(abs(erbRight(:, rightWithin)) * normalized_weights(weight(right25))));
values(3) = 0.5 * (high_frequency_error(selectedDb(:, :, 1), ...
    referenceDb(:, :, 1), frequencyHz, leftHemisphere, weight) + ...
    high_frequency_error(selectedDb(:, :, 2), referenceDb(:, :, 2), ...
    frequencyHz, rightHemisphere, weight));
methodIld = calculate_ild(hrir(:, :, 1), hrir(:, :, 2));
values(4) = mean(abs(methodIld(horizontal) - referenceIld(horizontal)));
assert(all(isfinite(values)), 'A strict metric is non-finite.');
end

function ild = calculate_ild(left, right)
ild = (10 * log10(sum(abs(left) .^ 2, 1) ./ ...
    sum(abs(right) .^ 2, 1))).';
end

function mask = great_circle_mask(azimuth, elevation, centerAzimuth, ...
        centerElevation, radius)
dotProduct = sind(elevation) .* sind(centerElevation) + ...
    cosd(elevation) .* cosd(centerElevation) .* ...
    cosd(azimuth - centerAzimuth);
mask = acosd(min(1, max(-1, dotProduct))) <= radius + 1e-10;
end

function weights = normalized_weights(weights)
weights = weights(:);
assert(all(isfinite(weights)) && all(weights > 0), 'Invalid weights.');
weights = weights / sum(weights);
end

function value = high_frequency_error(estimateDb, referenceDb, ...
        frequencyHz, directionMask, weights)
frequencyMask = frequencyHz > 10000 & frequencyHz <= 20000;
errorDb = abs(estimateDb(frequencyMask, directionMask) - ...
    referenceDb(frequencyMask, directionMask));
value = mean(errorDb * normalized_weights(weights(directionMask)));
end

function aggregate = aggregate_metrics(metricLong)
groups = unique(metricLong(:, {'Method', 'MethodLabel', ...
    'DirectionCount', 'Metric'}), 'rows', 'stable');
aggregate = table();
for rowIndex = 1:height(groups)
    mask = metricLong.Method == groups.Method(rowIndex) & ...
        metricLong.DirectionCount == groups.DirectionCount(rowIndex) & ...
        metricLong.Metric == groups.Metric(rowIndex);
    values = metricLong.Value_dB(mask);
    aggregate = [aggregate; table(groups.Method(rowIndex), ...
        groups.MethodLabel(rowIndex), groups.DirectionCount(rowIndex), ...
        groups.Metric(rowIndex), numel(values), mean(values), std(values), ...
        1.96 * std(values) / sqrt(numel(values)), ...
        'VariableNames', {'Method', 'MethodLabel', 'DirectionCount', ...
        'Metric', 'SubjectCount', 'Mean_dB', 'SD_dB', 'CI95HalfWidth_dB'})]; %#ok<AGROW>
end
end

function [robustness, pairwise] = robustness_statistics(metricLong, ...
        methodIds, methodLabels, metricIds, counts, replicates, seed)
rng(seed, 'twister');
x = log2(26 ./ counts(:));
robustness = table();
subjectIds = unique(metricLong.SubjectID, 'stable');
for metricIndex = 1:numel(metricIds)
    for methodIndex = 1:numel(methodIds)
        values = zeros(numel(subjectIds), numel(counts));
        for countIndex = 1:numel(counts)
            rows = metricLong(metricLong.Method == methodIds(methodIndex) & ...
                metricLong.Metric == metricIds(metricIndex) & ...
                metricLong.DirectionCount == counts(countIndex), :);
            rows = sortrows(rows, 'SubjectID');
            assert(isequal(rows.SubjectID, sort(subjectIds)), ...
                'Subject pairing mismatch.');
            values(:, countIndex) = rows.Value_dB;
        end
        degradation = values(:, 1) - values(:, end);
        slopes = zeros(numel(subjectIds), 1);
        for subjectIndex = 1:numel(subjectIds)
            coefficients = polyfit(x, values(subjectIndex, :).', 1);
            slopes(subjectIndex) = coefficients(1);
        end
        robustness = [robustness; table(methodIds(methodIndex), ...
            methodLabels(methodIndex), metricIds(metricIndex), ...
            mean(values(:, 1)), mean(values(:, end)), mean(degradation), ...
            std(degradation), mean(slopes), std(slopes), ...
            'VariableNames', {'Method', 'MethodLabel', 'Metric', ...
            'Q6Mean_dB', 'Q26Mean_dB', 'Q6MinusQ26Mean_dB', ...
            'Q6MinusQ26SD_dB', 'SlopePerHalvingMean_dB', ...
            'SlopePerHalvingSD_dB'})]; %#ok<AGROW>
    end
end

pairwise = table();
pairs = nchoosek(1:numel(methodIds), 2);
for metricIndex = 1:numel(metricIds)
    for pairIndex = 1:size(pairs, 1)
        a = pairs(pairIndex, 1);
        b = pairs(pairIndex, 2);
        [endpointA, degradationA] = paired_vectors(metricLong, ...
            methodIds(a), metricIds(metricIndex));
        [endpointB, degradationB] = paired_vectors(metricLong, ...
            methodIds(b), metricIds(metricIndex));
        for contrastIndex = 1:2
            if contrastIndex == 1
                difference = endpointA - endpointB;
                contrast = "Q6Endpoint";
            else
                difference = degradationA - degradationB;
                contrast = "Q6MinusQ26Degradation";
            end
            [lower, upper, pValue] = paired_bootstrap(difference, replicates);
            pairwise = [pairwise; table(metricIds(metricIndex), contrast, ...
                methodIds(a), methodIds(b), mean(difference), lower, upper, ...
                pValue, replicates, seed, ...
                'VariableNames', {'Metric', 'Contrast', 'MethodA', ...
                'MethodB', 'MeanDifferenceAminusB_dB', 'CI95Lower_dB', ...
                'CI95Upper_dB', 'TwoSidedBootstrapP', ...
                'BootstrapReplicates', 'Seed'})]; %#ok<AGROW>
        end
    end
end
end

function [endpoint, degradation] = paired_vectors(metricLong, method, metric)
q6 = sortrows(metricLong(metricLong.Method == method & ...
    metricLong.Metric == metric & metricLong.DirectionCount == 6, :), 'SubjectID');
q26 = sortrows(metricLong(metricLong.Method == method & ...
    metricLong.Metric == metric & metricLong.DirectionCount == 26, :), 'SubjectID');
assert(isequal(q6.SubjectID, q26.SubjectID), 'Endpoint pairing mismatch.');
endpoint = q6.Value_dB;
degradation = q6.Value_dB - q26.Value_dB;
end

function [lower, upper, pValue] = paired_bootstrap(difference, replicates)
n = numel(difference);
indices = randi(n, n, replicates);
bootstrapMeans = mean(difference(indices), 1);
interval = prctile(bootstrapMeans, [2.5, 97.5]);
lower = interval(1);
upper = interval(2);
pValue = min(1, 2 * min((sum(bootstrapMeans <= 0) + 1) / ...
    (replicates + 1), (sum(bootstrapMeans >= 0) + 1) / (replicates + 1)));
end

function plot_curves(aggregate, methodIds, methodLabels, metricIds, root)
figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1200, 850]);
layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
colors = lines(numel(methodIds));
for metricIndex = 1:numel(metricIds)
    axisHandle = nexttile(layout);
    hold(axisHandle, 'on');
    for methodIndex = 1:numel(methodIds)
        rows = aggregate(aggregate.Method == methodIds(methodIndex) & ...
            aggregate.Metric == metricIds(metricIndex), :);
        rows = sortrows(rows, 'DirectionCount');
        errorbar(axisHandle, rows.DirectionCount, rows.Mean_dB, ...
            rows.CI95HalfWidth_dB, '-o', 'Color', colors(methodIndex, :), ...
            'LineWidth', 1.6, 'MarkerFaceColor', colors(methodIndex, :));
    end
    set(axisHandle, 'XScale', 'log', 'XTick', [6, 14, 26]);
    grid(axisHandle, 'on');
    xlabel(axisHandle, 'Observed directions Q');
    ylabel(axisHandle, 'Error (dB)');
    title(axisHandle, strrep(metricIds(metricIndex), '_', ' '));
end
legend(axisHandle, cellstr(methodLabels), 'Location', 'best');
exportgraphics(figureHandle, fullfile(root, 'learned_sparsity_curves.png'), ...
    'Resolution', 220);
savefig(figureHandle, fullfile(root, 'learned_sparsity_curves.fig'));
close(figureHandle);
end

function write_report(path, robustness, pairwise, subjectCount)
handle = fopen(path, 'w');
assert(handle >= 0, 'Cannot write report: %s', path);
cleanup = onCleanup(@() fclose(handle));
fprintf(handle, '# SONICOM learned-method sparsity experiment\n\n');
fprintf(handle, 'Completed on %d fixed test subjects. All methods use the same 767-target mask.\n\n', subjectCount);
fprintf(handle, '## Robustness summary\n\n');
fprintf(handle, '| Method | Metric | Q6 | Q26 | Q6-Q26 | Slope / halving |\n');
fprintf(handle, '|---|---|---:|---:|---:|---:|\n');
for index = 1:height(robustness)
    fprintf(handle, '| %s | %s | %.3f | %.3f | %.3f | %.3f |\n', ...
        robustness.MethodLabel(index), robustness.Metric(index), ...
        robustness.Q6Mean_dB(index), robustness.Q26Mean_dB(index), ...
        robustness.Q6MinusQ26Mean_dB(index), ...
        robustness.SlopePerHalvingMean_dB(index));
end
fprintf(handle, '\n## Paired bootstrap\n\n');
fprintf(handle, 'Differences are Method A minus Method B; negative favors A. See `paired_bootstrap_comparisons.csv` for all %d contrasts.\n', height(pairwise));
clear cleanup;
end

function write_json(path, value)
handle = fopen(path, 'w');
assert(handle >= 0, 'Cannot write JSON: %s', path);
cleanup = onCleanup(@() fclose(handle));
fwrite(handle, jsonencode(value, 'PrettyPrint', true), 'char');
fwrite(handle, newline, 'char');
clear cleanup;
end
