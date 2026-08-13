function evaluate_sonicom_five_baseline_sparsity(subjectLimit, outputName, numWorkers)
%EVALUATE_SONICOM_FIVE_BASELINE_SPARSITY Evaluate five baselines over Q.

if nargin < 1 || isempty(subjectLimit)
    subjectLimit = inf;
end
if nargin < 2 || isempty(outputName)
    outputName = 'sonicom_five_baseline_sparsity_v1';
end
if nargin < 3 || isempty(numWorkers)
    numWorkers = 4;
end
validateattributes(subjectLimit, {'numeric'}, {'scalar', 'positive'});
validateattributes(outputName, {'char', 'string'}, {'scalartext'});
validateattributes(numWorkers, {'numeric'}, {'scalar', 'integer', 'positive'});

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
configPath = fullfile(projectRoot, 'configs', 'experiments', ...
    'sonicom_eight_method_sparsity_v1.json');
config = jsondecode(fileread(configPath));
assert(any(string(config.status) == ["preregistered", "running", "completed"]), ...
    'Unexpected frozen protocol status.');
splitTable = readtable(fullfile(projectRoot, config.dataset.split_file), ...
    'TextType', 'string');
subjects = splitTable(splitTable.split == "test", :);
assert(height(subjects) == 44, 'Expected 44 frozen SONICOM test subjects.');
subjects = subjects(1:min(height(subjects), floor(subjectLimit)), :);
referenceTable = readtable(fullfile(projectRoot, config.dataset.reference_grid), ...
    'TextType', 'string');
gridTable = readtable(fullfile(projectRoot, config.dataset.sparse_grid_file), ...
    'TextType', 'string');
counts = double(config.dataset.direction_counts(:).');
assert(height(referenceTable) == 793 && isequal(counts, [6, 14, 26]), ...
    'Frozen grid configuration was modified.');
fixedMask = logical(referenceTable.is_interpolation_evaluation);
assert(sum(fixedMask) == 767, 'The fixed evaluation mask must have 767 targets.');

artifactRoot = fullfile(projectRoot, 'artifacts', 'sparsity', ...
    'sonicom_learned_methods_sparsity_v1');
subjectsRoot = fullfile(artifactRoot, 'subjects');
assert(isfolder(subjectsRoot), 'Completed SONICOM sparsity caches are missing.');
outputRoot = fullfile(projectRoot, 'results', char(outputName));
figuresRoot = fullfile(outputRoot, 'figures');
if ~isfolder(figuresRoot)
    mkdir(figuresRoot);
end

supdeqDir = fullfile(projectRoot, 'external', 'SUpDEq');
compatibilityRoot = activate_findvoronoi_compatibility(projectRoot);
initialize_worker(supdeqDir, compatibilityRoot);
operators = build_operators(counts, gridTable, referenceTable, fixedMask);

requestedWorkers = min([numWorkers, height(subjects)]);
pool = gcp('nocreate');
if requestedWorkers > 1
    if isempty(pool)
        parpool('local', requestedWorkers);
    elseif pool.NumWorkers ~= requestedWorkers
        delete(pool);
        parpool('local', requestedWorkers);
    end
    workerEnvironment = parallel.pool.Constant( ...
        @() initialize_worker(supdeqDir, compatibilityRoot));
else
    workerEnvironment = [];
end

metricCells = cell(height(subjects), 1);
qualityCells = cell(height(subjects), 1);
if requestedWorkers > 1
    parfor subjectIndex = 1:height(subjects)
        workerEnvironment.Value;
        [metricCells{subjectIndex}, qualityCells{subjectIndex}] = ...
            evaluate_subject(subjects(subjectIndex, :), counts, operators, ...
            subjectsRoot, fixedMask);
    end
    delete(workerEnvironment);
else
    for subjectIndex = 1:height(subjects)
        [metricCells{subjectIndex}, qualityCells{subjectIndex}] = ...
            evaluate_subject(subjects(subjectIndex, :), counts, operators, ...
            subjectsRoot, fixedMask);
    end
end
metricLong = vertcat(metricCells{:});
qualityChecks = vertcat(qualityCells{:});
assert(height(metricLong) == height(subjects) * 3 * 5 * 4 && ...
    all(isfinite(metricLong.Value_dB)), 'Baseline metric table is incomplete.');
aggregate = aggregate_metrics(metricLong);
writetable(metricLong, fullfile(outputRoot, 'metric_long.csv'));
writetable(aggregate, fullfile(outputRoot, 'aggregate_metrics.csv'));
writetable(qualityChecks, fullfile(outputRoot, 'quality_checks.csv'));
plot_curves(aggregate, figuresRoot);
summary = struct('schema_version', '1.0', 'status', 'completed', ...
    'subject_count', height(subjects), 'direction_counts', counts, ...
    'fixed_evaluation_direction_count', sum(fixedMask), ...
    'methods', {cellstr(unique(metricLong.Method, 'stable'))}, ...
    'aggregate', table2struct(aggregate), ...
    'quality_checks', table2struct(qualityChecks));
write_json(fullfile(outputRoot, 'summary.json'), summary);
write_report(fullfile(outputRoot, 'README.md'), height(subjects));
fprintf('SONICOM five-baseline sparsity evaluation complete: %s\n', outputRoot);
end

function environment = initialize_worker(supdeqDir, compatibilityRoot)
originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir);
supdeq_start;
addpath(compatibilityRoot, '-begin');
clear findvoronoi;
environment = struct('initialized', true);
clear restoreDirectory;
end

function operators = build_operators(counts, gridTable, referenceTable, fixedMask)
referenceGrid = [double(referenceTable.azimuth_deg), ...
    double(referenceTable.colatitude_deg), ...
    double(referenceTable.solid_angle_weight)];
operators = cell(numel(counts), 1);
for countIndex = 1:numel(counts)
    count = counts(countIndex);
    sparseRows = gridTable(gridTable.direction_count == count, :);
    assert(height(sparseRows) == count, 'Invalid Q%d grid.', count);
    sparseGrid = [double(sparseRows.azimuth_deg), ...
        double(sparseRows.colatitude_deg)];
    sourceIndices = double(sparseRows.source_index_zero_based) + 1;
    operators{countIndex} = struct('count', count, ...
        'sparseGrid', sparseGrid, 'sourceIndices', sourceIndices, ...
        'referenceGrid', referenceGrid, 'fixedMask', fixedMask);
end
end

function [metricRows, qualityRows] = evaluate_subject(subjectRow, counts, ...
        operators, subjectsRoot, fixedMask)
methodIds = ["SHOnly", "SUpDEqSH", "SUpDEqNN", "SUpDEqBary", "MCA"];
methodLabels = ["SH only", "SUpDEq + SH", ...
    "SUpDEq + Natural Neighbor", "SUpDEq + Barycentric", "MCA"];
metricIds = ["MeasuredDomainERB", "Contralateral25ERB", ...
    "ContralateralHemisphereHighFrequency", "HorizontalILDMAE"];
subjectLabel = subjectRow.subject_id;
subjectId = sscanf(char(subjectLabel), 'P%d');
metricRows = table();
qualityRows = table();
for countIndex = 1:numel(counts)
    count = counts(countIndex);
    operator = operators{countIndex};
    cachePath = fullfile(subjectsRoot, char(subjectLabel), ...
        sprintf('q%d', count), 'cache.mat');
    assert(isfile(cachePath), 'Missing sparsity cache: %s', cachePath);
    loaded = load(cachePath, 'cache');
    cache = loaded.cache;
    assert(cache.directionCount == count && cache.split == "test" && ...
        isequal(double(cache.sourceIndices(:)), operator.sourceIndices(:)), ...
        'Cache/grid mismatch for %s Q%d.', subjectLabel, count);
    referenceLeft = double(cache.referenceLeft);
    referenceRight = double(cache.referenceRight);
    referenceHrir = double(cache.referenceHrir);
    frequencyHzAll = double(cache.frequencyHz(:));
    frequencyMask = logical(cache.frequencyMask(:));
    frequencyHz = frequencyHzAll(frequencyMask);
    samplingRateHz = double(cache.samplingRateHz);
    referenceDb = selected_db(referenceLeft, referenceRight, frequencyMask);
    directionFeatures = build_direction_features(cache.referenceGrid);
    referenceIld = calculate_ild(referenceHrir(:, :, 1), ...
        referenceHrir(:, :, 2));
    sparseHrtf = struct('HRTF_L', referenceLeft(operator.sourceIndices, :), ...
        'HRTF_R', referenceRight(operator.sourceIndices, :), ...
        'f', frequencyHzAll, 'fs', samplingRateHz, 'Nmax', 3, ...
        'FFToversize', double(cache.fftOversize), ...
        'samplingGrid', operator.sparseGrid);
    shOnly = supdeq_interpHRTF(sparseHrtf, operator.referenceGrid, ...
        'None', 'SH', nan, 0.09, 1e-2, true, 0, true, 'fadeDown');
    supdeqSh = supdeq_interpHRTF(sparseHrtf, operator.referenceGrid, ...
        'SUpDEq', 'SH', nan, 0.09, 1e-2, true, 0, true, 'fadeDown');
    nativeNn = supdeq_interpHRTF(sparseHrtf, operator.referenceGrid, ...
        'SUpDEq', 'NN', nan, 0.09, 0, true, 0, true, 'fadeDown');
    nativeBary = supdeq_interpHRTF(sparseHrtf, operator.referenceGrid, ...
        'SUpDEq', 'Bary', nan, 0.09, 0, true, 0, true, 'fadeDown');
    nnSpectra = cat(3, nativeNn.HRTF_L, nativeNn.HRTF_R);
    barySpectra = cat(3, nativeBary.HRTF_L, nativeBary.HRTF_R);
    assert(all(isfinite(nnSpectra), 'all') && ...
        all(isfinite(barySpectra), 'all'), ...
        'Native SUpDEq generated non-finite values for %s Q%d.', ...
        subjectLabel, count);
    methodSpectra = {cat(3, shOnly.HRTF_L, shOnly.HRTF_R), ...
        cat(3, supdeqSh.HRTF_L, supdeqSh.HRTF_R), ...
        nnSpectra, barySpectra, ...
        cat(3, double(cache.mcaLeft), double(cache.mcaRight))};
    for methodIndex = 1:numel(methodIds)
        [selectedDb, hrir] = method_representation( ...
            methodSpectra{methodIndex}, frequencyMask, 256);
        values = strict_metrics(selectedDb, hrir, referenceDb, ...
            referenceHrir, referenceIld, directionFeatures, fixedMask, ...
            frequencyHz, samplingRateHz);
        for metricIndex = 1:numel(metricIds)
            metricRows = [metricRows; table(subjectLabel, subjectId, count, ...
                methodIds(methodIndex), methodLabels(methodIndex), ...
                metricIds(metricIndex), values(metricIndex), ...
                'VariableNames', {'SubjectLabel', 'SubjectID', ...
                'DirectionCount', 'Method', 'MethodLabel', 'Metric', ...
                'Value_dB'})]; %#ok<AGROW>
        end
    end
    nnObservedError = max(abs(nnSpectra(operator.sourceIndices, :, :) - ...
        cat(3, sparseHrtf.HRTF_L, sparseHrtf.HRTF_R)), [], 'all');
    baryObservedError = max(abs(barySpectra(operator.sourceIndices, :, :) - ...
        cat(3, sparseHrtf.HRTF_L, sparseHrtf.HRTF_R)), [], 'all');
    qualityRows = [qualityRows; table(subjectLabel, subjectId, count, ...
        nnObservedError, baryObservedError, sum(fixedMask), ...
        'VariableNames', {'SubjectLabel', 'SubjectID', 'DirectionCount', ...
        'NaturalNeighborObservedHRTFMaxAbsError', ...
        'BarycentricObservedHRTFMaxAbsError', ...
        'EvaluationDirectionCount'})]; %#ok<AGROW>
    fprintf('Five baselines %s Q%d complete.\n', subjectLabel, count);
end
end

function compatibilityRoot = activate_findvoronoi_compatibility(projectRoot)
sourceFile = fullfile(projectRoot, 'external', 'SUpDEq', 'thirdParty', ...
    'sfs-matlab-2.5.0', 'SFS_general', 'findvoronoi.m');
sourceText = fileread(sourceFile);
oldText = 'for n = 1:size(idx)';
assert(numel(strfind(sourceText, oldText)) == 2, ... %#ok<STREMP>
    'Unexpected upstream findvoronoi revision.');
patchedText = strrep(sourceText, oldText, 'for n = 1:size(idx,1)');
compatibilityRoot = fullfile(projectRoot, 'artifacts', ...
    'matlab_compat', 'sfs_findvoronoi_r2026');
if ~isfolder(compatibilityRoot)
    mkdir(compatibilityRoot);
end
compatibilityFile = fullfile(compatibilityRoot, 'findvoronoi.m');
handle = fopen(compatibilityFile, 'w');
assert(handle >= 0, 'Could not create findvoronoi compatibility copy.');
cleanup = onCleanup(@() fclose(handle));
fprintf(handle, '%s', patchedText);
clear cleanup;
end

function selectedDb = selected_db(left, right, mask)
spectra = cat(3, left, right);
selectedDb = 20 * log10(max(abs(permute(spectra(:, mask, :), ...
    [2, 1, 3])), 1e-10));
end

function features = build_direction_features(grid)
azimuth = double(grid(:, 1));
elevation = 90 - double(grid(:, 2));
features = [azimuth.'; elevation.'; ...
    (cosd(elevation) .* cosd(azimuth)).'; ...
    (cosd(elevation) .* sind(azimuth)).'; sind(elevation).'; ...
    double(grid(:, 3)).'];
end

function [selectedDb, hrir] = method_representation(spectra, mask, hrirLength)
assert(isequal(size(spectra), [793, 513, 2]), ...
    'Unexpected interpolated spectrum shape.');
selectedDb = 20 * log10(max(abs(permute(spectra(:, mask, :), ...
    [2, 1, 3])), 1e-10));
hrir = zeros(hrirLength, 793, 2);
for ear = 1:2
    fullSpectrum = AKsingle2bothSidedSpectrum(spectra(:, :, ear).');
    oversizedHrir = real(ifft(fullSpectrum));
    hrir(:, :, ear) = oversizedHrir(1:hrirLength, :);
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
values(2) = 0.5 * (mean(abs(erbLeft(:, left25(fixedMask))) * ...
    normalized_weights(weight(left25))) + ...
    mean(abs(erbRight(:, right25(fixedMask))) * ...
    normalized_weights(weight(right25))));
values(3) = 0.5 * (high_frequency_error(selectedDb(:, :, 1), ...
    referenceDb(:, :, 1), frequencyHz, leftHemisphere, weight) + ...
    high_frequency_error(selectedDb(:, :, 2), referenceDb(:, :, 2), ...
    frequencyHz, rightHemisphere, weight));
methodIld = calculate_ild(hrir(:, :, 1), hrir(:, :, 2));
values(4) = mean(abs(methodIld(horizontal) - referenceIld(horizontal)));
assert(all(isfinite(values)), 'A strict baseline metric is non-finite.');
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
        'Metric', 'SubjectCount', 'Mean_dB', 'SD_dB', ...
        'CI95HalfWidth_dB'})]; %#ok<AGROW>
end
end

function plot_curves(aggregate, root)
methods = unique(aggregate(:, {'Method', 'MethodLabel'}), 'rows', 'stable');
metrics = unique(aggregate.Metric, 'stable');
figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1200, 850]);
layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
colors = lines(height(methods));
for metricIndex = 1:numel(metrics)
    axisHandle = nexttile(layout);
    hold(axisHandle, 'on');
    for methodIndex = 1:height(methods)
        rows = aggregate(aggregate.Method == methods.Method(methodIndex) & ...
            aggregate.Metric == metrics(metricIndex), :);
        rows = sortrows(rows, 'DirectionCount');
        errorbar(axisHandle, rows.DirectionCount, rows.Mean_dB, ...
            rows.CI95HalfWidth_dB, '-o', 'Color', colors(methodIndex, :), ...
            'LineWidth', 1.5, 'MarkerFaceColor', colors(methodIndex, :));
    end
    set(axisHandle, 'XScale', 'log', 'XTick', [6, 14, 26]);
    grid(axisHandle, 'on');
    xlabel(axisHandle, 'Observed directions Q');
    ylabel(axisHandle, 'Error (dB)');
    title(axisHandle, strrep(metrics(metricIndex), '_', ' '));
end
legend(axisHandle, cellstr(methods.MethodLabel), 'Location', 'best');
exportgraphics(figureHandle, fullfile(root, 'five_baseline_curves.png'), ...
    'Resolution', 220);
savefig(figureHandle, fullfile(root, 'five_baseline_curves.fig'));
close(figureHandle);
end

function write_report(path, subjectCount)
handle = fopen(path, 'w');
assert(handle >= 0, 'Cannot write report: %s', path);
cleanup = onCleanup(@() fclose(handle));
fprintf(handle, '# SONICOM five-baseline sparsity experiment\n\n');
fprintf(handle, 'Completed on %d fixed test subjects at Q6/Q14/Q26.\n', ...
    subjectCount);
fprintf(handle, 'Every method uses the same 767-target evaluation mask.\n');
fprintf(handle, 'See `aggregate_metrics.csv` and `metric_long.csv`.\n');
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
