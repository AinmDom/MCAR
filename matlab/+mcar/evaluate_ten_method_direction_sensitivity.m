function evaluate_ten_method_direction_sensitivity(subjectLimit, outputName)
%EVALUATE_TEN_METHOD_DIRECTION_SENSITIVITY Strict Q14/Q26/Q50 comparison.

if nargin < 1 || isempty(subjectLimit), subjectLimit = inf; end
if nargin < 2 || isempty(outputName)
    outputName = 'sonicom_ten_method_direction_sensitivity_v1';
end
validateattributes(subjectLimit, {'numeric'}, {'scalar', 'positive'});
validateattributes(outputName, {'char', 'string'}, {'scalartext'});

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
config = jsondecode(fileread(fullfile(projectRoot, 'configs', 'experiments', ...
    'sonicom_ten_method_direction_sensitivity_v1.json')));
assert(string(config.status) == "preregistered" && ...
    string(config.dataset.split) == "val" && ...
    ~config.inference.test_access_allowed, ...
    'Ten-method evaluation must remain preregistered and validation-only.');
splitTable = readtable(fullfile(projectRoot, config.dataset.split_file), ...
    'TextType', 'string');
subjects = splitTable(splitTable.split == "val", :);
assert(height(subjects) == 44, 'Expected 44 validation listeners.');
subjects = subjects(1:min(height(subjects), floor(subjectLimit)), :);
counts = double(config.dataset.direction_counts(:).');
assert(isequal(counts, [14, 26, 50]), 'Frozen direction counts changed.');
gridTable = readtable(fullfile(projectRoot, config.dataset.sparse_grid_file), ...
    'TextType', 'string');
q50 = double(gridTable.source_index_zero_based( ...
    gridTable.direction_count == 50)) + 1;
fixedMask = true(793, 1); fixedMask(q50) = false;
assert(sum(fixedMask) == 743, 'Fixed Q50-excluded mask changed.');

inputRoot = fullfile(projectRoot, config.dataset.prepared_input_root);
predictionRoot = fullfile(projectRoot, 'artifacts', 'sparsity', config.output_name);
boundedReport = jsondecode(fileread(fullfile(inputRoot, 'inference_report.json')));
fastReports = ["inference_hybrid_mcar_report.json", "inference_fspae_report.json"];
for reportName = fastReports
    report = jsondecode(fileread(fullfile(predictionRoot, reportName)));
    assert(string(report.status) == "completed" && ...
        report.test_subject_count_read == 0 && ...
        all(structfun(@(x) x <= report.q26_reproduction_tolerance, ...
        report.q26_reproduction_max_abs_error)), ...
        'Missing or failed frozen inference report: %s', reportName);
end
assert(string(boundedReport.status) == "completed" && ...
    boundedReport.test_subject_count_read == 0, ...
    'Completed Bounded sensitivity report is required.');

if iscell(config.methods)
    ranfIndex = find(cellfun(@(method) string(method.id) == "RANF", ...
        config.methods), 1);
else
    ranfIndex = find(string({config.methods.id}) == "RANF", 1);
end
assert(~isempty(ranfIndex), 'RANF method configuration is required.');
if iscell(config.methods)
    ranfMethod = config.methods{ranfIndex};
else
    ranfMethod = config.methods(ranfIndex);
end
ranfRoots = containers.Map({'14', '26', '50'}, { ...
    fullfile(projectRoot, ranfMethod.prediction_roots.q14), ...
    fullfile(projectRoot, ranfMethod.prediction_roots.q26), ...
    fullfile(projectRoot, ranfMethod.prediction_roots.q50)});
for count = counts
    manifestPath = fullfile(ranfRoots(num2str(count)), 'manifest.json');
    assert(isfile(manifestPath), 'Missing RANF Q%d manifest.', count);
    manifest = jsondecode(fileread(manifestPath));
    assert(string(manifest.split) == "val" && manifest.subject_count == 44, ...
        'RANF Q%d provenance mismatch.', count);
end

supdeqDir = fullfile(projectRoot, 'external', 'SUpDEq');
compatibilityRoot = activate_findvoronoi_compatibility(projectRoot);
initialize_worker(supdeqDir, compatibilityRoot);
operators = build_operators(counts, gridTable, fixedMask);

methodIds = ["SHOnly", "SUpDEqSH", "SUpDEqNN", "SUpDEqBary", ...
    "MCA", "MCARv351", "FSPAE", "RANF", "HYBRID", "BOUNDED"];
methodLabels = ["SH only", "SUpDEq + SH", "SUpDEq + Natural Neighbor", ...
    "SUpDEq + Barycentric", "MCA", "MCAR v3.5.1", "FSP-AE", ...
    "RANF", "Hybrid E190", "Bounded E25"];
metricIds = ["FullSphereERB", "Contralateral25ERB", ...
    "ContralateralHighFrequency", "HorizontalILDMAE"];
metricLabels = ["Full-sphere ERB", "Contralateral 25-deg ERB", ...
    "Contralateral HF", "Horizontal ILD MAE"];
metricLong = table(); qualityChecks = table();

for subjectIndex = 1:height(subjects)
    subjectLabel = subjects.subject_id(subjectIndex);
    subjectId = sscanf(char(subjectLabel), 'P%d');
    for countIndex = 1:numel(counts)
        count = counts(countIndex); operator = operators{countIndex};
        inputLevel = fullfile(inputRoot, 'subjects', char(subjectLabel), ...
            sprintf('q%d', count));
        predictionLevel = fullfile(predictionRoot, 'subjects', ...
            char(subjectLabel), sprintf('q%d', count));
        cachePath = fullfile(inputLevel, 'cache.mat');
        loaded = load(cachePath, 'cache'); cache = loaded.cache;
        operator.referenceGrid = double(cache.referenceGrid);
        assert(cache.directionCount == count && string(cache.split) == "val" && ...
            isequal(double(cache.sourceIndices(:)), operator.sourceIndices(:)), ...
            'Cache/grid provenance mismatch for %s Q%d.', subjectLabel, count);
        referenceLeft = double(cache.referenceLeft);
        referenceRight = double(cache.referenceRight);
        referenceHrir = double(cache.referenceHrir);
        frequencyAll = double(cache.frequencyHz(:));
        frequencyMask = logical(cache.frequencyMask(:));
        frequencyHz = frequencyAll(frequencyMask);
        samplingRateHz = double(cache.samplingRateHz);
        referenceDb = selected_db(referenceLeft, referenceRight, frequencyMask);
        features = build_direction_features(cache.referenceGrid);
        referenceIld = calculate_ild(referenceHrir(:, :, 1), ...
            referenceHrir(:, :, 2));

        sparse = struct('HRTF_L', referenceLeft(operator.sourceIndices, :), ...
            'HRTF_R', referenceRight(operator.sourceIndices, :), ...
            'f', frequencyAll, 'fs', samplingRateHz, 'Nmax', 3, ...
            'FFToversize', double(cache.fftOversize), ...
            'samplingGrid', operator.sparseGrid);
        shOnly = supdeq_interpHRTF(sparse, operator.referenceGrid, ...
            'None', 'SH', nan, 0.09, 1e-2, true, 0, true, 'fadeDown');
        supdeqSh = supdeq_interpHRTF(sparse, operator.referenceGrid, ...
            'SUpDEq', 'SH', nan, 0.09, 1e-2, true, 0, true, 'fadeDown');
        nativeNn = supdeq_interpHRTF(sparse, operator.referenceGrid, ...
            'SUpDEq', 'NN', nan, 0.09, 0, true, 0, true, 'fadeDown');
        nativeBary = supdeq_interpHRTF(sparse, operator.referenceGrid, ...
            'SUpDEq', 'Bary', nan, 0.09, 0, true, 0, true, 'fadeDown');
        spectraValues = {cat(3, shOnly.HRTF_L, shOnly.HRTF_R), ...
            cat(3, supdeqSh.HRTF_L, supdeqSh.HRTF_R), ...
            cat(3, nativeNn.HRTF_L, nativeNn.HRTF_R), ...
            cat(3, nativeBary.HRTF_L, nativeBary.HRTF_R), ...
            cat(3, double(cache.mcaLeft), double(cache.mcaRight))};
        dbValues = cell(1, 10); hrirValues = cell(1, 10);
        for methodIndex = 1:5
            [dbValues{methodIndex}, hrirValues{methodIndex}] = ...
                method_representation(spectraValues{methodIndex}, frequencyMask);
        end
        [dbValues{6}, hrirValues{6}] = load_residual( ...
            fullfile(predictionLevel, 'mcar_v351_prediction.h5'), cache);
        [dbValues{7}, hrirValues{7}, fspFrequencyError] = load_fsp( ...
            fullfile(predictionLevel, 'fspae_prediction.h5'), ...
            frequencyMask, frequencyAll);
        [dbValues{8}, hrirValues{8}, ranfGridError] = load_ranf( ...
            fullfile(ranfRoots(num2str(count)), 'subjects', ...
            char(subjectLabel), 'prediction.sofa'), cache.referenceGrid, ...
            frequencyMask);
        [dbValues{9}, hrirValues{9}] = load_residual( ...
            fullfile(predictionLevel, 'hybrid_e190_prediction.h5'), cache);
        [dbValues{10}, hrirValues{10}] = load_residual( ...
            fullfile(inputLevel, 'bounded_prediction.h5'), cache);
        for methodIndex = 1:numel(methodIds)
            values = strict_metrics(dbValues{methodIndex}, ...
                hrirValues{methodIndex}, referenceDb, referenceHrir, ...
                referenceIld, features, fixedMask, frequencyHz, samplingRateHz);
            for metricIndex = 1:numel(metricIds)
                metricLong = [metricLong; table(subjectLabel, subjectId, count, ...
                    methodIds(methodIndex), methodLabels(methodIndex), ...
                    metricIds(metricIndex), metricLabels(metricIndex), ...
                    values(metricIndex), 'VariableNames', {'SubjectLabel', ...
                    'SubjectID', 'DirectionCount', 'Method', 'MethodLabel', ...
                    'Metric', 'MetricLabel', 'Value_dB'})]; %#ok<AGROW>
            end
        end
        nnObserved = max(abs(spectraValues{3}(operator.sourceIndices, :, :) - ...
            cat(3, sparse.HRTF_L, sparse.HRTF_R)), [], 'all');
        baryObserved = max(abs(spectraValues{4}(operator.sourceIndices, :, :) - ...
            cat(3, sparse.HRTF_L, sparse.HRTF_R)), [], 'all');
        ranfObserved = max(abs(hrirValues{8}(:, operator.sourceIndices, :) - ...
            referenceHrir(:, operator.sourceIndices, :)), [], 'all');
        qualityChecks = [qualityChecks; table(subjectLabel, subjectId, count, ...
            nnObserved, baryObserved, fspFrequencyError, ranfGridError, ...
            ranfObserved, sum(fixedMask), ...
            all(cellfun(@(x) all(isfinite(x), 'all'), dbValues)), ...
            'VariableNames', {'SubjectLabel', 'SubjectID', 'DirectionCount', ...
            'NaturalNeighborObservedHRTFMaxAbsError', ...
            'BarycentricObservedHRTFMaxAbsError', ...
            'FSPAEFrequencyMaxAbsError_Hz', 'RANFGridMaxAbsError_deg', ...
            'RANFObservedHRIRMaxAbsError', 'EvaluationDirectionCount', ...
            'AllFinite'})]; %#ok<AGROW>
        fprintf('Ten-method sensitivity [%d/%d] %s Q%d\n', ...
            (subjectIndex - 1) * numel(counts) + countIndex, ...
            height(subjects) * numel(counts), subjectLabel, count);
    end
end

expectedRows = height(subjects) * 10 * 3 * 4;
assert(height(metricLong) == expectedRows && all(isfinite(metricLong.Value_dB)), ...
    'Ten-method metric table is incomplete.');
replicates = double(config.statistics.bootstrap_replicates);
seed = double(config.statistics.bootstrap_seed);
aggregate = aggregate_metrics(metricLong, methodIds, metricIds, counts, replicates, seed);
effects = direction_effects(metricLong, methodIds, metricIds, replicates, seed);
interactions = bounded_interactions(metricLong, methodIds, metricIds, replicates, seed);
assert(height(aggregate) == 120 && height(effects) == 80 && ...
    height(interactions) == 72, 'Frozen result-table cardinality failed.');
outputRoot = fullfile(projectRoot, 'results', char(outputName));
assert(~isfolder(outputRoot), 'Refusing to overwrite %s', outputRoot);
mkdir(outputRoot); figuresRoot = fullfile(outputRoot, 'figures'); mkdir(figuresRoot);
writetable(metricLong, fullfile(outputRoot, 'metric_long.csv'));
writetable(aggregate, fullfile(outputRoot, 'aggregate_metrics.csv'));
writetable(effects, fullfile(outputRoot, 'paired_direction_effects.csv'));
writetable(interactions, fullfile(outputRoot, 'bounded_interactions.csv'));
writetable(qualityChecks, fullfile(outputRoot, 'quality_checks.csv'));
plot_sensitivity(aggregate, methodIds, metricIds, metricLabels, figuresRoot);
summary = struct('schema_version', '1.0', 'status', 'completed', ...
    'split', 'val', 'subject_count', height(subjects), ...
    'method_count', numel(methodIds), 'direction_counts', counts, ...
    'fixed_evaluation_direction_count', sum(fixedMask), ...
    'metric_row_count', height(metricLong), ...
    'aggregate_row_count', height(aggregate), ...
    'direction_effect_row_count', height(effects), ...
    'bounded_interaction_row_count', height(interactions), ...
    'bootstrap_replicates', replicates, 'bootstrap_seed', seed, ...
    'all_finite', all(isfinite(metricLong.Value_dB)) && ...
        all(qualityChecks.AllFinite), 'test_subject_count_read', 0, ...
    'aggregate', table2struct(aggregate), ...
    'paired_direction_effects', table2struct(effects), ...
    'bounded_interactions', table2struct(interactions));
write_json(fullfile(outputRoot, 'summary.json'), summary);
fprintf('Ten-method direction sensitivity complete: %s\n', outputRoot);
end

function environment = initialize_worker(supdeqDir, compatibilityRoot)
originalDir = pwd; restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir); supdeq_start; addpath(compatibilityRoot, '-begin');
clear findvoronoi; environment = struct('initialized', true);
clear restoreDirectory;
end

function operators = build_operators(counts, gridTable, fixedMask)
operators = cell(numel(counts), 1);
for index = 1:numel(counts)
    count = counts(index); rows = gridTable(gridTable.direction_count == count, :);
    assert(height(rows) == count, 'Invalid Q%d grid.', count);
    referenceGrid = [double(rows.azimuth_deg), double(rows.colatitude_deg)];
    operators{index} = struct('sourceIndices', ...
        double(rows.source_index_zero_based) + 1, 'sparseGrid', referenceGrid, ...
        'referenceGrid', [], 'fixedMask', fixedMask);
end
% The full target grid is identical in every prepared cache; populate it in
% the caller from cache.referenceGrid to avoid a second grid source.
end

function compatibilityRoot = activate_findvoronoi_compatibility(projectRoot)
sourceFile = fullfile(projectRoot, 'external', 'SUpDEq', 'thirdParty', ...
    'sfs-matlab-2.5.0', 'SFS_general', 'findvoronoi.m');
sourceText = fileread(sourceFile); oldText = 'for n = 1:size(idx)';
assert(numel(strfind(sourceText, oldText)) == 2, ... %#ok<STREMP>
    'Unexpected upstream findvoronoi revision.');
compatibilityRoot = fullfile(projectRoot, 'artifacts', 'matlab_compat', ...
    'sfs_findvoronoi_r2026');
if ~isfolder(compatibilityRoot), mkdir(compatibilityRoot); end
path = fullfile(compatibilityRoot, 'findvoronoi.m');
handle = fopen(path, 'w'); assert(handle >= 0); cleanup = onCleanup(@() fclose(handle));
fprintf(handle, '%s', strrep(sourceText, oldText, 'for n = 1:size(idx,1)'));
clear cleanup;
end

function selectedDb = selected_db(left, right, mask)
spectra = cat(3, double(left), double(right));
selectedDb = 20 * log10(max(abs(permute(spectra(:, mask, :), ...
    [2, 1, 3])), 1e-10));
end

function features = build_direction_features(grid)
azimuth = double(grid(:, 1)); elevation = 90 - double(grid(:, 2));
features = [azimuth.'; elevation.'; ...
    (cosd(elevation) .* cosd(azimuth)).'; ...
    (cosd(elevation) .* sind(azimuth)).'; sind(elevation).'; ...
    double(grid(:, 3)).'];
end

function [selectedDb, hrir] = method_representation(spectra, mask)
assert(isequal(size(spectra), [793, 513, 2]));
selectedDb = 20 * log10(max(abs(permute(spectra(:, mask, :), ...
    [2, 1, 3])), 1e-10));
hrir = spectra_to_hrir(spectra);
end

function [selectedDb, hrir] = load_residual(path, cache)
residualDb = double(h5read(path, '/predicted_residual_db'));
assert(isequal(size(residualDb), [sum(cache.frequencyMask), 793, 2]) && ...
    all(isfinite(residualDb), 'all'), 'Invalid residual prediction: %s', path);
spectra = cat(3, double(cache.mcaLeft), double(cache.mcaRight));
selected = permute(spectra(:, cache.frequencyMask, :), [2, 1, 3]);
phase = angle(selected);
selectedDb = 20 * log10(max(abs(selected), 1e-10)) + residualDb;
for ear = 1:2
    value = spectra(:, :, ear).';
    value(cache.frequencyMask, :) = 10 .^ (selectedDb(:, :, ear) / 20) .* ...
        exp(1i * phase(:, :, ear));
    spectra(:, :, ear) = value.';
end
hrir = spectra_to_hrir(spectra);
end

function [selectedDb, hrir, frequencyError] = load_fsp(path, mask, allFrequency)
magnitude = double(h5read(path, '/predicted_magnitude_db'));
hrirRaw = double(h5read(path, '/predicted_hrir'));
frequency = double(h5read(path, '/frequency_hz'));
assert(isequal(size(magnitude), [512, 2, 793]) && ...
    isequal(size(hrirRaw), [256, 2, 793]), 'Unexpected FSP-AE shape.');
selected = find(mask);
selectedDb = permute(magnitude(selected - 1, :, :), [1, 3, 2]);
hrir = permute(hrirRaw, [1, 3, 2]);
frequencyError = max(abs(frequency(:) - double(allFrequency(2:end))));
assert(frequencyError <= 1e-3 && all(isfinite(selectedDb), 'all') && ...
    all(isfinite(hrir), 'all'), 'Invalid FSP-AE representation.');
end

function [selectedDb, hrir, gridError] = load_ranf(path, referenceGrid, mask)
sofa = SOFAload(path); positions = double(sofa.SourcePosition);
azimuth = double(referenceGrid(:, 1)); elevation = 90 - double(referenceGrid(:, 2));
gridError = max([abs(mod(positions(:, 1) - azimuth + 180, 360) - 180); ...
    abs(positions(:, 2) - elevation)]);
assert(gridError <= 1e-8, 'RANF direction order mismatch.');
hrir = permute(double(sofa.Data.IR), [3, 1, 2]);
spectra = complex(zeros(793, 513, 2));
for ear = 1:2
    full = fft(hrir(:, :, ear), 1024, 1);
    spectra(:, :, ear) = full(1:513, :).';
end
selectedDb = 20 * log10(max(abs(permute(spectra(:, mask, :), ...
    [2, 1, 3])), 1e-10));
end

function hrir = spectra_to_hrir(spectra)
hrir = zeros(256, 793, 2);
for ear = 1:2
    full = AKsingle2bothSidedSpectrum(spectra(:, :, ear).');
    value = real(ifft(full)); hrir(:, :, ear) = value(1:256, :);
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
values(1) = 0.5 * (mean(abs(leftErb) * normalized_weights(weights(fixedMask))) + ...
    mean(abs(rightErb) * normalized_weights(weights(fixedMask))));
values(2) = 0.5 * (mean(abs(leftErb(:, left25(fixedMask))) * ...
    normalized_weights(weights(left25))) + ...
    mean(abs(rightErb(:, right25(fixedMask))) * normalized_weights(weights(right25))));
values(3) = 0.5 * (high_frequency_error(selectedDb(:, :, 1), ...
    referenceDb(:, :, 1), frequencyHz, leftHemisphere, weights) + ...
    high_frequency_error(selectedDb(:, :, 2), referenceDb(:, :, 2), ...
    frequencyHz, rightHemisphere, weights));
ild = calculate_ild(hrir(:, :, 1), hrir(:, :, 2));
values(4) = mean(abs(ild(horizontal) - referenceIld(horizontal)));
assert(all(isfinite(values)), 'Non-finite strict metric.');
end

function aggregate = aggregate_metrics(long, methods, metrics, counts, replicates, seed)
rng(seed, 'twister'); aggregate = table();
for method = methods
    label = long.MethodLabel(find(long.Method == method, 1));
    for metric = metrics
        for count = counts
            values = long.Value_dB(long.Method == method & long.Metric == metric & ...
                long.DirectionCount == count);
            means = mean(values(randi(numel(values), numel(values), replicates)), 1);
            interval = prctile(means, [2.5, 97.5]);
            aggregate = [aggregate; table(method, label, count, metric, ...
                numel(values), mean(values), std(values), interval(1), interval(2), ...
                'VariableNames', {'Method', 'MethodLabel', 'DirectionCount', ...
                'Metric', 'SubjectCount', 'Mean_dB', 'SD_dB', ...
                'Bootstrap95Lower_dB', 'Bootstrap95Upper_dB'})]; %#ok<AGROW>
        end
    end
end
end

function effects = direction_effects(long, methods, metrics, replicates, seed)
rng(seed, 'twister'); effects = table();
for method = methods
    for metric = metrics
        center = sortrows(long(long.Method == method & long.Metric == metric & ...
            long.DirectionCount == 26, :), 'SubjectID');
        for count = [14, 50]
            current = sortrows(long(long.Method == method & long.Metric == metric & ...
                long.DirectionCount == count, :), 'SubjectID');
            assert(isequal(current.SubjectID, center.SubjectID));
            difference = current.Value_dB - center.Value_dB;
            means = mean(difference(randi(numel(difference), ...
                numel(difference), replicates)), 1);
            interval = prctile(means, [2.5, 97.5]);
            effects = [effects; table(method, count, 26, metric, ...
                mean(difference), interval(1), interval(2), ...
                sum(difference < 0), sum(difference == 0), sum(difference > 0), ...
                'VariableNames', {'Method', 'DirectionCount', ...
                'ReferenceDirectionCount', 'Metric', 'QMinusQ26Mean_dB', ...
                'Bootstrap95Lower_dB', 'Bootstrap95Upper_dB', ...
                'QWins', 'Ties', 'QLosses'})]; %#ok<AGROW>
        end
    end
end
end

function interactions = bounded_interactions(long, methods, metrics, replicates, seed)
rng(seed, 'twister'); interactions = table();
for method = methods(methods ~= "BOUNDED")
    for metric = metrics
        bounded26 = sortrows(long(long.Method == "BOUNDED" & ...
            long.Metric == metric & long.DirectionCount == 26, :), 'SubjectID');
        method26 = sortrows(long(long.Method == method & ...
            long.Metric == metric & long.DirectionCount == 26, :), 'SubjectID');
        for count = [14, 50]
            boundedQ = sortrows(long(long.Method == "BOUNDED" & ...
                long.Metric == metric & long.DirectionCount == count, :), 'SubjectID');
            methodQ = sortrows(long(long.Method == method & ...
                long.Metric == metric & long.DirectionCount == count, :), 'SubjectID');
            assert(isequal(boundedQ.SubjectID, methodQ.SubjectID) && ...
                isequal(boundedQ.SubjectID, bounded26.SubjectID) && ...
                isequal(boundedQ.SubjectID, method26.SubjectID));
            difference = (methodQ.Value_dB - method26.Value_dB) - ...
                (boundedQ.Value_dB - bounded26.Value_dB);
            means = mean(difference(randi(numel(difference), ...
                numel(difference), replicates)), 1);
            interval = prctile(means, [2.5, 97.5]);
            interactions = [interactions; table(method, count, metric, ...
                mean(difference), interval(1), interval(2), ...
                'VariableNames', {'Comparator', 'DirectionCount', 'Metric', ...
                'ComparatorMinusBoundedSensitivity_dB', ...
                'Bootstrap95Lower_dB', 'Bootstrap95Upper_dB'})]; %#ok<AGROW>
        end
    end
end
end

function plot_sensitivity(aggregate, methods, metrics, labels, root)
figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1300, 850]);
layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
colors = lines(numel(methods));
for metricIndex = 1:numel(metrics)
    axisHandle = nexttile(layout); hold(axisHandle, 'on');
    for methodIndex = 1:numel(methods)
        rows = sortrows(aggregate(aggregate.Method == methods(methodIndex) & ...
            aggregate.Metric == metrics(metricIndex), :), 'DirectionCount');
        errorbar(axisHandle, rows.DirectionCount, rows.Mean_dB, ...
            rows.Mean_dB - rows.Bootstrap95Lower_dB, ...
            rows.Bootstrap95Upper_dB - rows.Mean_dB, '-o', ...
            'Color', colors(methodIndex, :), 'LineWidth', 1.2, ...
            'MarkerFaceColor', colors(methodIndex, :));
    end
    xticks(axisHandle, [14, 26, 50]); grid(axisHandle, 'on'); box(axisHandle, 'on');
    xlabel(axisHandle, 'Observed directions'); ylabel(axisHandle, 'Error (dB)');
    title(axisHandle, labels(metricIndex));
end
legend(axisHandle, cellstr(aggregate.MethodLabel( ...
    aggregate.DirectionCount == 14 & aggregate.Metric == metrics(end))), ...
    'Location', 'eastoutside');
title(layout, 'Ten-method input-direction sensitivity (44 validation listeners)');
exportgraphics(figureHandle, fullfile(root, 'ten_method_direction_sensitivity.png'), ...
    'Resolution', 300);
exportgraphics(figureHandle, fullfile(root, 'ten_method_direction_sensitivity.pdf'), ...
    'ContentType', 'vector');
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
weights = weights(:); assert(all(isfinite(weights)) && all(weights > 0));
weights = weights / sum(weights);
end

function value = high_frequency_error(estimateDb, referenceDb, frequencyHz, mask, weights)
frequencyMask = frequencyHz > 10000 & frequencyHz <= 20000;
value = mean(abs(estimateDb(frequencyMask, mask) - ...
    referenceDb(frequencyMask, mask)) * normalized_weights(weights(mask)));
end

function write_json(path, value)
handle = fopen(path, 'w', 'n', 'UTF-8'); assert(handle >= 0);
cleanup = onCleanup(@() fclose(handle));
fprintf(handle, '%s\n', jsonencode(value)); clear cleanup;
end
