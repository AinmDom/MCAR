function export_sonicom_classical_secondary_validation(subjectLimit, outputName, ...
        publishResults, splitName, mcarPredictionName, allowTest, ...
        fspPredictionName, ranfPredictionName, predictionOutputName)
%EXPORT_SONICOM_CLASSICAL_SECONDARY_VALIDATION Export five classical baselines.
%
% Evaluates six established methods, plus optional FSP-AE and RANF
% predictions, with the same frozen SONICOM split and strict
% metrics used by evaluate_sonicom_validation_reconstruction:
%   SH only, SUpDEq + SH, SUpDEq + Natural Neighbor,
%   SUpDEq + Barycentric, MCA, and frozen MCAR v3.2.
%
% Natural-neighbor and barycentric weights are constructed with the exact
% geometry routines bundled by the upstream SUpDEq toolbox. Their batched
% application is checked against supdeq_interpHRTF on the first subject.
% Test evaluation requires splitName='test' and allowTest=true.

if nargin < 1 || isempty(subjectLimit)
    subjectLimit = inf;
end
if nargin < 2 || isempty(outputName)
    outputName = 'sonicom_supdeq_nn_bary_q26_strict_validation';
end
if nargin < 3 || isempty(publishResults)
    publishResults = true;
end
if nargin < 4 || isempty(splitName)
    splitName = 'val';
end
if nargin < 5 || isempty(mcarPredictionName)
    mcarPredictionName = 'sonicom_q26_validation_mlp_cnn_v32_locked_ild075';
end
if nargin < 6 || isempty(allowTest)
    allowTest = false;
end
if nargin < 7
    fspPredictionName = '';
end
if nargin < 8
    ranfPredictionName = '';
end
if nargin < 9 || isempty(predictionOutputName)
    predictionOutputName = ...
        'sonicom_classical_secondary_baselines_v1_validation';
end
validateattributes(subjectLimit, {'numeric'}, {'scalar', 'positive'});
validateattributes(outputName, {'char', 'string'}, {'scalartext'});
validateattributes(publishResults, {'logical', 'numeric'}, {'scalar'});
validateattributes(splitName, {'char', 'string'}, {'scalartext'});
validateattributes(mcarPredictionName, {'char', 'string'}, {'scalartext'});
validateattributes(allowTest, {'logical', 'numeric'}, {'scalar'});
if ~isempty(fspPredictionName)
    validateattributes(fspPredictionName, {'char', 'string'}, {'scalartext'});
end
if ~isempty(ranfPredictionName)
    validateattributes(ranfPredictionName, {'char', 'string'}, {'scalartext'});
end
outputName = char(outputName);
publishResults = logical(publishResults);
splitName = string(splitName);
mcarPredictionName = char(mcarPredictionName);
allowTest = logical(allowTest);
fspPredictionName = char(fspPredictionName);
includeFsp = ~isempty(fspPredictionName);
ranfPredictionName = char(ranfPredictionName);
includeRanf = ~isempty(ranfPredictionName);
predictionOutputName = char(predictionOutputName);
assert(any(splitName == ["val", "test"]), 'splitName must be val or test.');
assert(splitName ~= "test" || allowTest, ...
    'The locked SONICOM test split requires allowTest=true.');
assert((splitName == "val" && ~allowTest) || ...
    (splitName == "test" && allowTest), ...
    'allowTest must be false for val and true for test.');

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
supdeqDir = fullfile(projectRoot, 'external', 'SUpDEq');
datasetRoot = fullfile(projectRoot, 'data', 'processed', ...
    'sonicom_residual_q26_v1');
sofaRoot = fullfile(projectRoot, 'data', 'HRTF', ...
    'sonicom_measured_ffcmp_minphase_44k1', 'subjects');
configRoot = fullfile(projectRoot, 'configs', 'data');
splitFile = fullfile(configRoot, 'sonicom_subject_split_v1.csv');
q26File = fullfile(configRoot, 'sonicom_sparse_grid_q26_v1.csv');
referenceGridFile = fullfile(configRoot, 'sonicom_reference_grid_v1.csv');
mcarPredictionRoot = fullfile(projectRoot, 'artifacts', ...
    'reconstruction', mcarPredictionName);
fspPredictionRoot = fullfile(projectRoot, 'artifacts', ...
    'reconstruction', fspPredictionName);
ranfPredictionRoot = fullfile(projectRoot, 'artifacts', ...
    'reconstruction', ranfPredictionName);
classicalPredictionRoot = fullfile(projectRoot, 'artifacts', ...
    'reconstruction', predictionOutputName);
if publishResults
    outputRoot = fullfile(projectRoot, 'results', outputName);
else
    outputRoot = fullfile(projectRoot, 'artifacts', 'evaluation', outputName);
end
figuresRoot = fullfile(outputRoot, 'figures');

assert(isfolder(supdeqDir), 'SUpDEq directory not found: %s', supdeqDir);
assert(isfolder(datasetRoot), 'Dataset root not found: %s', datasetRoot);
assert(isfolder(sofaRoot), 'SOFA root not found: %s', sofaRoot);
assert(isfolder(mcarPredictionRoot), ...
    'MCAR prediction root not found: %s', mcarPredictionRoot);
assert(~includeFsp || isfolder(fspPredictionRoot), ...
    'FSP-AE prediction root not found: %s', fspPredictionRoot);
assert(~includeRanf || isfolder(ranfPredictionRoot), ...
    'RANF prediction root not found: %s', ranfPredictionRoot);
assert(isfile(splitFile) && isfile(q26File) && isfile(referenceGridFile), ...
    'A frozen SONICOM configuration file is missing.');
if ~isfolder(figuresRoot)
    mkdir(figuresRoot);
end
assert(~isfolder(classicalPredictionRoot), ...
    'Refusing to overwrite classical prediction root: %s', classicalPredictionRoot);
mkdir(classicalPredictionRoot);

originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir);
supdeq_start;
cd(projectRoot);
findvoronoiCompatibilityFile = activate_findvoronoi_compatibility(projectRoot);

splitTable = readtable(splitFile, 'TextType', 'string');
q26Table = readtable(q26File, 'TextType', 'string');
referenceGridTable = readtable(referenceGridFile, 'TextType', 'string');
splitRows = splitTable(splitTable.split == splitName, :);
assert(height(splitRows) == 44, 'Expected exactly 44 locked %s subjects.', splitName);
subjectCount = min(height(splitRows), floor(subjectLimit));
splitRows = splitRows(1:subjectCount, :);
assert(height(q26Table) == 26 && height(referenceGridTable) == 793, ...
    'Frozen Q26/reference grid dimensions are invalid.');

sparseGrid = [double(q26Table.azimuth_deg), ...
    double(q26Table.colatitude_deg)];
referenceGrid = [double(referenceGridTable.azimuth_deg), ...
    double(referenceGridTable.colatitude_deg), ...
    double(referenceGridTable.solid_angle_weight)];
q26Indices = double(q26Table.source_index_zero_based) + 1;
headRadiusM = 0.09;
tikhonovEpsilon = 1e-2;
fftOversize = 4;
samplingRateHz = 44100;
nfft = 1024;
frequencyHzAll = (0:(nfft / 2)).' * samplingRateHz / nfft;

fprintf('Building fixed Q26 Natural-Neighbor and Barycentric operators...\n');
configurationInterpolationMask = logical( ...
    referenceGridTable.is_interpolation_evaluation);
nnWeights = natural_neighbor_weights(sparseGrid, referenceGrid(:, 1:2), ...
    q26Indices, configurationInterpolationMask);
baryWeights = barycentric_weights(sparseGrid, referenceGrid(:, 1:2), ...
    q26Indices, configurationInterpolationMask);
assert(max(abs(sum(nnWeights, 2) - 1)) < 1e-10, ...
    'Natural-neighbor rows do not sum to one.');
assert(max(abs(sum(baryWeights, 2) - 1)) < 1e-10, ...
    'Barycentric rows do not sum to one.');
eqDataset = supdeq_getEqDataset(44, 2 * headRadiusM, nfft, samplingRateHz);
[eqSparseLeft, eqSparseRight] = supdeq_getEqTF( ...
    eqDataset, sparseGrid, 'DEG', 2, 'ak', false);
[eqTargetLeft, eqTargetRight] = supdeq_getEqTF( ...
    eqDataset, referenceGrid, 'DEG', 2, 'ak', false);

methodIds = ["SHOnly", "SUpDEqSH", "SUpDEqNN", ...
    "SUpDEqBary", "MCA"];
methodLabels = ["SH only", "SUpDEq + SH", ...
    "SUpDEq + Natural Neighbor", "SUpDEq + Barycentric", ...
    "MCA"];
if includeRanf
    methodIds(end + 1) = "RANF";
    methodLabels(end + 1) = "RANF";
end
methodIds(end + 1) = "MCARv32";
methodLabels(end + 1) = "MCAR v3.2";
if includeFsp
    methodIds(end + 1) = "FSPAE";
    methodLabels(end + 1) = "FSP-AE-Q26";
end
metricIds = ["FullSphereERB", "Contralateral25ERB", ...
    "ContralateralHighFrequency", "HorizontalILDMAE"];
metricLong = table();
qualityChecks = table();
perSubject = table(splitRows.subject_id, ...
    arrayfun(@(label) sscanf(char(label), 'P%d'), splitRows.subject_id), ...
    'VariableNames', {'SubjectLabel', 'SubjectID'});

for subjectIndex = 1:subjectCount
    subjectLabel = splitRows.subject_id(subjectIndex);
    sourceFile = fullfile(datasetRoot, 'subjects', char(subjectLabel), 'q26.h5');
    predictionFile = fullfile(mcarPredictionRoot, 'subjects', ...
        char(subjectLabel), 'prediction.h5');
    fspPredictionFile = fullfile(fspPredictionRoot, 'subjects', ...
        char(subjectLabel), 'prediction.h5');
    ranfPredictionFile = fullfile(ranfPredictionRoot, 'subjects', ...
        char(subjectLabel), 'prediction.sofa');
    sofaFile = fullfile(sofaRoot, sprintf( ...
        '%s_FreeFieldCompMinPhase_44kHz.sofa', subjectLabel));
    assert(isfile(sourceFile), 'Missing source HDF5: %s', sourceFile);
    assert(isfile(predictionFile), 'Missing MCAR prediction: %s', predictionFile);
    assert(~includeFsp || isfile(fspPredictionFile), ...
        'Missing FSP-AE prediction: %s', fspPredictionFile);
    assert(~includeRanf || isfile(ranfPredictionFile), ...
        'Missing RANF prediction: %s', ranfPredictionFile);
    assert(isfile(sofaFile), 'Missing SOFA: %s', sofaFile);
    assert(string(h5readatt(sourceFile, '/', 'split')) == splitName, ...
        'Source split mismatch for %s.', subjectLabel);
    assert(string(h5readatt(predictionFile, '/', 'split')) == splitName, ...
        'Prediction split mismatch for %s.', subjectLabel);
    assert(~includeFsp || string(h5readatt( ...
        fspPredictionFile, '/', 'split')) == splitName, ...
        'FSP-AE prediction split mismatch for %s.', subjectLabel);
    assert(abs(double(h5readatt(sourceFile, '/', 'tikhonov_epsilon')) - ...
        tikhonovEpsilon) < 1e-12, 'Tikhonov mismatch for %s.', subjectLabel);

    sofa = SOFAload(sofaFile);
    [referenceSpectra, referenceHrir] = reference_spectra( ...
        sofa, referenceGridTable, nfft);
    ranfSpectra = [];
    ranfHrir = [];
    ranfObservedError = NaN;
    if includeRanf
        ranfSofa = SOFAload(ranfPredictionFile);
        [ranfSpectra, ranfHrir] = reference_spectra( ...
            ranfSofa, referenceGridTable, nfft);
        ranfObservedError = max(abs( ...
            ranfHrir(:, q26Indices, :) - referenceHrir(:, q26Indices, :)), ...
            [], 'all');
        assert(ranfObservedError <= 1e-12, ...
            'RANF altered a Q26 observation for %s: %.6g.', ...
            subjectLabel, ranfObservedError);
    end
    sparseHrtf = struct( ...
        'HRTF_L', referenceSpectra(q26Indices, :, 1), ...
        'HRTF_R', referenceSpectra(q26Indices, :, 2), ...
        'f', frequencyHzAll, 'fs', samplingRateHz, 'Nmax', 3, ...
        'FFToversize', fftOversize, 'samplingGrid', sparseGrid, ...
        'sourceDistance', double(sofa.SourcePosition(1, 3)));

    shOnly = supdeq_interpHRTF(sparseHrtf, referenceGrid, ...
        'None', 'SH', nan, headRadiusM, tikhonovEpsilon, ...
        true, 0, true, 'fadeDown');
    supdeqSh = supdeq_interpHRTF(sparseHrtf, referenceGrid, ...
        'SUpDEq', 'SH', nan, headRadiusM, tikhonovEpsilon, ...
        true, 0, true, 'fadeDown');
    nnSpectra = apply_supdeq_operator(sparseHrtf, nnWeights, ...
        eqSparseLeft, eqSparseRight, eqTargetLeft, eqTargetRight);
    barySpectra = apply_supdeq_operator(sparseHrtf, baryWeights, ...
        eqSparseLeft, eqSparseRight, eqTargetLeft, eqTargetRight);

    [mcaSpectra, mcarSpectra, selectedIndices, interpolationMask, ...
        referenceDb, directionFeatures, referenceIldMetadata, ...
        referenceIldError] = load_mca_and_mcar( ...
        sourceFile, predictionFile, referenceHrir);
    methodSpectra = struct( ...
        'SHOnly', cat(3, shOnly.HRTF_L, shOnly.HRTF_R), ...
        'SUpDEqSH', cat(3, supdeqSh.HRTF_L, supdeqSh.HRTF_R), ...
        'SUpDEqNN', nnSpectra, ...
        'SUpDEqBary', barySpectra, ...
        'MCA', mcaSpectra, ...
        'MCARv32', mcarSpectra);
    classicalMethodIds = ["SHOnly", "SUpDEqSH", "SUpDEqNN", ...
        "SUpDEqBary", "MCA"];
    for classicalIndex = 1:numel(classicalMethodIds)
        classicalMethod = classicalMethodIds(classicalIndex);
        [classicalMagnitudeDb, classicalHrir] = method_representation( ...
            methodSpectra.(char(classicalMethod)), selectedIndices, 256);
        write_classical_prediction(classicalPredictionRoot, subjectLabel, ...
            classicalMethod, classicalMagnitudeDb, classicalHrir, ...
            frequencyHzAll(selectedIndices), splitName);
    end
    fspFrequencyError = NaN;
    if includeFsp
        [fspMagnitudeDb, fspHrir, fspFrequencyError] = ...
            load_fsp_prediction(fspPredictionFile, selectedIndices, ...
            frequencyHzAll);
    end

    nnParityError = NaN;
    baryParityError = NaN;
    if subjectIndex == 1
        nativeTargetGrid = referenceGrid(configurationInterpolationMask, :);
        nativeNn = supdeq_interpHRTF(sparseHrtf, nativeTargetGrid, ...
            'SUpDEq', 'NN', nan, headRadiusM, 0, ...
            true, 0, true, 'fadeDown');
        nativeBary = supdeq_interpHRTF(sparseHrtf, nativeTargetGrid, ...
            'SUpDEq', 'Bary', nan, headRadiusM, 0, ...
            true, 0, true, 'fadeDown');
        nnParityError = max(abs(nnSpectra(configurationInterpolationMask, :, :) - ...
            cat(3, nativeNn.HRTF_L, nativeNn.HRTF_R)), [], 'all');
        baryParityError = max(abs(barySpectra(configurationInterpolationMask, :, :) - ...
            cat(3, nativeBary.HRTF_L, nativeBary.HRTF_R)), [], 'all');
        assert(nnParityError < 1e-9 && baryParityError < 1e-9, ...
            'Batched NN/Bary implementation differs from upstream SUpDEq.');
    end

    metricValues = zeros(numel(methodIds), numel(metricIds));
    for methodIndex = 1:numel(methodIds)
        if methodIds(methodIndex) == "FSPAE"
            selectedMagnitudeDb = fspMagnitudeDb;
            hrir = fspHrir;
        elseif methodIds(methodIndex) == "RANF"
            selectedMagnitudeDb = 20 * log10(max(abs(permute( ...
                ranfSpectra(:, selectedIndices, :), [2, 1, 3])), 1e-10));
            hrir = ranfHrir;
        else
            [selectedMagnitudeDb, hrir] = method_representation( ...
                methodSpectra.(char(methodIds(methodIndex))), ...
                selectedIndices, 256);
        end
        if ~ismember(methodIds(methodIndex), classicalMethodIds)
            write_classical_prediction(classicalPredictionRoot, subjectLabel, ...
                methodIds(methodIndex), selectedMagnitudeDb, hrir, ...
                frequencyHzAll(selectedIndices), splitName);
        end
        metricValues(methodIndex, :) = strict_metrics( ...
            selectedMagnitudeDb, hrir, referenceDb, referenceHrir, ...
            referenceIldMetadata, directionFeatures, interpolationMask, ...
            frequencyHzAll(selectedIndices), samplingRateHz);
        for metricIndex = 1:numel(metricIds)
            metricLong = [metricLong; table(subjectLabel, ...
                perSubject.SubjectID(subjectIndex), methodIds(methodIndex), ...
                methodLabels(methodIndex), metricIds(metricIndex), ...
                metricValues(methodIndex, metricIndex), ...
                'VariableNames', {'SubjectLabel', 'SubjectID', ...
                'Method', 'MethodLabel', 'Metric', 'Value_dB'})]; %#ok<AGROW>
            variableName = sprintf('%s_%s_dB', ...
                methodIds(methodIndex), metricIds(metricIndex));
            perSubject.(variableName)(subjectIndex, 1) = ...
                metricValues(methodIndex, metricIndex);
        end
    end
    qualityChecks = [qualityChecks; table(subjectLabel, ...
        perSubject.SubjectID(subjectIndex), referenceIldError, ...
        nnParityError, baryParityError, ranfObservedError, ...
        fspFrequencyError, ...
        sum(interpolationMask), ...
        'VariableNames', {'SubjectLabel', 'SubjectID', ...
        'ReferenceILDMetadataMaxError_dB', ...
        'NaturalNeighborNativeParityMaxAbs', ...
        'BarycentricNativeParityMaxAbs', ...
        'RANFObservedHRIRMaxAbsError', ...
        'FSPAEFrequencyMaxAbsError_Hz', ...
        'InterpolationDirectionCount'})]; %#ok<AGROW>
    fprintf('SONICOM %s baseline [%d/%d]: %s\n', ...
        splitName, subjectIndex, subjectCount, subjectLabel);
end

aggregate = aggregate_metrics(metricLong, methodIds, methodLabels, metricIds);
paperTable = make_paper_table(aggregate, methodIds, methodLabels, metricIds);
writetable(perSubject, fullfile(outputRoot, 'per_subject_metrics.csv'));
writetable(metricLong, fullfile(outputRoot, 'metric_long.csv'));
writetable(qualityChecks, fullfile(outputRoot, 'quality_checks.csv'));
writetable(aggregate, fullfile(outputRoot, 'aggregate_metrics.csv'));
writetable(paperTable, fullfile(outputRoot, 'paper_comparison.csv'));
if includeRanf
    coreMethodIds = ["SHOnly", "SUpDEqSH", "MCA", "RANF", "MCARv32"];
    coreMetricVariables = reshape( ...
        coreMethodIds.' + "_" + metricIds + "_dB", 1, []);
    coreVariableNames = ["SubjectLabel", "SubjectID", coreMetricVariables];
    perSubjectCore = perSubject(:, coreVariableNames);
    paperCore = paperTable(ismember(paperTable.Method, coreMethodIds), :);
    writetable(perSubjectCore, fullfile( ...
        outputRoot, 'per_subject_core_comparison.csv'));
    writetable(paperCore, fullfile(outputRoot, 'paper_core_comparison.csv'));
end
plot_aggregate(aggregate, methodIds, methodLabels, metricIds, ...
    figuresRoot, splitName, subjectCount);

configuration = struct( ...
    'schemaVersion', '1.0', ...
    'createdOn', char(datetime('now', 'TimeZone', 'local')), ...
    'split', char(splitName), ...
    'testSubjectCountRead', subjectCount * double(splitName == "test"), ...
    'subjectCount', subjectCount, ...
    'sparseGrid', 'SONICOM-Q26-v1', ...
    'interpolationDirectionCount', 767, ...
    'headRadiusM', headRadiusM, ...
    'tikhonovEpsilonSH', tikhonovEpsilon, ...
    'mcarPredictionName', mcarPredictionName, ...
    'classicalPredictionName', predictionOutputName, ...
    'fspPredictionName', fspPredictionName, ...
    'ranfPredictionName', ranfPredictionName, ...
    'naturalNeighborImplementation', ...
        'SUpDEq findvoronoi weights; first-subject native parity checked', ...
    'barycentricImplementation', ...
        'SUpDEq TriangleRayIntersection weights; first-subject native parity checked', ...
    'findvoronoiCompatibilityFile', findvoronoiCompatibilityFile, ...
    'metrics', {cellstr(metricIds)});
summary = struct('schema_version', '1.0', 'status', 'completed', ...
    'split', char(splitName), ...
    'test_subject_count_read', subjectCount * double(splitName == "test"), ...
    'subject_count', subjectCount, ...
    'aggregate', table2struct(aggregate), ...
    'configuration', configuration);
write_json(fullfile(outputRoot, 'summary.json'), summary);
write_markdown_report(fullfile(outputRoot, 'README.md'), ...
    paperTable, configuration, qualityChecks);
fprintf('SONICOM interpolation baseline evaluation complete: %s\n', outputRoot);
disp(paperTable);
clear restoreDirectory;
end

function write_classical_prediction(root, subjectLabel, method, ...
        selectedMagnitudeDb, hrir, frequencyHz, splitName)
subjectRoot = fullfile(root, 'subjects', char(subjectLabel), char(method));
if ~isfolder(subjectRoot)
    mkdir(subjectRoot);
end
path = fullfile(subjectRoot, 'prediction.h5');
assert(~isfile(path), 'Refusing to overwrite prediction: %s', path);
assert(isequal(size(selectedMagnitudeDb), [463, 793, 2]), ...
    'Unexpected selected magnitude shape.');
assert(isequal(size(hrir), [256, 793, 2]), ...
    'Unexpected HRIR shape.');
assert(all(isfinite(selectedMagnitudeDb), 'all') && all(isfinite(hrir), 'all'), ...
    'Prediction contains non-finite values.');

% MATLAB reverses HDF5 dimension order relative to h5py. Writing the
% spectral tensor as [frequency,direction,ear] exposes [ear,direction,
% frequency] to Python; writing HRIR as [sample,ear,direction] exposes
% [direction,ear,sample].
h5create(path, '/predicted_magnitude_db', size(selectedMagnitudeDb), ...
    'Datatype', 'single');
h5write(path, '/predicted_magnitude_db', single(selectedMagnitudeDb));
hrirForHdf5 = permute(hrir, [1, 3, 2]);
h5create(path, '/predicted_hrir', size(hrirForHdf5), 'Datatype', 'single');
h5write(path, '/predicted_hrir', single(hrirForHdf5));
h5create(path, '/frequency_hz', size(frequencyHz), 'Datatype', 'double');
h5write(path, '/frequency_hz', double(frequencyHz));
h5writeatt(path, '/', 'split', char(splitName));
h5writeatt(path, '/', 'subject_id', char(subjectLabel));
h5writeatt(path, '/', 'method', char(method));
h5writeatt(path, '/', 'test_subject_count_read', ...
    int32(splitName == "test"));
end

function compatibilityFile = activate_findvoronoi_compatibility(projectRoot)
% The SFS 2.5.0 dependency uses `1:size(idx)`, which is rejected by current
% MATLAB releases because size(idx) is a two-element vector. Generate an
% artifacts-only shadow copy with the two loop bounds made explicit. This
% keeps external/SUpDEq untouched and preserves every numerical operation.
sourceFile = fullfile(projectRoot, 'external', 'SUpDEq', 'thirdParty', ...
    'sfs-matlab-2.5.0', 'SFS_general', 'findvoronoi.m');
assert(isfile(sourceFile), 'Upstream findvoronoi.m not found: %s', sourceFile);
sourceText = fileread(sourceFile);
oldText = 'for n = 1:size(idx)';
newText = 'for n = 1:size(idx,1)';
assert(numel(strfind(sourceText, oldText)) == 2, ... %#ok<STREMP>
    'Unexpected upstream findvoronoi.m revision; compatibility patch aborted.');
patchedText = strrep(sourceText, oldText, newText);
compatibilityRoot = fullfile(projectRoot, 'artifacts', ...
    'matlab_compat', 'sfs_findvoronoi_r2026');
if ~isfolder(compatibilityRoot)
    mkdir(compatibilityRoot);
end
compatibilityFile = fullfile(compatibilityRoot, 'findvoronoi.m');
file = fopen(compatibilityFile, 'w');
assert(file ~= -1, 'Could not create findvoronoi compatibility copy.');
cleanup = onCleanup(@() fclose(file));
fprintf(file, '%s', patchedText);
clear cleanup;
addpath(compatibilityRoot, '-begin');
clear findvoronoi;
resolvedFile = which('findvoronoi');
assert(strcmpi(resolvedFile, compatibilityFile), ...
    'Compatibility findvoronoi.m did not take path precedence.');
end

function weights = natural_neighbor_weights( ...
        samplingGrid, targetGrid, sourceTargetIndices, interpolationMask)
[sx, sy, sz] = spherical_to_cartesian(samplingGrid);
[tx, ty, tz] = spherical_to_cartesian(targetGrid);
sourceCartesian = [sx, sy, sz];
targetCartesian = [tx, ty, tz];
weights = zeros(size(targetGrid, 1), size(samplingGrid, 1));
for sourceIndex = 1:numel(sourceTargetIndices)
    weights(sourceTargetIndices(sourceIndex), sourceIndex) = 1;
end
interpolationIndices = find(interpolationMask);
for localIndex = 1:numel(interpolationIndices)
    index = interpolationIndices(localIndex);
    [neighbors, coefficients] = findvoronoi( ...
        sourceCartesian, targetCartesian(index, :));
    for coefficientIndex = 1:numel(neighbors)
        weights(index, neighbors(coefficientIndex)) = ...
            weights(index, neighbors(coefficientIndex)) + ...
            coefficients(coefficientIndex);
    end
end
end

function weights = barycentric_weights( ...
        samplingGrid, targetGrid, sourceTargetIndices, interpolationMask)
[sx, sy, sz] = spherical_to_cartesian(samplingGrid);
[tx, ty, tz] = spherical_to_cartesian(targetGrid);
sourceCartesian = [sx, sy, sz];
targetCartesian = [tx, ty, tz];
hullIndices = convhull(sourceCartesian(:, 1), sourceCartesian(:, 2), ...
    sourceCartesian(:, 3), 'simplify', true);
triangleA = sourceCartesian(hullIndices(:, 1), :);
triangleB = sourceCartesian(hullIndices(:, 2), :);
triangleC = sourceCartesian(hullIndices(:, 3), :);
weights = zeros(size(targetGrid, 1), size(samplingGrid, 1));
for sourceIndex = 1:numel(sourceTargetIndices)
    weights(sourceTargetIndices(sourceIndex), sourceIndex) = 1;
end
interpolationIndices = find(interpolationMask);
for localIndex = 1:numel(interpolationIndices)
    index = interpolationIndices(localIndex);
    [intersect, ~, u, v, ~] = TriangleRayIntersection( ...
        [0, 0, 0], targetCartesian(index, :), ...
        triangleA, triangleB, triangleC, 'border', 'inclusive');
    triangleIndex = find(intersect, 1, 'first');
    assert(~isempty(triangleIndex), ...
        'No barycentric triangle found for target direction %d.', index);
    coefficients = [1 - u(triangleIndex) - v(triangleIndex), ...
        u(triangleIndex), v(triangleIndex)];
    weights(index, hullIndices(triangleIndex, :)) = coefficients;
end
end

function [x, y, z] = spherical_to_cartesian(grid)
[x, y, z] = sph2cart(grid(:, 1) / 360 * 2 * pi, ...
    pi / 2 - grid(:, 2) / 360 * 2 * pi, 1);
end

function spectra = apply_supdeq_operator(sparseHrtf, weights, ...
        eqSparseLeft, eqSparseRight, eqTargetLeft, eqTargetRight)
left = (weights * (sparseHrtf.HRTF_L ./ eqSparseLeft)) .* eqTargetLeft;
right = (weights * (sparseHrtf.HRTF_R ./ eqSparseRight)) .* eqTargetRight;
spectra = cat(3, left, right);
assert(all(isfinite(spectra), 'all'), ...
    'SUpDEq interpolation generated non-finite values.');
end

function [spectra, hrir] = reference_spectra(sofa, referenceGridTable, nfft)
assert(strcmp(sofa.GLOBAL_SOFAConventions, 'SimpleFreeFieldHRIR'), ...
    'Unexpected SOFA convention.');
hrirRaw = double(sofa.Data.IR);
assert(isequal(size(hrirRaw), [793, 2, 256]), ...
    'Expected SONICOM Data.IR [793, 2, 256].');
sourcePosition = double(sofa.SourcePosition);
azimuthError = abs(mod(sourcePosition(:, 1) - ...
    double(referenceGridTable.azimuth_deg) + 180, 360) - 180);
elevationError = abs(sourcePosition(:, 2) - ...
    double(referenceGridTable.elevation_deg));
assert(max(azimuthError) < 1e-8 && max(elevationError) < 1e-8, ...
    'SOFA direction order differs from the frozen reference grid.');
hrir = permute(hrirRaw, [3, 1, 2]);
spectra = complex(zeros(793, nfft / 2 + 1, 2));
for ear = 1:2
    fullSpectrum = fft(hrir(:, :, ear), nfft, 1);
    spectra(:, :, ear) = fullSpectrum(1:(nfft / 2 + 1), :).';
end
end

function [mcaSpectra, mcarSpectra, selectedIndices, interpolationMask, ...
        referenceDb, directionFeatures, referenceIldMetadata, ...
        referenceIldError] = load_mca_and_mcar( ...
        sourceFile, predictionFile, referenceHrir)
mcaDb = double(h5read(sourceFile, '/mca_logmag_db'));
mcarResidualDb = double(h5read(predictionFile, '/predicted_residual_db'));
referenceDb = double(h5read(sourceFile, '/reference_logmag_db'));
directionFeatures = double(h5read(sourceFile, '/direction_features'));
interpolationMask = logical(h5read( ...
    sourceFile, '/interpolation_evaluation_mask'));
interpolationMask = interpolationMask(:);
selectedPhase = double(h5read( ...
    sourceFile, '/strict_ild/mca_selected_phase_rad'));
outsideReal = double(h5read( ...
    sourceFile, '/strict_ild/mca_outside_real'));
outsideImag = double(h5read( ...
    sourceFile, '/strict_ild/mca_outside_imag'));
selectedIndices = double(h5read(sourceFile, ...
    '/strict_ild/selected_bin_indices_zero_based')) + 1;
selectedIndices = selectedIndices(:);
outsideIndices = double(h5read(sourceFile, ...
    '/strict_ild/outside_bin_indices_zero_based')) + 1;
outsideIndices = outsideIndices(:);
referenceIldMetadata = double(h5read( ...
    sourceFile, '/strict_ild/reference_ild_db'));
referenceIldMetadata = referenceIldMetadata(:);
assert(isequal(size(mcaDb), [numel(selectedIndices), 793, 2]) && ...
    isequal(size(mcarResidualDb), size(mcaDb)), ...
    'MCA/MCAR tensor shape mismatch.');
mcaSpectra = reconstruct_spectra( ...
    mcaDb, selectedPhase, outsideReal, outsideImag, ...
    selectedIndices, outsideIndices);
mcarSpectra = reconstruct_spectra( ...
    mcaDb + mcarResidualDb, selectedPhase, outsideReal, outsideImag, ...
    selectedIndices, outsideIndices);
referenceIldRaw = calculate_ild(referenceHrir(:, :, 1), referenceHrir(:, :, 2));
referenceIldError = max(abs(referenceIldRaw - referenceIldMetadata));
assert(referenceIldError <= 2e-4, ...
    'Reference ILD metadata mismatch: %.6g dB.', referenceIldError);
end

function spectra = reconstruct_spectra(selectedDb, selectedPhase, ...
        outsideReal, outsideImag, selectedIndices, outsideIndices)
singleSidedCount = numel(selectedIndices) + numel(outsideIndices);
spectra = complex(zeros(size(selectedDb, 2), singleSidedCount, 2));
for ear = 1:2
    earSpectrum = complex(zeros(singleSidedCount, size(selectedDb, 2)));
    earSpectrum(selectedIndices, :) = 10 .^ (selectedDb(:, :, ear) / 20) .* ...
        exp(1i * selectedPhase(:, :, ear));
    earSpectrum(outsideIndices, :) = outsideReal(:, :, ear) + ...
        1i * outsideImag(:, :, ear);
    spectra(:, :, ear) = earSpectrum.';
end
end

function [selectedMagnitudeDb, hrir, frequencyError] = ...
        load_fsp_prediction(predictionFile, selectedIndices, frequencyHzAll)
% Python writes [direction, ear, frequency/sample]; h5read exposes reversed
% dimensions in MATLAB.
magnitudeRaw = double(h5read(predictionFile, ...
    '/predicted_magnitude_db'));
itd = double(h5read(predictionFile, '/predicted_itd_seconds'));
hrirRaw = double(h5read(predictionFile, '/predicted_hrir'));
frequencyHz = double(h5read(predictionFile, '/frequency_hz'));
assert(isequal(size(magnitudeRaw), [512, 2, 793]), ...
    'Unexpected FSP-AE magnitude shape.');
assert(numel(itd) == 793 && all(isfinite(itd), 'all'), ...
    'Unexpected FSP-AE ITD values.');
assert(isequal(size(hrirRaw), [256, 2, 793]), ...
    'Unexpected FSP-AE HRIR shape.');
frequencyHz = frequencyHz(:);
assert(numel(frequencyHz) == 512, ...
    'Unexpected FSP-AE frequency count.');
frequencyError = max(abs(frequencyHz - frequencyHzAll(2:end)));
assert(frequencyError <= 1e-3, ...
    'FSP-AE frequency grid differs from strict evaluation grid.');

magnitudeDb = permute(magnitudeRaw, [1, 3, 2]);
selectedMagnitudeDb = magnitudeDb(selectedIndices - 1, :, :);
hrir = permute(hrirRaw, [1, 3, 2]);
assert(isequal(size(selectedMagnitudeDb), ...
    [numel(selectedIndices), 793, 2]) && ...
    isequal(size(hrir), [256, 793, 2]) && ...
    all(isfinite(selectedMagnitudeDb), 'all') && ...
    all(isfinite(hrir), 'all'), ...
    'FSP-AE strict representation is invalid.');
end

function [selectedMagnitudeDb, hrir] = method_representation( ...
        spectra, selectedIndices, hrirLength)
assert(isequal(size(spectra), [793, 513, 2]), ...
    'Unexpected interpolated spectrum shape.');
selectedMagnitudeDb = 20 * log10(max( ...
    abs(permute(spectra(:, selectedIndices, :), [2, 1, 3])), 1e-10));
hrir = zeros(hrirLength, 793, 2);
for ear = 1:2
    fullSpectrum = AKsingle2bothSidedSpectrum(spectra(:, :, ear).');
    oversizedHrir = real(ifft(fullSpectrum));
    hrir(:, :, ear) = oversizedHrir(1:hrirLength, :);
end
end

function values = strict_metrics(selectedDb, hrir, referenceDb, ...
        referenceHrir, referenceIld, directionFeatures, ...
        interpolationMask, frequencyHz, samplingRateHz)
azimuthDeg = directionFeatures(1, :).';
elevationDeg = directionFeatures(2, :).';
y = directionFeatures(4, :).';
solidAngleWeight = directionFeatures(6, :).';
leftContra25Mask = interpolationMask & great_circle_mask( ...
    azimuthDeg, elevationDeg, 270, 0, 25);
rightContra25Mask = interpolationMask & great_circle_mask( ...
    azimuthDeg, elevationDeg, 90, 0, 25);
leftContraHemisphereMask = interpolationMask & y < -1e-12;
rightContraHemisphereMask = interpolationMask & y > 1e-12;
horizontalMask = interpolationMask & abs(elevationDeg) <= 1e-9;
assert(sum(interpolationMask) == 767 && any(horizontalMask), ...
    'Frozen strict direction masks are invalid.');

[erbLeft, ~] = AKerbError(hrir(:, interpolationMask, 1), ...
    referenceHrir(:, interpolationMask, 1), ...
    [50, samplingRateHz / 2], samplingRateHz);
erbRight = AKerbError(hrir(:, interpolationMask, 2), ...
    referenceHrir(:, interpolationMask, 2), ...
    [50, samplingRateHz / 2], samplingRateHz);
fullWeights = normalized_weights(solidAngleWeight(interpolationMask));
values(1) = 0.5 * (mean(abs(erbLeft) * fullWeights) + ...
    mean(abs(erbRight) * fullWeights));
leftWithinFull = leftContra25Mask(interpolationMask);
rightWithinFull = rightContra25Mask(interpolationMask);
leftWeights = normalized_weights(solidAngleWeight(leftContra25Mask));
rightWeights = normalized_weights(solidAngleWeight(rightContra25Mask));
values(2) = 0.5 * (...
    mean(abs(erbLeft(:, leftWithinFull)) * leftWeights) + ...
    mean(abs(erbRight(:, rightWithinFull)) * rightWeights));
values(3) = 0.5 * (...
    high_frequency_error(selectedDb(:, :, 1), referenceDb(:, :, 1), ...
        frequencyHz, leftContraHemisphereMask, solidAngleWeight) + ...
    high_frequency_error(selectedDb(:, :, 2), referenceDb(:, :, 2), ...
        frequencyHz, rightContraHemisphereMask, solidAngleWeight));
methodIld = calculate_ild(hrir(:, :, 1), hrir(:, :, 2));
values(4) = mean(abs(methodIld(horizontalMask) - referenceIld(horizontalMask)));
end

function ild = calculate_ild(leftHrir, rightHrir)
leftEnergy = sum(abs(leftHrir) .^ 2, 1);
rightEnergy = sum(abs(rightHrir) .^ 2, 1);
ild = (10 * log10(leftEnergy ./ rightEnergy)).';
end

function mask = great_circle_mask(azimuth, elevation, ...
        centerAzimuth, centerElevation, radiusDegrees)
dotProduct = sind(elevation) .* sind(centerElevation) + ...
    cosd(elevation) .* cosd(centerElevation) .* ...
    cosd(azimuth - centerAzimuth);
distance = acosd(min(1, max(-1, dotProduct)));
mask = distance <= radiusDegrees + 1e-10;
end

function weights = normalized_weights(weights)
weights = weights(:);
assert(all(isfinite(weights)) && all(weights > 0), ...
    'Direction weights must be finite and positive.');
weights = weights / sum(weights);
end

function value = high_frequency_error(estimateDb, referenceDb, ...
        frequencyHz, directionMask, solidAngleWeight)
frequencyMask = frequencyHz > 10000 & frequencyHz <= 20000;
weights = normalized_weights(solidAngleWeight(directionMask));
errorDb = abs(estimateDb(frequencyMask, directionMask) - ...
    referenceDb(frequencyMask, directionMask));
value = mean(errorDb * weights);
end

function aggregate = aggregate_metrics( ...
        metricLong, methodIds, methodLabels, metricIds)
aggregate = table();
for metricIndex = 1:numel(metricIds)
    mcaValues = metricLong.Value_dB( ...
        metricLong.Method == "MCA" & metricLong.Metric == metricIds(metricIndex));
    mcarValues = metricLong.Value_dB( ...
        metricLong.Method == "MCARv32" & metricLong.Metric == metricIds(metricIndex));
    for methodIndex = 1:numel(methodIds)
        values = metricLong.Value_dB(metricLong.Method == methodIds(methodIndex) & ...
            metricLong.Metric == metricIds(metricIndex));
        assert(numel(values) == numel(mcaValues), ...
            'Aggregate subject alignment failed.');
        row = table(methodIds(methodIndex), methodLabels(methodIndex), ...
            metricIds(metricIndex), numel(values), mean(values), std(values), ...
            median(values), 100 * (mean(mcaValues) - mean(values)) / mean(mcaValues), ...
            100 * (mean(values) - mean(mcarValues)) / mean(values), ...
            sum(values < mcaValues), sum(mcarValues < values), ...
            'VariableNames', {'Method', 'MethodLabel', 'Metric', ...
            'SubjectCount', 'Mean_dB', 'Std_dB', 'Median_dB', ...
            'ImprovementVsMCA_percent', 'MCARImprovementVsMethod_percent', ...
            'ImprovedVsMCA_SubjectCount', 'MCARBetter_SubjectCount'});
        aggregate = [aggregate; row]; %#ok<AGROW>
    end
end
end

function paperTable = make_paper_table( ...
        aggregate, methodIds, methodLabels, metricIds)
paperTable = table(methodIds.', methodLabels.', ...
    'VariableNames', {'Method', 'MethodLabel'});
for metricIndex = 1:numel(metricIds)
    means = zeros(numel(methodIds), 1);
    stds = zeros(numel(methodIds), 1);
    formatted = strings(numel(methodIds), 1);
    for methodIndex = 1:numel(methodIds)
        row = aggregate(aggregate.Method == methodIds(methodIndex) & ...
            aggregate.Metric == metricIds(metricIndex), :);
        means(methodIndex) = row.Mean_dB;
        stds(methodIndex) = row.Std_dB;
        formatted(methodIndex) = sprintf('%.3f +/- %.3f', ...
            row.Mean_dB, row.Std_dB);
    end
    paperTable.(sprintf('%s_Mean_dB', metricIds(metricIndex))) = means;
    paperTable.(sprintf('%s_Std_dB', metricIds(metricIndex))) = stds;
    paperTable.(sprintf('%s_MeanPlusMinusStd_dB', metricIds(metricIndex))) = formatted;
end
end

function plot_aggregate(aggregate, methodIds, methodLabels, metricIds, ...
        figuresRoot, splitName, subjectCount)
values = zeros(numel(metricIds), numel(methodIds));
for metricIndex = 1:numel(metricIds)
    for methodIndex = 1:numel(methodIds)
        row = aggregate(aggregate.Method == methodIds(methodIndex) & ...
            aggregate.Metric == metricIds(metricIndex), :);
        values(metricIndex, methodIndex) = row.Mean_dB;
    end
end
figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1600, 760]);
bar(values);
grid on;
xticks(1:numel(metricIds));
xticklabels(["Full ERB", "Contra25 ERB", "Contra HF", "Horizontal ILD"]);
ylabel('Mean error (dB)');
legend(methodLabels, 'Location', 'northoutside', ...
    'NumColumns', 3, 'Interpreter', 'none');
title(sprintf('SONICOM Q26 %s horizontal comparison (N=%d)', ...
    splitName, subjectCount));
exportgraphics(figureHandle, fullfile(figuresRoot, ...
    sprintf('%s%d_aggregate_baselines.png', splitName, subjectCount)), ...
    'Resolution', 180);
close(figureHandle);
end

function write_markdown_report(path, paperTable, configuration, qualityChecks)
file = fopen(path, 'w');
assert(file ~= -1, 'Could not open Markdown report: %s', path);
cleanup = onCleanup(@() fclose(file));
fprintf(file, '# SONICOM Q26 interpolation baselines\n\n');
fprintf(file, '- Split: `%s`\n', configuration.split);
fprintf(file, '- Subjects: %d\n', configuration.subjectCount);
fprintf(file, '- Sparse grid: `%s`\n', configuration.sparseGrid);
fprintf(file, '- Evaluation directions: %d (Q26 inputs excluded)\n', ...
    configuration.interpolationDirectionCount);
fprintf(file, '- SH Tikhonov epsilon: %.4g\n', configuration.tikhonovEpsilonSH);
fprintf(file, '- MCAR prediction: `%s`\n', configuration.mcarPredictionName);
if ~isempty(configuration.ranfPredictionName)
    fprintf(file, '- RANF prediction: `%s`\n', configuration.ranfPredictionName);
end
if ~isempty(configuration.fspPredictionName)
    fprintf(file, '- FSP-AE prediction: `%s`\n', configuration.fspPredictionName);
end
fprintf(file, '\n');
fprintf(file, '| Method | Full ERB | Contra25 ERB | Contra HF | Horizontal ILD |\n');
fprintf(file, '|---|---:|---:|---:|---:|\n');
for index = 1:height(paperTable)
    fprintf(file, '| %s | %s | %s | %s | %s |\n', ...
        paperTable.MethodLabel(index), ...
        paperTable.FullSphereERB_MeanPlusMinusStd_dB(index), ...
        paperTable.Contralateral25ERB_MeanPlusMinusStd_dB(index), ...
        paperTable.ContralateralHighFrequency_MeanPlusMinusStd_dB(index), ...
        paperTable.HorizontalILDMAE_MeanPlusMinusStd_dB(index));
end
fprintf(file, '\nAll entries are mean +/- subject standard deviation in dB; lower is better.\n\n');
fprintf(file, '## Reproducibility checks\n\n');
fprintf(file, 'Natural Neighbor and Barycentric use the upstream SUpDEq geometry routines. ');
fprintf(file, 'The first subject was also evaluated through the native `supdeq_interpHRTF` entry point.\n\n');
fprintf(file, '- Natural Neighbor maximum complex-spectrum parity error: %.3g\n', ...
    qualityChecks.NaturalNeighborNativeParityMaxAbs(1));
fprintf(file, '- Barycentric maximum complex-spectrum parity error: %.3g\n', ...
    qualityChecks.BarycentricNativeParityMaxAbs(1));
if ~isempty(configuration.ranfPredictionName)
    fprintf(file, '- RANF maximum Q26 observed-HRIR preservation error: %.3g\n', ...
        max(qualityChecks.RANFObservedHRIRMaxAbsError));
end
clear cleanup;
end

function write_json(path, value)
text = jsonencode(value, 'PrettyPrint', true);
file = fopen(path, 'w');
assert(file ~= -1, 'Could not open JSON output: %s', path);
cleanup = onCleanup(@() fclose(file));
fprintf(file, '%s\n', text);
clear cleanup;
end
