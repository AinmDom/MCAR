function outputPath = plot_fsp_ae_subject_hrtf_overview(splitName, ...
        outputName, fspPredictionName, mcarPredictionName, allowTest)
%PLOT_FSP_AE_SUBJECT_HRTF_OVERVIEW Plot one contralateral HRTF per subject.
%
% Each panel uses the left-ear direction nearest (270 deg, 0 deg) among
% strict interpolation-only directions and overlays Reference, MCA,
% MCAR v3.2, and FSP-AE-Q26 magnitude responses.

if nargin < 1 || isempty(splitName)
    splitName = 'val';
end
if nargin < 2 || isempty(outputName)
    outputName = 'sonicom_fsp_ae_q26_formal_validation';
end
if nargin < 3 || isempty(fspPredictionName)
    fspPredictionName = 'sonicom_fsp_ae_q26_formal_validation';
end
if nargin < 4 || isempty(mcarPredictionName)
    mcarPredictionName = ...
        'sonicom_q26_validation_mlp_cnn_v32_locked_ild075';
end
if nargin < 5 || isempty(allowTest)
    allowTest = false;
end

validateattributes(splitName, {'char', 'string'}, {'scalartext'});
validateattributes(outputName, {'char', 'string'}, {'scalartext'});
validateattributes(fspPredictionName, {'char', 'string'}, {'scalartext'});
validateattributes(mcarPredictionName, {'char', 'string'}, {'scalartext'});
validateattributes(allowTest, {'logical', 'numeric'}, {'scalar'});
splitName = string(splitName);
allowTest = logical(allowTest);
assert(any(splitName == ["val", "test"]), ...
    'splitName must be val or test.');
assert(splitName ~= "test" || allowTest, ...
    'The locked SONICOM test split requires allowTest=true.');

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
datasetRoot = fullfile(projectRoot, 'data', 'processed', ...
    'sonicom_residual_q26_v1');
splitFile = fullfile(projectRoot, 'configs', 'data', ...
    'sonicom_subject_split_v1.csv');
fspRoot = fullfile(projectRoot, 'artifacts', 'reconstruction', ...
    char(fspPredictionName));
mcarRoot = fullfile(projectRoot, 'artifacts', 'reconstruction', ...
    char(mcarPredictionName));
figuresRoot = fullfile(projectRoot, 'results', char(outputName), 'figures');

assert(isfolder(datasetRoot), 'Dataset root not found: %s', datasetRoot);
assert(isfile(splitFile), 'Split file not found: %s', splitFile);
assert(isfolder(fspRoot), 'FSP-AE prediction root not found: %s', fspRoot);
assert(isfolder(mcarRoot), 'MCAR prediction root not found: %s', mcarRoot);
if ~isfolder(figuresRoot)
    mkdir(figuresRoot);
end

splitTable = readtable(splitFile, 'TextType', 'string');
splitRows = splitTable(splitTable.split == splitName, :);
assert(height(splitRows) == 44, ...
    'Expected exactly 44 locked %s subjects.', splitName);

if splitName == "val"
    splitDisplayName = 'validation';
else
    splitDisplayName = 'test';
end
subjectCount = height(splitRows);
figurePrefix = sprintf('%s%d', splitDisplayName, subjectCount);
outputPath = fullfile(figuresRoot, ...
    [figurePrefix '_contralateral_hrtf_overview.png']);

samplingRateHz = 44100;
nfft = 1024;
frequencyHzAll = (0:(nfft / 2)).' * samplingRateHz / nfft;
overview = repmat(struct( ...
    'subjectLabel', "", 'frequencyHz', [], 'referenceDb', [], ...
    'mcaDb', [], 'mcarDb', [], 'fspDb', [], ...
    'azimuthDeg', NaN, 'elevationDeg', NaN), subjectCount, 1);

for subjectIndex = 1:subjectCount
    subjectLabel = splitRows.subject_id(subjectIndex);
    sourceFile = fullfile(datasetRoot, 'subjects', char(subjectLabel), ...
        'q26.h5');
    fspFile = fullfile(fspRoot, 'subjects', char(subjectLabel), ...
        'prediction.h5');
    mcarFile = fullfile(mcarRoot, 'subjects', char(subjectLabel), ...
        'prediction.h5');
    assert(isfile(sourceFile), 'Missing source HDF5: %s', sourceFile);
    assert(isfile(fspFile), 'Missing FSP-AE prediction: %s', fspFile);
    assert(isfile(mcarFile), 'Missing MCAR prediction: %s', mcarFile);
    assert(string(h5readatt(sourceFile, '/', 'split')) == splitName, ...
        'Source split mismatch for %s.', subjectLabel);
    assert(string(h5readatt(fspFile, '/', 'split')) == splitName, ...
        'FSP-AE split mismatch for %s.', subjectLabel);
    assert(string(h5readatt(mcarFile, '/', 'split')) == splitName, ...
        'MCAR split mismatch for %s.', subjectLabel);

    referenceDb = double(h5read(sourceFile, '/reference_logmag_db'));
    mcaDb = double(h5read(sourceFile, '/mca_logmag_db'));
    mcarResidualDb = double(h5read(mcarFile, '/predicted_residual_db'));
    directionFeatures = double(h5read(sourceFile, '/direction_features'));
    interpolationMask = logical(h5read( ...
        sourceFile, '/interpolation_evaluation_mask'));
    interpolationMask = interpolationMask(:);
    selectedIndices = double(h5read(sourceFile, ...
        '/strict_ild/selected_bin_indices_zero_based')) + 1;
    selectedIndices = selectedIndices(:);
    fspRaw = double(h5read(fspFile, '/predicted_magnitude_db'));
    fspMagnitudeDb = permute(fspRaw, [1, 3, 2]);
    fspSelectedDb = fspMagnitudeDb(selectedIndices - 1, :, :);

    assert(isequal(size(referenceDb), [numel(selectedIndices), 793, 2]), ...
        'Unexpected reference shape for %s.', subjectLabel);
    assert(isequal(size(mcaDb), size(referenceDb)) && ...
        isequal(size(mcarResidualDb), size(referenceDb)) && ...
        isequal(size(fspSelectedDb), size(referenceDb)), ...
        'Prediction tensor shape mismatch for %s.', subjectLabel);
    assert(all(isfinite(referenceDb), 'all') && ...
        all(isfinite(mcaDb), 'all') && ...
        all(isfinite(mcarResidualDb), 'all') && ...
        all(isfinite(fspSelectedDb), 'all'), ...
        'Non-finite HRTF magnitude for %s.', subjectLabel);

    azimuthDeg = directionFeatures(1, :).';
    elevationDeg = directionFeatures(2, :).';
    directionDistance = great_circle_distance( ...
        azimuthDeg, elevationDeg, 270, 0);
    directionDistance(~interpolationMask) = inf;
    [~, overviewIndex] = min(directionDistance);
    frequencyHz = frequencyHzAll(selectedIndices);
    plotMask = frequencyHz >= 100 & frequencyHz <= 20000;

    overview(subjectIndex).subjectLabel = subjectLabel;
    overview(subjectIndex).frequencyHz = frequencyHz(plotMask);
    overview(subjectIndex).referenceDb = ...
        referenceDb(plotMask, overviewIndex, 1);
    overview(subjectIndex).mcaDb = mcaDb(plotMask, overviewIndex, 1);
    overview(subjectIndex).mcarDb = ...
        mcaDb(plotMask, overviewIndex, 1) + ...
        mcarResidualDb(plotMask, overviewIndex, 1);
    overview(subjectIndex).fspDb = ...
        fspSelectedDb(plotMask, overviewIndex, 1);
    overview(subjectIndex).azimuthDeg = azimuthDeg(overviewIndex);
    overview(subjectIndex).elevationDeg = elevationDeg(overviewIndex);
    fprintf('FSP-AE %s HRTF overview [%d/%d]: %s\n', ...
        splitDisplayName, subjectIndex, subjectCount, subjectLabel);
end

figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [30, 30, 2200, 1900]);
layout = tiledlayout(figureHandle, 8, 6, ...
    'TileSpacing', 'compact', 'Padding', 'compact');
colors = struct( ...
    'reference', [0.15, 0.15, 0.15], ...
    'mca', [0.85, 0.33, 0.10], ...
    'mcar', [0.49, 0.18, 0.56], ...
    'fsp', [0.00, 0.65, 0.85]);
for subjectIndex = 1:subjectCount
    axisHandle = nexttile(layout);
    semilogx(axisHandle, overview(subjectIndex).frequencyHz, ...
        overview(subjectIndex).referenceDb, 'Color', colors.reference, ...
        'LineWidth', 0.9);
    hold(axisHandle, 'on');
    semilogx(axisHandle, overview(subjectIndex).frequencyHz, ...
        overview(subjectIndex).mcaDb, '--', 'Color', colors.mca, ...
        'LineWidth', 0.8);
    semilogx(axisHandle, overview(subjectIndex).frequencyHz, ...
        overview(subjectIndex).mcarDb, 'Color', colors.mcar, ...
        'LineWidth', 1.0);
    semilogx(axisHandle, overview(subjectIndex).frequencyHz, ...
        overview(subjectIndex).fspDb, 'Color', colors.fsp, ...
        'LineWidth', 1.3);
    grid(axisHandle, 'on');
    axisHandle.GridAlpha = 0.18;
    axisHandle.MinorGridAlpha = 0.08;
    xlim(axisHandle, [100, 20000]);
    ylim(axisHandle, [-60, 25]);
    title(axisHandle, sprintf('%s (%.0f deg, %.0f deg)', ...
        overview(subjectIndex).subjectLabel, ...
        overview(subjectIndex).azimuthDeg, ...
        overview(subjectIndex).elevationDeg), 'FontSize', 7);
    if subjectIndex == 1
        legend(axisHandle, ...
            ["Reference", "MCA", "MCAR v3.2", "FSP-AE-Q26"], ...
            'Location', 'southwest', 'FontSize', 6);
    end
end
title(layout, ['SONICOM ' splitDisplayName ...
    ': left-ear contralateral HRTF (nearest pure interpolation direction)'], ...
    'FontWeight', 'bold');
exportgraphics(figureHandle, outputPath, 'Resolution', 180);
close(figureHandle);
fprintf('Wrote FSP-AE subject HRTF overview: %s\n', outputPath);
end

function distanceDeg = great_circle_distance(azimuthDeg, elevationDeg, ...
        targetAzimuthDeg, targetElevationDeg)
azimuthRad = deg2rad(azimuthDeg);
elevationRad = deg2rad(elevationDeg);
targetAzimuthRad = deg2rad(targetAzimuthDeg);
targetElevationRad = deg2rad(targetElevationDeg);
cosineDistance = sin(elevationRad) * sin(targetElevationRad) + ...
    cos(elevationRad) .* cos(targetElevationRad) .* ...
    cos(azimuthRad - targetAzimuthRad);
distanceDeg = rad2deg(acos(max(-1, min(1, cosineDistance))));
end
