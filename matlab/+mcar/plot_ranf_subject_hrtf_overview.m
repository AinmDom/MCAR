function outputPaths = plot_ranf_subject_hrtf_overview( ...
        outputName, ranfPredictionName, mcarPredictionName)
%PLOT_RANF_SUBJECT_HRTF_OVERVIEW Plot contralateral HRTFs for 44 test subjects.
%
% For every frozen SONICOM test subject, the panel uses the left-ear
% direction nearest (270 deg, 0 deg) among strict interpolation-only
% directions and overlays Reference, MCA, RANF, and frozen MCAR v3.2.

if nargin < 1 || isempty(outputName)
    outputName = 'sonicom_ranf_q26_final_test';
end
if nargin < 2 || isempty(ranfPredictionName)
    ranfPredictionName = 'sonicom_ranf_q26_final_test';
end
if nargin < 3 || isempty(mcarPredictionName)
    mcarPredictionName = 'sonicom_q26_test_mlp_cnn_v32_locked_ild075';
end
validateattributes(outputName, {'char', 'string'}, {'scalartext'});
validateattributes(ranfPredictionName, {'char', 'string'}, {'scalartext'});
validateattributes(mcarPredictionName, {'char', 'string'}, {'scalartext'});
outputName = char(outputName);
ranfPredictionName = char(ranfPredictionName);
mcarPredictionName = char(mcarPredictionName);

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
datasetRoot = fullfile(projectRoot, 'data', 'processed', ...
    'sonicom_residual_q26_v1');
splitFile = fullfile(projectRoot, 'configs', 'data', ...
    'sonicom_subject_split_v1.csv');
ranfRoot = fullfile(projectRoot, 'artifacts', 'reconstruction', ...
    ranfPredictionName);
mcarRoot = fullfile(projectRoot, 'artifacts', 'reconstruction', ...
    mcarPredictionName);
figuresRoot = fullfile(projectRoot, 'results', outputName, 'figures');
sofaToolboxRoot = fullfile(projectRoot, 'external', 'SUpDEq', ...
    'thirdParty', 'SOFAtoolbox-2.1.5', 'SOFAtoolbox');

assert(isfolder(datasetRoot), 'Dataset root not found: %s', datasetRoot);
assert(isfile(splitFile), 'Split file not found: %s', splitFile);
assert(isfolder(ranfRoot), 'RANF prediction root not found: %s', ranfRoot);
assert(isfolder(mcarRoot), 'MCAR prediction root not found: %s', mcarRoot);
assert(isfolder(sofaToolboxRoot), ...
    'SOFA toolbox root not found: %s', sofaToolboxRoot);
if ~isfolder(figuresRoot)
    mkdir(figuresRoot);
end
addpath(sofaToolboxRoot);
SOFAstart;

splitTable = readtable(splitFile, 'TextType', 'string');
splitRows = splitTable(splitTable.split == "test", :);
assert(height(splitRows) == 44, ...
    'Expected exactly 44 locked test subjects.');

samplingRateHz = 44100;
nfft = 1024;
frequencyHzAll = (0:(nfft / 2)).' * samplingRateHz / nfft;
subjectCount = height(splitRows);
overview = repmat(struct( ...
    'subjectLabel', "", 'frequencyHz', [], 'referenceDb', [], ...
    'mcaDb', [], 'ranfDb', [], 'mcarDb', [], ...
    'azimuthDeg', NaN, 'elevationDeg', NaN), subjectCount, 1);

for subjectIndex = 1:subjectCount
    subjectLabel = splitRows.subject_id(subjectIndex);
    sourceFile = fullfile(datasetRoot, 'subjects', char(subjectLabel), ...
        'q26.h5');
    ranfFile = fullfile(ranfRoot, 'subjects', char(subjectLabel), ...
        'prediction.sofa');
    mcarFile = fullfile(mcarRoot, 'subjects', char(subjectLabel), ...
        'prediction.h5');
    assert(isfile(sourceFile), 'Missing source HDF5: %s', sourceFile);
    assert(isfile(ranfFile), 'Missing RANF SOFA: %s', ranfFile);
    assert(isfile(mcarFile), 'Missing MCAR prediction: %s', mcarFile);
    assert(string(h5readatt(sourceFile, '/', 'split')) == "test", ...
        'Source split mismatch for %s.', subjectLabel);
    assert(string(h5readatt(mcarFile, '/', 'split')) == "test", ...
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

    ranfSofa = SOFAload(ranfFile);
    ranfHrir = double(ranfSofa.Data.IR);
    assert(isequal(size(ranfHrir), [793, 2, 256]), ...
        'Unexpected RANF HRIR shape for %s.', subjectLabel);
    ranfPosition = double(ranfSofa.SourcePosition);
    azimuthDeg = directionFeatures(1, :).';
    elevationDeg = directionFeatures(2, :).';
    azimuthError = abs(mod(ranfPosition(:, 1) - azimuthDeg + 180, 360) - 180);
    elevationError = abs(ranfPosition(:, 2) - elevationDeg);
    assert(max(azimuthError) < 1e-8 && max(elevationError) < 1e-8, ...
        'RANF SOFA direction order mismatch for %s.', subjectLabel);

    directionDistance = great_circle_distance( ...
        azimuthDeg, elevationDeg, 270, 0);
    directionDistance(~interpolationMask) = inf;
    [~, overviewIndex] = min(directionDistance);
    assert(interpolationMask(overviewIndex), ...
        'Selected direction is not interpolation-only for %s.', subjectLabel);

    ranfSpectrum = fft(squeeze(ranfHrir(overviewIndex, 1, :)), nfft);
    ranfSelectedDb = 20 * log10(max(abs( ...
        ranfSpectrum(selectedIndices)), 1e-10));
    frequencyHz = frequencyHzAll(selectedIndices);
    plotMask = frequencyHz >= 100 & frequencyHz <= 20000;

    assert(isequal(size(referenceDb), [numel(selectedIndices), 793, 2]), ...
        'Unexpected reference shape for %s.', subjectLabel);
    assert(isequal(size(mcaDb), size(referenceDb)) && ...
        isequal(size(mcarResidualDb), size(referenceDb)), ...
        'MCA/MCAR tensor shape mismatch for %s.', subjectLabel);
    assert(all(isfinite(referenceDb), 'all') && ...
        all(isfinite(mcaDb), 'all') && ...
        all(isfinite(mcarResidualDb), 'all') && ...
        all(isfinite(ranfSelectedDb), 'all'), ...
        'Non-finite HRTF magnitude for %s.', subjectLabel);

    overview(subjectIndex).subjectLabel = subjectLabel;
    overview(subjectIndex).frequencyHz = frequencyHz(plotMask);
    overview(subjectIndex).referenceDb = ...
        referenceDb(plotMask, overviewIndex, 1);
    overview(subjectIndex).mcaDb = mcaDb(plotMask, overviewIndex, 1);
    overview(subjectIndex).ranfDb = ranfSelectedDb(plotMask);
    overview(subjectIndex).mcarDb = ...
        mcaDb(plotMask, overviewIndex, 1) + ...
        mcarResidualDb(plotMask, overviewIndex, 1);
    overview(subjectIndex).azimuthDeg = azimuthDeg(overviewIndex);
    overview(subjectIndex).elevationDeg = elevationDeg(overviewIndex);
    fprintf('RANF test HRTF overview [%d/%d]: %s\n', ...
        subjectIndex, subjectCount, subjectLabel);
end

englishPath = fullfile(figuresRoot, ...
    'test44_contralateral_hrtf_overview.png');
chinesePath = fullfile(figuresRoot, ...
    'test44_contralateral_hrtf_overview_zh.png');
render_overview(overview, englishPath, false);
render_overview(overview, chinesePath, true);
outputPaths = string({englishPath, chinesePath});
fprintf('Wrote RANF subject HRTF overviews:\n%s\n%s\n', ...
    englishPath, chinesePath);
end

function render_overview(overview, outputPath, useChinese)
figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [30, 30, 2200, 1900]);
layout = tiledlayout(figureHandle, 8, 6, ...
    'TileSpacing', 'compact', 'Padding', 'compact');
colors = struct( ...
    'reference', [0.15, 0.15, 0.15], ...
    'mca', [0.85, 0.33, 0.10], ...
    'ranf', [0.00, 0.65, 0.85], ...
    'mcar', [0.49, 0.18, 0.56]);
for subjectIndex = 1:numel(overview)
    axisHandle = nexttile(layout);
    referenceLine = semilogx(axisHandle, overview(subjectIndex).frequencyHz, ...
        overview(subjectIndex).referenceDb, 'Color', colors.reference, ...
        'LineWidth', 0.9);
    hold(axisHandle, 'on');
    mcaLine = semilogx(axisHandle, overview(subjectIndex).frequencyHz, ...
        overview(subjectIndex).mcaDb, '--', 'Color', colors.mca, ...
        'LineWidth', 0.8);
    ranfLine = semilogx(axisHandle, overview(subjectIndex).frequencyHz, ...
        overview(subjectIndex).ranfDb, 'Color', colors.ranf, ...
        'LineWidth', 1.3);
    mcarLine = semilogx(axisHandle, overview(subjectIndex).frequencyHz, ...
        overview(subjectIndex).mcarDb, 'Color', colors.mcar, ...
        'LineWidth', 1.0);
    hold(axisHandle, 'off');
    grid(axisHandle, 'on');
    axisHandle.GridAlpha = 0.18;
    axisHandle.MinorGridAlpha = 0.08;
    axisHandle.FontSize = 6.5;
    axisHandle.FontName = 'Arial';
    xlim(axisHandle, [100, 20000]);
    ylim(axisHandle, [-60, 25]);
    title(axisHandle, sprintf('%s (%.0f°, %.0f°)', ...
        overview(subjectIndex).subjectLabel, ...
        overview(subjectIndex).azimuthDeg, ...
        overview(subjectIndex).elevationDeg), ...
        'FontSize', 7, 'Interpreter', 'none');
    if subjectIndex == 1
        if useChinese
            legendLabels = ["参考真值", "MCA", "RANF", "MCAR v3.2"];
        else
            legendLabels = ["Reference", "MCA", "RANF", "MCAR v3.2"];
        end
        legend(axisHandle, ...
            [referenceLine, mcaLine, ranfLine, mcarLine], legendLabels, ...
            'Location', 'southwest', 'FontSize', 6, 'Interpreter', 'none');
    end
end
if useChinese
    layoutTitle = 'SONICOM 测试集：左耳对侧 HRTF（最近纯插值方向）';
    titleFont = 'Microsoft YaHei';
else
    layoutTitle = ['SONICOM test: left-ear contralateral HRTF ' ...
        '(nearest pure interpolation direction)'];
    titleFont = 'Arial';
end
title(layout, layoutTitle, 'FontWeight', 'bold', ...
    'FontName', titleFont, 'Interpreter', 'none');
exportgraphics(figureHandle, outputPath, 'Resolution', 180);
close(figureHandle);
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
