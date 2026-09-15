function outputs = generate_fsc_individual_reconstruction_figures(language)
%GENERATE_FSC_INDIVIDUAL_RECONSTRUCTION_FIGURES Create paper candidate figures.
% Uses only the frozen SONICOM Q26 test-derived HDF5 inputs and the
% preselected single-member FSC E190 prediction.  P0009 is the median-effect
% representative; P0060 has the strongest aggregate magnitude improvement.

if nargin < 1
    language = "en";
end
language = lower(string(language));
assert(ismember(language, ["en", "zh"]), 'Language must be "en" or "zh".');

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
dataRoot = fullfile(projectRoot, 'data', 'processed', 'sonicom_residual_q26_v1');
predictionRoot = fullfile(projectRoot, 'artifacts', 'reconstruction', ...
    'sonicom_fsc_q26_e190_single_seed20260822_test');
figureRoot = fullfile(projectRoot, 'paper write', 'grand_paper', 'figure');
subjects = ["P0009", "P0060"];
assert(isfolder(dataRoot) && isfolder(predictionRoot), 'Frozen HDF5 inputs are missing.');
if ~isfolder(figureRoot), mkdir(figureRoot); end

records = repmat(load_subject(dataRoot, predictionRoot, subjects(1)), 2, 1);
for index = 1:numel(subjects)
    records(index) = load_subject(dataRoot, predictionRoot, subjects(index));
end

suffix = "";
if language == "zh"
    suffix = "_zh";
end
spectraPath = fullfile(figureRoot, "fsc_two_subject_binaural_spectra" + suffix + ".png");
mapsPath = fullfile(figureRoot, "fsc_two_subject_horizontal_maps" + suffix + ".png");
draw_spectra(records, spectraPath, language);
draw_maps(records, mapsPath, language);
export_pdf_from_png_figure(records, figureRoot);

outputs = string({spectraPath, strrep(spectraPath, '.png', '.pdf'), ...
    mapsPath, strrep(mapsPath, '.png', '.pdf')});
fprintf('Generated FSC individual reconstruction figures:\n%s\n', join(outputs, '\n'));
end

function record = load_subject(dataRoot, predictionRoot, subject)
source = fullfile(dataRoot, 'subjects', char(subject), 'q26.h5');
prediction = fullfile(predictionRoot, 'subjects', char(subject), 'prediction.h5');
assert(isfile(source) && isfile(prediction), 'Missing HDF5 for %s.', subject);
assert(string(h5readatt(source, '/', 'split')) == "test", 'Source split mismatch.');
assert(string(h5readatt(prediction, '/', 'split')) == "test", 'Prediction split mismatch.');
record.subject = subject;
record.reference = double(h5read(source, '/reference_logmag_db'));
record.mca = double(h5read(source, '/mca_logmag_db'));
record.fsc = record.mca + double(h5read(prediction, '/predicted_residual_db'));
record.direction = double(h5read(source, '/direction_features'));
record.interpolationMask = logical(h5read(source, '/interpolation_evaluation_mask'));
record.selectedBins = double(h5read(source, '/strict_ild/selected_bin_indices_zero_based')) + 1;
record.frequencyHz = (record.selectedBins - 1) * 44100 / 1024;
assert(isequal(size(record.reference), size(record.mca), size(record.fsc)), ...
    'Tensor dimensions disagree for %s.', subject);
assert(all(isfinite(record.reference), 'all') && all(isfinite(record.mca), 'all') && ...
    all(isfinite(record.fsc), 'all'), 'Non-finite tensor for %s.', subject);
end

function draw_spectra(records, outputPath, language)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80, 80, 1640, 980]);
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
colors = struct('reference', [0.10, 0.10, 0.10], 'mca', [0.40, 0.45, 0.52], ...
    'fsc', [0.83, 0.18, 0.20]);
if language == "zh"
    earNames = ["左耳，270°", "右耳，90°"];
    measuredLabel = "实测";
    frequencyLabel = "频率 (Hz)";
    magnitudeLabel = "幅度 (dB)";
    fontName = "Microsoft YaHei";
else
    earNames = ["Left ear, 270 degrees", "Right ear, 90 degrees"];
    measuredLabel = "Measured";
    frequencyLabel = "Frequency (Hz)";
    magnitudeLabel = "Magnitude (dB)";
    fontName = "Arial";
end
targetAzimuth = [270, 90];
for row = 1:2
    for column = 1:2
        ax = nexttile(layout);
        record = records(row);
        directionIndex = nearest_interpolation_direction(record, targetAzimuth(column), 0);
        earIndex = column;
        frequencyMask = record.frequencyHz >= 100 & record.frequencyHz <= 20000;
        f = record.frequencyHz(frequencyMask);
        reference = record.reference(frequencyMask, directionIndex, earIndex);
        mca = record.mca(frequencyMask, directionIndex, earIndex);
        fsc = record.fsc(frequencyMask, directionIndex, earIndex);
        semilogx(ax, f, reference, 'Color', colors.reference, 'LineWidth', 1.5, 'DisplayName', measuredLabel);
        hold(ax, 'on');
        semilogx(ax, f, mca, '--', 'Color', colors.mca, 'LineWidth', 1.35, 'DisplayName', 'MCA');
        semilogx(ax, f, fsc, 'Color', colors.fsc, 'LineWidth', 1.65, 'DisplayName', 'FSC');
        xline(ax, 4000, ':', 'Color', [0.68 0.68 0.68], 'HandleVisibility', 'off');
        xline(ax, 18000, ':', 'Color', [0.68 0.68 0.68], 'HandleVisibility', 'off');
        hold(ax, 'off');
        xlim(ax, [100 20000]); ylim(ax, [-60 25]); grid(ax, 'on');
        ax.XScale = 'log'; ax.FontName = fontName; ax.FontSize = 11; ax.TickDir = 'out';
        ax.GridAlpha = 0.18; ax.MinorGridAlpha = 0.08;
        title(ax, sprintf('%s — %s', record.subject, earNames(column)), 'FontWeight', 'bold');
        xlabel(ax, frequencyLabel); ylabel(ax, magnitudeLabel);
        if row == 1 && column == 1
            legend(ax, 'Location', 'southwest', 'Box', 'off');
        end
    end
end
if language == "en"
    title(layout, {'Two-subject binaural HRTF reconstruction at contralateral interpolation directions', ...
        'P0009: median-effect representative; P0060: strongest aggregate magnitude improvement'}, ...
        'FontWeight', 'bold', 'FontName', fontName);
end
exportgraphics(fig, outputPath, 'Resolution', 300);
exportgraphics(fig, strrep(outputPath, '.png', '.pdf'), 'ContentType', 'vector');
close(fig);
end

function draw_maps(records, outputPath, language)
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [60, 60, 1800, 950]);
layout = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
errorLimit = 12;
if language == "zh"
    panelNames = {'实测左耳幅度', 'MCA绝对误差', 'FSC绝对误差'};
    azimuthLabel = '方位角 (°)';
    frequencyLabel = '频率 (Hz)';
    magnitudeLabel = '幅度 (dB)';
    errorLabel = '绝对误差 (dB)';
    fontName = 'Microsoft YaHei';
else
    panelNames = {'Measured left-ear magnitude', 'MCA absolute error', 'FSC absolute error'};
    azimuthLabel = 'Azimuth (degrees)';
    frequencyLabel = 'Frequency (Hz)';
    magnitudeLabel = 'Magnitude (dB)';
    errorLabel = 'Absolute error (dB)';
    fontName = 'Arial';
end
for row = 1:2
    record = records(row);
    [directions, sortOrder] = horizontal_directions(record);
    frequencyMask = record.frequencyHz >= 100 & record.frequencyHz <= 20000;
    frequencies = record.frequencyHz(frequencyMask);
    reference = record.reference(frequencyMask, directions, 1);
    mcaError = abs(record.mca(frequencyMask, directions, 1) - reference);
    fscError = abs(record.fsc(frequencyMask, directions, 1) - reference);
    reference = reference(:, sortOrder); mcaError = mcaError(:, sortOrder); fscError = fscError(:, sortOrder);
    azimuth = record.direction(1, directions(sortOrder));
    azimuth(azimuth > 180) = azimuth(azimuth > 180) - 360;
    valid = record.interpolationMask(directions(sortOrder));
    mcaError(:, ~valid) = NaN; fscError(:, ~valid) = NaN;
    panels = {reference, mcaError, fscError};
    for column = 1:3
        ax = nexttile(layout);
        imagesc(ax, azimuth, frequencies, panels{column});
        set(ax, 'YDir', 'normal', 'YScale', 'log');
        xlim(ax, [-180 180]); ylim(ax, [100 20000]);
        ax.XTick = -180:90:180; ax.FontName = fontName; ax.FontSize = 10; ax.TickDir = 'out';
        xlabel(ax, azimuthLabel); ylabel(ax, frequencyLabel);
        title(ax, sprintf('%s — %s', record.subject, panelNames{column}), 'FontWeight', 'bold');
        if column == 1
            colormap(ax, parula(256)); clim(ax, [-60 20]);
            cb = colorbar(ax); cb.Label.String = magnitudeLabel; cb.FontName = fontName;
        else
            colormap(ax, hot(256)); clim(ax, [0 errorLimit]);
            cb = colorbar(ax); cb.Label.String = errorLabel; cb.FontName = fontName;
        end
        hold(ax, 'on');
        xline(ax, -90, ':', 'Color', [0.1 0.1 0.1], 'LineWidth', 0.8, 'HandleVisibility', 'off');
        hold(ax, 'off');
    end
end
if language == "en"
    title(layout, {'Horizontal-plane left-ear HRTF reconstruction and absolute spectral error', ...
        'Only interpolation directions enter the error maps; dashed line marks the contralateral direction (270 degrees)'}, ...
        'FontWeight', 'bold', 'FontName', fontName);
end
exportgraphics(fig, outputPath, 'Resolution', 300);
exportgraphics(fig, strrep(outputPath, '.png', '.pdf'), 'ContentType', 'vector');
close(fig);
end

function export_pdf_from_png_figure(~, ~)
% PDF exports are written by each drawing routine alongside its PNG export.
end

function index = nearest_interpolation_direction(record, targetAzimuth, targetElevation)
azimuth = record.direction(1, :).'; elevation = record.direction(2, :).';
distance = great_circle_distance(azimuth, elevation, targetAzimuth, targetElevation);
distance(~record.interpolationMask(:)) = inf;
[~, index] = min(distance);
assert(isfinite(distance(index)), 'No interpolation direction found.');
end

function [indices, sortOrder] = horizontal_directions(record)
elevation = record.direction(2, :).';
indices = find(abs(elevation) < 1e-8);
assert(numel(indices) == 72, 'Expected 72 horizontal directions.');
azimuth = record.direction(1, indices).';
azimuth(azimuth > 180) = azimuth(azimuth > 180) - 360;
[~, sortOrder] = sort(azimuth);
end

function distanceDeg = great_circle_distance(azimuthDeg, elevationDeg, targetAzimuthDeg, targetElevationDeg)
azimuthRad = deg2rad(azimuthDeg); elevationRad = deg2rad(elevationDeg);
targetAzimuthRad = deg2rad(targetAzimuthDeg); targetElevationRad = deg2rad(targetElevationDeg);
cosineDistance = sin(elevationRad) * sin(targetElevationRad) + cos(elevationRad) .* cos(targetElevationRad) .* cos(azimuthRad - targetAzimuthRad);
distanceDeg = rad2deg(acos(max(-1, min(1, cosineDistance))));
end
