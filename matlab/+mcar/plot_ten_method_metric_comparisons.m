function plot_ten_method_metric_comparisons()
%PLOT_TEN_METHOD_METRIC_COMPARISONS Export one ten-method figure per metric.
%
% Creates 24 standalone ranked dot-and-whisker charts from committed CSV
% results. Each point is the 44-subject mean and each whisker is a fixed
% 10,000-resample subject-bootstrap 95% interval. No raw HRTF or test input
% is read by this visualization-only function.

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
dataRoot = fullfile(projectRoot, 'results', ...
    'sonicom_complete_ten_method_metric_figure_data_v1');
summaryFile = fullfile(dataRoot, 'metric_summary.csv');
indexFile = fullfile(dataRoot, 'figure_index.csv');
outputRoot = fullfile(projectRoot, 'outputs', ...
    'ten_method_metric_figures_v1');
partialRoot = outputRoot + ".partial";
assert(isfile(summaryFile) && isfile(indexFile), ...
    'Plotting summaries are missing.');
assert(~isfolder(outputRoot) && ~isfolder(partialRoot), ...
    'Refusing to overwrite visualization output.');
mkdir(fullfile(partialRoot, 'png'));
mkdir(fullfile(partialRoot, 'pdf'));

data = readtable(summaryFile, 'TextType', 'string');
index = readtable(indexFile, 'TextType', 'string');
assert(height(data) == 240 && height(index) == 24, ...
    'Expected 240 method rows and 24 figures.');
assert(all(isfinite(data.Mean)) && all(isfinite(data.Bootstrap95Lower)) && ...
    all(isfinite(data.Bootstrap95Upper)), 'Plot input is not finite.');

palette = orderedcolors("gem");
otherColor = palette(1, :);
hybridColor = palette(2, :);
boundedColor = palette(3, :);

for figureIndex = 1:height(index)
    figureNumber = index.FigureNumber(figureIndex);
    rows = data(data.FigureNumber == figureNumber, :);
    assert(height(rows) == 10 && numel(unique(rows.Method)) == 10, ...
        'Incomplete ten-method figure %d.', figureNumber);
    descriptive = rows.Direction(1) == "Descriptive only";
    tiedMeans = (max(rows.Mean) - min(rows.Mean)) <= ...
        1e-12 * max(1, max(abs(rows.Mean)));
    rankable = ~descriptive && ~tiedMeans;
    if ~rankable
        [~, order] = sort(rows.MethodOrder, 'ascend');
    else
        [~, order] = sort(rows.Mean, 'ascend');
    end
    rows = rows(order, :);

    fig = figure('Visible', 'off', 'Color', 'white', ...
        'Position', [100, 100, 1280, 800], ...
        'Name', char(index.Title(figureIndex)));
    ax = axes(fig);
    hold(ax, 'on');
    y = (1:10).';
    lowerError = rows.Mean - rows.Bootstrap95Lower;
    upperError = rows.Bootstrap95Upper - rows.Mean;
    for methodIndex = 1:10
        [color, marker] = method_style(rows.Method(methodIndex), ...
            otherColor, hybridColor, boundedColor);
        errorbar(ax, rows.Mean(methodIndex), y(methodIndex), ...
            lowerError(methodIndex), upperError(methodIndex), ...
            'horizontal', marker, 'Color', color, 'MarkerFaceColor', color, ...
            'MarkerEdgeColor', 'white', 'MarkerSize', 8, 'LineWidth', 1.8, ...
            'CapSize', 9);
    end

    low = min(rows.Bootstrap95Lower);
    high = max(rows.Bootstrap95Upper);
    span = max(high - low, max(abs([low, high])) * 0.08);
    if span <= eps
        span = 1;
    end
    left = low - 0.08 * span;
    right = high + 0.34 * span;
    labelAlignment = 'left';
    labelX = high + 0.06 * span;
    if descriptive
        left = max(0, low - 0.03);
        right = 100.0;
        labelX = low - 0.005;
        labelAlignment = 'right';
    elseif rows.DisplayUnit(1) == "%"
        left = max(0, left);
        right = min(100, right);
    end
    xlim(ax, [left, right]);
    ylim(ax, [0.35, 10.65]);
    ax.YDir = 'reverse';
    ax.YTick = y;
    methodLabels = cellstr(rows.MethodLabel);
    if rankable
        methodLabels{1} = [methodLabels{1}, '  - best mean'];
    end
    ax.YTickLabel = methodLabels;
    ax.TickLabelInterpreter = 'none';
    ax.FontName = 'Arial';
    ax.FontSize = 11;
    ax.TickDir = 'out';
    ax.Box = 'off';
    ax.XGrid = 'on';
    ax.YGrid = 'off';
    ax.GridLineStyle = ':';
    ax.GridAlpha = 0.25;
    ax.Layer = 'top';
    xlabel(ax, axis_label(rows.DisplayUnit(1)), 'FontWeight', 'bold');
    title(ax, index.Title(figureIndex), 'FontSize', 17, 'FontWeight', 'bold');
    subtitle(ax, tier_subtitle(rows.EvidenceTier(1), descriptive, tiedMeans), ...
        'FontSize', 11, 'FontWeight', 'normal');

    for methodIndex = 1:10
        text(ax, labelX, y(methodIndex), ...
            format_value(rows.Mean(methodIndex), rows.Endpoint(methodIndex), ...
            rows.DisplayUnit(methodIndex)), ...
            'FontName', 'Arial', 'FontSize', 10, ...
            'HorizontalAlignment', labelAlignment, 'VerticalAlignment', 'middle');
    end
    text(ax, 0.01, -0.10, ...
        'Point = mean; whisker = 95% subject-bootstrap CI (10,000 resamples); n = 44 per method.', ...
        'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 9, ...
        'Color', [0.25, 0.25, 0.25]);
    text(ax, 0.99, -0.10, ...
        'Circles: other methods   Diamond: Hybrid E190   Square: Bounded E25', ...
        'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 9, ...
        'HorizontalAlignment', 'right', 'Color', [0.25, 0.25, 0.25]);
    hold(ax, 'off');

    pngFile = fullfile(partialRoot, char(index.PngFile(figureIndex)));
    pdfFile = fullfile(partialRoot, char(index.PdfFile(figureIndex)));
    exportgraphics(fig, pngFile, 'Resolution', 300, 'BackgroundColor', 'white');
    exportgraphics(fig, pdfFile, 'ContentType', 'vector', ...
        'BackgroundColor', 'white');
    close(fig);
    fprintf('metric figure [%d/24]: %s\n', figureIndex, index.Endpoint(figureIndex));
end

copyfile(summaryFile, fullfile(partialRoot, 'metric_summary.csv'));
copyfile(indexFile, fullfile(partialRoot, 'figure_index.csv'));
write_readme(fullfile(partialRoot, 'README.md'), index);
pngFiles = dir(fullfile(partialRoot, 'png', '*.png'));
pdfFiles = dir(fullfile(partialRoot, 'pdf', '*.pdf'));
assert(numel(pngFiles) == 24 && numel(pdfFiles) == 24 && ...
    all([pngFiles.bytes] > 0) && all([pdfFiles.bytes] > 0), ...
    'Figure export completeness check failed.');
summary = struct('status', 'completed', 'figure_count', 24, ...
    'png_count', numel(pngFiles), 'pdf_count', numel(pdfFiles), ...
    'method_count_per_figure', 10, 'subject_count_per_method', 44, ...
    'all_files_nonempty', true, 'new_test_subject_count_read', 0);
write_json(fullfile(partialRoot, 'summary.json'), summary);
movefile(partialRoot, outputRoot);
fprintf('Ten-method metric visualization complete: %s\n', outputRoot);
end

function [color, marker] = method_style(method, otherColor, hybridColor, boundedColor)
if method == "HYBRID"
    color = hybridColor;
    marker = 'd';
elseif method == "BOUNDED"
    color = boundedColor;
    marker = 's';
else
    color = otherColor;
    marker = 'o';
end
end

function value = axis_label(unit)
if unit == "%"
    value = 'Rate / fraction (%)';
elseif unit == "us"
    value = 'Error (microseconds)';
elseif unit == "Hz"
    value = 'Error (Hz)';
else
    value = "Value (" + unit + ")";
end
end

function value = tier_subtitle(tier, descriptive, tiedMeans)
if descriptive
    direction = 'descriptive reference quantity; method-invariant by definition';
elseif tiedMeans
    direction = 'lower is better; all ten method means are tied';
else
    direction = 'lower is better';
end
value = tier + " | " + direction;
end

function label = format_value(value, endpoint, unit)
if unit == "%" && endpoint == "ReferenceNotchFraction"
    label = sprintf('%.4f%%', value);
elseif unit == "%"
    label = sprintf('%.2f%%', value);
elseif abs(value) >= 100
    label = sprintf('%.1f', value);
elseif abs(value) >= 10
    label = sprintf('%.2f', value);
elseif abs(value) >= 1
    label = sprintf('%.3f', value);
else
    label = sprintf('%.4f', value);
end
end

function write_readme(path, index)
file = fopen(path, 'w');
assert(file ~= -1, 'Could not create README: %s', path);
cleanup = onCleanup(@() fclose(file));
fprintf(file, '# Ten-method per-metric figures\n\n');
fprintf(file, ['Each standalone figure compares all ten registered methods. ', ...
    'Points are 44-subject means and whiskers are deterministic 95%% ', ...
    'subject-bootstrap intervals (10,000 resamples). Validation, frozen ', ...
    'engineering test, secondary, and deferred evidence remain separate.\n\n']);
fprintf(file, '| # | Evidence tier | Endpoint | PNG | Vector PDF |\n');
fprintf(file, '|---:|---|---|---|---|\n');
for row = 1:height(index)
    fprintf(file, '| %d | %s | `%s` | `%s` | `%s` |\n', ...
        index.FigureNumber(row), index.EvidenceTier(row), index.Endpoint(row), ...
        index.PngFile(row), index.PdfFile(row));
end
clear cleanup;
end

function write_json(path, value)
file = fopen(path, 'w');
assert(file ~= -1, 'Could not create JSON output: %s', path);
cleanup = onCleanup(@() fclose(file));
fprintf(file, '%s\n', jsonencode(value, 'PrettyPrint', true));
clear cleanup;
end
