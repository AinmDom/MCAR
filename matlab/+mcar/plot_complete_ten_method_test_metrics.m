function plot_complete_ten_method_test_metrics()
%PLOT_COMPLETE_TEN_METHOD_TEST_METRICS Export one ten-method chart per test metric.
%
% Reads only consolidated CSV summaries. Each point is the 44-subject mean;
% whiskers are deterministic 10,000-resample subject-bootstrap 95% intervals.

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
dataRoot = fullfile(projectRoot, 'results', 'sonicom_complete_ten_method_test_v1');
outputRoot = fullfile(projectRoot, 'outputs', 'ten_method_test_metric_figures_v1');
partialRoot = outputRoot + ".partial";
assert(isfile(fullfile(dataRoot, 'aggregate_metrics.csv')) && ...
    isfile(fullfile(dataRoot, 'figure_index.csv')), 'Consolidated test tables are missing.');
assert(~isfolder(outputRoot) && ~isfolder(partialRoot), 'Refusing to overwrite figure output.');
mkdir(fullfile(partialRoot, 'png'));
mkdir(fullfile(partialRoot, 'pdf'));

data = readtable(fullfile(dataRoot, 'aggregate_metrics.csv'), 'TextType', 'string');
index = readtable(fullfile(dataRoot, 'figure_index.csv'), 'TextType', 'string');
assert(height(data) == 200 && height(index) == 20, 'Expected 200 summary rows and 20 figures.');
assert(all(isfinite(data.Mean)) && all(isfinite(data.Bootstrap95Lower)) && ...
    all(isfinite(data.Bootstrap95Upper)), 'Plot input is not finite.');

theme = test_figure_theme();
for figureIndex = 1:height(index)
    endpoint = index.Endpoint(figureIndex);
    rows = data(data.Endpoint == endpoint, :);
    assert(height(rows) == 10 && numel(unique(rows.Method)) == 10, ...
        'Incomplete ten-method endpoint: %s', endpoint);
    percentage = rows.Unit(1) == "fraction";
    descriptive = rows.Direction(1) == "Descriptive only";
    if percentage
        rows.Mean = 100 .* rows.Mean;
        rows.Bootstrap95Lower = 100 .* rows.Bootstrap95Lower;
        rows.Bootstrap95Upper = 100 .* rows.Bootstrap95Upper;
    end
    tiedMeans = max(rows.Mean) - min(rows.Mean) <= 1e-12 * max(1, max(abs(rows.Mean)));
    if descriptive || tiedMeans
        [~, order] = sort(rows.MethodOrder, 'ascend');
    else
        [~, order] = sort(rows.Mean, 'ascend');
    end
    rows = rows(order, :);

    fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100, 100, 1500, 950]);
    ax = axes(fig); %#ok<LAXES> One independent publication figure per endpoint.
    colororder(ax, theme.plotColors);
    hold(ax, 'on');
    y = (1:10).';
    for rowIndex = 1:10
        [color, marker] = method_style(rows.Method(rowIndex), theme);
        errorbar(ax, rows.Mean(rowIndex), y(rowIndex), ...
            rows.Mean(rowIndex) - rows.Bootstrap95Lower(rowIndex), ...
            rows.Bootstrap95Upper(rowIndex) - rows.Mean(rowIndex), ...
            'horizontal', marker, 'Color', color, 'MarkerFaceColor', color, ...
            'MarkerEdgeColor', 'white', 'MarkerSize', 9, 'LineWidth', 1.9, 'CapSize', 10);
    end
    low = min(rows.Bootstrap95Lower);
    high = max(rows.Bootstrap95Upper);
    span = max(high - low, max(abs([low, high])) * 0.08);
    if span <= eps
        span = 1;
    end
    left = low - 0.08 * span;
    right = high + 0.34 * span;
    labelX = high + 0.06 * span;
    alignment = 'left';
    if descriptive
        left = max(0, low - 0.03 * span);
        right = min(100, high + 0.08 * span);
        labelX = low - 0.01 * span;
        alignment = 'right';
    elseif percentage
        left = max(0, left);
        right = min(100, right);
    end
    if right <= left
        right = left + 1;
    end
    xlim(ax, [left, right]);
    ylim(ax, [0.35, 10.65]);
    ax.YDir = 'reverse';
    ax.YTick = y;
    labels = cellstr(rows.MethodLabel);
    if ~descriptive && ~tiedMeans
        labels{1} = [labels{1}, '  - best mean'];
    end
    ax.YTickLabel = labels;
    ax.TickLabelInterpreter = 'none';
    ax.FontName = 'Arial';
    ax.FontSize = 12;
    ax.TickDir = 'out';
    ax.Box = 'off';
    ax.XGrid = 'on';
    ax.YGrid = 'off';
    ax.GridLineStyle = ':';
    ax.GridAlpha = 0.25;
    ax.Layer = 'top';
    xlabel(ax, axis_label(rows.Unit(1)), 'FontWeight', 'bold');
    title(ax, index.Title(figureIndex), 'FontSize', 18, 'FontWeight', 'bold');
    subtitle(ax, subtitle_text(descriptive, tiedMeans), 'FontSize', 11);
    for rowIndex = 1:10
        text(ax, labelX, y(rowIndex), format_value(rows.Mean(rowIndex), percentage), ...
            'FontName', 'Arial', 'FontSize', 10, 'HorizontalAlignment', alignment, ...
            'VerticalAlignment', 'middle');
    end
    text(ax, 0.01, -0.10, ...
        'Point = test mean; whisker = 95% subject-bootstrap CI (10,000 resamples); n = 44.', ...
        'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 9, 'Color', theme.note);
    text(ax, 0.99, -0.10, ...
        'Circle: baselines   Diamond: Hybrid E190   Square: Bounded E25', ...
        'Units', 'normalized', 'FontName', 'Arial', 'FontSize', 9, ...
        'HorizontalAlignment', 'right', 'Color', theme.note);
    hold(ax, 'off');
    exportgraphics(fig, fullfile(partialRoot, index.PngFile(figureIndex)), ...
        'Resolution', 300, 'BackgroundColor', 'white');
    exportgraphics(fig, fullfile(partialRoot, index.PdfFile(figureIndex)), ...
        'ContentType', 'vector', 'BackgroundColor', 'white');
    close(fig);
    fprintf('complete test metric figure [%d/20]: %s\n', figureIndex, endpoint);
end

copyfile(fullfile(dataRoot, 'aggregate_metrics.csv'), fullfile(partialRoot, 'aggregate_metrics.csv'));
copyfile(fullfile(dataRoot, 'figure_index.csv'), fullfile(partialRoot, 'figure_index.csv'));
pngFiles = dir(fullfile(partialRoot, 'png', '*.png'));
pdfFiles = dir(fullfile(partialRoot, 'pdf', '*.pdf'));
assert(numel(pngFiles) == 20 && numel(pdfFiles) == 20 && ...
    all([pngFiles.bytes] > 0) && all([pdfFiles.bytes] > 0), 'Figure export is incomplete.');
summary = struct('status', 'completed', 'split', 'test', 'figure_count', 20, ...
    'png_count', 20, 'pdf_count', 20, 'method_count_per_figure', 10, ...
    'subject_count_per_method', 44, 'all_files_nonempty', true, ...
    'new_test_subject_count_read', 0);
write_json(fullfile(partialRoot, 'summary.json'), summary);
movefile(partialRoot, outputRoot);
fprintf('Complete test metric figures written to %s\n', outputRoot);
end

function theme = test_figure_theme()
palette = orderedcolors("gem");
theme.plotColors = palette;
theme.other = palette(1, :);
theme.hybrid = palette(2, :);
theme.bounded = palette(3, :);
theme.note = [0.25, 0.25, 0.25];
end

function [color, marker] = method_style(method, theme)
if method == "HYBRID"
    color = theme.hybrid;
    marker = 'd';
elseif method == "BOUNDED"
    color = theme.bounded;
    marker = 's';
else
    color = theme.other;
    marker = 'o';
end
end

function value = axis_label(unit)
if unit == "fraction"
    value = 'Rate / fraction (%)';
elseif unit == "us"
    value = 'Error (microseconds)';
elseif unit == "Hz"
    value = 'Error (Hz)';
else
    value = "Value (" + unit + ")";
end
end

function value = subtitle_text(descriptive, tied)
if descriptive
    direction = 'descriptive reference quantity; method-invariant by definition';
elseif tied
    direction = 'lower is better; all ten means are tied';
else
    direction = 'lower is better; frozen engineering test / post-lock characterization';
end
value = "Test | " + direction;
end

function label = format_value(value, percentage)
if percentage
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

function write_json(path, value)
file = fopen(path, 'w');
assert(file ~= -1, 'Could not create JSON output: %s', path);
cleanup = onCleanup(@() fclose(file));
fprintf(file, '%s\n', jsonencode(value, 'PrettyPrint', true));
clear cleanup;
end
