function generate_film_secondary_metric_figures()
%GENERATE_FILM_SECONDARY_METRIC_FIGURES Plot frozen validation CSV outputs.

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
resultRoot = fullfile(projectRoot, 'results', ...
    'sonicom_film_secondary_metrics_v1_validation');
figureRoot = fullfile(resultRoot, 'figures');
assert(isfolder(resultRoot), 'Secondary result directory is missing.');
if ~isfolder(figureRoot), mkdir(figureRoot); end

plot_relative_mcar_effect(resultRoot, figureRoot);
plot_spatial_distance_profile(resultRoot, figureRoot);
end

function plot_relative_mcar_effect(resultRoot, figureRoot)
tail = readtable(fullfile(resultRoot, 'paired_tail_risk.csv'), ...
    'TextType', 'string');
aggregate = readtable(fullfile(resultRoot, 'aggregate_metrics.csv'), ...
    'TextType', 'string');
endpointIds = ["FullSphereLSD", "HFFirstDifferenceMAE", ...
    "HFSecondDifferenceMAE", "MultiScaleNotchDepthMAE", ...
    "ERBBandILDMean"];
endpointLabels = ["Full-sphere LSD", "HF first difference", ...
    "HF second difference", "Notch-depth MAE", "ERB-band ILD"];
relative = zeros(size(endpointIds));
lower = zeros(size(endpointIds));
upper = zeros(size(endpointIds));
for index = 1:numel(endpointIds)
    row = tail(tail.Baseline == "MCAR" & tail.Endpoint == endpointIds(index), :);
    base = aggregate(aggregate.Method == "MCAR" & ...
        aggregate.Endpoint == endpointIds(index), :);
    assert(height(row) == 1 && height(base) == 1, ...
        'Missing MCAR comparison for %s.', endpointIds(index));
    denominator = base.Mean;
    relative(index) = 100 * row.MeanDifference / denominator;
    lower(index) = 100 * row.Bootstrap95Lower / denominator;
    upper(index) = 100 * row.Bootstrap95Upper / denominator;
end

fig = figure('Visible', 'off', 'Color', 'white', ...
    'Position', [100, 100, 920, 470]);
ax = axes(fig);
hold(ax, 'on');
for index = 1:numel(endpointIds)
    plot(ax, [lower(index), upper(index)], [index, index], '-', ...
        'Color', [0.22, 0.34, 0.55], 'LineWidth', 2.0);
end
scatter(ax, relative, 1:numel(endpointIds), 58, ...
    [0.08, 0.36, 0.64], 'filled');
xline(ax, 0, '--', 'MCAR', 'LabelVerticalAlignment', 'bottom', ...
    'Color', [0.35, 0.35, 0.35], 'LineWidth', 1.1);
hold(ax, 'off');
ax.YTick = 1:numel(endpointIds);
ax.YTickLabel = endpointLabels;
ax.YDir = 'reverse';
ax.FontName = 'Arial';
ax.FontSize = 11;
ax.TickDir = 'out';
ax.Box = 'on';
grid(ax, 'on');
ax.YGrid = 'off';
xlabel(ax, 'BOUNDED minus MCAR (% of MCAR mean; negative is better)');
title(ax, 'Secondary validation effects relative to MCAR');
exportgraphics(ax, fullfile(figureRoot, ...
    'secondary_relative_to_mcar.png'), 'Resolution', 300);
exportgraphics(ax, fullfile(figureRoot, ...
    'secondary_relative_to_mcar.pdf'), 'ContentType', 'vector');
close(fig);
end

function plot_spatial_distance_profile(resultRoot, figureRoot)
spatial = readtable(fullfile(resultRoot, 'spatial_distance_bins.csv'), ...
    'TextType', 'string');
spatial = spatial(spatial.RecordType == "aggregate", :);
methodIds = ["BOUNDED", "HYBRID", "FILMENS", "MCAR"];
methodLabels = ["Bounded E25", "Hybrid E190", "FiLM E130", "MCAR v3.5.1"];
binIds = ["0_10", "10_20", "20_30", "30_180"];
binLabels = ["0-10", "10-20", "20-30", ">=30"];
colors = [0.08, 0.36, 0.64; 0.82, 0.33, 0.20; ...
    0.47, 0.36, 0.65; 0.30, 0.30, 0.30];

fig = figure('Visible', 'off', 'Color', 'white', ...
    'Position', [100, 100, 900, 500]);
ax = axes(fig);
hold(ax, 'on');
for methodIndex = 1:numel(methodIds)
    values = zeros(size(binIds));
    for binIndex = 1:numel(binIds)
        row = spatial(spatial.Method == methodIds(methodIndex) & ...
            spatial.DistanceBin_deg == binIds(binIndex), :);
        assert(height(row) == 1, 'Missing spatial aggregate row.');
        values(binIndex) = row.LSD_dB;
    end
    plot(ax, 1:numel(binIds), values, '-o', 'LineWidth', 1.8, ...
        'MarkerSize', 6, 'MarkerFaceColor', colors(methodIndex, :), ...
        'Color', colors(methodIndex, :), 'DisplayName', methodLabels(methodIndex));
end
hold(ax, 'off');
ax.XTick = 1:numel(binIds);
ax.XTickLabel = binLabels;
ax.FontName = 'Arial';
ax.FontSize = 11;
ax.TickDir = 'out';
ax.Box = 'on';
grid(ax, 'on');
xlabel(ax, 'Minimum angular distance to a Q26 observation (degrees)');
ylabel(ax, 'Full-spectrum LSD (dB)');
title(ax, 'Spatial interpolation error versus distance from Q26');
legend(ax, 'Location', 'northwest', 'Box', 'off');
exportgraphics(ax, fullfile(figureRoot, ...
    'spatial_lsd_by_q26_distance.png'), 'Resolution', 300);
exportgraphics(ax, fullfile(figureRoot, ...
    'spatial_lsd_by_q26_distance.pdf'), 'ContentType', 'vector');
close(fig);
end
