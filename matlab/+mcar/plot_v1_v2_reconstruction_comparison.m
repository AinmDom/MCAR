function plot_v1_v2_reconstruction_comparison(v1Name, v2Name)
%PLOT_V1_V2_RECONSTRUCTION_COMPARISON Compare strict test-set metrics.

if nargin < 1 || isempty(v1Name)
    v1Name = 'mlp_n03_v1';
end
if nargin < 2 || isempty(v2Name)
    v2Name = 'mlp_n03_v2';
end
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
resultsRoot = fullfile(projectRoot, 'results', 'residual_mlp');
v1Root = fullfile(resultsRoot, char(v1Name), 'evaluation');
v2Root = fullfile(resultsRoot, char(v2Name), 'evaluation');
v1 = readtable(fullfile(v1Root, 'aggregate_metrics.csv'), ...
    'TextType', 'string');
v2 = readtable(fullfile(v2Root, 'aggregate_metrics.csv'), ...
    'TextType', 'string');
assert(isequal(v1.Metric, v2.Metric), ...
    'v1 and v2 aggregate metric rows do not match.');

metricLabels = ["Full-sphere ERB"; "Contralateral 25° ERB"; ...
    "Contralateral >10 kHz"; "ILD MAE"];
absoluteValues = [v2.MCAMean_dB, v1.CorrectedMean_dB, ...
    v2.CorrectedMean_dB];
relativeValues = [v1.RelativeImprovement_percent, ...
    v2.RelativeImprovement_percent];

figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1500, 850]);
layout = tiledlayout(figureHandle, 1, 2, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

nexttile(layout);
bar(absoluteValues);
grid on;
ylabel('Mean test error (dB)');
title('Strict reconstruction metrics');
xticks(1:numel(metricLabels));
xticklabels(metricLabels);
xtickangle(20);
legend({'MCA', 'v1', 'v2 auditory-aware'}, 'Location', 'best');

nexttile(layout);
bar(relativeValues);
grid on;
ylabel('Improvement over MCA (%)');
title('Relative improvement on 12 unseen subjects');
xticks(1:numel(metricLabels));
xticklabels(metricLabels);
xtickangle(20);
legend({'v1', 'v2 auditory-aware'}, 'Location', 'best');

title(layout, 'Residual MLP v1 versus v2', 'FontWeight', 'bold');
outputFile = fullfile(v2Root, 'figures', ...
    'v1_v2_aggregate_comparison.png');
exportgraphics(figureHandle, outputFile, 'Resolution', 180);
close(figureHandle);
fprintf('Wrote %s\n', outputFile);
end
