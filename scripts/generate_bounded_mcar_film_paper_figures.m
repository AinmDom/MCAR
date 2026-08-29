function generate_bounded_mcar_film_paper_figures()
%GENERATE_BOUNDED_MCAR_FILM_PAPER_FIGURES Publication figures from frozen CSVs.

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
sourceRoot = fullfile(projectRoot, 'results', ...
    'sonicom_bounded_mcar_film_correction_final_e25_validation');
outputRoot = fullfile(projectRoot, 'results', ...
    'sonicom_bounded_mcar_film_correction_e25_paper');
assert(isfolder(outputRoot), 'Generate the paper tables before figures.');

aggregate = readtable(fullfile(sourceRoot, 'aggregate_metrics.csv'), ...
    'TextType', 'string');
paired = readtable(fullfile(sourceRoot, ...
    'paired_bootstrap_comparisons.csv'), 'TextType', 'string');

methodIds = ["HYBRID", "PARENT", "FILMENS", "MCAR"];
methodLabels = ["Bounded", "Hybrid", "FiLM", "MCAR"];
metricIds = ["FullSphereERB", "Contralateral25ERB", ...
    "ContralateralHighFrequency", "HorizontalILDMAE"];
metricLabels = ["Full-sphere ERB", "Contralateral-25 ERB", ...
    "Contralateral HF", "Horizontal ILD"];
colors = [37, 99, 235; 245, 158, 11; 124, 58, 237; 100, 116, 139] / 255;

fig = figure('Color', 'white', 'Position', [100, 100, 1100, 760]);
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for metricIndex = 1:numel(metricIds)
    ax = nexttile(layout);
    means = zeros(numel(methodIds), 1);
    deviations = zeros(numel(methodIds), 1);
    for methodIndex = 1:numel(methodIds)
        mask = aggregate.Method == methodIds(methodIndex) & ...
            aggregate.Metric == metricIds(metricIndex);
        assert(sum(mask) == 1, 'Missing aggregate row.');
        means(methodIndex) = aggregate.Mean_dB(mask);
        deviations(methodIndex) = aggregate.Std_dB(mask);
    end
    bars = bar(ax, 1:numel(methodIds), means, 0.68, ...
        'FaceColor', 'flat', 'EdgeColor', 'none');
    bars.CData = colors;
    hold(ax, 'on');
    errorbar(ax, 1:numel(methodIds), means, deviations, 'k', ...
        'LineStyle', 'none', 'LineWidth', 0.9, 'CapSize', 5);
    hold(ax, 'off');
    ax.XTick = 1:numel(methodIds);
    ax.XTickLabel = methodLabels;
    ax.FontName = 'Arial';
    ax.FontSize = 10;
    ax.TickDir = 'out';
    ax.Box = 'off';
    grid(ax, 'on');
    ax.XGrid = 'off';
    ax.GridAlpha = 0.18;
    title(ax, metricLabels(metricIndex));
    ylabel(ax, 'Error (dB, lower is better)');
end
title(layout, 'SONICOM Q26 validation performance (44 subjects)', ...
    'FontName', 'Arial', 'FontSize', 14);
exportgraphics(fig, fullfile(outputRoot, 'figure_1_main_validation.png'), ...
    'Resolution', 300);
exportgraphics(fig, fullfile(outputRoot, 'figure_1_main_validation.pdf'), ...
    'ContentType', 'vector');
close(fig);

baselineIds = ["PARENT", "FILMENS", "MCAR"];
baselineLabels = ["Spectral-CNN hybrid E190", ...
    "FiLM-SIREN E130 ensemble", "MCAR v3.5.1"];
baselineColors = colors(2:4, :);
offsets = [-0.22, 0, 0.22];
fig = figure('Color', 'white', 'Position', [100, 100, 1050, 620]);
ax = axes(fig);
hold(ax, 'on');
for baselineIndex = 1:numel(baselineIds)
    means = zeros(numel(metricIds), 1);
    lower = zeros(numel(metricIds), 1);
    upper = zeros(numel(metricIds), 1);
    for metricIndex = 1:numel(metricIds)
        mask = paired.Baseline == baselineIds(baselineIndex) & ...
            paired.Metric == metricIds(metricIndex);
        assert(sum(mask) == 1, 'Missing paired row.');
        means(metricIndex) = paired.HybridMinusBaselineMean_dB(mask);
        lower(metricIndex) = paired.Bootstrap95Lower_dB(mask);
        upper(metricIndex) = paired.Bootstrap95Upper_dB(mask);
    end
    y = (1:numel(metricIds))' + offsets(baselineIndex);
    errorbar(ax, means, y, means - lower, upper - means, 'horizontal', ...
        'LineStyle', 'none', 'Marker', 'o', 'MarkerSize', 6, ...
        'MarkerFaceColor', baselineColors(baselineIndex, :), ...
        'Color', baselineColors(baselineIndex, :), 'LineWidth', 1.2, ...
        'CapSize', 6, 'DisplayName', baselineLabels(baselineIndex));
end
xline(ax, 0, '--k', 'LineWidth', 1.0, 'HandleVisibility', 'off');
hold(ax, 'off');
ax.YTick = 1:numel(metricIds);
ax.YTickLabel = metricLabels;
ax.YDir = 'reverse';
ax.YLim = [0.5, numel(metricIds) + 0.5];
ax.FontName = 'Arial';
ax.FontSize = 11;
ax.TickDir = 'out';
ax.Box = 'off';
grid(ax, 'on');
ax.YGrid = 'off';
ax.GridAlpha = 0.18;
xlabel(ax, 'Candidate minus baseline (dB; negative favors candidate)');
title(ax, 'Paired mean differences with 95% bootstrap intervals');
legend(ax, 'Location', 'southoutside', 'NumColumns', 3, 'Box', 'off');
exportgraphics(fig, fullfile(outputRoot, 'figure_2_paired_effects.png'), ...
    'Resolution', 300);
exportgraphics(fig, fullfile(outputRoot, 'figure_2_paired_effects.pdf'), ...
    'ContentType', 'vector');
close(fig);
end
