%PLOT_RANF_HORIZONTAL_COMPARISON RANF versus MCA and frozen MCAR v3.2.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
resultRoot = fullfile(projectRoot, 'results', ...
    'sonicom_ranf_q26_final_test');
inputFile = fullfile(resultRoot, 'per_subject_metrics.csv');
figureRoot = fullfile(resultRoot, 'figures');
assert(isfile(inputFile), 'Missing unified per-subject table: %s', inputFile);
if ~isfolder(figureRoot)
    mkdir(figureRoot);
end

perSubject = readtable(inputFile, 'TextType', 'string');
assert(height(perSubject) == 44, 'Expected 44 frozen test subjects.');

metricIds = ["FullSphereERB", "Contralateral25ERB", ...
    "ContralateralHighFrequency", "HorizontalILDMAE"];
metricLabels = ["全空间 ERB", ...
    "对侧 25° ERB", ...
    "对侧高频", ...
    "水平面严格 ILD"];
metricLabelsPlain = metricLabels;

reductionVsMca = zeros(numel(metricIds), 1);
reductionVsMcar = zeros(numel(metricIds), 1);
winsVsMca = zeros(numel(metricIds), 1);
winsVsMcar = zeros(numel(metricIds), 1);
for metricIndex = 1:numel(metricIds)
    metric = metricIds(metricIndex);
    ranf = perSubject.("RANF_" + metric + "_dB");
    mca = perSubject.("MCA_" + metric + "_dB");
    mcar = perSubject.("MCARv32_" + metric + "_dB");
    assert(all(isfinite([ranf; mca; mcar])), ...
        'Non-finite value found for %s.', metric);
    reductionVsMca(metricIndex) = 100 * (mean(mca) - mean(ranf)) / mean(mca);
    reductionVsMcar(metricIndex) = 100 * (mean(mcar) - mean(ranf)) / mean(mcar);
    winsVsMca(metricIndex) = sum(ranf < mca);
    winsVsMcar(metricIndex) = sum(ranf < mcar);
end

figureData = table(metricLabelsPlain.', reductionVsMca, reductionVsMcar, ...
    winsVsMca, winsVsMcar, ...
    'VariableNames', {'Metric', 'RANFvsMCA_ErrorReduction_percent', ...
    'RANFvsMCARv32_ErrorReduction_percent', 'RANFWinsVsMCA', ...
    'RANFWinsVsMCARv32'});
writetable(figureData, fullfile(figureRoot, ...
    'ranf_vs_mca_mcar_error_reduction.csv'));

fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1650, 1060]);
ax = axes(fig, 'Position', [0.23, 0.22, 0.67, 0.60]);
hold(ax, 'on');

teal = [0.07, 0.70, 0.62];
orange = [1.00, 0.63, 0.00];
dark = [0.06, 0.09, 0.16];
muted = [0.32, 0.37, 0.46];
gridColor = [0.88, 0.90, 0.93];
groupCenters = (1:numel(metricIds)).';
seriesOffset = 0.14;
barHalfHeight = 0.085;

for metricIndex = 1:numel(metricIds)
    yMca = groupCenters(metricIndex) - seriesOffset;
    yMcar = groupCenters(metricIndex) + seriesOffset;
    draw_bar(ax, reductionVsMca(metricIndex), yMca, ...
        barHalfHeight, teal);
    draw_bar(ax, reductionVsMcar(metricIndex), yMcar, ...
        barHalfHeight, orange);
    draw_value_label(ax, reductionVsMca(metricIndex), yMca, dark);
    draw_value_label(ax, reductionVsMcar(metricIndex), yMcar, dark);
    text(ax, 36.2, groupCenters(metricIndex), ...
        sprintf('优于 MCAR：%d/44', winsVsMcar(metricIndex)), ...
        'Color', muted, 'FontName', 'Microsoft YaHei', 'FontSize', 10.5, ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
end

zeroLine = xline(ax, 0, '-', 'LineWidth', 2.2, 'Color', dark);
zeroLine.HandleVisibility = 'off';

xlim(ax, [-30, 47]);
ylim(ax, [0.45, 4.55]);
ax.YDir = 'reverse';
ax.YTick = groupCenters;
ax.YTickLabel = metricLabels;
ax.TickLabelInterpreter = 'none';
ax.XTick = -30:10:40;
ax.XTickLabel = compose('%+d%%', ax.XTick);
ax.XGrid = 'on';
ax.YGrid = 'off';
ax.GridColor = gridColor;
ax.GridAlpha = 1;
ax.LineWidth = 0.8;
ax.Box = 'off';
ax.TickDir = 'out';
ax.TickLength = [0, 0];
ax.FontName = 'Microsoft YaHei';
ax.FontSize = 11;
ax.XColor = muted;
ax.YColor = dark;
ax.Layer = 'bottom';
ax.YAxis.FontWeight = 'bold';

text(ax, 0.5, 1.155, 'RANF 的优势与局限', ...
    'Units', 'normalized', 'Color', dark, ...
    'FontName', 'Microsoft YaHei', 'Interpreter', 'none', ...
    'FontSize', 22, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
text(ax, 0.5, 1.105, ...
    'RANF 的误差降低率：正值表示更优，负值表示更差', ...
    'Units', 'normalized', 'Color', muted, ...
    'FontName', 'Microsoft YaHei', 'Interpreter', 'none', ...
    'FontSize', 11.5, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom');

legendHandles = gobjects(2, 1);
legendHandles(1) = patch(ax, NaN, NaN, teal, 'EdgeColor', 'none', ...
    'DisplayName', 'RANF 相对 MCA');
legendHandles(2) = patch(ax, NaN, NaN, orange, 'EdgeColor', 'none', ...
    'DisplayName', 'RANF 相对 MCAR v3.2');
lg = legend(ax, legendHandles, 'Location', 'southoutside', ...
    'Orientation', 'horizontal', 'NumColumns', 2);
lg.Box = 'off';
lg.FontName = 'Microsoft YaHei';
lg.Interpreter = 'none';
lg.FontSize = 11;
lg.TextColor = dark;

text(ax, 0.5, -0.205, ...
    '固定的 44 名 SONICOM 测试被试；767 个仅插值方向。', ...
    'Units', 'normalized', 'Color', muted, ...
    'FontName', 'Microsoft YaHei', 'Interpreter', 'none', ...
    'FontSize', 9.5, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'top');

hold(ax, 'off');
pngPath = fullfile(figureRoot, 'ranf_vs_mca_mcar_error_reduction.png');
pdfPath = fullfile(figureRoot, 'ranf_vs_mca_mcar_error_reduction.pdf');
exportgraphics(fig, pngPath, 'Resolution', 300);
exportgraphics(fig, pdfPath, 'ContentType', 'vector');
close(fig);

fprintf('RANF comparison figure written to:\n%s\n%s\n', pngPath, pdfPath);
disp(figureData);

function handle = draw_bar(ax, value, y, halfHeight, color)
%DRAW_BAR Draw one horizontal bar from zero to value.
handle = patch(ax, [0, value, value, 0], ...
    [y - halfHeight, y - halfHeight, y + halfHeight, y + halfHeight], ...
    color, 'EdgeColor', 'none');
end

function draw_value_label(ax, value, y, color)
%DRAW_VALUE_LABEL Place a signed percentage just beyond a bar tip.
if value >= 0
    x = value + 0.7;
    alignment = 'left';
else
    x = value - 0.7;
    alignment = 'right';
end
text(ax, x, y, sprintf('%+.1f%%', value), ...
    'Color', color, 'FontName', 'Microsoft YaHei', 'FontSize', 10.5, ...
    'FontWeight', 'bold', 'HorizontalAlignment', alignment, ...
    'VerticalAlignment', 'middle');
end
