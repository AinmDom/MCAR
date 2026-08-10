function plot_sonicom_metric_overview(perSubjectOrCsv, figuresRoot, ...
        includeV3, includeV31, comparisonLabel, splitDisplayName, figurePrefix)
%PLOT_SONICOM_METRIC_OVERVIEW Plot strict reconstruction metrics from a table or CSV.
%
% This public plotting entry point lets an existing evaluation be redrawn
% without reading HRTF data or rerunning reconstruction.

if istable(perSubjectOrCsv)
    perSubject = perSubjectOrCsv;
else
    perSubject = readtable(perSubjectOrCsv);
end
if ~isfolder(figuresRoot)
    mkdir(figuresRoot);
end

figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [60, 60, 1800, 1200]);
layout = tiledlayout(figureHandle, 2, 2, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

values = [perSubject.MCAFullSphereERB_dB, ...
    perSubject.MLPv1FullSphereERB_dB, ...
    perSubject.MLPv2FullSphereERB_dB];
values = append_optional_methods(values, perSubject, includeV3, includeV31, ...
    'MLPCNNv3FullSphereERB_dB', 'MLPCNNv31FullSphereERB_dB');
plot_grouped_metric(nexttile(layout), values, 'Full-sphere ERB', ...
    'ERB error (dB)', includeV3, includeV31, comparisonLabel, splitDisplayName);

values = [perSubject.MCAContralateral25ERB_dB, ...
    perSubject.MLPv1Contralateral25ERB_dB, ...
    perSubject.MLPv2Contralateral25ERB_dB];
values = append_optional_methods(values, perSubject, includeV3, includeV31, ...
    'MLPCNNv3Contralateral25ERB_dB', ...
    'MLPCNNv31Contralateral25ERB_dB');
plot_grouped_metric(nexttile(layout), values, ...
    'Contralateral 25-degree ERB', 'ERB error (dB)', ...
    includeV3, includeV31, comparisonLabel, splitDisplayName);

values = [perSubject.MCAContralateralHighFrequency_dB, ...
    perSubject.MLPv1ContralateralHighFrequency_dB, ...
    perSubject.MLPv2ContralateralHighFrequency_dB];
values = append_optional_methods(values, perSubject, includeV3, includeV31, ...
    'MLPCNNv3ContralateralHighFrequency_dB', ...
    'MLPCNNv31ContralateralHighFrequency_dB');
plot_grouped_metric(nexttile(layout), values, ...
    'Contralateral high frequency', 'Magnitude error (dB)', ...
    includeV3, includeV31, comparisonLabel, splitDisplayName);

values = [perSubject.MCAHorizontalILDMAE_dB, ...
    perSubject.MLPv1HorizontalILDMAE_dB, ...
    perSubject.MLPv2HorizontalILDMAE_dB];
values = append_optional_methods(values, perSubject, includeV3, includeV31, ...
    'MLPCNNv3HorizontalILDMAE_dB', 'MLPCNNv31HorizontalILDMAE_dB');
plot_grouped_metric(nexttile(layout), values, ...
    'Horizontal-plane strict ILD', 'ILD MAE (dB)', ...
    includeV3, includeV31, comparisonLabel, splitDisplayName);

title(layout, sprintf('SONICOM %s: strict reconstruction metrics', ...
    splitDisplayName), 'FontWeight', 'bold');
exportgraphics(figureHandle, fullfile(figuresRoot, ...
    [figurePrefix '_metric_overview.png']), 'Resolution', 180);
close(figureHandle);
end

function values = append_optional_methods(values, perSubject, ...
        includeV3, includeV31, v3Variable, v31Variable)
if includeV3
    values(:, end + 1) = perSubject.(v3Variable);
end
if includeV31
    values(:, end + 1) = perSubject.(v31Variable);
end
end

function plot_grouped_metric(axisHandle, values, titleText, yLabelText, ...
        includeV3, includeV31, comparisonLabel, splitDisplayName)
plot(axisHandle, values(:, 1), '-', 'LineWidth', 1.0);
hold(axisHandle, 'on');
plot(axisHandle, values(:, 2), '-', 'LineWidth', 1.0);
plot(axisHandle, values(:, 3), '-', 'LineWidth', 1.2);
if includeV3
    plot(axisHandle, values(:, 4), '-', 'LineWidth', 1.4);
end
if includeV31
    plot(axisHandle, values(:, 5), '-', 'LineWidth', 1.6);
end
grid(axisHandle, 'on');
if size(values, 1) == 1
    xlim(axisHandle, [0.5, 1.5]);
else
    xlim(axisHandle, [1, size(values, 1)]);
end
xlabel(axisHandle, sprintf('%s subject index', splitDisplayName));
ylabel(axisHandle, yLabelText);
title(axisHandle, titleText);
labels = ["MCA", "MLP v1", "MLP v2"];
if includeV3
    labels(end + 1) = "MLP+CNN v3";
end
if includeV31
    labels(end + 1) = string(comparisonLabel);
end
legend(axisHandle, labels, 'Location', 'best');
end
