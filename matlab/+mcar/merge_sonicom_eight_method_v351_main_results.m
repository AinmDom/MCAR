function merge_sonicom_eight_method_v351_main_results()
%MERGE_SONICOM_EIGHT_METHOD_V351_MAIN_RESULTS Promote and compare v3.5.1.
%
% This result-level merge replaces the frozen MCAR v3.2 epoch-39 rows in the
% prior eight-method table with frozen MCAR v3.5.1 rows. It does not read
% HRTFs, predictions, or checkpoints, and it leaves all seven baselines exact.

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
sourceRoot = fullfile(projectRoot, 'results', ...
    'sonicom_eight_method_q26_epoch39_final_test');
v351Root = fullfile(projectRoot, 'results', ...
    'sonicom_mlp_cnn_q26_v351_vs_v32e39_frozen_test_strict');
outputRoot = fullfile(projectRoot, 'results', ...
    'sonicom_eight_method_q26_v351_main_test');
figuresRoot = fullfile(outputRoot, 'figures');
if ~isfolder(figuresRoot)
    mkdir(figuresRoot);
end

sourceLong = readtable(fullfile(sourceRoot, 'metric_long.csv'), TextType='string');
candidate = readtable(fullfile(v351Root, 'per_subject_metrics.csv'), ...
    TextType='string');
candidateQuality = readtable(fullfile(v351Root, 'quality_checks.csv'), ...
    TextType='string');

methodIds = ["MCARv351" "FSPAE" "RANF" "MCA" "SUpDEqBary" ...
    "SUpDEqNN" "SUpDEqSH" "SHOnly"];
methodLabels = ["MCAR v3.5.1" "FSP-AE" "RANF" "MCA" ...
    "SUpDEq + Barycentric" "SUpDEq + Natural Neighbor" ...
    "SUpDEq + SH" "SH only"];
metricIds = ["FullSphereERB" "Contralateral25ERB" ...
    "ContralateralHighFrequency" "HorizontalILDMAE"];
metricLabels = ["Full-sphere ERB" "Contralateral 25° ERB" ...
    "Contralateral high-frequency" "Horizontal ILD MAE"];
candidateVariables = [
    "MLPCNNv31FullSphereERB_dB"
    "MLPCNNv31Contralateral25ERB_dB"
    "MLPCNNv31ContralateralHighFrequency_dB"
    "MLPCNNv31HorizontalILDMAE_dB"
];

assert(height(sourceLong) == 44 * 8 * 4, ...
    'Legacy eight-method table must contain 44 x 8 x 4 rows.');
assert(~anymissing(sourceLong) && all(isfinite(sourceLong.Value_dB)), ...
    'Legacy eight-method table contains missing or nonfinite values.');
assert_no_duplicate_keys(sourceLong);
assert(height(candidate) == 44 && ~anymissing(candidate), ...
    'v3.5.1 result table must contain 44 complete subjects.');
candidateNumeric = table2array(candidate(:, vartype('numeric')));
assert(all(isfinite(candidateNumeric), 'all'), ...
    'v3.5.1 result table contains nonfinite numeric values.');
assert(numel(unique(candidate.SubjectID)) == 44, ...
    'v3.5.1 SubjectID values must be unique.');

sourceSubjects = sortrows(unique(sourceLong(:, ["SubjectLabel" "SubjectID"])), ...
    "SubjectID");
candidateSubjects = sortrows(candidate(:, ["SubjectLabel" "SubjectID"]), ...
    "SubjectID");
assert(isequal(sourceSubjects, candidateSubjects), ...
    'Legacy and v3.5.1 subject keys do not match.');

otherLong = sourceLong(sourceLong.Method ~= "MCARv32", :);
candidateLong = table();
for metricIndex = 1:numel(metricIds)
    block = table(candidate.SubjectLabel, candidate.SubjectID, ...
        repmat("MCARv351", height(candidate), 1), ...
        repmat("MCAR v3.5.1", height(candidate), 1), ...
        repmat(metricIds(metricIndex), height(candidate), 1), ...
        candidate.(candidateVariables(metricIndex)), ...
        VariableNames=sourceLong.Properties.VariableNames);
    candidateLong = [candidateLong; block]; %#ok<AGROW>
end
assert(height(candidateLong) == 44 * 4, ...
    'v3.5.1 long table must contain 44 x 4 rows.');

metricLong = [otherLong; candidateLong];
[presentMethod, methodOrder] = ismember(metricLong.Method, methodIds);
[presentMetric, metricOrder] = ismember(metricLong.Metric, metricIds);
assert(all(presentMethod) && all(presentMetric), 'Unexpected method or metric ID.');
metricLong.MethodOrder = methodOrder;
metricLong.MetricOrder = metricOrder;
metricLong = sortrows(metricLong, ["SubjectID" "MethodOrder" "MetricOrder"]);
metricLong(:, ["MethodOrder" "MetricOrder"]) = [];
assert(height(metricLong) == 44 * 8 * 4, ...
    'New eight-method table must contain 44 x 8 x 4 rows.');
assert_no_duplicate_keys(metricLong);

sourceOther = sortrows(otherLong, ["SubjectLabel" "Method" "Metric"]);
mergedOther = metricLong(metricLong.Method ~= "MCARv351", :);
mergedOther = sortrows(mergedOther, ["SubjectLabel" "Method" "Metric"]);
assert(isequal(sourceOther, mergedOther), ...
    'At least one of the seven baseline rows changed during the merge.');

aggregate = groupsummary(metricLong, ["Method" "MethodLabel" "Metric"], ...
    ["mean" "std" "median"], "Value_dB");
aggregate = renamevars(aggregate, ...
    ["GroupCount" "mean_Value_dB" "std_Value_dB" "median_Value_dB"], ...
    ["SubjectCount" "Mean_dB" "Std_dB" "Median_dB"]);
[~, aggregate.MethodOrder] = ismember(aggregate.Method, methodIds);
[~, aggregate.MetricOrder] = ismember(aggregate.Metric, metricIds);
aggregate = sortrows(aggregate, ["MetricOrder" "MethodOrder"]);
aggregate(:, ["MethodOrder" "MetricOrder"]) = [];
assert(height(aggregate) == 32 && all(aggregate.SubjectCount == 44), ...
    'Aggregate table must contain 32 complete method-metric rows.');

perSubject = make_per_subject_table(metricLong, sourceSubjects, ...
    methodIds, metricIds);
paperTable = make_paper_table(aggregate, methodIds, methodLabels, metricIds);
ranking = make_ranking_table(aggregate, metricIds, metricLabels);

candidateQuality = sortrows(candidateQuality, "SubjectID");
assert(height(candidateQuality) == 44 && ...
    isequal(candidateQuality(:, ["SubjectLabel" "SubjectID"]), sourceSubjects), ...
    'v3.5.1 quality-check subjects do not align.');
candidateQuality.LegacySevenBaselinesUnchanged = true(height(candidateQuality), 1);

writetable(metricLong, fullfile(outputRoot, 'metric_long.csv'));
writetable(perSubject, fullfile(outputRoot, 'per_subject_metrics.csv'));
writetable(candidateQuality, fullfile(outputRoot, 'quality_checks.csv'));
writetable(aggregate, fullfile(outputRoot, 'aggregate_metrics.csv'));
writetable(paperTable, fullfile(outputRoot, 'paper_comparison.csv'));
writetable(ranking, fullfile(outputRoot, 'metric_ranking.csv'));

plot_comparison(aggregate, metricIds, metricLabels, figuresRoot);
write_readme(fullfile(outputRoot, 'README.md'), paperTable, ranking);

configuration = struct( ...
    'schema_version', '1.0', ...
    'created_on', char(datetime('now', TimeZone='local')), ...
    'merge_type', 'frozen result-level replacement of MCAR v3.2 e39 by v3.5.1', ...
    'split', 'test', ...
    'test_subject_count_read_during_merge', 0, ...
    'subject_count', 44, ...
    'source_eight_method_metric_long_sha256', ...
        'BF1D27D161E7445D6AECC1F4656CB13407AAB0F4C3A69F29DEA2BA60471753F6', ...
    'v351_per_subject_sha256', ...
        '1DD5DF1C1E1056C8C2BD99AC0541DFA8211BA05A52C39153CDDDEFC9033BB81A', ...
    'legacy_seven_baselines_unchanged', true, ...
    'current_main_model', 'MCAR v3.5.1', ...
    'methods', {cellstr(methodLabels)}, ...
    'metrics', {cellstr(metricLabels)});
summary = struct( ...
    'schema_version', '1.0', ...
    'status', 'completed', ...
    'split', 'test', ...
    'test_subject_count_read_during_merge', 0, ...
    'subject_count', 44, ...
    'method_count', 8, ...
    'metric_count', 4, ...
    'metric_long_row_count', height(metricLong), ...
    'nonfinite_metric_count', sum(~isfinite(metricLong.Value_dB)), ...
    'legacy_seven_baselines_unchanged', true, ...
    'aggregate', table2struct(aggregate), ...
    'ranking', table2struct(ranking), ...
    'configuration', configuration);
write_json(fullfile(outputRoot, 'summary.json'), summary);

fprintf('Eight-method v3.5.1 main-model merge completed: %s\n', outputRoot);
disp(paperTable);
end

function assert_no_duplicate_keys(data)
keys = data(:, ["SubjectLabel" "Method" "Metric"]);
assert(height(unique(keys, 'rows')) == height(keys), ...
    'Duplicate subject-method-metric keys detected.');
end

function perSubject = make_per_subject_table(metricLong, subjects, methodIds, metricIds)
perSubject = subjects;
for methodId = methodIds
    for metricId = metricIds
        rows = metricLong(metricLong.Method == methodId & ...
            metricLong.Metric == metricId, ...
            ["SubjectLabel" "SubjectID" "Value_dB"]);
        rows = sortrows(rows, "SubjectID");
        assert(isequal(rows(:, ["SubjectLabel" "SubjectID"]), subjects), ...
            'Per-subject alignment failed for %s / %s.', methodId, metricId);
        perSubject.(methodId + "_" + metricId + "_dB") = rows.Value_dB;
    end
end
end

function paperTable = make_paper_table(aggregate, methodIds, methodLabels, metricIds)
paperTable = table(methodIds.', methodLabels.', ...
    VariableNames={'Method', 'MethodLabel'});
for metricId = metricIds
    means = zeros(numel(methodIds), 1);
    stds = zeros(numel(methodIds), 1);
    formatted = strings(numel(methodIds), 1);
    for methodIndex = 1:numel(methodIds)
        row = aggregate(aggregate.Method == methodIds(methodIndex) & ...
            aggregate.Metric == metricId, :);
        means(methodIndex) = row.Mean_dB;
        stds(methodIndex) = row.Std_dB;
        formatted(methodIndex) = sprintf('%.3f +/- %.3f', ...
            row.Mean_dB, row.Std_dB);
    end
    paperTable.(metricId + "_Mean_dB") = means;
    paperTable.(metricId + "_Std_dB") = stds;
    paperTable.(metricId + "_MeanPlusMinusStd_dB") = formatted;
end
end

function ranking = make_ranking_table(aggregate, metricIds, metricLabels)
ranking = table();
for metricIndex = 1:numel(metricIds)
    rows = aggregate(aggregate.Metric == metricIds(metricIndex), ...
        ["Method" "MethodLabel" "Mean_dB" "Std_dB"]);
    rows = sortrows(rows, "Mean_dB", "ascend");
    block = table(repmat(metricLabels(metricIndex), height(rows), 1), ...
        (1:height(rows)).', rows.Method, rows.MethodLabel, rows.Mean_dB, rows.Std_dB, ...
        VariableNames={'Metric', 'Rank', 'Method', 'MethodLabel', ...
        'Mean_dB', 'Std_dB'});
    ranking = [ranking; block]; %#ok<AGROW>
end
end

function plot_comparison(aggregate, metricIds, metricLabels, figuresRoot)
figureHandle = figure(Visible='off', Color='w', Position=[100 100 1900 1180]);
layout = tiledlayout(figureHandle, 2, 2, TileSpacing='compact', Padding='compact');
title(layout, sprintf(['SONICOM Q26 Test: Eight-Method Comparison (N=44)\n' ...
    'Mean +/- subject SD; lower is better; current main model highlighted in red']), ...
    FontSize=17, FontWeight='bold');

neutralColor = [0.72 0.75 0.79];
mainColor = [0.82 0.16 0.18];
fspColor = [0.24 0.47 0.72];
for metricIndex = 1:numel(metricIds)
    rows = aggregate(aggregate.Metric == metricIds(metricIndex), ...
        ["Method" "MethodLabel" "Mean_dB" "Std_dB"]);
    rows = sortrows(rows, "Mean_dB", "ascend");
    y = (1:height(rows)).';
    colors = repmat(neutralColor, height(rows), 1);
    colors(rows.Method == "MCARv351", :) = repmat(mainColor, ...
        nnz(rows.Method == "MCARv351"), 1);
    colors(rows.Method == "FSPAE", :) = repmat(fspColor, ...
        nnz(rows.Method == "FSPAE"), 1);

    ax = nexttile(layout);
    bars = barh(ax, y, rows.Mean_dB, 0.68);
    bars.FaceColor = 'flat';
    bars.CData = colors;
    bars.EdgeColor = 'none';
    hold(ax, 'on');
    errorbar(ax, rows.Mean_dB, y, rows.Std_dB, 'horizontal', ...
        LineStyle='none', Color=[0.24 0.26 0.29], LineWidth=0.8, CapSize=4);
    maximumWithError = max(rows.Mean_dB + rows.Std_dB);
    textOffset = maximumWithError * 0.012;
    text(ax, rows.Mean_dB + textOffset, y, compose('%.3f', rows.Mean_dB), ...
        FontSize=9, Color=[0.12 0.12 0.12], VerticalAlignment='middle');
    hold(ax, 'off');

    ax.YTick = y;
    ax.YTickLabel = rows.MethodLabel;
    ax.YDir = 'reverse';
    ax.FontName = 'Arial';
    ax.FontSize = 10;
    ax.TickDir = 'out';
    ax.Box = 'off';
    ax.XGrid = 'on';
    ax.YGrid = 'off';
    ax.GridAlpha = 0.18;
    ax.Layer = 'top';
    xlim(ax, [0 maximumWithError * 1.14]);
    xlabel(ax, 'Mean error (dB)');
    title(ax, metricLabels(metricIndex), FontWeight='bold');
end

pngPath = fullfile(figuresRoot, 'test44_eight_method_four_metrics.png');
pdfPath = fullfile(figuresRoot, 'test44_eight_method_four_metrics.pdf');
exportgraphics(figureHandle, pngPath, Resolution=300);
exportgraphics(figureHandle, pdfPath, ContentType='vector');
close(figureHandle);
end

function write_readme(path, paperTable, ranking)
file = fopen(path, 'w');
assert(file ~= -1, 'Could not open README: %s', path);
cleanup = onCleanup(@() fclose(file));
fprintf(file, '# SONICOM Q26 eight-method comparison with MCAR v3.5.1\n\n');
fprintf(file, '- Current engineering main model: MCAR v3.5.1\n');
fprintf(file, '- Frozen test subjects: 44\n');
fprintf(file, '- Result merge reads HRTFs/predictions/test subjects: 0\n');
fprintf(file, '- The seven non-MCAR baseline rows are byte-for-value unchanged.\n\n');
fprintf(file, '| Method | Full-sphere ERB | Contra-25 ERB | Contra HF | Horizontal ILD |\n');
fprintf(file, '|---|---:|---:|---:|---:|\n');
for index = 1:height(paperTable)
    fprintf(file, '| %s | %s | %s | %s | %s |\n', ...
        paperTable.MethodLabel(index), ...
        paperTable.FullSphereERB_MeanPlusMinusStd_dB(index), ...
        paperTable.Contralateral25ERB_MeanPlusMinusStd_dB(index), ...
        paperTable.ContralateralHighFrequency_MeanPlusMinusStd_dB(index), ...
        paperTable.HorizontalILDMAE_MeanPlusMinusStd_dB(index));
end
fprintf(file, '\nValues are mean +/- subject standard deviation in dB; lower is better.\n\n');
mainRanks = ranking(ranking.Method == "MCARv351", :);
fprintf(file, 'MCAR v3.5.1 ranks %s across the four metrics.\n', ...
    strjoin(string(mainRanks.Rank).', '/'));
clear cleanup;
end

function write_json(path, value)
file = fopen(path, 'w');
assert(file ~= -1, 'Could not open JSON output: %s', path);
cleanup = onCleanup(@() fclose(file));
fprintf(file, '%s\n', jsonencode(value, PrettyPrint=true));
clear cleanup;
end
