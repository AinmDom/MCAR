function report = summarize_v351_frozen_test()
%SUMMARIZE_V351_FROZEN_TEST Validate and compare v3.5.1 with v3.2 e39.

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
resultRoot = fullfile(projectRoot, 'results', ...
    'sonicom_mlp_cnn_q26_v351_vs_v32e39_frozen_test_strict');
inputPath = fullfile(resultRoot, 'per_subject_metrics.csv');
outputCsv = fullfile(resultRoot, 'paired_comparison.csv');
outputJson = fullfile(resultRoot, 'paired_comparison.json');

T = readtable(inputPath, TextType='string');
assert(height(T) == 44, 'Expected 44 test subjects, found %d.', height(T));
assert(numel(unique(T.SubjectID)) == 44, 'SubjectID values must be unique.');
assert(~anymissing(T), 'Per-subject result table contains missing values.');
numericValues = table2array(T(:, vartype('numeric')));
assert(all(isfinite(numericValues), 'all'), ...
    'Per-subject result table contains nonfinite numeric values.');

metricNames = [
    "full_sphere_erb"
    "contralateral_25deg_erb"
    "contralateral_high_frequency"
    "horizontal_ild_mae"
];
mainVariables = [
    "MLPCNNv3FullSphereERB_dB"
    "MLPCNNv3Contralateral25ERB_dB"
    "MLPCNNv3ContralateralHighFrequency_dB"
    "MLPCNNv3HorizontalILDMAE_dB"
];
candidateVariables = [
    "MLPCNNv31FullSphereERB_dB"
    "MLPCNNv31Contralateral25ERB_dB"
    "MLPCNNv31ContralateralHighFrequency_dB"
    "MLPCNNv31HorizontalILDMAE_dB"
];

metricCount = numel(metricNames);
v32Mean = zeros(metricCount, 1);
v351Mean = zeros(metricCount, 1);
meanDifference = zeros(metricCount, 1);
relativeImprovement = zeros(metricCount, 1);
improvedSubjectCount = zeros(metricCount, 1);
ciLow = zeros(metricCount, 1);
ciHigh = zeros(metricCount, 1);
pairedTP = zeros(metricCount, 1);
cohenDz = zeros(metricCount, 1);

for metricIndex = 1:metricCount
    mainValues = T.(mainVariables(metricIndex));
    candidateValues = T.(candidateVariables(metricIndex));
    difference = candidateValues - mainValues;
    v32Mean(metricIndex) = mean(mainValues);
    v351Mean(metricIndex) = mean(candidateValues);
    meanDifference(metricIndex) = mean(difference);
    relativeImprovement(metricIndex) = ...
        100 * (v32Mean(metricIndex) - v351Mean(metricIndex)) ...
        / v32Mean(metricIndex);
    improvedSubjectCount(metricIndex) = nnz(candidateValues < mainValues);
    differenceStd = std(difference, 0);
    standardError = differenceStd / sqrt(height(T));
    degreesOfFreedom = height(T) - 1;
    tStatistic = mean(difference) / standardError;
    pairedTP(metricIndex) = betainc( ...
        degreesOfFreedom / (degreesOfFreedom + tStatistic^2), ...
        degreesOfFreedom / 2, 0.5);
    betaQuantile = betaincinv(0.05, degreesOfFreedom / 2, 0.5);
    criticalT = sqrt(degreesOfFreedom * (1 / betaQuantile - 1));
    ciLow(metricIndex) = mean(difference) - criticalT * standardError;
    ciHigh(metricIndex) = mean(difference) + criticalT * standardError;
    cohenDz(metricIndex) = mean(difference) / differenceStd;
end

comparison = table(metricNames, v32Mean, v351Mean, meanDifference, ...
    relativeImprovement, improvedSubjectCount, ciLow, ciHigh, pairedTP, ...
    cohenDz, VariableNames=[
        "Metric"
        "V32Epoch39Mean_dB"
        "V351Mean_dB"
        "MeanDifference_dB"
        "RelativeImprovement_percent"
        "ImprovedSubjectCountOf44"
        "CI95Low_dB"
        "CI95High_dB"
        "PairedT_P"
        "CohenDz"
    ]);
writetable(comparison, outputCsv);

report = struct( ...
    'schema_version', '1.0', ...
    'status', 'completed', ...
    'comparison', 'v351_frozen_vs_v32_epoch39_frozen', ...
    'split', 'test', ...
    'test_subject_count_read', 44, ...
    'table_height', height(T), ...
    'table_width', width(T), ...
    'subject_ids_unique', true, ...
    'missing_value_count', sum(ismissing(T), 'all'), ...
    'all_numeric_values_finite', true, ...
    'metrics', table2struct(comparison), ...
    'model_or_weight_updates_after_test', false);
fid = fopen(outputJson, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(report, PrettyPrint=true));

disp(jsonencode(report, PrettyPrint=true));
end
