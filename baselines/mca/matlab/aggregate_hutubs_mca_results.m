function aggregate_hutubs_mca_results()
%AGGREGATE_HUTUBS_MCA_RESULTS Aggregate all 96-subject MCA batch results.
% Produces paper-comparable cross-subject tables and figures from the
% checkpoint files written by run_hutubs_mca_batch.

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(fileparts(scriptDir)));
batchRoot = fullfile(projectRoot, 'artifacts', ...
    'mca_reproduction', 'hutubs_mca_batch');
subjectsRoot = fullfile(batchRoot, 'subjects');
outputDir = fullfile(projectRoot, 'results', ...
    'baselines', 'mca', 'hutubs_batch');
subjectIds = 1:96;
sparseOrders = 1:10;
methodNames = {'SH', 'Conventional', 'MCA'};
earNames = {'Left', 'Right'};
regionNames = {'FullSphere', 'Frontal25deg', 'Contralateral25deg'};

assert(isfolder(subjectsRoot), 'Batch subject directory not found: %s', subjectsRoot);
if ~isfolder(outputDir)
    mkdir(outputDir);
end
legacyOutputNames = {
    'all_subject_order_erb_summary.csv'
    'aggregate_erb_by_order.csv'
    'aggregate_frequency_by_order.csv'};
for legacyIndex = 1:numel(legacyOutputNames)
    legacyFile = fullfile(outputDir, legacyOutputNames{legacyIndex});
    if isfile(legacyFile)
        delete(legacyFile);
    end
end

%% Load all per-order summaries and frequency curves.
erbTables = cell(numel(subjectIds) * numel(sparseOrders), 1);
binauralTables = cell(size(erbTables));
paperERBTables = cell(size(erbTables));
frequencyValues = [];
frequencyHz = [];
tableIndex = 0;

for subjectIndex = 1:numel(subjectIds)
    subjectId = subjectIds(subjectIndex);
    ordersDir = fullfile(subjectsRoot, sprintf('pp%d', subjectId), 'orders');
    for orderIndex = 1:numel(sparseOrders)
        sparseOrder = sparseOrders(orderIndex);
        tableIndex = tableIndex + 1;
        baseName = sprintf('n%02d', sparseOrder);
        erbFile = fullfile(ordersDir, [baseName '_erb_summary.csv']);
        binauralFile = fullfile(ordersDir, [baseName '_binaural_summary.csv']);
        resultFile = fullfile(ordersDir, [baseName '_results.mat']);
        assert(isfile(erbFile), 'Missing ERB summary: %s', erbFile);
        assert(isfile(binauralFile), 'Missing binaural summary: %s', binauralFile);
        assert(isfile(resultFile), 'Missing result MAT: %s', resultFile);

        erbTables{tableIndex} = readtable(erbFile, 'TextType', 'string');
        binauralTables{tableIndex} = readtable(binauralFile, 'TextType', 'string');
        loaded = load(resultFile, 'frequencyCurves', 'spatialErrors', 'denseGrid');
        paperERBTables{tableIndex} = make_paper_erb_rows( ...
            subjectId, sparseOrder, loaded.spatialErrors, loaded.denseGrid, ...
            methodNames, earNames, regionNames);

        if isempty(frequencyValues)
            frequencyHz = loaded.frequencyCurves.SH.erbFrequencyHz(:);
            frequencyValues = nan( ...
                numel(subjectIds), numel(sparseOrders), numel(methodNames), ...
                numel(earNames), numel(frequencyHz));
        end
        for methodIndex = 1:numel(methodNames)
            methodName = methodNames{methodIndex};
            frequencyValues(subjectIndex, orderIndex, methodIndex, 1, :) = ...
                reshape(loaded.frequencyCurves.(methodName).leftFullErbDb, ...
                1, 1, 1, 1, []);
            frequencyValues(subjectIndex, orderIndex, methodIndex, 2, :) = ...
                reshape(loaded.frequencyCurves.(methodName).rightFullErbDb, ...
                1, 1, 1, 1, []);
        end
    end
end

allERBQuadratureWeighted = vertcat(erbTables{:});
allERB = vertcat(paperERBTables{:});
allBinaural = vertcat(binauralTables{:});
allERB.Method = string(allERB.Method);
allERB.Ear = string(allERB.Ear);
allERB.Region = string(allERB.Region);
allERBQuadratureWeighted.Method = string(allERBQuadratureWeighted.Method);
allERBQuadratureWeighted.Ear = string(allERBQuadratureWeighted.Ear);
allERBQuadratureWeighted.Region = string(allERBQuadratureWeighted.Region);
allBinaural.Method = string(allBinaural.Method);

assert(height(allERB) == 96 * 10 * 3 * 2 * 3, ...
    'Unexpected combined ERB row count: %d', height(allERB));
assert(height(allBinaural) == 96 * 10 * 3, ...
    'Unexpected combined binaural row count: %d', height(allBinaural));
assert(all(isfinite(allERB.MeanERBError_dB)), 'ERB table contains nonfinite values.');
assert(all(isfinite(allBinaural.MeanAbsoluteILDError_dB)), ...
    'Binaural table contains nonfinite ILD values.');
assert(all(isfinite(allBinaural.MeanAbsoluteITDError_us)), ...
    'Binaural table contains nonfinite ITD values.');

writetable(allERB, fullfile(outputDir, ...
    'all_subject_order_erb_paper_unweighted.csv'));
writetable(allERBQuadratureWeighted, fullfile(outputDir, ...
    'all_subject_order_erb_quadrature_weighted.csv'));
writetable(allBinaural, fullfile(outputDir, 'all_subject_order_binaural_summary.csv'));

%% Cross-subject means and standard deviations.
erbAggregate = aggregate_erb(allERB, sparseOrders, methodNames, earNames, regionNames);
binauralAggregate = aggregate_binaural(allBinaural, sparseOrders, methodNames);
frequencyAggregate = aggregate_frequency( ...
    frequencyValues, frequencyHz, sparseOrders, methodNames, earNames);
improvementTable = calculate_mca_improvements( ...
    allERB, sparseOrders, earNames, regionNames);

writetable(erbAggregate, fullfile(outputDir, ...
    'aggregate_erb_by_order_paper_unweighted.csv'));
writetable(binauralAggregate, fullfile(outputDir, 'aggregate_binaural_by_order.csv'));
writetable(frequencyAggregate, fullfile(outputDir, ...
    'aggregate_frequency_by_order_quadrature_weighted.csv'));
writetable(improvementTable, fullfile(outputDir, 'mca_improvement_by_order.csv'));

%% Paper-comparable figures.
plot_region_error(erbAggregate, outputDir);
plot_full_sphere_error(erbAggregate, outputDir);
plot_binaural_error(binauralAggregate, outputDir);
plot_frequency_error(frequencyAggregate, outputDir);

save(fullfile(outputDir, 'aggregate_results.mat'), ...
    'allERB', 'allERBQuadratureWeighted', 'allBinaural', ...
    'erbAggregate', 'binauralAggregate', ...
    'frequencyAggregate', 'improvementTable', 'frequencyValues', ...
    'frequencyHz', 'subjectIds', 'sparseOrders', 'methodNames', ...
    'earNames', 'regionNames', '-v7.3');

%% Compact numerical report.
keyFindings = select_key_findings(improvementTable);
writetable(keyFindings, fullfile(outputDir, 'key_findings.csv'));
write_report(keyFindings, outputDir);

fprintf('\nAggregation complete. Output: %s\n', outputDir);
disp(keyFindings);
end

function rows = make_paper_erb_rows(subjectId, sparseOrder, spatialErrors, ...
        denseGrid, methods, ears, regions)
frontMask = great_circle_mask(denseGrid, [0, 90], 25);
contraMasks = {
    great_circle_mask(denseGrid, [270, 90], 25)
    great_circle_mask(denseGrid, [90, 90], 25)};
regionMasks = {true(size(frontMask)), frontMask, []};
rowCells = cell(numel(methods) * numel(ears) * numel(regions), 1);
rowIndex = 0;
for methodIndex = 1:numel(methods)
    method = methods{methodIndex};
    for earIndex = 1:numel(ears)
        if earIndex == 1
            values = spatialErrors.(method).leftMeanErbDb;
        else
            values = spatialErrors.(method).rightMeanErbDb;
        end
        regionMasks{3} = contraMasks{earIndex};
        for regionIndex = 1:numel(regions)
            mask = regionMasks{regionIndex};
            rowIndex = rowIndex + 1;
            rowCells{rowIndex} = table( ...
                subjectId, sparseOrder, string(method), ...
                string(ears{earIndex}), string(regions{regionIndex}), ...
                sum(mask), mean(values(mask)), ...
                'VariableNames', {'SubjectID', 'SparseOrder', 'Method', ...
                'Ear', 'Region', 'DirectionCount', 'MeanERBError_dB'});
        end
    end
end
rows = vertcat(rowCells{:});
end

function mask = great_circle_mask(grid, center, radiusDegrees)
dotProduct = sind(grid(:, 2)) .* sind(center(2)) .* ...
    cosd(grid(:, 1) - center(1)) + cosd(grid(:, 2)) .* cosd(center(2));
distanceDegrees = acosd(max(-1, min(1, dotProduct)));
mask = distanceDegrees <= radiusDegrees;
end

function output = aggregate_erb(allERB, orders, methods, ears, regions)
rows = cell(numel(orders) * numel(methods) * numel(ears) * numel(regions), 1);
rowIndex = 0;
for sparseOrder = orders
    for methodIndex = 1:numel(methods)
        for earIndex = 1:numel(ears)
            for regionIndex = 1:numel(regions)
                mask = allERB.SparseOrder == sparseOrder & ...
                    allERB.Method == methods{methodIndex} & ...
                    allERB.Ear == ears{earIndex} & ...
                    allERB.Region == regions{regionIndex};
                values = allERB.MeanERBError_dB(mask);
                assert(numel(values) == 96, ...
                    'Expected 96 ERB values for N=%d, %s, %s, %s.', ...
                    sparseOrder, methods{methodIndex}, ears{earIndex}, regions{regionIndex});
                rowIndex = rowIndex + 1;
                rows{rowIndex} = table( ...
                    sparseOrder, string(methods{methodIndex}), ...
                    string(ears{earIndex}), string(regions{regionIndex}), ...
                    numel(values), mean(values), std(values), median(values), ...
                    min(values), max(values), ...
                    'VariableNames', {'SparseOrder', 'Method', 'Ear', 'Region', ...
                    'SubjectCount', 'MeanERBError_dB', 'SD_ERBError_dB', ...
                    'MedianERBError_dB', 'MinERBError_dB', 'MaxERBError_dB'});
            end
        end
    end
end
output = vertcat(rows{:});
end

function output = aggregate_binaural(allBinaural, orders, methods)
rows = cell(numel(orders) * numel(methods), 1);
rowIndex = 0;
for sparseOrder = orders
    for methodIndex = 1:numel(methods)
        mask = allBinaural.SparseOrder == sparseOrder & ...
            allBinaural.Method == methods{methodIndex};
        ild = allBinaural.MeanAbsoluteILDError_dB(mask);
        itd = allBinaural.MeanAbsoluteITDError_us(mask);
        assert(numel(ild) == 96, ...
            'Expected 96 binaural values for N=%d, %s.', ...
            sparseOrder, methods{methodIndex});
        rowIndex = rowIndex + 1;
        rows{rowIndex} = table( ...
            sparseOrder, string(methods{methodIndex}), numel(ild), ...
            mean(ild), std(ild), median(ild), ...
            mean(itd), std(itd), median(itd), ...
            'VariableNames', {'SparseOrder', 'Method', 'SubjectCount', ...
            'MeanAbsoluteILDError_dB', 'SD_AbsoluteILDError_dB', ...
            'MedianAbsoluteILDError_dB', 'MeanAbsoluteITDError_us', ...
            'SD_AbsoluteITDError_us', 'MedianAbsoluteITDError_us'});
    end
end
output = vertcat(rows{:});
end

function output = aggregate_frequency(values, frequencyHz, orders, methods, ears)
rowCount = numel(orders) * numel(methods) * numel(ears) * numel(frequencyHz);
rows = cell(rowCount, 1);
rowIndex = 0;
for orderIndex = 1:numel(orders)
    for methodIndex = 1:numel(methods)
        for earIndex = 1:numel(ears)
            subjectCurves = squeeze(values(:, orderIndex, methodIndex, earIndex, :));
            meanCurve = mean(subjectCurves, 1);
            sdCurve = std(subjectCurves, 0, 1);
            for frequencyIndex = 1:numel(frequencyHz)
                rowIndex = rowIndex + 1;
                rows{rowIndex} = table( ...
                    orders(orderIndex), string(methods{methodIndex}), ...
                    string(ears{earIndex}), frequencyHz(frequencyIndex), ...
                    meanCurve(frequencyIndex), sdCurve(frequencyIndex), ...
                    'VariableNames', {'SparseOrder', 'Method', 'Ear', ...
                    'FrequencyHz', 'MeanERBError_dB', 'SD_ERBError_dB'});
            end
        end
    end
end
output = vertcat(rows{:});
end

function output = calculate_mca_improvements(allERB, orders, ears, regions)
rows = cell(numel(orders) * numel(ears) * numel(regions), 1);
rowIndex = 0;
for sparseOrder = orders
    for earIndex = 1:numel(ears)
        for regionIndex = 1:numel(regions)
            baseMask = allERB.SparseOrder == sparseOrder & ...
                allERB.Ear == ears{earIndex} & ...
                allERB.Region == regions{regionIndex};
            conventional = sortrows(allERB(baseMask & ...
                allERB.Method == "Conventional", :), 'SubjectID');
            mca = sortrows(allERB(baseMask & allERB.Method == "MCA", :), 'SubjectID');
            assert(isequal(conventional.SubjectID, mca.SubjectID) && height(mca) == 96, ...
                'Subject pairing failed for N=%d, %s, %s.', ...
                sparseOrder, ears{earIndex}, regions{regionIndex});
            difference = conventional.MeanERBError_dB - mca.MeanERBError_dB;
            relative = 100 * difference ./ conventional.MeanERBError_dB;
            rowIndex = rowIndex + 1;
            rows{rowIndex} = table( ...
                sparseOrder, string(ears{earIndex}), string(regions{regionIndex}), ...
                mean(conventional.MeanERBError_dB), mean(mca.MeanERBError_dB), ...
                mean(difference), std(difference), mean(relative), ...
                sum(difference > 0), ...
                'VariableNames', {'SparseOrder', 'Ear', 'Region', ...
                'ConventionalMeanError_dB', 'MCAMeanError_dB', ...
                'MeanImprovement_dB', 'SD_Improvement_dB', ...
                'MeanRelativeImprovement_percent', 'SubjectsImproved'});
        end
    end
end
output = vertcat(rows{:});
end

function plot_region_error(data, outputDir)
figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100 100 940 580]);
hold on;
series = {
    'Conventional', 'Frontal25deg', [0.00 0.45 0.74], '--', 'Frontal - conventional'
    'Conventional', 'Contralateral25deg', [0.85 0.33 0.10], '--', 'Contralateral - conventional'
    'MCA', 'Frontal25deg', [0.00 0.45 0.74], '-', 'Frontal - MCA'
    'MCA', 'Contralateral25deg', [0.85 0.33 0.10], '-', 'Contralateral - MCA'};
for index = 1:size(series, 1)
    rows = data(data.Method == series{index, 1} & ...
        data.Ear == "Left" & data.Region == series{index, 2}, :);
    rows = sortrows(rows, 'SparseOrder');
    errorbar(rows.SparseOrder, rows.MeanERBError_dB, rows.SD_ERBError_dB, ...
        'Color', series{index, 3}, 'LineStyle', series{index, 4}, ...
        'LineWidth', 1.8, 'Marker', 'o', 'MarkerSize', 5, ...
        'DisplayName', series{index, 5});
end
grid on;
xlim([1 10]);
xticks(1:10);
xlabel('Sparse SH order N');
ylabel('Mean auditory-band magnitude error (dB)');
title('HUTUBS simulated: left-ear regional error, mean \pm SD across 96 subjects');
legend('Location', 'northeast');
export_figure(figureHandle, outputDir, 'fig05_region_magnitude_error');
end

function plot_full_sphere_error(data, outputDir)
figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100 100 1120 500]);
colors = lines(3);
methods = ["SH", "Conventional", "MCA"];
ears = ["Left", "Right"];
labels = ["SH only", "SUpDEq + SH", "MCA"];
for earIndex = 1:2
    subplot(1, 2, earIndex);
    hold on;
    for methodIndex = 1:3
        rows = data(data.Method == methods(methodIndex) & ...
            data.Ear == ears(earIndex) & data.Region == "FullSphere", :);
        rows = sortrows(rows, 'SparseOrder');
        errorbar(rows.SparseOrder, rows.MeanERBError_dB, rows.SD_ERBError_dB, ...
            'Color', colors(methodIndex, :), 'LineWidth', 1.6, ...
            'Marker', 'o', 'DisplayName', labels(methodIndex));
    end
    grid on;
    xlim([1 10]);
    xticks(1:10);
    xlabel('Sparse SH order N');
    ylabel('Mean auditory-band magnitude error (dB)');
    title(sprintf('%s ear - full sphere', ears(earIndex)));
    legend('Location', 'northeast');
end
sgtitle('HUTUBS simulated: full-sphere error, mean \pm SD across 96 subjects');
export_figure(figureHandle, outputDir, 'full_sphere_magnitude_error');
end

function plot_binaural_error(data, outputDir)
figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100 100 1120 500]);
colors = lines(3);
methods = ["SH", "Conventional", "MCA"];
labels = ["SH only", "SUpDEq + SH", "MCA"];
for metricIndex = 1:2
    subplot(1, 2, metricIndex);
    hold on;
    for methodIndex = 1:numel(methods)
        rows = sortrows(data(data.Method == methods(methodIndex), :), 'SparseOrder');
        if metricIndex == 1
            means = rows.MeanAbsoluteILDError_dB;
            deviations = rows.SD_AbsoluteILDError_dB;
        else
            means = rows.MeanAbsoluteITDError_us;
            deviations = rows.SD_AbsoluteITDError_us;
        end
        errorbar(rows.SparseOrder, means, deviations, ...
            'Color', colors(methodIndex, :), 'LineWidth', 1.6, ...
            'Marker', 'o', 'DisplayName', labels(methodIndex));
    end
    grid on;
    xlim([1 10]);
    xticks(1:10);
    xlabel('Sparse SH order N');
    if metricIndex == 1
        ylabel('Mean absolute ILD error (dB)');
        title('ILD error');
    else
        ylabel('Mean absolute ITD error (\mus)');
        title('ITD error');
    end
    legend('Location', 'northeast');
end
sgtitle('HUTUBS simulated horizontal plane, mean \pm SD across 96 subjects');
export_figure(figureHandle, outputDir, 'binaural_error_by_order');
end

function plot_frequency_error(data, outputDir)
figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100 100 1120 500]);
orders = [3 6];
colors = lines(3);
methods = ["SH", "Conventional", "MCA"];
labels = ["SH only", "SUpDEq + SH", "MCA"];
for orderIndex = 1:numel(orders)
    subplot(1, 2, orderIndex);
    hold on;
    for methodIndex = 1:numel(methods)
        rows = data(data.SparseOrder == orders(orderIndex) & ...
            data.Method == methods(methodIndex) & data.Ear == "Left", :);
        rows = sortrows(rows, 'FrequencyHz');
        semilogx(rows.FrequencyHz, rows.MeanERBError_dB, ...
            'Color', colors(methodIndex, :), 'LineWidth', 1.7, ...
            'DisplayName', labels(methodIndex));
    end
    grid on;
    xlim([50 20000]);
    xlabel('ERB center frequency (Hz)');
    ylabel('Mean absolute magnitude error (dB)');
    title(sprintf('Left ear, N = %d', orders(orderIndex)));
    legend('Location', 'northwest');
end
sgtitle('HUTUBS simulated full-sphere frequency error across 96 subjects');
export_figure(figureHandle, outputDir, 'frequency_error_n3_n6');
end

function export_figure(figureHandle, outputDir, baseName)
exportgraphics(figureHandle, fullfile(outputDir, [baseName '.png']), ...
    'Resolution', 200);
savefig(figureHandle, fullfile(outputDir, [baseName '.fig']));
close(figureHandle);
end

function findings = select_key_findings(improvements)
selected = improvements( ...
    improvements.Ear == "Left" & ...
    (improvements.Region == "FullSphere" | ...
    improvements.Region == "Frontal25deg" | ...
    improvements.Region == "Contralateral25deg"), :);
findings = sortrows(selected, {'Region', 'SparseOrder'});
end

function write_report(findings, outputDir)
reportFile = fullfile(outputDir, 'aggregation_report.txt');
fileId = fopen(reportFile, 'w');
assert(fileId >= 0, 'Could not create report: %s', reportFile);
closeFile = onCleanup(@() fclose(fileId));
fprintf(fileId, 'HUTUBS MCA reproduction aggregation\n');
fprintf(fileId, 'Subjects: 96\nSparse orders: 1-10\n\n');
for region = ["FullSphere", "Frontal25deg", "Contralateral25deg"]
    rows = findings(findings.Region == region, :);
    [maximumImprovement, index] = max(rows.MeanImprovement_dB);
    fprintf(fileId, '%s: maximum MCA improvement %.6f dB at N=%d; ', ...
        region, maximumImprovement, rows.SparseOrder(index));
    fprintf(fileId, '%d/96 subjects improved at that order.\n', ...
        rows.SubjectsImproved(index));
end
clear closeFile;
end
