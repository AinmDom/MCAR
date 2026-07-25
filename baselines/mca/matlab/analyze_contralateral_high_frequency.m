function analyze_contralateral_high_frequency()
%ANALYZE_CONTRALATERAL_HIGH_FREQUENCY Quantify contralateral HRTF errors.
%
% Reproduces the KU100 MCA demo interpolation (Lebedev Ns = 3, Nd = 44),
% then compares SH-only, conventional SUpDEq + SH, and MCA against the
% dense reference for frequencies strictly above 10 kHz. The left-ear
% contralateral region is the right hemisphere; the right-ear
% contralateral region is the left hemisphere. Directions on the sagittal
% median plane are excluded.
%
% Outputs are written below:
%   artifacts/figures/mca_demo_ku100_ns3_nd44/contralateral_high_frequency/
%     contralateral_high_frequency_summary.csv
%     contralateral_high_frequency_improvement.csv
%     contralateral_high_frequency_statistics.mat
%     contralateral_high_frequency_error.png
%     contralateral_high_frequency_error.fig

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(fileparts(scriptDir)));
supdeqDir = fullfile(projectRoot, 'external', 'SUpDEq');
outputDir = fullfile(projectRoot, 'artifacts', 'figures', ...
    'mca_demo_ku100_ns3_nd44', 'contralateral_high_frequency');

if ~isfolder(supdeqDir)
    error('SUpDEq directory was not found: %s', supdeqDir);
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir);
supdeq_start;

%% Reproduce the MCA demo baseline datasets.
Ns = 3;
Nd = 44;
headRadius = 0.0875;
highFrequencyCutoffHz = 10000;
magnitudeFloorDb = -200;
magnitudeFloor = 10^(magnitudeFloorDb / 20);

sgS = supdeq_lebedev([], Ns);
sgD = supdeq_lebedev([], Nd);
sparseHRTF = supdeq_getSparseDataset(sgS, Ns, 44, 'ku100');
fs = sparseHRTF.fs;

interpHRTF_sh = supdeq_interpHRTF( ...
    sparseHRTF, sgD, 'None', 'SH', nan, headRadius);
interpHRTF_con = supdeq_interpHRTF( ...
    sparseHRTF, sgD, 'SUpDEq', 'SH', nan, headRadius);
interpHRTF_mca = supdeq_interpHRTF( ...
    sparseHRTF, sgD, 'SUpDEq', 'SH', inf, headRadius);
refHRTF = supdeq_getSparseDataset(sgD, Nd, 44, 'ku100');

frequencyHz = interpHRTF_mca.f(:).';
highFrequencyMask = frequencyHz > highFrequencyCutoffHz & ...
    frequencyHz <= fs / 2;
if ~any(highFrequencyMask)
    error('No frequency bins exist above %.0f Hz.', highFrequencyCutoffHz);
end

%% Define contralateral hemispheres from the Cartesian lateral coordinate.
% SUpDEq uses azimuth 0=front, 90=left, 180=back, 270=right. The second
% grid column is colatitude (0=north pole, 90=horizontal, 180=south pole).
azimuthDeg = mod(sgD(:, 1), 360);
colatitudeDeg = sgD(:, 2);
lateralCoordinate = sind(colatitudeDeg) .* sind(azimuthDeg);
medianPlaneTolerance = 1e-12;
leftContralateralMask = lateralCoordinate < -medianPlaneTolerance;
rightContralateralMask = lateralCoordinate > medianPlaneTolerance;

if size(sgD, 2) >= 3
    directionWeights = sgD(:, 3);
else
    directionWeights = ones(size(sgD, 1), 1);
end
if any(directionWeights < 0) || sum(directionWeights) <= 0
    error('Sampling-grid weights must be nonnegative with a positive sum.');
end

%% Calculate absolute log-magnitude error in dB.
methodNames = {'SH', 'Conventional', 'MCA'};
methodData = {
    interpHRTF_sh
    interpHRTF_con
    interpHRTF_mca};

analysis = struct();
analysis.configuration = struct( ...
    'dataset', 'KU100 HRIR_L2702.sofa', ...
    'sourceLebedevOrder', Ns, ...
    'targetLebedevOrder', Nd, ...
    'headRadiusM', headRadius, ...
    'samplingRateHz', fs, ...
    'highFrequencyCutoffHz', highFrequencyCutoffHz, ...
    'frequencyRule', 'f > 10000 Hz and f <= Nyquist', ...
    'magnitudeFloorDb', magnitudeFloorDb, ...
    'meanRule', 'Lebedev-weighted over directions; uniform over frequency bins', ...
    'medianRule', 'Unweighted median over direction-frequency samples', ...
    'leftContralateralRule', 'negative Cartesian lateral coordinate (right hemisphere)', ...
    'rightContralateralRule', 'positive Cartesian lateral coordinate (left hemisphere)');
analysis.frequencyHz = frequencyHz(highFrequencyMask);
analysis.samplingGrid = sgD;
analysis.leftContralateralMask = leftContralateralMask;
analysis.rightContralateralMask = rightContralateralMask;

regionNames = {'LeftEar', 'RightEar', 'Binaural'};
summaryRegion = strings(9, 1);
summaryMethod = strings(9, 1);
summaryDirectionCount = zeros(9, 1);
summaryFrequencyBinCount = zeros(9, 1);
summaryWeightedMeanDb = zeros(9, 1);
summaryMedianDb = zeros(9, 1);

summaryRow = 0;
for methodIndex = 1:numel(methodNames)
    methodName = methodNames{methodIndex};
    current = methodData{methodIndex};

    leftErrorDb = absolute_log_magnitude_error( ...
        current.HRTF_L(leftContralateralMask, highFrequencyMask), ...
        refHRTF.HRTF_L(leftContralateralMask, highFrequencyMask), ...
        magnitudeFloor);
    rightErrorDb = absolute_log_magnitude_error( ...
        current.HRTF_R(rightContralateralMask, highFrequencyMask), ...
        refHRTF.HRTF_R(rightContralateralMask, highFrequencyMask), ...
        magnitudeFloor);

    leftWeights = directionWeights(leftContralateralMask);
    rightWeights = directionWeights(rightContralateralMask);
    binauralErrorDb = [leftErrorDb; rightErrorDb];
    binauralWeights = [leftWeights; rightWeights];

    errorsByRegion = {leftErrorDb, rightErrorDb, binauralErrorDb};
    weightsByRegion = {leftWeights, rightWeights, binauralWeights};

    for regionIndex = 1:numel(regionNames)
        regionName = regionNames{regionIndex};
        errorDb = errorsByRegion{regionIndex};
        weights = weightsByRegion{regionIndex};
        stats = summarize_error(errorDb, weights);

        analysis.errors.(regionName).(methodName) = errorDb;
        analysis.curves.(regionName).(methodName) = stats.frequencyMeanDb;
        analysis.statistics.(regionName).(methodName) = stats;

        summaryRow = summaryRow + 1;
        summaryRegion(summaryRow) = regionName;
        summaryMethod(summaryRow) = methodName;
        summaryDirectionCount(summaryRow) = size(errorDb, 1);
        summaryFrequencyBinCount(summaryRow) = size(errorDb, 2);
        summaryWeightedMeanDb(summaryRow) = stats.weightedMeanDb;
        summaryMedianDb(summaryRow) = stats.medianDb;
    end
end

summaryTable = table( ...
    summaryRegion, summaryMethod, summaryDirectionCount, ...
    summaryFrequencyBinCount, summaryWeightedMeanDb, summaryMedianDb, ...
    'VariableNames', {'Region', 'Method', 'DirectionCount', ...
    'FrequencyBinCount', 'WeightedMeanAbsoluteError_dB', ...
    'MedianAbsoluteError_dB'});

%% Compare conventional interpolation directly with MCA.
improvementRegion = string(regionNames(:));
baselineMeanDb = zeros(3, 1);
proposedMeanDb = zeros(3, 1);
meanImprovementDb = zeros(3, 1);
relativeMeanImprovementPercent = zeros(3, 1);
baselineMedianDb = zeros(3, 1);
proposedMedianDb = zeros(3, 1);
medianImprovementDb = zeros(3, 1);
relativeMedianImprovementPercent = zeros(3, 1);

for regionIndex = 1:numel(regionNames)
    regionName = regionNames{regionIndex};
    conventionalStats = analysis.statistics.(regionName).Conventional;
    mcaStats = analysis.statistics.(regionName).MCA;

    baselineMeanDb(regionIndex) = conventionalStats.weightedMeanDb;
    proposedMeanDb(regionIndex) = mcaStats.weightedMeanDb;
    meanImprovementDb(regionIndex) = ...
        baselineMeanDb(regionIndex) - proposedMeanDb(regionIndex);
    relativeMeanImprovementPercent(regionIndex) = 100 * ...
        meanImprovementDb(regionIndex) / baselineMeanDb(regionIndex);

    baselineMedianDb(regionIndex) = conventionalStats.medianDb;
    proposedMedianDb(regionIndex) = mcaStats.medianDb;
    medianImprovementDb(regionIndex) = ...
        baselineMedianDb(regionIndex) - proposedMedianDb(regionIndex);
    relativeMedianImprovementPercent(regionIndex) = 100 * ...
        medianImprovementDb(regionIndex) / baselineMedianDb(regionIndex);
end

improvementTable = table( ...
    improvementRegion, baselineMeanDb, proposedMeanDb, meanImprovementDb, ...
    relativeMeanImprovementPercent, baselineMedianDb, proposedMedianDb, ...
    medianImprovementDb, relativeMedianImprovementPercent, ...
    'VariableNames', {'Region', 'ConventionalMeanError_dB', ...
    'MCAMeanError_dB', 'MeanImprovement_dB', ...
    'RelativeMeanImprovement_percent', 'ConventionalMedianError_dB', ...
    'MCAMedianError_dB', 'MedianImprovement_dB', ...
    'RelativeMedianImprovement_percent'});

analysis.summaryTable = summaryTable;
analysis.improvementTable = improvementTable;

%% Export tables, raw statistics, and a paper-ready comparison figure.
writetable(summaryTable, fullfile(outputDir, ...
    'contralateral_high_frequency_summary.csv'));
writetable(improvementTable, fullfile(outputDir, ...
    'contralateral_high_frequency_improvement.csv'));
save(fullfile(outputDir, 'contralateral_high_frequency_statistics.mat'), ...
    'analysis', '-v7.3');

figureHandle = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100 100 900 560]);
semilogx(analysis.frequencyHz, analysis.curves.Binaural.SH, ...
    'LineWidth', 1.6);
hold on;
semilogx(analysis.frequencyHz, analysis.curves.Binaural.Conventional, ...
    'LineWidth', 1.6);
semilogx(analysis.frequencyHz, analysis.curves.Binaural.MCA, ...
    'LineWidth', 1.8);
grid on;
xlim([highFrequencyCutoffHz fs / 2]);
xlabel('Frequency (Hz)');
ylabel('Weighted mean absolute log-magnitude error (dB)');
title('KU100 contralateral high-frequency HRTF error');
legend('SH only', 'SUpDEq + SH', 'MCA', 'Location', 'best');

exportgraphics(figureHandle, fullfile(outputDir, ...
    'contralateral_high_frequency_error.png'), 'Resolution', 200);
savefig(figureHandle, fullfile(outputDir, ...
    'contralateral_high_frequency_error.fig'));
close(figureHandle);

fprintf('\nContralateral high-frequency summary (> %.0f Hz):\n', ...
    highFrequencyCutoffHz);
disp(summaryTable);
fprintf('\nConventional to MCA improvement:\n');
disp(improvementTable);
fprintf('Outputs written to: %s\n', outputDir);

clear restoreDirectory;
end

function errorDb = absolute_log_magnitude_error(estimate, reference, floorValue)
estimateDb = 20 * log10(max(abs(estimate), floorValue));
referenceDb = 20 * log10(max(abs(reference), floorValue));
errorDb = abs(estimateDb - referenceDb);
end

function stats = summarize_error(errorDb, directionWeights)
directionWeights = directionWeights(:);
normalizedWeights = directionWeights / sum(directionWeights);
stats.weightedMeanDb = sum(normalizedWeights .* mean(errorDb, 2));
stats.medianDb = median(errorDb(:));
stats.frequencyMeanDb = sum(errorDb .* normalizedWeights, 1);
stats.directionMeanDb = mean(errorDb, 2);
end
