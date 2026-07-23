function run_hutubs_mca_p91_n3()
%RUN_HUTUBS_MCA_P91_N3 First strict MCA-paper reproduction checkpoint.
% Reproduces the paper configuration for HUTUBS simulated subject 91 with
% a Lebedev N=3 sparse input (26 directions), and saves local outputs under
% reproduce/pp91_n3/. This workspace is intentionally ignored by Git.

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
supdeqDir = fullfile(projectRoot, 'SUpDEq-master');
hutubsDir = fullfile(projectRoot, 'data', 'HRTF', 'hutubs');
outputDir = fullfile(scriptDir, 'pp91_n3');

subjectId = 91;
referenceOrder = 35;
sparseOrder = 3;
denseFliegeOrder = 29;
fftOversize = 4;

if ~isfolder(supdeqDir)
    error('SUpDEq directory not found: %s', supdeqDir);
end
if ~isfolder(hutubsDir)
    error('HUTUBS directory not found: %s', hutubsDir);
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir);
supdeq_start;

%% Load and validate the simulated HUTUBS reference data.
sofaFile = fullfile(hutubsDir, sprintf('pp%d_HRIRs_simulated.sofa', subjectId));
if ~isfile(sofaFile)
    error('Simulated SOFA file not found: %s', sofaFile);
end
sofa = SOFAload(sofaFile);
if ~strcmp(sofa.GLOBAL_SOFAConventions, 'SimpleFreeFieldHRIR')
    error('Unexpected SOFA convention: %s', sofa.GLOBAL_SOFAConventions);
end

sourceGridCanonical = supdeq_lebedev([], referenceOrder);
sofaGrid = [mod(sofa.SourcePosition(:, 1), 360), 90 - sofa.SourcePosition(:, 2)];
azimuthError = abs(mod(sofaGrid(:, 1) - sourceGridCanonical(:, 1) + 180, 360) - 180);
colatitudeError = abs(sofaGrid(:, 2) - sourceGridCanonical(:, 2));
if size(sofaGrid, 1) ~= size(sourceGridCanonical, 1) || ...
        max(azimuthError) > 1e-8 || max(colatitudeError) > 1e-8
    error(['HUTUBS source directions do not match the expected Lebedev N=%d ', ...
        'grid. Max azimuth/colatitude errors: %.3g / %.3g degree.'], ...
        referenceOrder, max(azimuthError), max(colatitudeError));
end
sourceGrid = [sofaGrid, sourceGridCanonical(:, 3)];

%% Derive the individual rigid-sphere radius exactly as in the paper.
anthropometry = readtable(fullfile(hutubsDir, 'AntrhopometricMeasures.csv'));
subjectRow = anthropometry(anthropometry.SubjectID == subjectId, :);
if height(subjectRow) ~= 1
    error('Expected exactly one anthropometry row for subject %d.', subjectId);
end
headRadius = supdeq_optRadius( ...
    subjectRow.x1 / 100, subjectRow.x2 / 100, subjectRow.x3 / 100, 'Algazi');

%% Build a full-spherical N=35 reference and resample it to both grids.
referenceSfd = supdeq_sofa2sfd(sofa, referenceOrder, sourceGrid, ...
    fftOversize, 'ak');
sparseGrid = supdeq_lebedev([], sparseOrder);
denseGrid = supdeq_fliege(denseFliegeOrder);

[sparseLeft, sparseRight] = supdeq_getArbHRTF( ...
    referenceSfd, sparseGrid, 'DEG', 2, 'ak');
[referenceLeft, referenceRight] = supdeq_getArbHRTF( ...
    referenceSfd, denseGrid, 'DEG', 2, 'ak');

sparseHRTF = struct( ...
    'HRTF_L', sparseLeft, ...
    'HRTF_R', sparseRight, ...
    'f', referenceSfd.f, ...
    'fs', referenceSfd.fs, ...
    'Nmax', sparseOrder, ...
    'FFToversize', referenceSfd.FFToversize, ...
    'samplingGrid', sparseGrid);
referenceHRTF = struct( ...
    'HRTF_L', referenceLeft, ...
    'HRTF_R', referenceRight, ...
    'f', referenceSfd.f, ...
    'fs', referenceSfd.fs, ...
    'Nmax', denseFliegeOrder, ...
    'FFToversize', referenceSfd.FFToversize, ...
    'samplingGrid', denseGrid);
[referenceHRTF.HRIR_L, referenceHRTF.HRIR_R] = hrtf_to_hrir( ...
    referenceHRTF.HRTF_L, referenceHRTF.HRTF_R, referenceHRTF.FFToversize);

%% Paper baselines: SH, conventional time-aligned SH, and MCA.
interpSH = supdeq_interpHRTF( ...
    sparseHRTF, denseGrid, 'None', 'SH', nan, headRadius);
interpConventional = supdeq_interpHRTF( ...
    sparseHRTF, denseGrid, 'SUpDEq', 'SH', nan, headRadius);
interpMCA = supdeq_interpHRTF( ...
    sparseHRTF, denseGrid, 'SUpDEq', 'SH', inf, headRadius);

%% ERB-band magnitude error and LSD, with full/front/contralateral regions.
methods = {'SH', 'Conventional', 'MCA'};
results = {interpSH, interpConventional, interpMCA};
directionWeights = denseGrid(:, 3) / sum(denseGrid(:, 3));
frontMask = great_circle_mask(denseGrid, [0, 90], 25);
leftContraMask = great_circle_mask(denseGrid, [270, 90], 25);
rightContraMask = great_circle_mask(denseGrid, [90, 90], 25);

summary = table();
frequencyCurves = struct();
for methodIndex = 1:numel(methods)
    methodName = methods{methodIndex};
    current = results{methodIndex};
    [erbLeft, erbFrequencies] = AKerbError( ...
        current.HRIR_L, referenceHRTF.HRIR_L, [50 referenceHRTF.fs / 2], referenceHRTF.fs);
    erbRight = AKerbError( ...
        current.HRIR_R, referenceHRTF.HRIR_R, [50 referenceHRTF.fs / 2], referenceHRTF.fs);
    lsdLeft = supdeq_calcLSD_HRIR( ...
        current.HRIR_L, referenceHRTF.HRIR_L, referenceHRTF.fs, 16);
    lsdRight = supdeq_calcLSD_HRIR( ...
        current.HRIR_R, referenceHRTF.HRIR_R, referenceHRTF.fs, 16);

    frequencyCurves.(methodName).erbFrequencyHz = erbFrequencies;
    frequencyCurves.(methodName).leftFullErbDb = weighted_direction_mean(abs(erbLeft), directionWeights);
    frequencyCurves.(methodName).rightFullErbDb = weighted_direction_mean(abs(erbRight), directionWeights);
    frequencyCurves.(methodName).leftLsdFrequencyHz = lsdLeft.f;
    frequencyCurves.(methodName).leftLsdDb = lsdLeft.lsd_freq;
    frequencyCurves.(methodName).rightLsdFrequencyHz = lsdRight.f;
    frequencyCurves.(methodName).rightLsdDb = lsdRight.lsd_freq;

    summary = [summary; make_summary_rows(subjectId, sparseOrder, methodName, 'Left', ...
        abs(erbLeft), directionWeights, frontMask, leftContraMask); ...
        make_summary_rows(subjectId, sparseOrder, methodName, 'Right', ...
        abs(erbRight), directionWeights, frontMask, rightContraMask)]; %#ok<AGROW>
end

configuration = struct( ...
    'subjectId', subjectId, ...
    'sourceFile', sofaFile, ...
    'referenceLebedevOrder', referenceOrder, ...
    'sparseLebedevOrder', sparseOrder, ...
    'sparseDirectionCount', size(sparseGrid, 1), ...
    'denseFliegeOrder', denseFliegeOrder, ...
    'denseDirectionCount', size(denseGrid, 1), ...
    'samplingRateHz', referenceHRTF.fs, ...
    'fftOversize', fftOversize, ...
    'headRadiusM', headRadius, ...
    'headRadiusCm', 100 * headRadius, ...
    'mcaMaximumBoostDb', inf, ...
    'mcaMinimumPhase', true, ...
    'mcaLimitBelowSpatialAliasingFrequency', true, ...
    'mcaAliasingFade', 'fadeDown', ...
    'regionRadiusDegrees', 25, ...
    'sourceGridMaxAzimuthMismatchDeg', max(azimuthError), ...
    'sourceGridMaxColatitudeMismatchDeg', max(colatitudeError));

writetable(summary, fullfile(outputDir, 'pp91_n3_erb_summary.csv'));
save(fullfile(outputDir, 'pp91_n3_results.mat'), ...
    'configuration', 'summary', 'frequencyCurves', 'sparseGrid', 'denseGrid', ...
    'frontMask', 'leftContraMask', 'rightContraMask', '-v7.3');
plot_frequency_curves(frequencyCurves, outputDir);

fprintf('\nHUTUBS MCA p91 / N=3 checkpoint complete.\n');
fprintf('Individual Algazi head radius: %.4f cm\n', configuration.headRadiusCm);
fprintf('Outputs: %s\n', outputDir);
disp(summary);
end

function [leftHrir, rightHrir] = hrtf_to_hrir(leftHrtf, rightHrtf, fftOversize)
leftHrir = real(ifft(AKsingle2bothSidedSpectrum(leftHrtf.')));
rightHrir = real(ifft(AKsingle2bothSidedSpectrum(rightHrtf.')));
hrirLength = size(leftHrir, 1) / fftOversize;
leftHrir = leftHrir(1:hrirLength, :);
rightHrir = rightHrir(1:hrirLength, :);
end

function mask = great_circle_mask(grid, center, radiusDegrees)
azimuth = grid(:, 1);
colatitude = grid(:, 2);
centerAzimuth = center(1);
centerColatitude = center(2);
dotProduct = sind(colatitude) .* sind(centerColatitude) .* ...
    cosd(azimuth - centerAzimuth) + cosd(colatitude) .* cosd(centerColatitude);
distanceDegrees = acosd(min(1, max(-1, dotProduct)));
mask = distanceDegrees <= radiusDegrees + 1e-10;
end

function meanByFrequency = weighted_direction_mean(errorDb, weights)
meanByFrequency = errorDb * weights(:);
end

function rows = make_summary_rows(subjectId, sparseOrder, method, ear, errorDb, weights, frontMask, contraMask)
regions = {'FullSphere', 'Frontal25deg', 'Contralateral25deg'};
masks = {true(size(weights)), frontMask, contraMask};
rows = table();
for regionIndex = 1:numel(regions)
    mask = masks{regionIndex};
    regionWeights = weights(mask);
    regionWeights = regionWeights / sum(regionWeights);
    meanByFrequency = weighted_direction_mean(errorDb(:, mask), regionWeights);
    rows = [rows; table(subjectId, sparseOrder, string(method), string(ear), ...
        string(regions{regionIndex}), sum(mask), mean(meanByFrequency), ...
        'VariableNames', {'SubjectID', 'SparseOrder', 'Method', 'Ear', ...
        'Region', 'DirectionCount', 'MeanERBError_dB'})]; %#ok<AGROW>
end
end

function plot_frequency_curves(curves, outputDir)
methodNames = {'SH', 'Conventional', 'MCA'};
figureHandle = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 940 520]);
for methodIndex = 1:numel(methodNames)
    methodName = methodNames{methodIndex};
    semilogx(curves.(methodName).erbFrequencyHz, curves.(methodName).leftFullErbDb, ...
        'LineWidth', 1.6);
    hold on;
end
grid on;
xlim([50 20000]);
xlabel('ERB center frequency (Hz)');
ylabel('Left-ear weighted absolute magnitude error (dB)');
title('HUTUBS simulated pp91: MCA reproduction, Lebedev N = 3');
legend('SH only', 'SUpDEq + SH', 'MCA', 'Location', 'best');
exportgraphics(figureHandle, fullfile(outputDir, 'pp91_n3_left_erb_error.png'), 'Resolution', 200);
savefig(figureHandle, fullfile(outputDir, 'pp91_n3_left_erb_error.fig'));
close(figureHandle);
end
