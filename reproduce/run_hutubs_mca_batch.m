function run_hutubs_mca_batch(subjectIds, sparseOrders, numWorkers, outputName)
%RUN_HUTUBS_MCA_BATCH Reproduce MCA on HUTUBS simulated HRTFs in parallel.
%
% Defaults:
%   subjectIds   = 1:96
%   sparseOrders = 1:10
%   numWorkers   = 6
%   outputName   = 'hutubs_mca_batch'
%
% Parallelization is across subjects. Each subject's N=35 reference is
% constructed once, then all requested sparse orders are processed in
% sequence. Every order is checkpointed independently for safe resumption.

if nargin < 1 || isempty(subjectIds)
    subjectIds = 1:96;
end
if nargin < 2 || isempty(sparseOrders)
    sparseOrders = 1:10;
end
if nargin < 3 || isempty(numWorkers)
    numWorkers = 6;
end
if nargin < 4 || isempty(outputName)
    outputName = 'hutubs_mca_batch';
end

subjectIds = unique(subjectIds(:).', 'stable');
sparseOrders = unique(sparseOrders(:).', 'stable');
validateattributes(subjectIds, {'numeric'}, ...
    {'integer', '>=', 1, '<=', 96, 'nonempty'});
validateattributes(sparseOrders, {'numeric'}, ...
    {'integer', '>=', 1, '<=', 10, 'nonempty'});
validateattributes(numWorkers, {'numeric'}, ...
    {'scalar', 'integer', '>=', 1});

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
supdeqDir = fullfile(projectRoot, 'SUpDEq-master');
hutubsDir = fullfile(projectRoot, 'data', 'HRTF', 'hutubs');
outputRoot = fullfile(scriptDir, outputName);
subjectsRoot = fullfile(outputRoot, 'subjects');

assert(isfolder(supdeqDir), 'SUpDEq directory not found: %s', supdeqDir);
assert(isfolder(hutubsDir), 'HUTUBS directory not found: %s', hutubsDir);
if ~isfolder(subjectsRoot)
    mkdir(subjectsRoot);
end

runConfiguration = struct( ...
    'subjectIds', subjectIds, ...
    'sparseOrders', sparseOrders, ...
    'numWorkers', numWorkers, ...
    'referenceLebedevOrder', 35, ...
    'denseFliegeOrder', 29, ...
    'denseDirectionCount', 900, ...
    'horizontalDirectionCount', 360, ...
    'fftOversize', 4, ...
    'mcaMaximumBoostDb', inf, ...
    'mcaMinimumPhase', true, ...
    'mcaLimitBelowSpatialAliasingFrequency', true, ...
    'mcaAliasingFade', 'fadeDown', ...
    'regionRadiusDegrees', 25, ...
    'itdThresholdDb', -20, ...
    'itdUpsamplingFactor', 10, ...
    'itdLowpassHz', 3000, ...
    'missingAnthropometryPolicy', ...
        'Mean Algazi radius across subjects with public x1/x2/x3 values', ...
    'matlabVersion', version, ...
    'startedAt', datetime('now', 'TimeZone', 'local'));
save(fullfile(outputRoot, 'run_configuration.mat'), 'runConfiguration');

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', min(numWorkers, numel(subjectIds)));
elseif pool.NumWorkers ~= min(numWorkers, numel(subjectIds))
    delete(pool);
    pool = parpool('local', min(numWorkers, numel(subjectIds)));
end
addAttachedFiles(pool, {mfilename('fullpath')});
workerEnvironment = parallel.pool.Constant(@() initialize_worker(supdeqDir));

fprintf('Starting HUTUBS MCA reproduction: %d subjects, %d orders, %d workers.\n', ...
    numel(subjectIds), numel(sparseOrders), pool.NumWorkers);

statusCells = cell(numel(subjectIds), 1);
parfor subjectIndex = 1:numel(subjectIds)
    workerEnvironment.Value;
    subjectId = subjectIds(subjectIndex);
    try
        statusCells{subjectIndex} = process_subject( ...
            subjectId, sparseOrders, hutubsDir, subjectsRoot, runConfiguration);
    catch exception
        statusCells{subjectIndex} = failure_status(subjectId, exception);
        write_failure_report(subjectId, exception, subjectsRoot);
    end
end
delete(workerEnvironment);

statusTable = vertcat(statusCells{:});
writetable(statusTable, fullfile(outputRoot, 'batch_status.csv'));
save(fullfile(outputRoot, 'batch_status.mat'), 'statusTable');

runConfiguration.finishedAt = datetime('now', 'TimeZone', 'local');
runConfiguration.completedSubjectCount = sum(statusTable.Success);
runConfiguration.failedSubjectCount = sum(~statusTable.Success);
save(fullfile(outputRoot, 'run_configuration.mat'), 'runConfiguration');

fprintf('\nHUTUBS MCA batch finished: %d succeeded, %d failed.\n', ...
    runConfiguration.completedSubjectCount, runConfiguration.failedSubjectCount);
fprintf('Output root: %s\n', outputRoot);
disp(statusTable);
end

function environment = initialize_worker(supdeqDir)
originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir);
supdeq_start;
environment = struct('worker', getCurrentTask(), 'initialized', true);
clear restoreDirectory;
end

function status = process_subject(subjectId, sparseOrders, hutubsDir, subjectsRoot, config)
subjectStart = tic;
subjectName = sprintf('pp%d', subjectId);
subjectDir = fullfile(subjectsRoot, subjectName);
ordersDir = fullfile(subjectDir, 'orders');
if ~isfolder(ordersDir)
    mkdir(ordersDir);
end

logFile = fullfile(subjectDir, 'subject.log');
diary(logFile);
stopDiary = onCleanup(@() diary('off'));
fprintf('\n[%s] Starting subject %d.\n', char(datetime('now')), subjectId);

anthropometry = readtable(fullfile(hutubsDir, 'AntrhopometricMeasures.csv'));
[headRadius, headRadiusSource] = resolve_head_radius(anthropometry, subjectId);

sofaFile = fullfile(hutubsDir, sprintf('pp%d_HRIRs_simulated.sofa', subjectId));
assert(isfile(sofaFile), 'Simulated SOFA file not found: %s', sofaFile);
sofa = SOFAload(sofaFile);
assert(strcmp(sofa.GLOBAL_SOFAConventions, 'SimpleFreeFieldHRIR'), ...
    'Unexpected SOFA convention: %s', sofa.GLOBAL_SOFAConventions);

sourceGridCanonical = supdeq_lebedev([], config.referenceLebedevOrder);
sofaGrid = [mod(sofa.SourcePosition(:, 1), 360), 90 - sofa.SourcePosition(:, 2)];
assert(size(sofaGrid, 1) == size(sourceGridCanonical, 1), ...
    'Subject %d does not contain the expected 1730 simulated directions.', subjectId);
azimuthError = abs(mod(sofaGrid(:, 1) - sourceGridCanonical(:, 1) + 180, 360) - 180);
colatitudeError = abs(sofaGrid(:, 2) - sourceGridCanonical(:, 2));
assert(max(azimuthError) <= 1e-8 && max(colatitudeError) <= 1e-8, ...
    'Subject %d source grid does not match Lebedev N=35.', subjectId);
sourceGrid = [sofaGrid, sourceGridCanonical(:, 3)];

referenceSfd = supdeq_sofa2sfd(sofa, config.referenceLebedevOrder, ...
    sourceGrid, config.fftOversize, 'ak');
denseGrid = supdeq_fliege(config.denseFliegeOrder);
horizontalGrid = [(0:359).', 90 * ones(360, 1), ones(360, 1) / 360];
targetGrid = [denseGrid; horizontalGrid];
denseCount = size(denseGrid, 1);
horizontalIndices = denseCount + (1:size(horizontalGrid, 1));

[referenceLeft, referenceRight] = supdeq_getArbHRTF( ...
    referenceSfd, targetGrid, 'DEG', 2, 'ak');
[referenceHrirLeft, referenceHrirRight] = hrtf_to_hrir( ...
    referenceLeft, referenceRight, referenceSfd.FFToversize);

referenceDenseLeft = referenceHrirLeft(:, 1:denseCount);
referenceDenseRight = referenceHrirRight(:, 1:denseCount);
referenceHorizontalLeft = referenceHrirLeft(:, horizontalIndices);
referenceHorizontalRight = referenceHrirRight(:, horizontalIndices);
[referenceIld, referenceItdUs] = horizontal_cues( ...
    referenceHorizontalLeft, referenceHorizontalRight, referenceSfd.fs, config);

directionWeights = denseGrid(:, 3) / sum(denseGrid(:, 3));
frontMask = great_circle_mask(denseGrid, [0, 90], config.regionRadiusDegrees);
leftContraMask = great_circle_mask(denseGrid, [270, 90], config.regionRadiusDegrees);
rightContraMask = great_circle_mask(denseGrid, [90, 90], config.regionRadiusDegrees);

subjectSummary = table();
subjectBinauralSummary = table();
completedOrders = false(size(sparseOrders));
for orderIndex = 1:numel(sparseOrders)
    sparseOrder = sparseOrders(orderIndex);
    orderFile = fullfile(ordersDir, sprintf('n%02d_results.mat', sparseOrder));
    orderSummaryFile = fullfile(ordersDir, sprintf('n%02d_erb_summary.csv', sparseOrder));
    orderBinauralFile = fullfile(ordersDir, sprintf('n%02d_binaural_summary.csv', sparseOrder));

    if isfile(orderFile) && isfile(orderSummaryFile) && isfile(orderBinauralFile)
        loaded = load(orderFile, 'summary', 'binauralSummary');
        subjectSummary = [subjectSummary; loaded.summary]; %#ok<AGROW>
        subjectBinauralSummary = [subjectBinauralSummary; loaded.binauralSummary]; %#ok<AGROW>
        completedOrders(orderIndex) = true;
        fprintf('[%s] Subject %d, N=%d already complete; skipping.\n', ...
            char(datetime('now')), subjectId, sparseOrder);
        continue;
    end

    fprintf('[%s] Subject %d, N=%d started.\n', ...
        char(datetime('now')), subjectId, sparseOrder);
    orderStart = tic;
    sparseGrid = supdeq_lebedev([], sparseOrder);
    [sparseLeft, sparseRight] = supdeq_getArbHRTF( ...
        referenceSfd, sparseGrid, 'DEG', 2, 'ak');
    sparseHRTF = struct( ...
        'HRTF_L', sparseLeft, ...
        'HRTF_R', sparseRight, ...
        'f', referenceSfd.f, ...
        'fs', referenceSfd.fs, ...
        'Nmax', sparseOrder, ...
        'FFToversize', referenceSfd.FFToversize, ...
        'samplingGrid', sparseGrid);

    methods = {'SH', 'Conventional', 'MCA'};
    methodResults = {
        supdeq_interpHRTF(sparseHRTF, targetGrid, 'None', 'SH', nan, headRadius)
        supdeq_interpHRTF(sparseHRTF, targetGrid, 'SUpDEq', 'SH', nan, headRadius)
        supdeq_interpHRTF(sparseHRTF, targetGrid, 'SUpDEq', 'SH', inf, headRadius)};

    summary = table();
    binauralSummary = table();
    frequencyCurves = struct();
    spatialErrors = struct();
    horizontalCues = struct( ...
        'azimuthDegrees', (0:359).', ...
        'referenceILD_dB', referenceIld, ...
        'referenceITD_us', referenceItdUs);

    for methodIndex = 1:numel(methods)
        methodName = methods{methodIndex};
        current = methodResults{methodIndex};
        currentDenseLeft = current.HRIR_L(:, 1:denseCount);
        currentDenseRight = current.HRIR_R(:, 1:denseCount);

        [erbLeft, erbFrequencies] = AKerbError( ...
            currentDenseLeft, referenceDenseLeft, ...
            [50 referenceSfd.fs / 2], referenceSfd.fs);
        erbRight = AKerbError( ...
            currentDenseRight, referenceDenseRight, ...
            [50 referenceSfd.fs / 2], referenceSfd.fs);

        frequencyCurves.(methodName).erbFrequencyHz = erbFrequencies;
        frequencyCurves.(methodName).leftFullErbDb = ...
            weighted_direction_mean(abs(erbLeft), directionWeights);
        frequencyCurves.(methodName).rightFullErbDb = ...
            weighted_direction_mean(abs(erbRight), directionWeights);
        spatialErrors.(methodName).leftMeanErbDb = mean(abs(erbLeft), 1).';
        spatialErrors.(methodName).rightMeanErbDb = mean(abs(erbRight), 1).';

        summary = [summary; make_summary_rows( ...
            subjectId, sparseOrder, methodName, 'Left', abs(erbLeft), ...
            directionWeights, frontMask, leftContraMask); ...
            make_summary_rows( ...
            subjectId, sparseOrder, methodName, 'Right', abs(erbRight), ...
            directionWeights, frontMask, rightContraMask)]; %#ok<AGROW>

        currentHorizontalLeft = current.HRIR_L(:, horizontalIndices);
        currentHorizontalRight = current.HRIR_R(:, horizontalIndices);
        [currentIld, currentItdUs] = horizontal_cues( ...
            currentHorizontalLeft, currentHorizontalRight, referenceSfd.fs, config);
        ildError = currentIld - referenceIld;
        itdErrorUs = currentItdUs - referenceItdUs;
        horizontalCues.(methodName).ILD_dB = currentIld;
        horizontalCues.(methodName).ITD_us = currentItdUs;
        horizontalCues.(methodName).ILDError_dB = ildError;
        horizontalCues.(methodName).ITDError_us = itdErrorUs;
        binauralSummary = [binauralSummary; table( ...
            subjectId, sparseOrder, string(methodName), ...
            mean(abs(ildError)), max(abs(ildError)), ...
            mean(abs(itdErrorUs)), max(abs(itdErrorUs)), ...
            'VariableNames', {'SubjectID', 'SparseOrder', 'Method', ...
            'MeanAbsoluteILDError_dB', 'MaxAbsoluteILDError_dB', ...
            'MeanAbsoluteITDError_us', 'MaxAbsoluteITDError_us'})]; %#ok<AGROW>
    end

    orderConfiguration = struct( ...
        'subjectId', subjectId, ...
        'sparseOrder', sparseOrder, ...
        'sparseDirectionCount', size(sparseGrid, 1), ...
        'headRadiusM', headRadius, ...
        'headRadiusCm', 100 * headRadius, ...
        'headRadiusSource', headRadiusSource, ...
        'samplingRateHz', referenceSfd.fs, ...
        'spatialAliasingFrequencyHz', ...
            sparseOrder * 343 / (2 * pi * headRadius), ...
        'elapsedSeconds', toc(orderStart), ...
        'completedAt', datetime('now', 'TimeZone', 'local'));
    save(orderFile, 'orderConfiguration', 'summary', 'binauralSummary', ...
        'frequencyCurves', 'spatialErrors', 'horizontalCues', ...
        'denseGrid', 'horizontalGrid', '-v7.3');
    writetable(summary, orderSummaryFile);
    writetable(binauralSummary, orderBinauralFile);

    subjectSummary = [subjectSummary; summary]; %#ok<AGROW>
    subjectBinauralSummary = [subjectBinauralSummary; binauralSummary]; %#ok<AGROW>
    completedOrders(orderIndex) = true;
    fprintf('[%s] Subject %d, N=%d complete in %.1f seconds.\n', ...
        char(datetime('now')), subjectId, sparseOrder, orderConfiguration.elapsedSeconds);
    clear methodResults;
end

writetable(subjectSummary, fullfile(subjectDir, 'subject_erb_summary.csv'));
writetable(subjectBinauralSummary, fullfile(subjectDir, 'subject_binaural_summary.csv'));
subjectConfiguration = struct( ...
    'subjectId', subjectId, ...
    'sourceFile', sofaFile, ...
    'headRadiusM', headRadius, ...
    'headRadiusCm', 100 * headRadius, ...
    'headRadiusSource', headRadiusSource, ...
    'requestedOrders', sparseOrders, ...
    'completedOrders', sparseOrders(completedOrders), ...
    'sourceGridMaxAzimuthMismatchDeg', max(azimuthError), ...
    'sourceGridMaxColatitudeMismatchDeg', max(colatitudeError), ...
    'elapsedSeconds', toc(subjectStart), ...
    'completedAt', datetime('now', 'TimeZone', 'local'));
save(fullfile(subjectDir, 'subject_complete.mat'), ...
    'subjectConfiguration', 'subjectSummary', 'subjectBinauralSummary');

fprintf('[%s] Subject %d finished in %.1f seconds.\n', ...
    char(datetime('now')), subjectId, subjectConfiguration.elapsedSeconds);
status = table(subjectId, true, numel(sparseOrders), ...
    subjectConfiguration.elapsedSeconds, "", ...
    'VariableNames', {'SubjectID', 'Success', 'CompletedOrderCount', ...
    'ElapsedSeconds', 'ErrorMessage'});
clear stopDiary;
end

function status = failure_status(subjectId, exception)
status = table(subjectId, false, 0, NaN, string(exception.message), ...
    'VariableNames', {'SubjectID', 'Success', 'CompletedOrderCount', ...
    'ElapsedSeconds', 'ErrorMessage'});
end

function write_failure_report(subjectId, exception, subjectsRoot)
subjectDir = fullfile(subjectsRoot, sprintf('pp%d', subjectId));
if ~isfolder(subjectDir)
    mkdir(subjectDir);
end
failure = struct( ...
    'subjectId', subjectId, ...
    'message', exception.message, ...
    'identifier', exception.identifier, ...
    'stack', exception.stack, ...
    'failedAt', datetime('now', 'TimeZone', 'local'));
save(fullfile(subjectDir, 'failure.mat'), 'failure');
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
dotProduct = sind(colatitude) .* sind(center(2)) .* ...
    cosd(azimuth - center(1)) + cosd(colatitude) .* cosd(center(2));
distanceDegrees = acosd(min(1, max(-1, dotProduct)));
mask = distanceDegrees <= radiusDegrees + 1e-10;
end

function meanByFrequency = weighted_direction_mean(errorDb, weights)
meanByFrequency = errorDb * weights(:);
end

function rows = make_summary_rows(subjectId, sparseOrder, method, ear, ...
        errorDb, weights, frontMask, contraMask)
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

function [ild, itdUs] = horizontal_cues(leftHrir, rightHrir, fs, config)
leftEnergy = sum(abs(leftHrir).^2, 1);
rightEnergy = sum(abs(rightHrir).^2, 1);
ild = 10 * log10(leftEnergy ./ rightEnergy).';
itdUs = zeros(size(leftHrir, 2), 1);
for directionIndex = 1:size(leftHrir, 2)
    [~, itdUs(directionIndex)] = supdeq_calcITD( ...
        [leftHrir(:, directionIndex), rightHrir(:, directionIndex)], ...
        fs, config.itdThresholdDb, config.itdUpsamplingFactor, config.itdLowpassHz);
end
end

function [headRadius, source] = resolve_head_radius(anthropometry, subjectId)
subjectRow = anthropometry(anthropometry.SubjectID == subjectId, :);
assert(height(subjectRow) == 1, ...
    'Expected exactly one anthropometry row for subject %d.', subjectId);
if all(isfinite([subjectRow.x1, subjectRow.x2, subjectRow.x3]))
    headRadius = supdeq_optRadius( ...
        subjectRow.x1 / 100, subjectRow.x2 / 100, subjectRow.x3 / 100, 'Algazi');
    source = "Individual public x1/x2/x3";
    return;
end

validMask = isfinite(anthropometry.x1) & ...
    isfinite(anthropometry.x2) & isfinite(anthropometry.x3);
validRows = anthropometry(validMask, :);
validRadii = arrayfun(@(rowIndex) supdeq_optRadius( ...
    validRows.x1(rowIndex) / 100, validRows.x2(rowIndex) / 100, ...
    validRows.x3(rowIndex) / 100, 'Algazi'), (1:height(validRows)).');
headRadius = mean(validRadii);
source = sprintf("Imputed mean Algazi radius from %d public subjects", ...
    height(validRows));
warning('HUTUBS:MissingAnthropometry', ...
    ['Subject %d has no public x1/x2/x3 values; using the mean ', ...
    'Algazi radius %.6f m from %d subjects.'], ...
    subjectId, headRadius, height(validRows));
end
