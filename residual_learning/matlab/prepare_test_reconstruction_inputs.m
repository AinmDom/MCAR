function prepare_test_reconstruction_inputs(numWorkers, outputName)
%PREPARE_TEST_RECONSTRUCTION_INPUTS Cache test-set MCA data for inference.
%
% The cache contains the complex MCA/reference HRTFs required to preserve
% MCA phase after residual magnitude correction. The HDF5 model input covers
% the 900-point Fliege evaluation grid plus a 360-point horizontal ring.
%
% Defaults:
%   numWorkers = 4
%   outputName = 'mlp_n03_v1'

if nargin < 1 || isempty(numWorkers)
    numWorkers = 4;
end
if nargin < 2 || isempty(outputName)
    outputName = 'mlp_n03_v1';
end
validateattributes(numWorkers, {'numeric'}, ...
    {'scalar', 'integer', '>=', 1});
validateattributes(outputName, {'char', 'string'}, {'scalartext'});
outputName = char(outputName);

scriptDir = fileparts(mfilename('fullpath'));
stageRoot = fileparts(scriptDir);
projectRoot = fileparts(stageRoot);
supdeqDir = fullfile(projectRoot, 'SUpDEq-master');
hutubsDir = fullfile(projectRoot, 'data', 'HRTF', 'hutubs');
splitFile = fullfile(stageRoot, 'config', 'subject_split_v1.csv');
outputRoot = fullfile(stageRoot, 'reconstruction', outputName);
subjectsRoot = fullfile(outputRoot, 'subjects');

assert(isfolder(supdeqDir), 'SUpDEq directory not found: %s', supdeqDir);
assert(isfolder(hutubsDir), 'HUTUBS directory not found: %s', hutubsDir);
assert(isfile(splitFile), 'Subject split file not found: %s', splitFile);
if ~isfolder(subjectsRoot)
    mkdir(subjectsRoot);
end

splitTable = readtable(splitFile, 'TextType', 'string');
testSubjects = splitTable.subject_id(splitTable.split == "test").';
assert(numel(testSubjects) == 12, 'Expected exactly 12 test subjects.');

configuration = struct( ...
    'subjectIds', testSubjects, ...
    'sparseOrder', 3, ...
    'referenceLebedevOrder', 35, ...
    'denseFliegeOrder', 29, ...
    'denseDirectionCount', 900, ...
    'horizontalDirectionCount', 360, ...
    'fftOversize', 4, ...
    'frequencyMinimumHz', 50, ...
    'frequencyMaximumHz', 20000, ...
    'magnitudeFloorDb', -200, ...
    'numWorkers', min(numWorkers, numel(testSubjects)), ...
    'matlabVersion', version, ...
    'startedAt', datetime('now', 'TimeZone', 'local'));
save(fullfile(outputRoot, 'preparation_configuration.mat'), 'configuration');

statusCells = cell(numel(testSubjects), 1);
if numWorkers == 1
    initialize_worker(supdeqDir);
    for subjectIndex = 1:numel(testSubjects)
        statusCells{subjectIndex} = prepare_subject( ...
            testSubjects(subjectIndex), hutubsDir, subjectsRoot, configuration);
    end
else
    requestedWorkers = min(numWorkers, numel(testSubjects));
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local', requestedWorkers);
    elseif pool.NumWorkers ~= requestedWorkers
        delete(pool);
        pool = parpool('local', requestedWorkers);
    end
    addAttachedFiles(pool, {mfilename('fullpath')});
    workerEnvironment = parallel.pool.Constant( ...
        @() initialize_worker(supdeqDir));
    parfor subjectIndex = 1:numel(testSubjects)
        workerEnvironment.Value;
        statusCells{subjectIndex} = prepare_subject( ...
            testSubjects(subjectIndex), hutubsDir, subjectsRoot, configuration);
    end
    delete(workerEnvironment);
end

statusTable = vertcat(statusCells{:});
writetable(statusTable, fullfile(outputRoot, 'preparation_status.csv'));
configuration.finishedAt = datetime('now', 'TimeZone', 'local');
configuration.completedSubjectCount = sum(statusTable.Success);
configuration.failedSubjectCount = sum(~statusTable.Success);
save(fullfile(outputRoot, 'preparation_configuration.mat'), 'configuration');
fprintf('Prepared %d/%d test subjects under %s.\n', ...
    configuration.completedSubjectCount, numel(testSubjects), outputRoot);
disp(statusTable);
end

function environment = initialize_worker(supdeqDir)
originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir);
supdeq_start;
environment = struct('initialized', true);
clear restoreDirectory;
end

function status = prepare_subject(subjectId, hutubsDir, subjectsRoot, config)
started = tic;
subjectDir = fullfile(subjectsRoot, sprintf('pp%d', subjectId));
if ~isfolder(subjectDir)
    mkdir(subjectDir);
end
cacheFile = fullfile(subjectDir, 'complex_cache.mat');
inputFile = fullfile(subjectDir, 'model_input.h5');
try
    if isfile(cacheFile) && is_complete_hdf5(inputFile)
        fprintf('Subject %d reconstruction input already complete; skipping.\n', ...
            subjectId);
        status = make_status(subjectId, true, toc(started), "Skipped complete");
        return;
    end

    anthropometry = readtable(fullfile(hutubsDir, 'AntrhopometricMeasures.csv'));
    [headRadius, headRadiusSource] = resolve_head_radius( ...
        anthropometry, subjectId);
    sofaFile = fullfile(hutubsDir, sprintf( ...
        'pp%d_HRIRs_simulated.sofa', subjectId));
    assert(isfile(sofaFile), 'Simulated SOFA not found: %s', sofaFile);
    sofa = SOFAload(sofaFile);

    sourceGridCanonical = supdeq_lebedev([], config.referenceLebedevOrder);
    sofaGrid = [mod(sofa.SourcePosition(:, 1), 360), ...
        90 - sofa.SourcePosition(:, 2)];
    azimuthError = abs(mod(sofaGrid(:, 1) - ...
        sourceGridCanonical(:, 1) + 180, 360) - 180);
    colatitudeError = abs(sofaGrid(:, 2) - sourceGridCanonical(:, 2));
    assert(max(azimuthError) <= 1e-8 && max(colatitudeError) <= 1e-8, ...
        'Subject %d source grid does not match Lebedev N=35.', subjectId);
    sourceGrid = [sofaGrid, sourceGridCanonical(:, 3)];
    referenceSfd = supdeq_sofa2sfd( ...
        sofa, config.referenceLebedevOrder, sourceGrid, ...
        config.fftOversize, 'ak');

    denseGrid = supdeq_fliege(config.denseFliegeOrder);
    horizontalGrid = [(0:359).', 90 * ones(360, 1), ...
        ones(360, 1) / 360];
    targetGrid = [denseGrid; horizontalGrid];
    [referenceLeft, referenceRight] = supdeq_getArbHRTF( ...
        referenceSfd, targetGrid, 'DEG', 2, 'ak');

    sparseGrid = supdeq_lebedev([], config.sparseOrder);
    [sparseLeft, sparseRight] = supdeq_getArbHRTF( ...
        referenceSfd, sparseGrid, 'DEG', 2, 'ak');
    sparseHrtf = struct( ...
        'HRTF_L', sparseLeft, ...
        'HRTF_R', sparseRight, ...
        'f', referenceSfd.f, ...
        'fs', referenceSfd.fs, ...
        'Nmax', config.sparseOrder, ...
        'FFToversize', referenceSfd.FFToversize, ...
        'samplingGrid', sparseGrid);
    mca = supdeq_interpHRTF( ...
        sparseHrtf, targetGrid, 'SUpDEq', 'SH', inf, headRadius);

    frequencyHz = referenceSfd.f(:);
    frequencyMask = frequencyHz >= config.frequencyMinimumHz & ...
        frequencyHz <= config.frequencyMaximumHz;
    [mcaDb, correctionDb] = model_spectral_tensors( ...
        mca, frequencyMask, config.magnitudeFloorDb);
    directionFeatures = build_direction_features(targetGrid);

    cache = struct( ...
        'schemaVersion', '1.0', ...
        'subjectId', subjectId, ...
        'sparseOrder', config.sparseOrder, ...
        'headRadiusM', headRadius, ...
        'headRadiusSource', headRadiusSource, ...
        'samplingRateHz', referenceSfd.fs, ...
        'fftOversize', referenceSfd.FFToversize, ...
        'frequencyHz', frequencyHz, ...
        'frequencyMask', frequencyMask, ...
        'denseGrid', denseGrid, ...
        'horizontalGrid', horizontalGrid, ...
        'referenceLeft', single(referenceLeft), ...
        'referenceRight', single(referenceRight), ...
        'mcaLeft', single(mca.HRTF_L), ...
        'mcaRight', single(mca.HRTF_R), ...
        'createdAt', datetime('now', 'TimeZone', 'local'));
    temporaryCache = [cacheFile '.partial'];
    if isfile(temporaryCache)
        delete(temporaryCache);
    end
    save(temporaryCache, 'cache', '-v7.3');
    movefile(temporaryCache, cacheFile, 'f');

    write_model_input(inputFile, mcaDb, correctionDb, ...
        directionFeatures, frequencyHz(frequencyMask), subjectId, config);
    status = make_status(subjectId, true, toc(started), "");
    fprintf('Subject %d reconstruction input prepared in %.1f s.\n', ...
        subjectId, status.ElapsedSeconds);
catch exception
    failure = struct('subjectId', subjectId, ...
        'message', exception.message, 'identifier', exception.identifier, ...
        'stack', exception.stack, ...
        'failedAt', datetime('now', 'TimeZone', 'local'));
    save(fullfile(subjectDir, 'preparation_failure.mat'), 'failure');
    status = make_status(subjectId, false, toc(started), string(exception.message));
    fprintf(2, 'Subject %d preparation failed: %s\n', ...
        subjectId, exception.message);
end
end

function [mcaDb, correctionDb] = model_spectral_tensors( ...
        mca, frequencyMask, magnitudeFloorDb)
magnitudeFloor = 10 ^ (magnitudeFloorDb / 20);
mcaSpectrum = cat(3, ...
    mca.HRTF_L(:, frequencyMask).', ...
    mca.HRTF_R(:, frequencyMask).');
correctionSpectrum = mca.p.corrFilt_lim(frequencyMask, :, :);
assert(isequal(size(mcaSpectrum), size(correctionSpectrum)), ...
    'MCA and correction tensors have different dimensions.');
mcaDb = single(20 * log10(max(abs(mcaSpectrum), magnitudeFloor)));
correctionDb = single(20 * log10(max( ...
    abs(correctionSpectrum), magnitudeFloor)));
end

function features = build_direction_features(grid)
azimuthDegrees = grid(:, 1);
elevationDegrees = 90 - grid(:, 2);
x = cosd(elevationDegrees) .* cosd(azimuthDegrees);
y = cosd(elevationDegrees) .* sind(azimuthDegrees);
z = sind(elevationDegrees);
features = single([azimuthDegrees.'; elevationDegrees.'; ...
    x.'; y.'; z.'; grid(:, 3).']);
end

function write_model_input(outputFile, mcaDb, correctionDb, ...
        directionFeatures, frequencyHz, subjectId, config)
partialFile = [outputFile '.partial'];
if isfile(partialFile)
    delete(partialFile);
end
if isfile(outputFile)
    delete(outputFile);
end
cleanupPartial = onCleanup(@() delete_if_present(partialFile));

write_spectral_dataset(partialFile, '/mca_logmag_db', mcaDb);
write_spectral_dataset(partialFile, '/correction_logmag_db', correctionDb);
h5create(partialFile, '/direction_features', size(directionFeatures), ...
    'Datatype', 'single', 'ChunkSize', [6, 128], 'Deflate', 4);
h5write(partialFile, '/direction_features', directionFeatures);
frequencyHz = single(frequencyHz(:).');
h5create(partialFile, '/frequency_hz', size(frequencyHz), ...
    'Datatype', 'single');
h5write(partialFile, '/frequency_hz', frequencyHz);

h5writeatt(partialFile, '/', 'schema_version', '1.0');
h5writeatt(partialFile, '/', 'complete', int32(1));
h5writeatt(partialFile, '/', 'subject_id', int32(subjectId));
h5writeatt(partialFile, '/', 'sparse_order', int32(config.sparseOrder));
h5writeatt(partialFile, '/', 'dense_direction_count', ...
    int32(config.denseDirectionCount));
h5writeatt(partialFile, '/', 'horizontal_direction_count', ...
    int32(config.horizontalDirectionCount));
h5writeatt(partialFile, '/', 'python_spectral_layout', ...
    'ear,direction,frequency');
h5writeatt(partialFile, '/', 'direction_feature_order', ...
    'azimuth_deg,elevation_deg,x,y,z,grid_weight');
movefile(partialFile, outputFile, 'f');
clear cleanupPartial;
end

function write_spectral_dataset(file, path, values)
valueSize = size(values);
h5create(file, path, valueSize, 'Datatype', 'single', ...
    'ChunkSize', [min(64, valueSize(1)), ...
    min(128, valueSize(2)), 1], 'Deflate', 4);
h5write(file, path, values);
end

function complete = is_complete_hdf5(file)
complete = false;
if ~isfile(file)
    return;
end
try
    complete = h5readatt(file, '/', 'complete') == 1;
catch
    complete = false;
end
end

function delete_if_present(file)
if isfile(file)
    delete(file);
end
end

function status = make_status(subjectId, success, elapsedSeconds, errorMessage)
status = table(subjectId, success, elapsedSeconds, string(errorMessage), ...
    'VariableNames', {'SubjectID', 'Success', 'ElapsedSeconds', ...
    'ErrorMessage'});
end

function [headRadius, source] = resolve_head_radius(anthropometry, subjectId)
subjectRow = anthropometry(anthropometry.SubjectID == subjectId, :);
assert(height(subjectRow) == 1, ...
    'Expected exactly one anthropometry row for subject %d.', subjectId);
if all(isfinite([subjectRow.x1, subjectRow.x2, subjectRow.x3]))
    headRadius = supdeq_optRadius( ...
        subjectRow.x1 / 100, subjectRow.x2 / 100, ...
        subjectRow.x3 / 100, 'Algazi');
    source = "Individual public x1/x2/x3";
    return;
end
validMask = isfinite(anthropometry.x1) & ...
    isfinite(anthropometry.x2) & isfinite(anthropometry.x3);
validRows = anthropometry(validMask, :);
validRadii = arrayfun(@(rowIndex) supdeq_optRadius( ...
    validRows.x1(rowIndex) / 100, validRows.x2(rowIndex) / 100, ...
    validRows.x3(rowIndex) / 100, 'Algazi'), ...
    (1:height(validRows)).');
headRadius = mean(validRadii);
source = sprintf( ...
    "Imputed mean Algazi radius from %d public subjects", height(validRows));
warning('HUTUBS:MissingAnthropometry', ...
    'Subject %d uses imputed head radius %.6f m.', subjectId, headRadius);
end
