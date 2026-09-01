function prepare_sonicom_learned_sparsity(configFile, subjectLimit, numWorkers)
%PREPARE_SONICOM_LEARNED_SPARSITY Build MCAR inputs for configured nested Q.

if nargin < 1 || isempty(configFile)
    configFile = fullfile('configs', 'experiments', ...
        'sonicom_learned_methods_sparsity_v1.json');
end
if nargin < 2 || isempty(subjectLimit)
    subjectLimit = inf;
end
if nargin < 3 || isempty(numWorkers)
    numWorkers = 4;
end
validateattributes(subjectLimit, {'numeric'}, {'scalar', 'positive'});
validateattributes(numWorkers, {'numeric'}, {'scalar', 'integer', 'positive'});

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
configPath = resolve_path(projectRoot, configFile);
config = jsondecode(fileread(configPath));
assert(any(string(config.status) == ["preregistered", "running", "completed"]), ...
    'Unexpected experiment status.');
splitTable = readtable(fullfile(projectRoot, config.dataset.split_file), ...
    'TextType', 'string');
splitRows = splitTable(splitTable.split == string(config.dataset.split), :);
assert(height(splitRows) == config.dataset.subject_count, ...
    'Frozen SONICOM subject count mismatch.');
splitRows = splitRows(1:min(height(splitRows), floor(subjectLimit)), :);
gridTable = readtable(fullfile(projectRoot, config.dataset.sparse_grid_file), ...
    'TextType', 'string');
referenceTable = readtable(fullfile(projectRoot, config.dataset.reference_grid), ...
    'TextType', 'string');
counts = double(config.dataset.direction_counts(:).');
expectedSplit = string(config.dataset.split);
assert(~isempty(counts) && all(counts > 0) && ...
    all(counts == floor(counts)) && issorted(counts) && ...
    height(referenceTable) == config.dataset.reference_direction_count, ...
    'Configured sparsity or reference grid is invalid.');
for count = counts
    rows = gridTable(gridTable.direction_count == count, :);
    assert(height(rows) == count && numel(unique(rows.source_index_zero_based)) == count, ...
        'Invalid Q%d grid.', count);
end

supdeqDir = fullfile(projectRoot, 'external', 'SUpDEq');
sofaRoot = fullfile(projectRoot, 'data', 'HRTF', ...
    'sonicom_measured_ffcmp_minphase_44k1', 'subjects');
outputRoot = fullfile(projectRoot, 'artifacts', 'sparsity', config.output_name);
subjectsRoot = fullfile(outputRoot, 'subjects');
if ~isfolder(subjectsRoot)
    mkdir(subjectsRoot);
end

requestedWorkers = min([numWorkers, height(splitRows)]);
pool = gcp('nocreate');
if requestedWorkers > 1
    if isempty(pool)
        parpool('local', requestedWorkers);
    elseif pool.NumWorkers ~= requestedWorkers
        delete(pool);
        parpool('local', requestedWorkers);
    end
    workerEnvironment = parallel.pool.Constant(@() initialize_worker(supdeqDir));
else
    initialize_worker(supdeqDir);
    workerEnvironment = [];
end

statusCells = cell(height(splitRows), 1);
if requestedWorkers > 1
    parfor subjectIndex = 1:height(splitRows)
        workerEnvironment.Value;
        statusCells{subjectIndex} = process_subject(splitRows(subjectIndex, :), ...
            counts, gridTable, referenceTable, sofaRoot, subjectsRoot, ...
            expectedSplit);
    end
    delete(workerEnvironment);
else
    for subjectIndex = 1:height(splitRows)
        statusCells{subjectIndex} = process_subject(splitRows(subjectIndex, :), ...
            counts, gridTable, referenceTable, sofaRoot, subjectsRoot, ...
            expectedSplit);
    end
end
status = vertcat(statusCells{:});
writetable(status, fullfile(outputRoot, 'preparation_status.csv'));
assert(all(status.Success), 'At least one SONICOM sparsity input failed.');
fprintf('Prepared %d SONICOM subjects x %d sparsity levels.\n', ...
    height(splitRows), numel(counts));
end

function environment = initialize_worker(supdeqDir)
originalDir = pwd;
restoreDirectory = onCleanup(@() cd(originalDir));
cd(supdeqDir);
supdeq_start;
environment = struct('initialized', true);
clear restoreDirectory;
end

function status = process_subject(splitRow, counts, gridTable, ...
        referenceTable, sofaRoot, subjectsRoot, expectedSplit)
subjectLabel = splitRow.subject_id;
subjectRoot = fullfile(subjectsRoot, char(subjectLabel));
if ~isfolder(subjectRoot)
    mkdir(subjectRoot);
end
sofaPath = fullfile(sofaRoot, sprintf( ...
    '%s_FreeFieldCompMinPhase_44kHz.sofa', subjectLabel));
sofa = SOFAload(sofaPath);
assert(size(sofa.Data.IR, 1) == 793 && size(sofa.Data.IR, 2) == 2, ...
    'Unexpected SONICOM HRIR shape for %s.', subjectLabel);
samplingRateHz = double(sofa.Data.SamplingRate);
nfft = 1024;
fftOversize = 4;
referenceSpectrum = fft(double(sofa.Data.IR), nfft, 3);
referenceLeft = squeeze(referenceSpectrum(:, 1, 1:(nfft / 2 + 1)));
referenceRight = squeeze(referenceSpectrum(:, 2, 1:(nfft / 2 + 1)));
referenceHrir = permute(double(sofa.Data.IR), [3, 1, 2]);
referenceGrid = [double(referenceTable.azimuth_deg), ...
    double(referenceTable.colatitude_deg), ...
    double(referenceTable.solid_angle_weight)];
frequencyHz = (0:(nfft / 2)).' * samplingRateHz / nfft;
frequencyMask = frequencyHz >= 50 & frequencyHz <= 20000;
directionFeatures = build_direction_features(referenceGrid);
rows = table();

for count = counts
    started = tic;
    levelRoot = fullfile(subjectRoot, sprintf('q%d', count));
    if ~isfolder(levelRoot)
        mkdir(levelRoot);
    end
    cacheFile = fullfile(levelRoot, 'cache.mat');
    inputFile = fullfile(levelRoot, 'model_input.h5');
    try
        if isfile(cacheFile) && is_complete_hdf5(inputFile, count)
            rows = [rows; status_row(subjectLabel, count, true, toc(started), ...
                "skipped")]; %#ok<AGROW>
            continue;
        end
        sparseRows = gridTable(gridTable.direction_count == count, :);
        sourceIndices = double(sparseRows.source_index_zero_based) + 1;
        sparseGrid = [double(sparseRows.azimuth_deg), ...
            double(sparseRows.colatitude_deg)];
        sparseHrtf = struct('HRTF_L', referenceLeft(sourceIndices, :), ...
            'HRTF_R', referenceRight(sourceIndices, :), 'f', frequencyHz, ...
            'fs', samplingRateHz, 'Nmax', 3, 'FFToversize', fftOversize, ...
            'samplingGrid', sparseGrid);
        mca = supdeq_interpHRTF(sparseHrtf, referenceGrid, ...
            'SUpDEq', 'SH', inf, 0.09, 1e-2, true, 0, true, 'fadeDown');
        [mcaDb, correctionDb] = model_tensors(mca, frequencyMask);
        cache = struct('schemaVersion', '1.0', ...
            'subjectLabel', subjectLabel, 'split', splitRow.split, ...
            'directionCount', count, 'sourceIndices', sourceIndices, ...
            'frequencyHz', frequencyHz, 'frequencyMask', frequencyMask, ...
            'referenceGrid', referenceGrid, ...
            'referenceLeft', single(referenceLeft), ...
            'referenceRight', single(referenceRight), ...
            'referenceHrir', single(referenceHrir), ...
            'mcaLeft', single(mca.HRTF_L), 'mcaRight', single(mca.HRTF_R), ...
            'samplingRateHz', samplingRateHz, 'fftOversize', fftOversize);
        temporaryCache = [cacheFile '.partial'];
        save(temporaryCache, 'cache', '-v7.3');
        movefile(temporaryCache, cacheFile, 'f');
        write_model_input(inputFile, mcaDb, correctionDb, ...
            directionFeatures, frequencyHz(frequencyMask), subjectLabel, ...
            count, expectedSplit);
        rows = [rows; status_row(subjectLabel, count, true, toc(started), ...
            "")]; %#ok<AGROW>
    catch exception
        rows = [rows; status_row(subjectLabel, count, false, toc(started), ...
            string(exception.message))]; %#ok<AGROW>
    end
end
status = rows;
end

function [mcaDb, correctionDb] = model_tensors(mca, frequencyMask)
mcaSpectrum = cat(3, mca.HRTF_L(:, frequencyMask).', ...
    mca.HRTF_R(:, frequencyMask).');
correctionSpectrum = mca.p.corrFilt_lim(frequencyMask, :, :);
mcaDb = single(20 * log10(max(abs(mcaSpectrum), 1e-10)));
correctionDb = single(20 * log10(max(abs(correctionSpectrum), 1e-10)));
assert(isequal(size(mcaDb), size(correctionDb)) && ...
    all(isfinite(mcaDb), 'all') && all(isfinite(correctionDb), 'all'), ...
    'Invalid MCAR model tensors.');
end

function features = build_direction_features(grid)
azimuth = grid(:, 1);
elevation = 90 - grid(:, 2);
features = single([azimuth.'; elevation.'; ...
    (cosd(elevation) .* cosd(azimuth)).'; ...
    (cosd(elevation) .* sind(azimuth)).'; sind(elevation).'; grid(:, 3).']);
end

function write_model_input(path, mcaDb, correctionDb, directions, ...
        frequencyHz, subjectLabel, count, split)
partial = [path '.partial'];
delete_if_present(partial);
delete_if_present(path);
write_spectral(partial, '/mca_logmag_db', mcaDb);
write_spectral(partial, '/correction_logmag_db', correctionDb);
h5create(partial, '/direction_features', size(directions), ...
    'Datatype', 'single', 'ChunkSize', [6, 128], 'Deflate', 4);
h5write(partial, '/direction_features', directions);
frequencyHz = single(frequencyHz(:).');
h5create(partial, '/frequency_hz', size(frequencyHz), 'Datatype', 'single');
h5write(partial, '/frequency_hz', frequencyHz);
h5writeatt(partial, '/', 'complete', int32(1));
h5writeatt(partial, '/', 'subject_id', ...
    int32(str2double(extractAfter(subjectLabel, 1))));
h5writeatt(partial, '/', 'subject_label', char(subjectLabel));
h5writeatt(partial, '/', 'split', char(split));
h5writeatt(partial, '/', 'sparse_order', int32(3));
h5writeatt(partial, '/', 'sparse_direction_count', int32(count));
movefile(partial, path, 'f');
end

function write_spectral(path, dataset, values)
h5create(path, dataset, size(values), 'Datatype', 'single', ...
    'ChunkSize', [64, 128, 1], 'Deflate', 4);
h5write(path, dataset, values);
end

function complete = is_complete_hdf5(path, count)
complete = false;
if ~isfile(path)
    return;
end
try
    complete = h5readatt(path, '/', 'complete') == 1 && ...
        h5readatt(path, '/', 'sparse_direction_count') == count;
catch
    complete = false;
end
end

function row = status_row(subjectLabel, count, success, elapsed, message)
row = table(string(subjectLabel), count, success, elapsed, string(message), ...
    'VariableNames', {'SubjectLabel', 'DirectionCount', 'Success', ...
    'ElapsedSeconds', 'Message'});
end

function path = resolve_path(projectRoot, path)
path = char(path);
if ~isfile(path)
    path = fullfile(projectRoot, path);
end
assert(isfile(path), 'Configuration not found: %s', path);
end

function delete_if_present(path)
if isfile(path)
    delete(path);
end
end
