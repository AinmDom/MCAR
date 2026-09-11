function prepare_ari_fsc_split(preparedRoot, outputRoot, split)
%PREPARE_ARI_FSC_SPLIT Batch ARI train/validation MCA feature preparation.
% Test is never an accepted value. Existing complete output files are skipped.

split = string(split);
assert(any(split == ["train", "val"]), ...
    'TEST ACCESS REFUSED: ARI MCA batch preparation accepts only train or val.');
inputDirectory = fullfile(char(preparedRoot), char(split));
destinationDirectory = fullfile(char(outputRoot), char(split));
assert(isfolder(inputDirectory), 'Prepared split directory not found: %s', inputDirectory);
if ~isfolder(destinationDirectory), mkdir(destinationDirectory); end
files = dir(fullfile(inputDirectory, 'nh*.h5'));
assert(~isempty(files), 'No prepared ARI subjects found for split %s.', split);
status = table('Size', [numel(files), 4], ...
    'VariableTypes', {'string', 'logical', 'double', 'string'}, ...
    'VariableNames', {'SubjectLabel', 'Success', 'ElapsedSeconds', 'Message'});
for index = 1:numel(files)
    source = fullfile(files(index).folder, files(index).name);
    destination = fullfile(destinationDirectory, files(index).name);
    status.SubjectLabel(index) = erase(string(files(index).name), '.h5');
    started = tic;
    try
        if isfile(destination) && h5readatt(destination, '/', 'complete') == 1
            status.Success(index) = true;
            status.Message(index) = "skipped_existing_complete";
        else
            mcar.prepare_ari_fsc_subject(source, destination);
            status.Success(index) = true;
            status.Message(index) = "created";
        end
    catch exception
        status.Success(index) = false;
        status.Message(index) = string(exception.message);
    end
    status.ElapsedSeconds(index) = toc(started);
    writetable(status(1:index, :), fullfile(destinationDirectory, 'preparation_status.csv'));
    if ~status.Success(index)
        error('ARI feature preparation failed for %s: %s', status.SubjectLabel(index), status.Message(index));
    end
end
end
