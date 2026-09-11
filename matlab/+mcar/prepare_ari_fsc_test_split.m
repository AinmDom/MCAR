function prepare_ari_fsc_test_split(preparedRoot, outputRoot)
%PREPARE_ARI_FSC_TEST_SPLIT Locked derived-feature preparation for ARI test.
inputDirectory = char(preparedRoot);
destinationDirectory = char(outputRoot);
assert(isfolder(inputDirectory), 'Prepared locked-test directory not found.');
assert(~isfolder(destinationDirectory), 'Refusing to overwrite locked-test features.');
mkdir(destinationDirectory);
files = dir(fullfile(inputDirectory, 'nh*.h5'));
assert(numel(files) == 22, 'Expected exactly 22 frozen ARI test subjects.');
status = table('Size',[22,4], 'VariableTypes',{'string','logical','double','string'}, ...
    'VariableNames',{'SubjectLabel','Success','ElapsedSeconds','Message'});
for index = 1:22
    source = fullfile(files(index).folder, files(index).name);
    destination = fullfile(destinationDirectory, files(index).name);
    status.SubjectLabel(index) = erase(string(files(index).name), '.h5');
    started = tic;
    try
        mcar.prepare_ari_fsc_test_subject(source, destination);
        status.Success(index) = true; status.Message(index) = "created";
    catch exception
        status.Message(index) = string(exception.message);
    end
    status.ElapsedSeconds(index) = toc(started);
    writetable(status(1:index,:), fullfile(destinationDirectory,'preparation_status.csv'));
    assert(status.Success(index), 'ARI test feature preparation failed for %s: %s', ...
        status.SubjectLabel(index), status.Message(index));
end
end
